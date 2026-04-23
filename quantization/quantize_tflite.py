"""
Chuyển model Keras CNN 2D sang TFLite + quantization.
Tiền xử lý CSV: `preprocess_iv.process_data_to_2d_inputs` — mặc định
`window_size=5`, `grid_size=64` (cùng chuẩn Trainuytin / quantize).
Đổi lưới: --grid 64|32|16 (64 = Trainuytin; 32/16 = Train3_32 / Train3_16).

Ví dụ:
  python quantize_tflite.py --model cnn2d_jam.h5 --csv-dir "D:\\train" --quant dynamic
  python quantize_tflite.py --model cnn2d_jam_g32.h5 --grid 32 --csv-dir "D:\\train" --quant int8 --eval-csv-dir "D:\\test"
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# Cùng thư mục với script
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from preprocess_iv import process_data_to_2d_inputs


def load_keras_model_compatible(path: pathlib.Path, grid_size: int) -> tf.keras.Model:
    """
    Load .h5 / .keras.
    Nếu lỗi (vd. Dense có quantization_config từ Keras mới): dựng lại graph
    Trainuytin (64) / Train3_32 (32) / Train3_16 (16) rồi load_weights.
    """
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        pass

    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))

    if grid_size == 64:
        import Trainuytin as m

        shape = (64, 64, 1)
    elif grid_size == 32:
        import Train3_32 as m

        shape = (32, 32, 1)
    elif grid_size == 16:
        import Train3_16 as m

        shape = (16, 16, 1)
    else:
        raise ValueError(f"grid_size không hỗ trợ: {grid_size}")

    model = m.build_2d_cnn(input_shape_2d=shape)
    model.load_weights(str(path))
    return model


def _collect_windows_from_csv_dir(
    folder: str, window_size: int, enable_plot: bool, grid_size: int
):
    X_list, y_list = [], []
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    for fp in paths:
        df = pd.read_csv(fp)
        x_chunk, y_chunk = process_data_to_2d_inputs(
            df,
            window_size=window_size,
            grid_size=grid_size,
            enable_plot=enable_plot,
        )
        X_list.extend(x_chunk)
        y_list.extend(y_chunk)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def convert_dynamic_range(model: tf.keras.Model, out_path: pathlib.Path) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    return tflite_model


def convert_int8_full(
    model: tf.keras.Model,
    out_path: pathlib.Path,
    representative_X: np.ndarray,
    max_rep: int,
) -> bytes:
    if max_rep <= 0:
        n = len(representative_X)
    else:
        n = min(max_rep, len(representative_X))
    indices = np.random.permutation(len(representative_X))[:n]
    X_ref = representative_X

    def representative_dataset():
        for idx in indices:
            yield [X_ref[idx : idx + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    return tflite_model


def _dequantize_output(out_q, out_details):
    out = out_q.astype(np.float32)
    scale, zp = out_details["quantization"]
    if scale != 0:
        out = (out - zp) * scale
    return out


def predict_tflite_prob(interpreter: tf.lite.Interpreter, x_hw: np.ndarray) -> float:
    """x_hw: shape (H, W, 1) float32, khớp đầu vào model."""
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    x = x_hw.astype(np.float32)
    if input_details["dtype"] == np.uint8:
        scale, zp = input_details["quantization"]
        x = np.clip(x / scale + zp, 0, 255)
    xb = np.expand_dims(x, 0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], xb)
    interpreter.invoke()
    out_q = interpreter.get_tensor(output_details["index"])
    if output_details["dtype"] == np.uint8:
        out = _dequantize_output(out_q, output_details)
    else:
        out = out_q.astype(np.float32)
    return float(out.reshape(-1)[0])


def evaluate_tflite(
    tflite_path: pathlib.Path,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> tuple[float, float]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    preds = []
    for i in range(len(X)):
        p = predict_tflite_prob(interpreter, X[i])
        preds.append(1 if p >= threshold else 0)
    preds = np.array(preds, dtype=np.int32)
    acc = float(np.mean(preds == y))
    return acc, float(np.mean(np.abs(preds - y)))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Keras CNN2D -> TFLite + quantization")
    p.add_argument("--model", required=True, help="Đường dẫn file .h5 hoặc .keras")
    p.add_argument(
        "--csv-dir",
        required=True,
        help="Thư mục chứa CSV để lấy mẫu representative (và trùng pipeline tiền xử lý train)",
    )
    p.add_argument(
        "--quant",
        choices=["dynamic", "int8"],
        default="dynamic",
        help="dynamic: chỉ Optimize.DEFAULT | int8: full integer uint8 I/O (cần rep dataset)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Trùng với Train3.main (mặc định 5)",
    )
    p.add_argument(
        "--grid",
        type=int,
        choices=[16, 32, 64],
        default=64,
        help="Kích thước lưới histogram / đầu vào CNN (64=Train3, 32=Train3_32, 16=Train3_16)",
    )
    p.add_argument(
        "--max-rep",
        type=int,
        default=200,
        help="Số mẫu tối đa cho representative_dataset (int8). Dùng 0 để lấy toàn bộ.",
    )
    p.add_argument(
        "--out-dir",
        default=".",
        help="Thư mục ghi file .tflite",
    )
    p.add_argument(
        "--eval-csv-dir",
        default="",
        help="Nếu có: đánh giá accuracy TFLite trên CSV trong thư mục này",
    )
    args = p.parse_args()

    model_path = pathlib.Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Không thấy model: {model_path}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = model_path.stem
    out_dynamic = out_dir / f"{stem}_dynamic.tflite"
    out_int8 = out_dir / f"{stem}_int8.tflite"

    print("-> Đang load Keras model...")
    model = load_keras_model_compatible(model_path, grid_size=args.grid)

    inp = model.input_shape
    if inp is not None and len(inp) == 4:
        print(f"   input_shape: {inp}")

    print(f"-> Đang đọc CSV & tiền xử lý từ: {args.csv_dir} (grid={args.grid})")
    X_rep, y_rep = _collect_windows_from_csv_dir(
        args.csv_dir,
        window_size=args.window_size,
        enable_plot=False,
        grid_size=args.grid,
    )
    if len(X_rep) == 0:
        raise SystemExit("Không tạo được mẫu nào từ csv-dir. Kiểm tra file CSV / Event / Uwave.")

    print(f"   Số mẫu sau tiền xử lý: {len(X_rep)}, shape 1 mẫu: {X_rep[0].shape}")

    print("-> Convert TFLite (dynamic range)...")
    convert_dynamic_range(model, out_dynamic)
    print(f"   Đã ghi: {out_dynamic} ({out_dynamic.stat().st_size / 1024 / 1024:.2f} MB)")

    if args.quant == "int8":
        print("-> Convert TFLite (full INT8, uint8 I/O)...")
        try:
            convert_int8_full(model, out_int8, X_rep, max_rep=args.max_rep)
            print(f"   Đã ghi: {out_int8} ({out_int8.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"   Lỗi INT8 (một số layer không hỗ trợ): {e}")
            print("   Giữ bản dynamic đã export; có thể thử chỉ float16 hoặc chỉ Optimize.DEFAULT.")

    if args.eval_csv_dir.strip():
        eval_dir = args.eval_csv_dir.strip()
        print(f"-> Đánh giá trên: {eval_dir}")
        X_ev, y_ev = _collect_windows_from_csv_dir(
            eval_dir,
            window_size=args.window_size,
            enable_plot=False,
            grid_size=args.grid,
        )
        if len(X_ev) == 0:
            print("   Không có mẫu eval.")
        else:
            acc_d, _ = evaluate_tflite(out_dynamic, X_ev, y_ev)
            print(f"   Dynamic TFLite accuracy: {acc_d * 100:.2f}% (n={len(y_ev)})")
            if args.quant == "int8" and out_int8.is_file():
                acc_i, _ = evaluate_tflite(out_int8, X_ev, y_ev)
                print(f"   INT8 TFLite accuracy:     {acc_i * 100:.2f}% (n={len(y_ev)})")

    print("-> Xong.")


if __name__ == "__main__":
    main()
