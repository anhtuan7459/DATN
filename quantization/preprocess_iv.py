"""
Tiền xử lý quỹ đạo I–V → ảnh 2D (histogram 64×64 hoặc grid_size khác), đồng bộ với
Trainuytin.py (mặc định window_size=5). Train3.py vẫn có thể dùng window_size=10
trong chữ ký hàm — khi train/quantize cần truyền cùng window_size.

Dùng cho quantize_tflite.py và thử nhanh: `python preprocess_iv.py`.

Nếu đổi logic cốt lõi: cập nhật Trainuytin.py, Train3.py (và Train3_32/16 nếu dùng chung).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. HÀM VẼ ĐỒ THỊ 1D (giống Trainuytin / Train3)
# =========================================================
def plot_raw_1d_signals(df_sliced, state_col, u_col, i_col):
    plt.figure(figsize=(15, 5))

    u_norm = df_sliced[u_col] / (df_sliced[u_col].abs().max() + 1e-9)
    i_norm = df_sliced[i_col] / (df_sliced[i_col].abs().max() + 1e-9)

    plt.plot(u_norm, label=f"{u_col} (Normalized)", alpha=0.7, color="blue")
    plt.plot(i_norm, label=f"{i_col} (Normalized)", alpha=0.7, color="orange")

    jam_indices = df_sliced.index[df_sliced[state_col] == "JAM"].tolist()
    if jam_indices:
        start_jam = jam_indices[0]
        plt.axvline(x=start_jam, color="red", linestyle="--", linewidth=2, label="Start of JAM Event")

    plt.title("Biểu đồ tín hiệu 1D (Toàn bộ File)", fontsize=14, fontweight="bold")
    plt.xlabel("Index (Data Points)")
    plt.ylabel("Biên độ (Chuẩn hóa)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


# =========================================================
# 1. HÀM TIỀN XỬ LÝ — CHỈ XUẤT ẢNH 2D TỪ I–V (khớp Trainuytin.py)
# =========================================================
def process_data_to_2d_inputs(
    df, num_points=50, grid_size=64, window_size=5, enable_plot=True
):
    state_col = "Event" if "Event" in df.columns else "EVENT"
    u_col_2d = "Uwave" if "Uwave" in df.columns else "Voltage[V]"
    i_col_2d = "Iwave" if "Iwave" in df.columns else "Current[A]"

    df_sliced = df.copy().reset_index(drop=True)
    events = df_sliced[state_col].values

    if enable_plot and "first_plot_done" not in globals():
        plot_raw_1d_signals(df_sliced, state_col, u_col_2d, i_col_2d)
        global first_plot_done
        first_plot_done = True

    u_raw = df_sliced[u_col_2d].values
    i_raw = df_sliced[i_col_2d].values

    zero_crossings = np.where((u_raw[:-1] < 0) & (u_raw[1:] >= 0))[0]

    cycles_raw = []
    cycles_labels = []

    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i + 1]
        if end - start < 10:
            continue

        old_idx = np.linspace(0, 1, end - start)
        new_idx = np.linspace(0, 1, num_points)

        interp_u = np.interp(new_idx, old_idx, u_raw[start:end])
        interp_i = np.interp(new_idx, old_idx, i_raw[start:end])

        cycles_raw.append(np.column_stack((interp_u, interp_i)))

        cycle_events = events[start:end]
        majority_event = max(set(cycle_events), key=list(cycle_events).count)
        cycles_labels.append(majority_event)

    X_2d_list, y_list = [], []
    bins = np.linspace(-1.1, 1.1, grid_size + 1)

    for i in range(0, len(cycles_raw), window_size):
        chunk_raw = cycles_raw[i : i + window_size]
        chunk_labels = cycles_labels[i : i + window_size]

        if len(chunk_raw) < window_size:
            continue

        majority_label = max(set(chunk_labels), key=chunk_labels.count)

        # Trung bình theo các chu kỳ trong cửa sổ (cùng pha)
        chunk_array = np.stack(chunk_raw, axis=0)  # (window_size, num_points, 2)
        mean_cycle = np.mean(chunk_array, axis=0)  # (num_points, 2)

        u_window = mean_cycle[:, 0]
        i_window = mean_cycle[:, 1]

        # Chuẩn hóa biên độ theo từng cửa sổ
        u_norm = u_window / (np.max(np.abs(u_window)) + 1e-9)
        i_norm = i_window / (np.max(np.abs(i_window)) + 1e-9)

        heatmap, _, _ = np.histogram2d(u_norm, i_norm, bins=(bins, bins))

        window_2d = heatmap / (heatmap.max() + 1e-9)
        window_2d = window_2d[..., np.newaxis]  # (grid_size, grid_size, 1)

        X_2d_list.append(window_2d)
        y_list.append(1 if str(majority_label).upper() == "JAM" else 0)

    return X_2d_list, y_list


if __name__ == "__main__":
    print(__doc__)
    print("--- Demo: CSV giả -> số mẫu & shape ảnh ---\n")
    t = np.linspace(0, 200 * np.pi, 8000)
    u = 220 * np.sin(t)
    rng = np.random.default_rng(0)
    i = np.where(
        t < 100 * np.pi,
        3.6 * np.sin(t + np.pi / 6) + rng.normal(0, 0.2, len(t)),
        15.0 * np.sin(t + np.pi / 4) + rng.normal(0, 1.0, len(t)) + 2.0,
    )
    event = np.where(t < 100 * np.pi, "NORMAL", "JAM")
    df_demo = pd.DataFrame({"Uwave": u, "Iwave": i, "Event": event})

    X_list, y_list = process_data_to_2d_inputs(df_demo, window_size=5)
    X = np.array(X_list)
    y = np.array(y_list)
    print(f"Số mẫu: {len(y)}")
    if len(X):
        print(f"Shape 1 ảnh (grid mặc định 64): {X[0].shape}")
        print(f"Nhãn mẫu (0=NORMAL, 1=JAM): {y[: min(10, len(y))]}")
