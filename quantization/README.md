# Quantization Toolkit

Thu muc nay chua cac script quantization de chuyen model Keras sang TFLite (uu tien INT8 voi uint8 I/O cho firmware STM32 CubeAI).

## Tep tin
- `quantize_tflite.py`: script chinh de quantize va evaluate model.
- `preprocess_iv.py`: tien xu ly I-V (dong bo voi pipeline train).
- `Trainuytin.py`: dung de rebuild model graph khi load `.h5` cu bi loi tuong thich.

## Cai dat
```bash
pip install tensorflow numpy pandas
```

## Cach chay nhanh
```bash
python quantize_tflite.py \
  --model "d:/path/to/cnn2d_jam80.h5" \
  --csv-dir "d:/path/to/calib_csv_folder" \
  --quant int8 \
  --max-rep 0 \
  --out-dir "d:/path/to/output" \
  --eval-csv-dir "d:/path/to/test_csv_folder"
```

## Ghi chu tham so
- `--max-rep 0`: dung toan bo samples sau tien xu ly de calibration.
- `--quant int8`: full integer quantization, I/O la `uint8`.
- Co the bo `--eval-csv-dir` neu khong can tinh accuracy sau khi export.

## Dau ra
- `<model>_dynamic.tflite`
- `<model>_int8.tflite` (khi dung `--quant int8`)
