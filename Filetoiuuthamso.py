import os
import glob
import random
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold

import optuna

# =========================================================
# 1. CỐ ĐỊNH SEED
# =========================================================
def set_global_determinism(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except: pass

set_global_determinism(seed=42)

# =========================================================
# 2. TIỀN XỬ LÝ SIÊU TỐC TỐI ƯU CHO OPTUNA
# =========================================================
def extract_1d_cycles_from_df(df, num_points=50):
    state_col = 'Event' if 'Event' in df.columns else 'EVENT'
    u_col_2d = 'Uwave' if 'Uwave' in df.columns else 'Voltage[V]'
    i_col_2d = 'Iwave' if 'Iwave' in df.columns else 'Current[A]'

    df_sliced = df.copy().reset_index(drop=True)
    events = df_sliced[state_col].values
    u_raw = df_sliced[u_col_2d].values
    i_raw = df_sliced[i_col_2d].values

    zero_crossings = np.where((u_raw[:-1] < 0) & (u_raw[1:] >= 0))[0]
    cycles_raw, cycles_labels = [], []

    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i+1]
        if end - start < 10: continue

        old_idx = np.linspace(0, 1, end - start)
        new_idx = np.linspace(0, 1, num_points)
        interp_u = np.interp(new_idx, old_idx, u_raw[start:end])
        interp_i = np.interp(new_idx, old_idx, i_raw[start:end])

        cycles_raw.append(np.column_stack((interp_u, interp_i)))
        cycle_events = events[start:end]
        cycles_labels.append(max(set(cycle_events), key=list(cycle_events).count))

    return cycles_raw, cycles_labels

def pre_load_all_files_to_1d(file_list):
    dataset_1d = []
    for f in file_list:
        try:
            df = pd.read_csv(f)
            c_raw, c_labels = extract_1d_cycles_from_df(df)
            dataset_1d.append((c_raw, c_labels))
        except Exception: pass
    return dataset_1d

def build_2d_dataset_from_1d(dataset_1d, window_size, overlap_ratio, grid_size=64):
    X_2d, y = [], []
    bins = np.linspace(-1.1, 1.1, grid_size + 1)

    stride = max(1, int(window_size * (1 - overlap_ratio)))

    for cycles_raw, cycles_labels in dataset_1d:
        for i in range(0, len(cycles_raw) - window_size + 1, stride):
            chunk_raw = cycles_raw[i : i + window_size]
            chunk_labels = cycles_labels[i : i + window_size]

            chunk_array = np.stack(chunk_raw, axis=0)
            mean_cycle = np.mean(chunk_array, axis=0)
            u_norm = mean_cycle[:, 0] / (np.max(np.abs(mean_cycle[:, 0])) + 1e-9)
            i_norm = mean_cycle[:, 1] / (np.max(np.abs(mean_cycle[:, 1])) + 1e-9)

            heatmap, _, _ = np.histogram2d(u_norm, i_norm, bins=(bins, bins))
            window_2d = heatmap / (heatmap.max() + 1e-9)

            X_2d.append(window_2d[..., np.newaxis])
            majority_label = max(set(chunk_labels), key=chunk_labels.count)
            y.append(1 if str(majority_label).upper() == 'JAM' else 0)

    return np.array(X_2d), np.array(y)

# =========================================================
# 3. KIẾN TRÚC CNN ĐỘNG
# =========================================================
def build_dynamic_cnn(params, input_shape=(64, 64, 1)):
    input_2d = Input(shape=input_shape)

    x = Conv2D(filters=params['conv1'], kernel_size=(3, 3), activation='relu')(input_2d)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=params['conv2'], kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=params['conv3'], kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(params['dense'], activation='relu')(x)
    x = Dropout(params['dropout'])(x)

    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_2d, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =========================================================
# 4. HÀM MỤC TIÊU CỦA OPTUNA (GỘP TÌM EPOCHS)
# =========================================================
def objective(trial, train_1d_data):
    w_size = trial.suggest_int('window_size', 3, 10)
    overlap = trial.suggest_categorical('overlap', [0.0, 0.25, 0.5])

    X_cv, y_cv = build_2d_dataset_from_1d(train_1d_data, w_size, overlap)

    params = {
        'conv1': trial.suggest_categorical('conv1', [16, 32]),
        'conv2': trial.suggest_categorical('conv2', [32, 64]),
        'conv3': trial.suggest_categorical('conv3', [64, 128]),
        'dense': trial.suggest_categorical('dense', [64, 128]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    best_epochs_list = []

    for train_idx, val_idx in kf.split(X_cv):
        tf.keras.backend.clear_session()
        X_train, X_val = X_cv[train_idx], X_cv[val_idx]
        y_train, y_val = y_cv[train_idx], y_cv[val_idx]

        model = build_dynamic_cnn(params)
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                         epochs=30, batch_size=32, callbacks=[early_stop], verbose=0)

        # Bắt lại số Epoch dừng sớm tốt nhất của Fold này
        actual_epochs = len(hist.history['loss'])
        best_epoch = actual_epochs - 3 if early_stop.stopped_epoch > 0 else actual_epochs
        best_epochs_list.append(best_epoch)

        y_pred = (model.predict(X_val, verbose=0) >= 0.5).astype(int).flatten()
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

    # Ghi nhớ lại số Epoch trung bình của lần chạy (trial) này
    trial.set_user_attr("avg_best_epoch", int(np.mean(best_epochs_list)))

    return np.mean(f1_scores)

# =========================================================
# 5. HÀM MAIN: THỰC CHIẾN, ĐÁNH NHANH THẮNG NHANH
# =========================================================
def main():
    print("=== TỐI ƯU OPTUNA & HUẤN LUYỆN THỰC CHIẾN ===")

    train_folder = input("Nhập đường dẫn thư mục TRAIN (14 files): ")
    val_folder = input("Nhập đường dẫn thư mục VALIDATION (2 files): ")
    test_folder = input("Nhập đường dẫn thư mục TEST (4 files): ")

    # Gộp 16 file Train/Val làm 1 hồ chứa để Optuna xào nấu
    cv_files = glob.glob(os.path.join(train_folder, '*.csv')) + glob.glob(os.path.join(val_folder, '*.csv'))
    test_files = glob.glob(os.path.join(test_folder, '*.csv'))

    print("\n[1/3] Đọc dữ liệu thô (1D) lên RAM để tăng tốc...")
    cv_1d_data = pre_load_all_files_to_1d(cv_files)
    test_1d_data = pre_load_all_files_to_1d(test_files)

    print("\n[2/3] Optuna đang chạy tìm Siêu tham số & Epochs (K=3)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Tắt bớt log rác của Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, cv_1d_data), n_trials=20)

    best_params = study.best_params
    # Lấy ra số Epoch tối ưu do Optuna chốt lại ở Trial vô địch
    optimal_epochs = study.best_trial.user_attrs["avg_best_epoch"]

    print("\n🔥 KẾT QUẢ TỐI ƯU HÓA 🔥")
    for key, value in best_params.items():
        print(f" - {key}: {value}")
    print(f" - Số Epochs tối ưu: {optimal_epochs}")

    print("\n[3/3] Trích xuất Ảnh 2D theo Window/Overlap vô địch và Train Final Model...")
    X_cv_final, y_cv_final = build_2d_dataset_from_1d(cv_1d_data, best_params['window_size'], best_params['overlap'])
    X_test_final, y_test_final = build_2d_dataset_from_1d(test_1d_data, best_params['window_size'], best_params['overlap'])

    tf.keras.backend.clear_session()
    final_model = build_dynamic_cnn(best_params)

    # Train 1 lèo trên toàn bộ X_cv_final với số epoch chuẩn, không cần tập Val nữa
    final_model.fit(X_cv_final, y_cv_final, epochs=optimal_epochs, batch_size=32, verbose=1)

    print("\n=== KẾT QUẢ TEST TRÊN 4 FILE CẤT TỦ ===")
    y_pred = (final_model.predict(X_test_final) >= 0.5).astype(int).flatten()
    print(classification_report(y_test_final, y_pred, target_names=['NORMAL', 'JAM']))
    print("\n✅ Xong! Mô hình final_model đã sẵn sàng để đem đi triển khai.")

if __name__ == "__main__":
    main()