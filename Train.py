import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Thư viện Machine Learning (Thêm KFold)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold

# BỘ THÔNG SỐ VÔ ĐỊCH TỪ OPTUNA CỦA BẠN
BEST_PARAMS = {
    'conv1_filters': 32,
    'conv2_filters': 64,
    'conv3_filters': 128,
    'dense_units': 64,
    'dropout_rate': 0.327, # Làm tròn từ 0.32704
    'learning_rate': 0.0001 # Làm tròn từ 0.0001007
}

# =========================================================
# CÁC HÀM TIỀN XỬ LÝ VÀ GRAD-CAM GIỮ NGUYÊN HOÀN TOÀN
# =========================================================
def plot_raw_1d_signals(df_sliced, state_col, u_col, i_col):
    plt.figure(figsize=(15, 5))
    u_norm = df_sliced[u_col] / (df_sliced[u_col].abs().max() + 1e-9)
    i_norm = df_sliced[i_col] / (df_sliced[i_col].abs().max() + 1e-9)
    plt.plot(u_norm, label=f'{u_col} (Normalized)', alpha=0.7, color='blue')
    plt.plot(i_norm, label=f'{i_col} (Normalized)', alpha=0.7, color='orange')
    jam_indices = df_sliced.index[df_sliced[state_col] == 'JAM'].tolist()
    if jam_indices:
        start_jam = jam_indices[0]
        plt.axvline(x=start_jam, color='red', linestyle='--', linewidth=2, label='Start of JAM Event')
    plt.title("Biểu đồ tín hiệu 1D (Toàn bộ File)", fontsize=14, fontweight='bold')
    plt.xlabel("Index (Data Points)")
    plt.ylabel("Biên độ (Chuẩn hóa)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

def process_data_to_2d_inputs(df, num_points=50, grid_size=64, window_size=30):
    state_col = 'Event' if 'Event' in df.columns else 'EVENT'
    u_col_2d = 'Uwave' if 'Uwave' in df.columns else 'Voltage[V]'
    i_col_2d = 'Iwave' if 'Iwave' in df.columns else 'Current[A]'
    df_sliced = df.copy().reset_index(drop=True)
    events = df_sliced[state_col].values

    if 'first_plot_done' not in globals():
        plot_raw_1d_signals(df_sliced, state_col, u_col_2d, i_col_2d)
        global first_plot_done
        first_plot_done = True

    u_raw = df_sliced[u_col_2d].values
    i_raw = df_sliced[i_col_2d].values
    zero_crossings = np.where((u_raw[:-1] < 0) & (u_raw[1:] >= 0))[0]

    cycles_raw = []
    cycles_labels = []

    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i+1]
        if end - start < 10: continue
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
        if len(chunk_raw) < window_size: continue
        majority_label = max(set(chunk_labels), key=chunk_labels.count)

        chunk_array = np.stack(chunk_raw, axis=0)
        mean_cycle = np.mean(chunk_array, axis=0)
        u_window = mean_cycle[:, 0]
        i_window = mean_cycle[:, 1]
        u_norm = u_window / (np.max(np.abs(u_window)) + 1e-9)
        i_norm = i_window / (np.max(np.abs(i_window)) + 1e-9)

        heatmap, _, _ = np.histogram2d(u_norm, i_norm, bins=(bins, bins))
        window_2d = heatmap / (heatmap.max() + 1e-9)
        window_2d = window_2d[..., np.newaxis]
        X_2d_list.append(window_2d)
        y_list.append(1 if str(majority_label).upper() == 'JAM' else 0)

    return X_2d_list, y_list

def load_files_to_dataset(file_list, window_size):
    X_2d, y = [], []
    for idx, file_path in enumerate(file_list, 1):
        try:
            df = pd.read_csv(file_path)
            x2_chunk, y_chunk = process_data_to_2d_inputs(df, window_size=window_size)
            X_2d.extend(x2_chunk)
            y.extend(y_chunk)
        except Exception as e:
            pass
    return np.array(X_2d), np.array(y)

# =========================================================
# XÂY DỰNG MÔ HÌNH CNN VỚI THÔNG SỐ CỐ ĐỊNH
# =========================================================
def build_2d_cnn(input_shape_2d=(64, 64, 1)):
    input_2d = Input(shape=input_shape_2d)

    x = Conv2D(filters=BEST_PARAMS['conv1_filters'], kernel_size=(3, 3), activation='relu')(input_2d)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=BEST_PARAMS['conv2_filters'], kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=BEST_PARAMS['conv3_filters'], kernel_size=(3, 3), activation='relu', name="last_conv2d_layer")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    flat_2d = Flatten()(x)

    x = Dense(BEST_PARAMS['dense_units'], activation='relu')(flat_2d)
    x = Dropout(BEST_PARAMS['dropout_rate'])(x)

    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="Output")(x)

    model = Model(inputs=input_2d, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=BEST_PARAMS['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def make_gradcam_heatmap(input_2d_array, model, last_conv_layer_name="last_conv2d_layer"):
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(input_2d_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam_overlay(X_test, y_test, model, num_samples=3):
    normal_indices = np.where(y_test == 0)[0]
    jam_indices = np.where(y_test == 1)[0]
    n_normal = min(num_samples, len(normal_indices))
    n_jam = min(num_samples, len(jam_indices))
    n_cols = max(n_normal, n_jam)

    if n_cols == 0: return

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    if n_cols == 1: axes = axes.reshape(2, 1)
    fig.suptitle('EXPLAINABLE AI: BẢN ĐỒ NHIỆT GRAD-CAM QUỸ ĐẠO I-V', fontsize=16, fontweight='bold', y=1.02)

    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_normal:
            idx = normal_indices[i]
            x_sample = np.expand_dims(X_test[idx], axis=0)
            heatmap = make_gradcam_heatmap(x_sample, model)
            img_original = X_test[idx][:, :, 0]
            ax.imshow(img_original.T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            ax.imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')
            pred_score = model.predict(x_sample, verbose=0)[0][0]
            ax.set_title(f"NORMAL (Mẫu {i+1})\nAI Dự đoán JAM: {pred_score*100:.1f}%", fontsize=11, color='green' if pred_score < 0.5 else 'red')
        else: ax.axis('off')

    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_jam:
            idx = jam_indices[i]
            x_sample = np.expand_dims(X_test[idx], axis=0)
            heatmap = make_gradcam_heatmap(x_sample, model)
            img_original = X_test[idx][:, :, 0]
            ax.imshow(img_original.T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            ax.imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')
            pred_score = model.predict(x_sample, verbose=0)[0][0]
            ax.set_title(f"JAM (Mẫu {i+1})\nAI Dự đoán JAM: {pred_score*100:.1f}%", fontsize=11, color='red' if pred_score >= 0.5 else 'green')
        else: ax.axis('off')
    plt.tight_layout()
    plt.show()

# =========================================================
# HÀM MAIN: K-FOLD CROSS VALIDATION
# =========================================================
def main():
    print("=== HUẤN LUYỆN K-FOLD VỚI BỘ THÔNG SỐ OPTUNA ===")

    # 1. Quản lý File: Gộp Train và Valid thành 1 hồ chứa (CV)
    train_folder_path = input("Nhập đường dẫn Folder chứa file TRAIN (14 files): ")
    val_folder_path = input("Nhập đường dẫn Folder chứa file VALIDATION (2 files): ")
    test_folder_path = input("Nhập đường dẫn Folder chứa file TEST (4 files): ")
    window_size = 5

    train_files = glob.glob(os.path.join(train_folder_path, '*.csv'))
    val_files = glob.glob(os.path.join(val_folder_path, '*.csv'))
    cv_files = train_files + val_files # Gộp chung lại thành 16 files
    test_files = glob.glob(os.path.join(test_folder_path, '*.csv'))

    print(f"\n-> Đã tìm thấy {len(cv_files)} files cho K-Fold (Train+Val) và {len(test_files)} files TEST.")

    print("\n=== ĐANG TRÍCH XUẤT ẢNH 2D CHO TẬP K-FOLD (CV) ===")
    X_cv, y_cv = load_files_to_dataset(cv_files, window_size)

    print("\n=== ĐANG TRÍCH XUẤT ẢNH 2D CHO TẬP TEST ===")
    X_test, y_test = load_files_to_dataset(test_files, window_size)

    # 2. Khởi tạo K-Fold 5 vòng
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_no = 1
    cv_scores = []

    print(f"\n=== BẮT ĐẦU CHẠY K-FOLD {k_folds} VÒNG ===")
    for train_index, val_index in kf.split(X_cv):
        print(f"\n-> Đang huấn luyện Vòng (Fold) {fold_no} ...")

        # Giải phóng RAM cực kỳ quan trọng khi chạy nhiều model liên tiếp
        tf.keras.backend.clear_session()

        # Cắt dữ liệu thành phần học và phần thi cho vòng này
        X_fold_train, X_fold_val = X_cv[train_index], X_cv[val_index]
        y_fold_train, y_fold_val = y_cv[train_index], y_cv[val_index]

        model = build_2d_cnn()

        # Huấn luyện ngắn gọn trong nội bộ Fold
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=20,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0 # Tắt bớt log để màn hình đỡ rối
        )

        # Chấm điểm Fold này
        scores = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        acc = scores[1] * 100
        print(f"   + Kết quả Vòng {fold_no}: Accuracy = {acc:.2f}%")
        cv_scores.append(acc)
        fold_no += 1

    print("\n=== TỔNG KẾT K-FOLD ===")
    print(f"Điểm trung bình hệ thống: {np.mean(cv_scores):.2f}% (+/- {np.std(cv_scores):.2f}%)")

    # 3. Train Mô hình Final trên TOÀN BỘ dữ liệu CV để đi thi
    print("\n-> Đang huấn luyện MÔ HÌNH CUỐI CÙNG (Final Model) trên 100% dữ liệu CV...")
    tf.keras.backend.clear_session()
    final_model = build_2d_cnn()

    # Không còn tập Val nội bộ nữa, train thẳng trên X_cv
    history = final_model.fit(
        X_cv, y_cv,
        epochs=25,
        batch_size=32,
        verbose=1
    )

    # 4. Kỳ thi cuối cùng trên 4 File Test Cất Tủ
    print("\n-> Đang đánh giá Final Model trên 4 Files Test...")
    y_pred_prob = final_model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    print("\n=== BÁO CÁO PHÂN LOẠI CUỐI CÙNG ===")
    print(classification_report(y_test, y_pred, target_names=['NORMAL', 'JAM']))

    # Vẽ Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'JAM'], yticklabels=['NORMAL', 'JAM'])
    plt.title('Ma trận nhầm lẫn (Tập Test Cất Tủ)')
    plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.show()

    display_gradcam_overlay(X_test, y_test, final_model, num_samples=3)

if __name__ == "__main__":
    main()