import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

# Thư viện Machine Learning
from sklearn.metrics import classification_report, confusion_matrix
# BỔ SUNG CÔNG CỤ TÍNH TRỌNG SỐ TỰ ĐỘNG
from sklearn.utils.class_weight import compute_class_weight 

# =========================================================
# 1. HÀM TIỀN XỬ LÝ (TRÍCH XUẤT CẢ 2D VÀ 1D)
# =========================================================
def process_data_multimodal(df, num_points=50, grid_size=64, window_size=10):
    state_col = 'Event' if 'Event' in df.columns else 'EVENT'
    u_col_2d = 'Uwave' if 'Uwave' in df.columns else 'Voltage[V]'
    i_col_2d = 'Iwave' if 'Iwave' in df.columns else 'Current[A]'

    ignore_cols = ['Date', 'Time', state_col]
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    df_sliced = df.copy().reset_index(drop=True)
    events = df_sliced[state_col].values

    u_raw = df_sliced[u_col_2d].values
    zero_crossings = np.where((u_raw[:-1] < 0) & (u_raw[1:] >= 0))[0]

    cycles_raw = []
    cycles_labels = []

    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i+1]
        if end - start < 10: continue

        old_idx = np.linspace(0, 1, end - start)
        new_idx = np.linspace(0, 1, num_points)

        cycle_features = []
        for col in feature_cols:
            interp_feat = np.interp(new_idx, old_idx, df_sliced[col].values[start:end])
            cycle_features.append(interp_feat)

        cycles_raw.append(np.column_stack(cycle_features))

        cycle_events = events[start:end]
        majority_event = max(set(cycle_events), key=list(cycle_events).count)
        cycles_labels.append(majority_event)

    X_2d_list, X_1d_list, y_list = [], [], []
    bins = np.linspace(-1.1, 1.1, grid_size + 1)

    u_idx = feature_cols.index(u_col_2d)
    i_idx = feature_cols.index(i_col_2d)

    for i in range(0, len(cycles_raw), window_size):
        chunk_raw = cycles_raw[i : i + window_size]
        chunk_labels = cycles_labels[i : i + window_size]

        if len(chunk_raw) < window_size:
            continue

        majority_label = max(set(chunk_labels), key=chunk_labels.count)
        chunk_array = np.stack(chunk_raw, axis=0) 

        # 1D
        window_1d = np.mean(chunk_array, axis=0)
        window_1d = (window_1d - np.mean(window_1d, axis=0)) / (np.std(window_1d, axis=0) + 1e-9)
        X_1d_list.append(window_1d)

        # 2D
        mean_cycle = np.mean(chunk_array, axis=0)
        u_window = mean_cycle[:, u_idx]
        i_window = mean_cycle[:, i_idx]

        u_norm = u_window / (np.max(np.abs(u_window)) + 1e-9)
        i_norm = i_window / (np.max(np.abs(i_window)) + 1e-9)

        heatmap, _, _ = np.histogram2d(u_norm, i_norm, bins=(bins, bins))
        window_2d = heatmap / (heatmap.max() + 1e-9)
        window_2d = window_2d[..., np.newaxis]
        X_2d_list.append(window_2d)

        y_list.append(1 if str(majority_label).upper() == 'JAM' else 0)

    return X_2d_list, X_1d_list, y_list

# =========================================================
# 2. XÂY DỰNG MÔ HÌNH HYBRID (CNN 2D + CNN 1D)
# =========================================================
def build_hybrid_model(input_shape_2d=(64, 64, 1), input_shape_1d=(250, 7)):
    # Nhánh 2D
    input_2d = Input(shape=input_shape_2d, name="Input_2D")
    x2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_2d)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name="last_conv2d_layer")(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    flat_2d = Flatten()(x2)

    # Nhánh 1D
    input_1d = Input(shape=input_shape_1d, name="Input_1D")
    x1 = Conv1D(filters=32, kernel_size=5, activation='relu')(input_1d)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Conv1D(filters=64, kernel_size=5, activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    flat_1d = Flatten()(x1)

    # Nối
    merged = Concatenate()([flat_2d, flat_1d])

    z = Dense(128, activation='relu')(merged)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, activation='sigmoid', name="Output")(z)

    model = Model(inputs=[input_2d, input_1d], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# =========================================================
# 3. HÀM TÍNH TOÁN VÀ VẼ BẢN ĐỒ NHIỆT (GRAD-CAM)
# =========================================================
def make_gradcam_heatmap(input_data, model, last_conv_layer_name="last_conv2d_layer"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(input_data)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam_overlay(X_test_2d, X_test_1d, y_test, model, num_samples=3):
    print(f"\n-> Đang tạo Bản đồ nhiệt Grad-CAM ({num_samples} ảnh mỗi nhãn)...")
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
        # NORMAL
        ax = axes[0, i]
        if i < n_normal:
            idx = normal_indices[i]
            x_sample_2d = np.expand_dims(X_test_2d[idx], axis=0)
            x_sample_1d = np.expand_dims(X_test_1d[idx], axis=0)

            heatmap = make_gradcam_heatmap([x_sample_2d, x_sample_1d], model)
            img_original = X_test_2d[idx][:, :, 0]

            ax.imshow(img_original.T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            ax.imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')

            pred_score = model.predict([x_sample_2d, x_sample_1d], verbose=0)[0][0]
            ax.set_title(f"NORMAL (Mẫu {i+1})\nAI Dự đoán JAM: {pred_score*100:.1f}%", fontsize=11, color='green' if pred_score < 0.5 else 'red')
            ax.grid(True, linestyle=':', alpha=0.2)
        else: ax.axis('off')

        # JAM
        ax = axes[1, i]
        if i < n_jam:
            idx = jam_indices[i]
            x_sample_2d = np.expand_dims(X_test_2d[idx], axis=0)
            x_sample_1d = np.expand_dims(X_test_1d[idx], axis=0)

            heatmap = make_gradcam_heatmap([x_sample_2d, x_sample_1d], model)
            img_original = X_test_2d[idx][:, :, 0]

            ax.imshow(img_original.T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            ax.imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')

            pred_score = model.predict([x_sample_2d, x_sample_1d], verbose=0)[0][0]
            ax.set_title(f"JAM (Mẫu {i+1})\nAI Dự đoán JAM: {pred_score*100:.1f}%", fontsize=11, color='red' if pred_score >= 0.5 else 'green')
            ax.grid(True, linestyle=':', alpha=0.2)
        else: ax.axis('off')

    plt.tight_layout()
    plt.show()

# =========================================================
# 4. HÀM CHÍNH VÀ CÁC HÀM PHỤ TRỢ
# =========================================================
def load_files_to_dataset(file_list, window_size):
    X_2d_all, X_1d_all, y_all = [], [], []
    for idx, file_path in enumerate(file_list, 1):
        print(f"   + Xử lý file [{idx}/{len(file_list)}]: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            x2_chunk, x1_chunk, y_chunk = process_data_multimodal(df, window_size=window_size)
            X_2d_all.extend(x2_chunk)
            X_1d_all.extend(x1_chunk)
            y_all.extend(y_chunk)
        except Exception as e:
            print(f"   ! Lỗi khi đọc file {file_path}: {e}")
    return np.array(X_2d_all), np.array(X_1d_all), np.array(y_all)

def main():
    print("=== HUẤN LUYỆN HYBRID MODEL (CNN 2D + CNN 1D) ===")

    train_folder_path = input("Nhập đường dẫn Folder chứa file TRAIN (Bỏ trống để dùng giả lập): ")
    window_size = 5
    num_points = 50

    if train_folder_path.strip() == "":
        print("\n-> Đang tạo dữ liệu giả lập (Bao gồm nhiều cột)...")
        def gen_sim_data(num_files):
            x2_list, x1_list, y_list = [], [], []
            for _ in range(num_files):
                t = np.linspace(0, 1000 * np.pi, 20000)
                u = 220 * np.sin(t)
                i = np.where(t < 500 * np.pi,
                             3.6 * np.sin(t + np.pi/6) + np.random.normal(0, 0.2, len(t)),
                             15.0 * np.sin(t + np.pi/4) + np.random.normal(0, 1.0, len(t)) + 2.0)

                power = u * i
                energy = np.cumsum(power)
                phase = np.full(len(t), 0.5)
                event = ['NORMAL' if time < 500 * np.pi else 'JAM' for time in t]

                df_sim = pd.DataFrame({
                    'Voltage[V]': u, 'Current[A]': i, 'Power[W]': power,
                    'Energy[u]': energy, 'Phase[rad]': phase,
                    'Uwave': u, 'Iwave': i, 'Event': event
                })

                x2, x1, y = process_data_multimodal(df_sim, num_points=num_points, window_size=window_size)
                x2_list.extend(x2); x1_list.extend(x1); y_list.extend(y)
            return np.array(x2_list), np.array(x1_list), np.array(y_list)

        X_train_2d, X_train_1d, y_train = gen_sim_data(3)
        X_val_2d, X_val_1d, y_val = gen_sim_data(1)
        X_test_2d, X_test_1d, y_test = gen_sim_data(1)

    else:
        val_folder_path = input("Nhập đường dẫn Folder chứa file VALIDATION (VAL): ")
        test_folder_path = input("Nhập đường dẫn Folder chứa file TEST: ")

        train_files = glob.glob(os.path.join(train_folder_path, '*.csv'))
        val_files = glob.glob(os.path.join(val_folder_path, '*.csv'))
        test_files = glob.glob(os.path.join(test_folder_path, '*.csv'))

        if len(train_files) == 0:
            print(f"Không tìm thấy file CSV nào trong thư mục TRAIN: {train_folder_path}!")
            return
        if len(val_files) == 0:
            print(f"Không tìm thấy file CSV nào trong thư mục VALIDATION: {val_folder_path}!")
            return
        if len(test_files) == 0:
            print(f"Không tìm thấy file CSV nào trong thư mục TEST: {test_folder_path}!")
            return

        print(f"\n-> Đã tìm thấy {len(train_files)} files TRAIN, {len(val_files)} files VAL và {len(test_files)} files TEST.")

        print("\n=== ĐANG TẠO TẬP TRAIN ===")
        X_train_2d, X_train_1d, y_train = load_files_to_dataset(train_files, window_size)

        print("\n=== ĐANG TẠO TẬP VALIDATION ===")
        X_val_2d, X_val_1d, y_val = load_files_to_dataset(val_files, window_size)

        print("\n=== ĐANG TẠO TẬP TEST ===")
        X_test_2d, X_test_1d, y_test = load_files_to_dataset(test_files, window_size)

        if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
            print("\nDữ liệu Train, Val hoặc Test bị rỗng. Vui lòng kiểm tra lại file CSV của bạn!")
            return

    # --------------------------------------------------
    # TÍNH TOÁN CLASS WEIGHTS ĐỂ PHẠT HÀM LOSS 
    # --------------------------------------------------
    print(f"\n-> TỔNG QUAN DỮ LIỆU:")
    print(f"   + TRAIN: {len(y_train)} mẫu ảnh (NORMAL: {np.sum(y_train==0)}, JAM: {np.sum(y_train==1)})")
    
    # Tính trọng số tự động dựa trên độ lệch dữ liệu
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = {classes[i]: weights[i] for i in range(len(classes))}
    
    # Nếu bạn muốn ép phạt JAM cực nặng, có thể bỏ comment dòng dưới để ghi đè (VD: JAM quan trọng gấp 3 lần NORMAL)
    # class_weights_dict = {0: 1.0, 1: 3.0} 
    
    print(f"   -> Đã áp dụng Trọng số Hàm Loss: NORMAL={class_weights_dict[0]:.2f}, JAM={class_weights_dict.get(1, 1.0):.2f}")
    
    print(f"   + Kích thước X_2d: {X_train_2d.shape}")
    print(f"   + Kích thước X_1d: {X_train_1d.shape}")

    # Khởi tạo mô hình
    print(f"\n-> Khởi tạo mô hình Hybrid...")
    input_shape_1d = (X_train_1d.shape[1], X_train_1d.shape[2])
    model = build_hybrid_model(input_shape_2d=(64, 64, 1), input_shape_1d=input_shape_1d)

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Đưa class_weights_dict vào model.fit
    history = model.fit(
        x=[X_train_2d, X_train_1d],
        y=y_train,
        validation_data=([X_val_2d, X_val_1d], y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weights_dict, # <-- ĐÃ BỔ SUNG Ở ĐÂY
        verbose=1
    )

    # Đánh giá
    print("\n-> Đang đánh giá trên tập Test ...")
    y_pred_prob = model.predict([X_test_2d, X_test_1d])
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    print("\n=== BÁO CÁO PHÂN LOẠI ===")
    print(classification_report(y_test, y_pred, target_names=['NORMAL', 'JAM']))

    # ==============================================================
    # VẼ ĐỒ THỊ HISTORY VÀ MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
    # ==============================================================
    print("\n-> Đang xuất đồ thị đánh giá mô hình...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Đồ thị Training History
    axes[0].plot(history.history['accuracy'], label='Train Acc', color='#1f77b4', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', color='#ff7f0e', linestyle='--', linewidth=2)
    axes[0].plot(history.history['loss'], label='Train Loss', color='#2ca02c', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#d62728', linestyle='--', linewidth=2)
    axes[0].set_title('Quá trình học (Training History)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Giá trị', fontsize=12)
    axes[0].legend(loc='center right')
    axes[0].grid(True, linestyle=':', alpha=0.7)

    # 2. Ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['NORMAL', 'JAM'], yticklabels=['NORMAL', 'JAM'],
                annot_kws={"size": 15, "weight": "bold"})
    axes[1].set_title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Mô hình Dự đoán', fontsize=12)
    axes[1].set_ylabel('Nhãn Thực tế', fontsize=12)

    plt.tight_layout()
    plt.show()

    # BẢN ĐỒ NHIỆT
    display_gradcam_overlay(X_test_2d, X_test_1d, y_test, model, num_samples=3)

if __name__ == "__main__":
    main()