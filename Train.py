import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Thư viện Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.callbacks import EarlyStopping

# Thư viện Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# 0. HÀM VẼ ĐỒ THỊ 1D
# =========================================================
def plot_raw_1d_signals(df_sliced, state_col, col_u, col_i):
    plt.figure(figsize=(15, 5))
    u_norm = df_sliced[col_u] / (df_sliced[col_u].abs().max() + 1e-9)
    i_norm = df_sliced[col_i] / (df_sliced[col_i].abs().max() + 1e-9)

    plt.plot(u_norm, label='Uwave (Normalized)', alpha=0.7, color='blue')
    plt.plot(i_norm, label='Iwave (Normalized)', alpha=0.7, color='orange')

    jam_indices = df_sliced.index[df_sliced[state_col] == 'JAM'].tolist()
    if jam_indices:
        start_jam = jam_indices[0]
        plt.axvline(x=start_jam, color='red', linestyle='--', linewidth=2, label='Start of JAM Event')
        start_unknown = max(0, start_jam - 1000)
        plt.axvspan(start_unknown, start_jam, color='yellow', alpha=0.3, label='UNKNOWN Buffer')

    plt.title("Biểu đồ tín hiệu 1D (U & I) và ranh giới các trạng thái", fontsize=14, fontweight='bold')
    plt.xlabel("Index (Data Points)")
    plt.ylabel("Biên độ (Chuẩn hóa)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# =========================================================
# 1. HÀM TIỀN XỬ LÝ
# =========================================================
def process_data_to_hybrid_inputs(df, num_points=50, grid_size=64, window_size=5):
    state_col = 'Event' if 'Event' in df.columns else 'EVENT'
    col_u = 'Uwave' if 'Uwave' in df.columns else 'Voltage[V]'
    col_i = 'Iwave' if 'Iwave' in df.columns else 'Current[A]'

    jam_indices = df.index[df[state_col] == 'JAM'].tolist()
    if not jam_indices:
        print(" -> File không có nhãn JAM, bỏ qua.")
        return [], [], []

    first_jam = jam_indices[0]
    start_idx = max(0, first_jam - 6000)
    end_idx = min(len(df), first_jam + 6000)

    df_sliced = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    new_first_jam = first_jam - start_idx
    start_unknown = max(0, new_first_jam - 1000)
    df_sliced.loc[start_unknown : new_first_jam - 1, state_col] = 'UNKNOWN'

    if 'first_plot_done' not in globals():
        plot_raw_1d_signals(df_sliced, state_col, col_u, col_i)
        global first_plot_done
        first_plot_done = True

    u_norm = df_sliced[col_u].values / (df_sliced[col_u].abs().max() + 1e-9)
    i_norm = df_sliced[col_i].values / (df_sliced[col_i].abs().max() + 1e-9)
    events = df_sliced[state_col].values

    zero_crossings = np.where((u_norm[:-1] < 0) & (u_norm[1:] >= 0))[0]
    cycles_1d = []
    cycles_labels = []

    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i+1]
        if end - start < 10: continue

        old_idx = np.linspace(0, 1, end - start)
        new_idx = np.linspace(0, 1, num_points)
        interp_u = np.interp(new_idx, old_idx, u_norm[start:end])
        interp_i = np.interp(new_idx, old_idx, i_norm[start:end])

        cycles_1d.append(np.column_stack((interp_u, interp_i)))

        cycle_events = events[start:end]
        majority_event = max(set(cycle_events), key=list(cycle_events).count)
        cycles_labels.append(majority_event)

    X_1d_list, X_2d_list, y_list = [], [], []
    bins = np.linspace(-1.1, 1.1, grid_size + 1)

    for i in range(0, len(cycles_1d), window_size):
        chunk_1d = cycles_1d[i : i + window_size]
        chunk_labels = cycles_labels[i : i + window_size]

        if len(chunk_1d) < window_size:
            continue

        majority_label = max(set(chunk_labels), key=chunk_labels.count)

        if str(majority_label).upper() == 'UNKNOWN':
            continue

        window_1d = np.vstack(chunk_1d)
        X_1d_list.append(window_1d)

        u_window = window_1d[:, 0]
        i_window = window_1d[:, 1]
        heatmap, _, _ = np.histogram2d(u_window, i_window, bins=(bins, bins))
        window_2d = np.where(heatmap > 0, 1.0, 0.0)[..., np.newaxis]
        X_2d_list.append(window_2d)

        y_list.append(1 if str(majority_label).upper() == 'JAM' else 0)

    return X_1d_list, X_2d_list, y_list

# =========================================================
# 2. XÂY DỰNG MÔ HÌNH HYBRID (1-D + 2-D CNN)
# =========================================================
def build_hybrid_cnn(input_shape_1d=(250, 2), input_shape_2d=(64, 64, 1)):
    input_1d = Input(shape=input_shape_1d, name="1D_Time_Series")
    x1 = Conv1D(filters=32, kernel_size=5, activation='relu')(input_1d)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Conv1D(filters=64, kernel_size=5, activation='relu')(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)
    flat_1d = Flatten()(x1)

    input_2d = Input(shape=input_shape_2d, name="2D_Trajectory_Image")
    x2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_2d)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    # QUAN TRỌNG: Đặt tên cho lớp Conv2D cuối cùng để trích xuất Grad-CAM
    x2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name="last_conv2d_layer")(x2)
    x2 = MaxPooling2D(pool_size=(2, 2))(x2)
    flat_2d = Flatten()(x2)

    merged = concatenate([flat_1d, flat_2d])

    z = Dense(128, activation='relu')(merged)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, activation='sigmoid', name="Output")(z)

    model = Model(inputs=[input_1d, input_2d], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# =========================================================
# 3. HÀM TÍNH TOÁN VÀ VẼ BẢN ĐỒ NHIỆT (GRAD-CAM)
# =========================================================
def make_gradcam_heatmap(input_1d_array, input_2d_array, model, last_conv_layer_name="last_conv2d_layer"):
    """Tạo ma trận bản đồ nhiệt bằng kỹ thuật Grad-CAM"""
    # Tạo một mô hình phụ lấy đầu ra là lớp Conv2D cuối cùng và kết quả dự đoán
    grad_model = tf.keras.models.Model(
        [model.inputs[0], model.inputs[1]], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Tính toán Gradient (Đạo hàm)
    with tf.GradientTape() as tape:
        inputs = [input_1d_array, input_2d_array]
        last_conv_layer_output, preds = grad_model(inputs)
        # Vì đây là bài toán phân loại nhị phân (Sigmoid), ta lấy luôn giá trị dự đoán
        class_channel = preds[:, 0]

    # Tính gradient của điểm số dự đoán với lớp Feature Map cuối cùng
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Tính trung bình gradient (Trọng số quan trọng của từng bộ lọc)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Nhân các Feature Map với trọng số quan trọng tương ứng
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Lọc bỏ giá trị âm (Chỉ quan tâm tới các pixel đóng góp tích cực vào quyết định KẸT)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam_overlay(X1_test, X2_test, y_test, model):
    """Hiển thị ảnh quỹ đạo gốc và phủ bản đồ nhiệt Grad-CAM lên trên"""
    print("\n-> Đang tạo Bản đồ nhiệt Grad-CAM để giải thích mô hình...")
    
    # Lấy 1 mẫu NORMAL và 1 mẫu JAM từ tập Test
    normal_idx = np.where(y_test == 0)[0][0]
    jam_idx = np.where(y_test == 1)[0][0]

    samples = [normal_idx, jam_idx]
    titles = ['NORMAL (Nhãn 0)', 'JAM (Nhãn 1)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('EXPLAINABLE AI: BẢN ĐỒ NHIỆT GRAD-CAM (Mạng CNN đang tập trung vào đâu?)', fontsize=14, fontweight='bold')

    for i, idx in enumerate(samples):
        # Lấy dữ liệu 1 mẫu (Thêm chiều batch size ở đầu)
        x1_sample = np.expand_dims(X1_test[idx], axis=0)
        x2_sample = np.expand_dims(X2_test[idx], axis=0)

        # Tạo heatmap
        heatmap = make_gradcam_heatmap(x1_sample, x2_sample, model)
        
        # Ảnh gốc (Bỏ chiều channel)
        img_original = X2_test[idx][:, :, 0]

        # Vẽ ảnh gốc (Quỹ đạo trắng đen)
        axes[i].imshow(img_original.T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
        
        # Phủ bản đồ nhiệt lên trên (Dùng tham số alpha để làm trong suốt)
        # Interpolation 'bilinear' giúp làm mịn bản đồ nhiệt từ kích thước nhỏ (vd: 15x15) lên 64x64
        axes[i].imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')
        
        # Tính toán xem mô hình tự tin bao nhiêu %
        pred_score = model.predict([x1_sample, x2_sample], verbose=0)[0][0]
        
        axes[i].set_title(f"{titles[i]} | AI Dự đoán JAM: {pred_score*100:.1f}%")
        axes[i].set_xlabel('U (chuẩn hóa)')
        axes[i].set_ylabel('I (chuẩn hóa)')
        axes[i].grid(True, color='black', linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.show()

# =========================================================
# 4. HÀM CHÍNH
# =========================================================
def main():
    print("=== HUẤN LUYỆN HYBRID CNN (1D + 2D) - WINDOW 5 - CẮT 6000 ĐIỂM ===")
    folder_path = input("Nhập đường dẫn Folder chứa file CSV (Bỏ trống để dùng giả lập): ")
    window_size = 5

    X1_all, X2_all, y_all = [], [], []

    # --- BƯỚC 1: TẠO DATASET ---
    if folder_path.strip() == "":
        print("-> Đang tạo dữ liệu giả lập...")
        for _ in range(3):
            t = np.linspace(0, 1000 * np.pi, 20000)
            u = 220 * np.sin(t)
            i = np.where(t < 500 * np.pi,
                         3.6 * np.sin(t + np.pi/6) + np.random.normal(0, 0.2, len(t)),
                         15.0 * np.sin(t + np.pi/4) + np.random.normal(0, 1.0, len(t)) + 2.0)
            event = ['NORMAL' if time < 500 * np.pi else 'JAM' for time in t]

            df_sim = pd.DataFrame({'Uwave': u, 'Iwave': i, 'Event': event})
            x1, x2, y = process_data_to_hybrid_inputs(df_sim, window_size=window_size)
            X1_all.extend(x1); X2_all.extend(x2); y_all.extend(y)
    else:
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        if not csv_files:
            print(f"Không tìm thấy file .csv nào trong {folder_path}!")
            return

        for idx, file_path in enumerate(csv_files, 1):
            print(f"-> Đang xử lý file [{idx}/{len(csv_files)}]: {os.path.basename(file_path)}")
            try:
                df = pd.read_csv(file_path)
                x1, x2, y = process_data_to_hybrid_inputs(df, window_size=window_size)
                X1_all.extend(x1); X2_all.extend(x2); y_all.extend(y)
            except Exception as e:
                print(f"Lỗi: {e}")

    X1_all, X2_all, y_all = np.array(X1_all), np.array(X2_all), np.array(y_all)

    print(f"\n-> Tổng số mẫu (Windows = {window_size}): {len(y_all)}")
    if len(y_all) == 0:
        print("Tập dữ liệu rỗng. Kết thúc!")
        return
    print(f"-> Số mẫu NORMAL (0): {np.sum(y_all == 0)} | Số mẫu JAM (1): {np.sum(y_all == 1)}")

    # --- BƯỚC 2: TRAIN-TEST SPLIT ---
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
        X1_all, X2_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # --- BƯỚC 3: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH ---
    print(f"\n-> Khởi tạo mô hình Hybrid CNN (1D: {X1_all.shape[1:]}, 2D: {X2_all.shape[1:]})...")
    model = build_hybrid_cnn(input_shape_1d=(window_size * 50, 2), input_shape_2d=(64, 64, 1))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x=[X1_train, X2_train],
        y=y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # --- BƯỚC 4: ĐÁNH GIÁ VÀ TRỰC QUAN HÓA ---
    print("\n-> Đang đánh giá trên tập Test...")
    y_pred_prob = model.predict([X1_test, X2_test])
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

    print("\n=== BÁO CÁO PHÂN LOẠI ===")
    print(classification_report(y_test, y_pred, target_names=['NORMAL', 'JAM']))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(history.history['accuracy'], label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc', linestyle='--')
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss', linestyle='--')
    axes[0].set_title('Quá trình học (Training History)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Giá trị')
    axes[0].legend(); axes[0].grid(True)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['NORMAL', 'JAM'], yticklabels=['NORMAL', 'JAM'])
    axes[1].set_title('Ma trận nhầm lẫn (Confusion Matrix)')
    axes[1].set_xlabel('Dự đoán'); axes[1].set_ylabel('Thực tế')

    plt.tight_layout()
    plt.show()

    # === GỌI HÀM VẼ BẢN ĐỒ NHIỆT GRAD-CAM Ở ĐÂY ===
    display_gradcam_overlay(X1_test, X2_test, y_test, model)
    # ===============================================

    print("\n-> Đang đóng gói và tải mô hình về máy...")
    model.save('hybrid_cnn_jam.h5')
    from google.colab import files
    files.download('hybrid_cnn_jam.h5')
    print("-> Xong! Vui lòng kiểm tra thư mục Download trên máy tính.")

if __name__ == "__main__":
    main()