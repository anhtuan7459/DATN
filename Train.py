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

# Thư viện Machine Learning
from sklearn.metrics import classification_report, confusion_matrix

# =========================================================
# 0. HÀM VẼ ĐỒ THỊ 1D
# =========================================================
def plot_raw_1d_signals(df_sliced, state_col, u_col, i_col):
    plt.figure(figsize=(15, 5))
    
    # Chỉ vẽ trực quan nên có thể chuẩn hóa nhanh
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

# =========================================================
# 1. HÀM TIỀN XỬ LÝ (CHỈ XUẤT ẢNH 2D TỪ I-V)
# =========================================================
def process_data_to_2d_inputs(df, num_points=50, grid_size=64, window_size=5):
    state_col = 'Event' if 'Event' in df.columns else 'EVENT'
    u_col_2d = 'Uwave' if 'Uwave' in df.columns else 'Voltage[V]'
    i_col_2d = 'Iwave' if 'Iwave' in df.columns else 'Current[A]'

    df_sliced = df.copy().reset_index(drop=True)
    events = df_sliced[state_col].values

    jam_starts = np.where((events[:-1] != 'JAM') & (events[1:] == 'JAM'))[0] + 1
    if len(jam_starts) == 0 and 'JAM' in events:
        jam_starts = [np.where(events == 'JAM')[0][0]]

    for start_jam in jam_starts:
        start_unknown = max(0, start_jam - 1000)
        events[start_unknown:start_jam] = 'UNKNOWN'
    
    df_sliced[state_col] = events

    if 'first_plot_done' not in globals():
        plot_raw_1d_signals(df_sliced, state_col, u_col_2d, i_col_2d)
        global first_plot_done
        first_plot_done = True

    # KHÔNG chuẩn hóa toàn file nữa, lấy giá trị thô để tính toán
    u_raw = df_sliced[u_col_2d].values
    i_raw = df_sliced[i_col_2d].values
    
    # Tìm điểm cắt 0 trên tín hiệu U
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

        if len(chunk_raw) < window_size:
            continue

        majority_label = max(set(chunk_labels), key=chunk_labels.count)

        if str(majority_label).upper() == 'UNKNOWN':
            continue

        # --- CẢI TIẾN: Lấy trung bình theo pha thay vì vstack ---
        chunk_array = np.stack(chunk_raw, axis=0) # Shape: (window_size, num_points, 2)
        mean_cycle = np.mean(chunk_array, axis=0) # Shape: (num_points, 2)

        u_window = mean_cycle[:, 0]
        i_window = mean_cycle[:, 1]

        # --- CẢI TIẾN: Chuẩn hóa theo từng window ---
        u_norm = u_window / (np.max(np.abs(u_window)) + 1e-9)
        i_norm = i_window / (np.max(np.abs(i_window)) + 1e-9)

        # Tính biểu đồ nhiệt
        heatmap, _, _ = np.histogram2d(u_norm, i_norm, bins=(bins, bins))
        
        # Cường độ sáng của điểm ảnh cũng được chuẩn hóa cục bộ
        window_2d = heatmap / (heatmap.max() + 1e-9) 
        window_2d = window_2d[..., np.newaxis] # Expand dims cho Keras (64, 64, 1)
        
        X_2d_list.append(window_2d)
        y_list.append(1 if str(majority_label).upper() == 'JAM' else 0)

    return X_2d_list, y_list

# =========================================================
# 2. XÂY DỰNG MÔ HÌNH CNN 2D THUẦN TÚY
# =========================================================
def build_2d_cnn(input_shape_2d=(64, 64, 1)):
    input_2d = Input(shape=input_shape_2d, name="2D_Trajectory_Image")
    
    x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_2d)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name="last_conv2d_layer")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    flat_2d = Flatten()(x)

    z = Dense(128, activation='relu')(flat_2d)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu')(z)
    output = Dense(1, activation='sigmoid', name="Output")(z)

    model = Model(inputs=input_2d, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# =========================================================
# 3. HÀM TÍNH TOÁN VÀ VẼ BẢN ĐỒ NHIỆT (GRAD-CAM)
# =========================================================
def make_gradcam_heatmap(input_2d_array, model, last_conv_layer_name="last_conv2d_layer"):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

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

def display_gradcam_overlay(X_test, y_test, model):
    print("\n-> Đang tạo Bản đồ nhiệt Grad-CAM để giải thích mô hình...")

    normal_indices = np.where(y_test == 0)[0]
    jam_indices = np.where(y_test == 1)[0]
    
    if len(normal_indices) == 0 or len(jam_indices) == 0:
        print("Không đủ cả nhãn NORMAL và JAM trong tập Test để vẽ Grad-CAM!")
        return

    normal_idx = normal_indices[0]
    jam_idx = jam_indices[0]

    samples = [normal_idx, jam_idx]
    titles = ['NORMAL (Nhãn 0)', 'JAM (Nhãn 1)']

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('EXPLAINABLE AI: BẢN ĐỒ NHIỆT GRAD-CAM QUỸ ĐẠO I-V', fontsize=14, fontweight='bold')

    for i, idx in enumerate(samples):
        x_sample = np.expand_dims(X_test[idx], axis=0)

        heatmap = make_gradcam_heatmap(x_sample, model)
        img_original = X_test[idx][:, :, 0]

        axes[i].imshow(img_original.T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
        axes[i].imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')

        pred_score = model.predict(x_sample, verbose=0)[0][0]

        axes[i].set_title(f"{titles[i]} | AI Dự đoán JAM: {pred_score*100:.1f}%")
        axes[i].set_xlabel('U (chuẩn hóa)')
        axes[i].set_ylabel('I (chuẩn hóa)')
        axes[i].grid(True, color='black', linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.show()

# =========================================================
# 4. HÀM CHÍNH VÀ CÁC HÀM PHỤ TRỢ 
# =========================================================
def load_files_to_dataset(file_list, window_size):
    X_2d, y = [], []
    for idx, file_path in enumerate(file_list, 1):
        print(f"   + Xử lý file [{idx}/{len(file_list)}]: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path)
            x2_chunk, y_chunk = process_data_to_2d_inputs(df, window_size=window_size)
            X_2d.extend(x2_chunk)
            y.extend(y_chunk)
            print(f"     -> Trích xuất được {len(y_chunk)} mẫu ảnh 2D.")
        except Exception as e:
            print(f"   ! Lỗi khi đọc file {file_path}: {e}")
    return np.array(X_2d), np.array(y)

def main():
    print("=== HUẤN LUYỆN CNN 2D (QUỸ ĐẠO I-V) ===")
    
    train_folder_path = input("Nhập đường dẫn Folder chứa file TRAIN (Bỏ trống để dùng giả lập): ")
    window_size = 5

    if train_folder_path.strip() == "":
        print("\n-> Đang tạo dữ liệu giả lập...")
        def gen_sim_data(num_files):
            x2_list, y_list = [], []
            for _ in range(num_files):
                t = np.linspace(0, 1000 * np.pi, 20000)
                u = 220 * np.sin(t)
                i = np.where(t < 500 * np.pi,
                             3.6 * np.sin(t + np.pi/6) + np.random.normal(0, 0.2, len(t)),
                             15.0 * np.sin(t + np.pi/4) + np.random.normal(0, 1.0, len(t)) + 2.0)
                event = ['NORMAL' if time < 500 * np.pi else 'JAM' for time in t]
                
                df_sim = pd.DataFrame({'Uwave': u, 'Iwave': i, 'Event': event})
                
                x2, y = process_data_to_2d_inputs(df_sim, window_size=window_size)
                x2_list.extend(x2); y_list.extend(y)
            return np.array(x2_list), np.array(y_list)

        X_train, y_train = gen_sim_data(3)
        X_test, y_test = gen_sim_data(1)
        
    else:
        test_folder_path = input("Nhập đường dẫn Folder chứa file TEST: ")
        
        train_files = glob.glob(os.path.join(train_folder_path, '*.csv'))
        test_files = glob.glob(os.path.join(test_folder_path, '*.csv'))
        
        if len(train_files) == 0:
            print(f"Không tìm thấy file CSV nào trong thư mục TRAIN: {train_folder_path}!")
            return
            
        if len(test_files) == 0:
            print(f"Không tìm thấy file CSV nào trong thư mục TEST: {test_folder_path}!")
            return
            
        print(f"\n-> Đã tìm thấy {len(train_files)} files TRAIN và {len(test_files)} files TEST.")

        print("\n=== ĐANG TẠO TẬP TRAIN ===")
        X_train, y_train = load_files_to_dataset(train_files, window_size)

        print("\n=== ĐANG TẠO TẬP TEST ===")
        X_test, y_test = load_files_to_dataset(test_files, window_size)

    if len(y_train) == 0 or len(y_test) == 0:
        print("\nDữ liệu Train hoặc Test bị rỗng. Vui lòng kiểm tra lại file CSV của bạn!")
        return

    print(f"\n-> TỔNG QUAN DỮ LIỆU SAU KHI GỘP TẤT CẢ:")
    print(f"   + TRAIN: {len(y_train)} mẫu ảnh (NORMAL: {np.sum(y_train==0)}, JAM: {np.sum(y_train==1)})")
    print(f"   + TEST:  {len(y_test)} mẫu ảnh (NORMAL: {np.sum(y_test==0)}, JAM: {np.sum(y_test==1)})")

    # --- HUẤN LUYỆN MÔ HÌNH ---
    print(f"\n-> Khởi tạo mô hình CNN 2D...")
    model = build_2d_cnn(input_shape_2d=(64, 64, 1))

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        x=X_train,
        y=y_train,
        validation_split=0.2, 
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # --- ĐÁNH GIÁ TRÊN TẬP TEST ---
    print("\n-> Đang đánh giá trên tập Test ...")
    y_pred_prob = model.predict(X_test)
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

    # BẢN ĐỒ NHIỆT
    display_gradcam_overlay(X_test, y_test, model)

    print("\n-> Đang đóng gói và tải mô hình về máy...")
    model.save('cnn2d_jam.h5')
    try:
        from google.colab import files
        files.download('cnn2d_jam.h5')
        print("-> Xong! Vui lòng kiểm tra thư mục Download trên máy tính.")
    except:
        print("-> Mô hình đã được lưu tại thư mục hiện tại: cnn2d_jam.h5")

if __name__ == "__main__":
    main()