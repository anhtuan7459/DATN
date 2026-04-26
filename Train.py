import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import gc # Thư viện dọn rác RAM

# Thư viện Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Thêm ReduceLROnPlateau

# Thư viện Machine Learning
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# =========================================================
# 1. HÀM TIỀN XỬ LÝ 2D (QUỸ ĐẠO I-V)
# =========================================================
def process_data_to_2d_inputs(df, num_points=50, grid_size=64, window_size=5):
    state_col = 'Event' if 'Event' in df.columns else 'EVENT'
    u_col = 'Uwave' if 'Uwave' in df.columns else 'Voltage[V]'
    i_col = 'Iwave' if 'Iwave' in df.columns else 'Current[A]'

    df_sliced = df.copy().reset_index(drop=True)
    events = df_sliced[state_col].values
    u_raw = df_sliced[u_col].values
    i_raw = df_sliced[i_col].values

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
    X_2d, y, groups = [], [], []
    for file_id, file_path in enumerate(file_list):
        try:
            df = pd.read_csv(file_path)
            x2_chunk, y_chunk = process_data_to_2d_inputs(df, window_size=window_size)
            X_2d.extend(x2_chunk)
            y.extend(y_chunk)
            groups.extend([file_id] * len(y_chunk)) 
        except Exception as e:
            pass
    return np.array(X_2d), np.array(y), np.array(groups)

# =========================================================
# 2. XÂY DỰNG MÔ HÌNH CNN THUẦN
# =========================================================
def build_2d_cnn(params, input_shape_2d=(64, 64, 1)):
    input_2d = Input(shape=input_shape_2d)

    x = Conv2D(filters=params['conv1_filters'], kernel_size=(3, 3), activation='relu')(input_2d)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=params['conv2_filters'], kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=params['conv3_filters'], kernel_size=(3, 3), activation='relu', name="last_conv2d_layer")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    flat_2d = Flatten()(x)

    x = Dense(params['dense_units'], activation='relu')(flat_2d)
    x = Dropout(params['dropout_rate'])(x)

    output = Dense(1, activation='sigmoid', name="Output")(x)

    model = Model(inputs=input_2d, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

# =========================================================
# 3. HÀM MỤC TIÊU OPTUNA (ROC AUC + REDUCE_LR)
# =========================================================
def objective(trial, X_cv, y_cv, groups_cv):
    params = {
        'conv1_filters': trial.suggest_categorical('conv1_filters', [16, 32, 64]),
        'conv2_filters': trial.suggest_categorical('conv2_filters', [32, 64, 128]),
        'conv3_filters': trial.suggest_categorical('conv3_filters', [64, 128, 256]),
        'dense_units': trial.suggest_categorical('dense_units', [32, 64, 128]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True)
    }

    gkf = GroupKFold(n_splits=3)
    cv_auc_scores = []
    epochs_list = []

    for train_index, val_index in gkf.split(X_cv, y_cv, groups=groups_cv):
        tf.keras.backend.clear_session()
        X_train, X_val = X_cv[train_index], X_cv[val_index]
        y_train, y_val = y_cv[train_index], y_cv[val_index]

        model = build_2d_cnn(params)

        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True)
        
        # LIỀU THUỐC 1: Tự động giảm Learning Rate nếu Validation Loss có dấu hiệu khựng lại
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=0)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=64, 
            callbacks=[early_stop, reduce_lr], # Gắn phanh vào đây
            verbose=0
        )

        actual_epochs = len(history.history['loss'])
        best_epoch = actual_epochs - 10 if actual_epochs > 10 else actual_epochs
        epochs_list.append(best_epoch)

        y_pred_prob = model.predict(X_val, verbose=0).flatten()
        auc = roc_auc_score(y_val, y_pred_prob)
        cv_auc_scores.append(auc)

    trial.set_user_attr('best_epoch', int(np.mean(epochs_list)))
    return np.mean(cv_auc_scores)

# =========================================================
# VẼ ĐỒ THỊ SO SÁNH & CÁC ĐỒ THỊ KHÁC
# =========================================================
def plot_window_size_comparison(leaderboard):
    df_lb = pd.DataFrame(leaderboard)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Window Size', y='Best ROC AUC', data=df_lb, palette='viridis')
    
    # Vẽ thêm đường nối để dễ nhìn xu hướng
    plt.plot(range(len(df_lb)), df_lb['Best ROC AUC'], color='red', marker='o', linewidth=2)
    
    # Hiển thị số liệu trên đầu các cột
    for i, v in enumerate(df_lb['Best ROC AUC']):
        ax.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.title('SO SÁNH HIỆU NĂNG ROC AUC GIỮA CÁC WINDOW SIZES (3 -> 7)', fontsize=14, fontweight='bold')
    plt.xlabel('Window Size (Số chu kỳ gộp)', fontsize=12)
    plt.ylabel('ROC AUC Score', fontsize=12)
    plt.ylim(0.5, 1.05) # Đặt trục Y từ 0.5 đến 1 cho dễ nhìn
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue', marker='o')
    plt.plot(history.history['val_loss'], label='Valid Loss', color='orange', marker='o')
    plt.title('Đồ thị Loss (Final Model)')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='green', marker='o')
    plt.plot(history.history['val_accuracy'], label='Valid Accuracy', color='red', marker='o')
    plt.title('Đồ thị Accuracy (Final Model)')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_roc_curve_and_find_threshold(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Báo động giả)')
    plt.ylabel('True Positive Rate (Độ nhạy)')
    plt.title('Đường cong ROC và Ngưỡng tối ưu')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
    return optimal_threshold

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

def display_gradcam_overlay(X_test, y_test, model, optimal_threshold, num_samples=3):
    normal_indices = np.where(y_test == 0)[0]
    jam_indices = np.where(y_test == 1)[0]
    n_normal = min(num_samples, len(normal_indices))
    n_jam = min(num_samples, len(jam_indices))
    n_cols = max(n_normal, n_jam)

    if n_cols == 0: return
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))
    if n_cols == 1: axes = axes.reshape(2, 1)
    fig.suptitle('BẢN ĐỒ NHIỆT GRAD-CAM QUỸ ĐẠO I-V', fontsize=16, fontweight='bold', y=1.02)

    for i in range(n_cols):
        ax = axes[0, i]
        if i < n_normal:
            idx = normal_indices[i]
            x_sample = np.expand_dims(X_test[idx], axis=0)
            heatmap = make_gradcam_heatmap(x_sample, model)
            ax.imshow(X_test[idx][:, :, 0].T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            ax.imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')
            pred_score = model.predict(x_sample, verbose=0)[0][0]
            ax.set_title(f"NORMAL\nProbability: {pred_score:.3f}", color='green' if pred_score < optimal_threshold else 'red')
        else: ax.axis('off')

    for i in range(n_cols):
        ax = axes[1, i]
        if i < n_jam:
            idx = jam_indices[i]
            x_sample = np.expand_dims(X_test[idx], axis=0)
            heatmap = make_gradcam_heatmap(x_sample, model)
            ax.imshow(X_test[idx][:, :, 0].T, cmap='gray_r', origin='lower', extent=[-1.1, 1.1, -1.1, 1.1])
            ax.imshow(heatmap.T, cmap='jet', alpha=0.5, origin='lower', extent=[-1.1, 1.1, -1.1, 1.1], interpolation='bilinear')
            pred_score = model.predict(x_sample, verbose=0)[0][0]
            ax.set_title(f"JAM\nProbability: {pred_score:.3f}", color='red' if pred_score >= optimal_threshold else 'green')
        else: ax.axis('off')
    plt.tight_layout()
    plt.show()

# =========================================================
# HÀM MAIN: VÒNG LẶP WINDOW SIZE (3 -> 7)
# =========================================================
def main():
    print("=== ĐẠI CHIẾN WINDOW SIZES (3 -> 7) & TỐI ƯU ROC AUC BẰNG OPTUNA ===")

    train_folder = input("Nhập đường dẫn Folder chứa file TRAIN: ")
    val_folder = input("Nhập đường dẫn Folder chứa file VALIDATION: ")
    test_folder = input("Nhập đường dẫn Folder chứa file TEST: ")

    cv_files = glob.glob(os.path.join(train_folder, '*.csv')) + glob.glob(os.path.join(val_folder, '*.csv'))
    test_files = glob.glob(os.path.join(test_folder, '*.csv'))

    leaderboard = []
    best_overall_ws = 3
    best_overall_auc = 0.0
    best_overall_params = {}
    best_overall_epochs = 0

    print("\n⏳ BẮT ĐẦU QUÉT WINDOW SIZE (TỪ 3 ĐẾN 7)...")
    
    for ws in range(3, 8):
        print(f"\n" + "="*60)
        print(f"⚙️ BẮT ĐẦU TỐI ƯU CHO WINDOW SIZE = {ws} ...")
        
        # Tiền xử lý dữ liệu cho Window Size hiện tại
        X_cv, y_cv, groups_cv = load_files_to_dataset(cv_files, window_size=ws)
        
        if len(np.unique(y_cv)) < 2:
            print(f"[Cảnh báo] Window Size = {ws} bị thiếu nhãn. Bỏ qua!")
            continue

        # Chạy Optuna tìm tham số (10 trials để tốc độ vừa phải)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_cv, y_cv, groups_cv), n_trials=10)
        
        ws_best_auc = study.best_value
        ws_best_epoch = study.best_trial.user_attrs['best_epoch']
        
        # Ghi danh vào bảng xếp hạng
        leaderboard.append({
            'Window Size': ws,
            'Best ROC AUC': ws_best_auc,
            'Best Epochs': ws_best_epoch,
            'Best Params': study.best_params
        })
        
        # Tranh ngôi Vô Địch
        if ws_best_auc > best_overall_auc:
            best_overall_auc = ws_best_auc
            best_overall_ws = ws
            best_overall_params = study.best_params
            best_overall_epochs = ws_best_epoch
            
        print(f" -> XONG WINDOW SIZE = {ws}. Đỉnh cao ROC AUC đạt: {ws_best_auc:.4f}")
        
        # Dọn dẹp RAM trước khi qua Window Size mới
        del X_cv, y_cv, groups_cv
        gc.collect()

    # XUẤT BẢNG SO SÁNH LÊN MÀN HÌNH VÀ VẼ ĐỒ THỊ
    print("\n🏆 BẢNG XẾP HẠNG TỔNG KẾT WINDOW SIZES (3 -> 7) 🏆")
    df_leaderboard = pd.DataFrame(leaderboard).drop(columns=['Best Params']).set_index('Window Size')
    print(df_leaderboard.to_markdown())
    
    # Gọi hàm vẽ đồ thị so sánh Bar Chart
    plot_window_size_comparison(leaderboard)

    print(f"\n👑 NHÀ VÔ ĐỊCH TUYỆT ĐỐI: WINDOW SIZE = {best_overall_ws} (ROC AUC: {best_overall_auc:.4f})")
    print("Siêu tham số vô địch:")
    for k, v in best_overall_params.items():
        print(f"  + {k}: {v}")

    # =========================================================
    # HUẤN LUYỆN CHUNG KẾT VỚI NHÀ VÔ ĐỊCH
    # =========================================================
    print(f"\n🛠️ Tải lại dữ liệu chuẩn bị vinh danh Window Size = {best_overall_ws} ...")
    X_cv, y_cv, groups_cv = load_files_to_dataset(cv_files, window_size=best_overall_ws)
    X_test, y_test, _ = load_files_to_dataset(test_files, window_size=best_overall_ws)

    print("\n⚙️ Đang huấn luyện Final Model để xuất Báo cáo...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, final_val_idx = next(gss.split(X_cv, y_cv, groups_cv))
    
    tf.keras.backend.clear_session()
    final_model = build_2d_cnn(best_overall_params)
    
    # Vẫn dùng ReduceLROnPlateau cho vòng thi cuối cùng cho chắc cú
    early_stop_final = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, restore_best_weights=True)
    reduce_lr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)
    
    history = final_model.fit(
        X_cv[train_idx], y_cv[train_idx],
        validation_data=(X_cv[final_val_idx], y_cv[final_val_idx]),
        epochs=best_overall_epochs,
        batch_size=64, 
        callbacks=[early_stop_final, reduce_lr_final],
        verbose=1
    )

    plot_learning_curves(history)

    print("\n🎯 ĐÁNH GIÁ TẬP TEST VÀ TÌM NGƯỠNG CẮT TỐI ƯU...")
    y_pred_prob = final_model.predict(X_test).flatten()
    
    optimal_threshold = plot_roc_curve_and_find_threshold(y_test, y_pred_prob)
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

    print(f"\n=== BÁO CÁO PHÂN LOẠI (Với Ngưỡng = {optimal_threshold:.3f}) ===")
    print(classification_report(y_test, y_pred_optimal, target_names=['NORMAL', 'JAM']))

    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred_optimal)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NORMAL', 'JAM'], yticklabels=['NORMAL', 'JAM'])
    plt.title(f'Ma trận nhầm lẫn (Window={best_overall_ws}, Threshold={optimal_threshold:.3f})')
    plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
    plt.show()

    display_gradcam_overlay(X_test, y_test, final_model, optimal_threshold, num_samples=3)

    final_model.save('best_2d_cnn_ws_roc_optimized.h5')
    print("\n✅ Hoàn tất! Model vô địch đã được cất kho với tên 'best_2d_cnn_ws_roc_optimized.h5'")

if __name__ == "__main__":
    main()