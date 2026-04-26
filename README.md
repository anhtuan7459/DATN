![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png)
![alt text](image-6.png)

## Tổng quan về quá trình huấn luyện (`Train.py`)

File `Train.py` mô tả quy trình huấn luyện mô hình Deep Learning (Mạng nơ-ron tích chập - CNN 2D) nhằm phân loại và phát hiện sự cố kẹt (JAM) dựa trên tín hiệu dòng điện và điện áp. Quá trình bao gồm các bước chính thống qua phương pháp biến đổi chu kỳ dòng - áp thành ảnh học sâu:

### 1. Tiền xử lý dữ liệu (Chuyển đổi tín hiệu 1D sang ảnh Quỹ đạo I-V 2D)
- **Tách chu kỳ:** Tín hiệu áp (`Uwave` / `Voltage`) và dòng (`Iwave` / `Current`) được cắt theo các điểm giao không (zero-crossings) của áp để trích xuất từng chu kỳ.
- **Nội suy & Gom lưới (`window_size`):** Các chu kỳ được chuẩn hóa nội suy về 50 điểm và tự động gom nhóm theo cửa sổ trượt (window size biến thiên từ 3 đến 7 chu kỳ).
- **Trích xuất Heatmap:** Chu kỳ dòng - áp sau khi chuẩn hóa tỷ lệ được đưa lên lưới tọa độ 2D tạo thành biểu diễn ảnh không gian (Heatmap) kích thước 64x64, qua đó biểu diễn quỹ đạo I-V. Đặc trưng nhị phân được gán nhãn `1` (JAM) và `0` (Normal).

### 2. Kiến trúc Mô hình (2D CNN)
- Hệ thống sử dụng một mạng nơ-ron tích chập 2D gồm 3 khối `Conv2D` đi kèm `MaxPooling2D` giúp trích lặp linh hoạt các kết cấu hình học độc lạ trên bức ảnh biểu đồ I-V.
- Mạng truyền thẳng `Dense`, kết hợp giải pháp tránh học vẹt bằng tổ hợp `Dropout`, xuất ra giá trị nhị phân đánh giá qua hàm Sigmoid.

### 3. Công cụ tối ưu mô hình mạnh mẽ
- **Tránh rò rỉ dữ liệu (Data Leakage):** Thay vì chia dữ liệu thông thường, kịch bản dùng `GroupKFold` 3-Fold tách biệt tập Train/Val dựa trên nguồn gốc từng File dữ liệu.
- **So khớp với Optuna:** Khung tối ưu Optuna được thiết lập để tự động dò tìm dải tham số cấu hình mạnh nhất (Filters, Dropout, Dense nodes, Learning Rate) cho ra đánh giá ROC AUC cao nhất. 
- **Huấn luyện tự động:** Cấu hình tự cài số lùi tham số `ReduceLROnPlateau` (giảm learning rate khi mất phương hướng hội tụ) và ngắt sớm `EarlyStopping`.

### 4. Đánh giá và Lựa chọn
Quá trình sẽ đánh giá đồng thời toàn bộ các mức cửa sổ `window_size` (3 đến 7) để trích dẫn ra khung chuẩn nhất. Tự động vẽ và trả về:
- Đồ thị so sánh `ROC AUC` của từng kích thước nhóm chu kỳ khác nhau.
- Learning curves về *Loss* và *Accuracy* để đối chiếu độ phập phù của dữ liệu trong quá trình luyện mạng.
- Gợi ý Ngưỡng tối ưu (Optimal Threshold) cực đại bằng tiêu chuẩn Youden's J statistic thông qua đồ thị ROC curve.

### 5. Xây dựng Bản đồ nhiệt giải thích (Explainable AI - Grad-CAM)
- Chức năng cực kỳ quan trọng được áp dụng ở pha cuối là **Grad-CAM**. Hàm này tô màu (Heatmap Overlay) trực tiếp vào vùng không gian Quỹ đạo 2D nơi mạng CNN ấn định đặc trưng kẹt (JAM), qua đó giúp giải thích lý do tại sao dòng và áp tại những vị trí đó lại xuất hiện sai phạm.
