# 🚀 Hệ thống Nhận diện Kẹt tải Động cơ Thời gian thực (Real-time AI Jam Detection)

Dự án Firmware nhúng trên vi điều khiển **STM32H562RGTX** kết hợp AI (TinyML) để phát hiện trạng thái kẹt (Jam) của tải động cơ điện xoay chiều thông qua tín hiệu dòng điện (I) và điện áp (U) trích xuất từ IC đo lường **BL0940**.

---

## 🛠 1. Kiến trúc Hệ thống (Hardware & OS)
- **Vi điều khiển (MCU):** STM32H562RGTX (Lõi Cortex-M33 tốc độ cao, tích hợp phần cứng FPU).
- **Cảm biến:** IC đo công suất và điện năng **BL0940** (Giao tiếp SPI 1kHz qua DMA).
- **Hệ điều hành tích hợp:** Azure RTOS (ThreadX) đa luồng.
- **Trí tuệ nhân tạo (TinyML):** X-CUBE-AI (Chạy mô hình mạng nén CNN 2D kích thước ~327KB Flash, dùng lượng tử hóa Int8).
- **Giao thức ngoại vi:** USB Device (CDC-ACM) ảo hóa COM Port để truyền Data AI thời gian thực lên máy tính, UART Debug.

---

## 📂 2. Cấu trúc Thư mục Dự án

- `Core/Src/` & `Core/Inc/`: Chứa mã nguồn logic chính của ứng dụng.
  - **`bl0940_driver.c`**: Driver giao tiếp SPI đọc dòng/áp thô 20-bit mỗi 1 mili-giây.
  - **`ai_preprocess.c`**: Thu thập dữ liệu dao động xoay chiều, tìm điểm Zero-crossing cắt gốc 0V, nội suy đồ thị và vẽ bản đồ tương quan U-I dưới dạng Histogram ảnh 64x64.
  - **`jam_ai.c`**: Nạp tấm ảnh vào lõi mạng Neural Network (X-CUBE-AI), chạy phép tính nhận diện khả năng Kẹt (Jam score), xuất ra tỷ lệ % dự đoán.
  - **`jam_led.c`**: Đọc trạng thái kẹt để đá còi/nháy đèn LED (25Hz) cảnh báo.
  - **`jam_detect.c`**: Giao diện đóng gói API chính điều phối các luồng chạy.
  - **`app_threadx.c`**: Phân chia tài nguyên RAM và điều phối độ ưu tiên ngắt cho các luồng hệ điều hành RTOS.
- `X-CUBE-AI/App/`: Chứa mô hình AI "bê-tông hóa", mảng weights tĩnh và các hàm kích hoạt Activation kích thước giấy nháp 24KB RAM.
- `USBX/`: Thư viện lõi xử lý USB Device chuẩn ngắt tốc độ Full-Speed hỗ trợ kết nối Data trực tiếp lên PC.
- `quantization/`: Bộ log mô tả độ nén và giới hạn của mô hình ban đầu khi build xuống Chip.

---

## ⚙️ 3. Quy trình Luồng Data (Data Flow)

1. **Ngắt Timer đo đạc (1kHz):** Mỗi 1ms, TIM1 kích hoạt SPI DMA lấy tín hiệu gốc từ cảm biến BL0940. Cất vào 3 bộ đệm quay vòng Circular Buffers (Triple Buffering).
2. **Tiền xử lý (Thread Ưu tiên 18):** Lấy Data từ hàng đợi, tách lấy 5 chu kỳ AC 50Hz hoàn chỉnh gần nhất. Cân bằng biên độ và chuyển hóa mảng 5 chu kỳ đó thành 1 ma trận điểm ảnh 64x64 bin.
3. **Suy luận AI (Thread Ưu tiên 19):** Chạy hàm `ai_network_run()`. Bộ xử lý quét ma trận trong vài mili-giây. Trả về kết quả 0~255 (Tương đương 0% -> 100%).
   - Nếu *Score ≥ 50%* ➔ Phát tín hiệu `Jam = 1` (Có Kẹt Tải).
   - Nếu *Score < 50%* ➔ Động cơ `Jam = 0` (Bình Thường).
4. **Hành động & Phản hồi (Thread Ưu tiên 20 & USB):** Phản hồi đèn LED báo động trên chân GPIO. Đóng gói kết quả gửi qua cổng USB COM về phía máy tính để Script Python (ai_live_deploy) hiển thị theo giời gian thực.

---

## 🚀 4. Hướng dẫn Biên dịch & Nạp Code
1. Mở file project cấu hình `.ioc` `STM32h5_Bl0940.ioc` bằng phần mềm **STM32CubeIDE**.
2. Click chuột phải vào Project Name trong tab Project Explorer $\rightarrow$ Chọn **Build Project**.
3. Kết nối board mạch dùng mạch nạp J-Link / ST-Link. Click vào biểu tượng con bọ xanh (Debug) hoặc **Run** để tiến hành flash firmware xuống bộ nhớ ROM Flash (Tại địa chỉ `0x08000000`).
4. Cắm cáp USB Type-C thứ 2 lên phần cấp nguồn USB của board vi điều khiển nối vào Máy tính. Windows sẽ nhận diện tự động thành thiết bị `USB Serial Device (COMx)`.
5. Mở Script Python hoặc mở phần mềm Terminal (2000000 baud) để theo dõi dòng log trạng thái của AI!
