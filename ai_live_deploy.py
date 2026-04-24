import serial
import struct
import math
import numpy as np
import time
import threading
from collections import deque
from tensorflow.keras.models import load_model
import os

# Tắt cảnh báo rác của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. CẤU HÌNH PHẦN CỨNG & KHO CHỨA
# ==========================================
PORT = 'COM5'  
BAUDRATE = 2000000
RECORD_COUNT = 64
RECORD_SIZE = 36
PAYLOAD_SIZE = RECORD_COUNT * RECORD_SIZE
HEADER = b'\xAA\x55'

# Kho chứa dữ liệu U và I (Hàng đợi vòng)
MAX_LEN = 5000 
buffer_u = deque(maxlen=MAX_LEN)
buffer_i = deque(maxlen=MAX_LEN)
lock = threading.Lock()

# Định mức hệ thống (DÙNG ĐỂ CHUẨN HÓA DỮ LIỆU)
# VUI LÒNG ĐỔI SỐ NÀY THEO THỰC TẾ MÁY CỦA BẠN ĐỂ AI NHẬN DIỆN CHUẨN XÁC
U_MAX_RATED = 320.0 # Ví dụ: Điện áp đỉnh là 350V
I_MAX_RATED = 7  # Ví dụ: Dòng điện đỉnh là 20A

# ==========================================
# 2. LUỒNG 1: NGƯỜI ĐỌC CẢM BIẾN (UART READER)
# ==========================================
def find_header(ser):
    while True:
        b = ser.read(1)
        if b == b'\xAA':
            b2 = ser.read(1)
            if b2 == b'\x55':
                return True

def parse_record(data):
    try:
        # Unpack 7 biến float (4 bytes * 7 = 28 bytes)
        voltage, current, power, energy, phase, i_wave, u_wave = struct.unpack('<7f', data[:28])
        return voltage, current, power, energy, phase, i_wave, u_wave
    except struct.error:
        return None

def serial_reader_thread():
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        time.sleep(1)
        ser.reset_input_buffer()
        print(f"[Luồng 1] Đã kết nối thành công với {PORT} ở tốc độ {BAUDRATE} baud.")
    except Exception as e:
        print(f"[Luồng 1] LỖI MỞ CỔNG COM: {e}")
        return

    while True:
        try:
            # 1. Tìm đúng Header đồng bộ
            find_header(ser)

            # 2. Đọc Payload và CRC
            payload = ser.read(PAYLOAD_SIZE)
            crc = ser.read(1)

            if len(payload) != PAYLOAD_SIZE or len(crc) != 1:
                continue

            # 3. Kiểm tra CRC
            calc_crc = sum(payload) & 0xFF
            if crc[0] != calc_crc:
                continue

            # 4. Giải mã 64 bản ghi và cất vào Kho (Buffer)
            temp_u = []
            temp_i = []
            
            for i in range(RECORD_COUNT):
                offset = i * RECORD_SIZE
                record = payload[offset:offset + RECORD_SIZE]
                parsed = parse_record(record)
                
                if not parsed:
                    continue
                
                voltage, current, power, energy, phase, i_wave, u_wave = parsed
                
                if any(math.isnan(x) for x in [i_wave, u_wave]):
                    continue
                    
                temp_u.append(u_wave)
                temp_i.append(i_wave)

            # Nhét một lúc 64 điểm vào kho để tiết kiệm thời gian đóng/mở khóa
            if temp_u and temp_i:
                with lock:
                    buffer_u.extend(temp_u)
                    buffer_i.extend(temp_i)

        except Exception as e:
            # Bỏ qua lỗi vặt để luồng chạy liên tục
            pass

# ==========================================
# 3. LUỒNG 2: CHUYÊN GIA AI (PREDICTOR)
# ==========================================
def extract_cycles_live(u_data, i_data, num_points=50):
    """Trích xuất chu kỳ từ mảng dữ liệu hiện tại"""
    threshold = 0.00
    zero_crossings = np.where((u_data[:-1] < -threshold) & (u_data[1:] >= threshold))[0]
    cycles_1d = []

    for i in range(len(zero_crossings) - 1):
        start, end = zero_crossings[i], zero_crossings[i+1]
        cycle_i = i_data[start:end]
        if len(cycle_i) < 10: continue

        old_idx = np.linspace(0, 1, len(cycle_i))
        new_idx = np.linspace(0, 1, num_points)
        cycles_1d.append(np.column_stack((
            np.interp(new_idx, old_idx, u_data[start:end]),
            np.interp(new_idx, old_idx, cycle_i)
        )))
    return cycles_1d

def ai_predictor_thread():
    print("[Luồng 2] Đang nạp mô hình Hybrid CNN...")
    try:
        model = load_model('hybrid_cnn_jam.h5')
        print("[Luồng 2] Mô hình đã sẵn sàng hoạt động!")
    except Exception as e:
        print(f"[Luồng 2] LỖI tải mô hình: {e}")
        return
    
    grid_size = 64
    bins = np.linspace(-1.1, 1.1, grid_size + 1)
    
    while True:
        # 1. Chụp nhanh dữ liệu từ Kho
        with lock:
            if len(buffer_u) < 1000: # Cần tích đủ điểm mới bắt đầu chạy AI
                time.sleep(0.05)
                continue
            u_arr = np.array(buffer_u)
            i_arr = np.array(buffer_i)

        # 2. CHUẨN HÓA DỮ LIỆU THEO BIÊN ĐỘ TỨC THỜI CỦA CỬA SỔ
        # Dùng epsilon để tránh chia cho 0 khi tín hiệu gần như phẳng.
        u_norm = u_arr / (np.max(np.abs(u_arr)) + 1e-9)
        i_norm = i_arr / (np.max(np.abs(i_arr)) + 1e-9)
        
        # 3. Trích xuất chu kỳ
        cycles = extract_cycles_live(u_norm, i_norm)
        
        # 4. Dự đoán nếu cắt đủ 5 chu kỳ
        if len(cycles) >= 5:
            recent_5_cycles = cycles[-5:]
            
            # Xử lý 1D
            window_1d = np.vstack(recent_5_cycles)
            
            # Xử lý 2D
            u_window, i_window = window_1d[:, 0], window_1d[:, 1]
            heatmap, _, _ = np.histogram2d(u_window, i_window, bins=(bins, bins))
            window_2d = np.where(heatmap > 0, 1.0, 0.0)[..., np.newaxis]
            
            X1_pred = np.expand_dims(window_1d, axis=0)
            X2_pred = np.expand_dims(window_2d, axis=0)
            
            # Đưa vào AI phán đoán
            pred_prob = model.predict([X1_pred, X2_pred], verbose=0)[0][0]
            
            # 5. Hiển thị cảnh báo trực tiếp (Dùng \r để ghi đè dòng Terminal)
            if pred_prob >= 0.5:
                print(f"\r[ {time.strftime('%H:%M:%S')} ] 🚨 ĐỘNG CƠ BỊ KẸT TẢI (JAM)!!! Xác suất: {pred_prob*100:.1f}%       ", end="", flush=True)
            else:
                print(f"\r[ {time.strftime('%H:%M:%S')} ] ✅ BÌNH THƯỜNG. Trạng thái ổn định ({pred_prob*100:.1f}%)         ", end="", flush=True)
                
        # Tạm nghỉ để nhường CPU cho luồng đọc UART và tạo độ trễ phân tích mượt mà
        time.sleep(0.1) 

# ==========================================
# KHỞI CHẠY HỆ THỐNG
# ==========================================
if __name__ == "__main__":
    print("=== KHỞI ĐỘNG HỆ THỐNG AI REAL-TIME ===")
    
    # Kích hoạt Luồng 1 (Lính gác UART) chạy ngầm
    t1 = threading.Thread(target=serial_reader_thread, daemon=True)
    t1.start()
    
    # Chạy Luồng 2 (Chuyên gia AI) trên luồng chính
    try:
        ai_predictor_thread()
    except KeyboardInterrupt:
        print("\n\n[🛑] Đã tắt hệ thống AI an toàn.")
        ai_predictor_thread()
    except KeyboardInterrupt:
        print("\n\n[🛑] Đã tắt hệ thống AI an toàn.")