import serial
import struct
import math
from datetime import datetime
import time
import keyboard  # <--- Thêm thư viện này

# ⚙️ Cấu hình giao thức
PORT = 'COM5'  
BAUDRATE = 2000000
RECORD_COUNT = 64
RECORD_SIZE = 36
PAYLOAD_SIZE = RECORD_COUNT * RECORD_SIZE
FRAME_SIZE = 2 + PAYLOAD_SIZE + 1  # Header + Payload + CRC
HEADER = b'\xAA\x55'
MAX_PACKETS = 1406    # Ghi ~ 89.984s

# Biến toàn cục để lưu trạng thái nhãn
current_label = "NORMAL"

def update_label(e):
    global current_label
    if e.name == 'j':
        current_label = "JAM"
        print(f"\n[⚠️] ĐÃ ĐÁNH DẤU: KẸT (JAM) lúc {datetime.now().strftime('%H:%M:%S')}")
    elif e.name == 'n':
        current_label = "NORMAL"
        print(f"\n[✅] ĐÃ ĐÁNH DẤU: BÌNH THƯỜNG (NORMAL) lúc {datetime.now().strftime('%H:%M:%S')}")

# Đăng ký sự kiện nhấn phím
keyboard.on_press_key('j', update_label)
keyboard.on_press_key('n', update_label)

def parse_record(data):
    try:
        voltage, current, power, energy, phase, i_wave, u_wave = struct.unpack('<7f', data[:28])
        return voltage, current, power, energy, phase, i_wave, u_wave
    except struct.error as e:
        print(f"[⚠️] Parse error: {e}")
        return None

def create_new_csv_file():
    now = datetime.now()
    filename = now.strftime("./Data_%Y%m%d_%H%M%S.csv")
    f = open(filename, "w", encoding="utf-8")
    # Thêm cột "Event" vào header
    f.write("Date,Time,Voltage[V],Current[A],Power[W],Energy[u],Phase[rad],Iwave,Uwave,Event\n")
    print(f"[💾] Logging to: {filename}")
    print("[⌨️] Nhấn 'j' để đánh dấu KẸT, 'n' để đánh dấu BÌNH THƯỜNG")
    return f

def find_header(ser):
    while True:
        b = ser.read(1)
        if b == b'\xAA':
            b2 = ser.read(1)
            if b2 == b'\x55':
                return True

def main():
    file = None
    try:
        with serial.Serial(PORT, BAUDRATE, timeout=1) as ser:
            time.sleep(1)
            ser.reset_input_buffer()

            file = create_new_csv_file()
            packet_count = 0

            while packet_count < MAX_PACKETS:
                find_header(ser)

                payload = ser.read(PAYLOAD_SIZE)
                crc = ser.read(1)

                if len(payload) != PAYLOAD_SIZE or len(crc) != 1:
                    continue

                calc_crc = sum(payload) & 0xFF
                if crc[0] != calc_crc:
                    continue

                current_now = datetime.now()
                date_str = current_now.strftime("%Y-%m-%d")
                time_str = current_now.strftime("%H:%M:%S.%f")[:-3] 

                # Chốt nhãn cho toàn bộ 64 bản ghi trong gói này
                label_to_write = current_label 

                for i in range(RECORD_COUNT):
                    offset = i * RECORD_SIZE
                    record = payload[offset:offset + RECORD_SIZE]
                    parsed = parse_record(record)
                    
                    if not parsed:
                        continue

                    voltage, current, power, energy, phase, i_wave, u_wave = parsed

                    if any(math.isnan(x) for x in [voltage, current, power, energy, phase]):
                        continue

                    # Ghi thêm nhãn Event vào cuối mỗi dòng
                    file.write(f"{date_str},{time_str},{voltage:.2f},{current:.4f},{power:.2f},{energy:.4f},{phase:.4f},{i_wave:.4f},{u_wave:.4f},{label_to_write}\n")

                file.flush()
                packet_count += 1
                # In trạng thái hiện tại ra console để bạn dễ theo dõi khi đang làm thí nghiệm
                print(f"\r[✍️] Gói {packet_count}/{MAX_PACKETS} | Trạng thái: {label_to_write}          ", end="")

            print(f"\n[✅] Đã ghi xong {MAX_PACKETS} gói.")

    except serial.SerialException as e:
        print(f"[💥] Serial error: {e}")
    except KeyboardInterrupt:
        print("\n[🛑] Stopped by user.")
    finally:
        if file:
            file.close()


if __name__ == "__main__":
    main()