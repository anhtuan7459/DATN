# BL0940 JamLed - Firmware STM32H562RGTX

Firmware phát hiện hiện tượng **JAM (kẹt)** thời gian thực trên bo STM32H562RGTX, sử dụng IC đo công suất **BL0940** và mô hình **CNN 2D** triển khai bằng **X-CUBE-AI**.

## Kiến trúc tổng thể

```
BL0940 (SPI, 1 kHz) ──► Raw Iwave/Uwave
                            │
                            ▼
                  Scale vật lý (V, A)
                            │
                            ▼
           AI Preprocess (zero-cross, resample,
            windowing 5 cycles, histogram 2D 64×64)
                            │
                            ▼
                CNN 2D int8 (X-CUBE-AI)
                            │
                            ▼
                JAM / Normal + Confidence
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        LED PB2 (JAM nháy)       USB CDC-ACM (log PC)
```

## Chức năng chính

- Đọc 6 thanh ghi BL0940 qua SPI DMA ở nhịp 1 kHz (Iwave, Uwave, IRMS, VRMS, WATT, PHASE).
- Chuyển đổi raw → đơn vị vật lý (Volt/Ampere) cho AI.
- Tiền xử lý tín hiệu (zero-crossing, resampling 50 điểm/chu kỳ, phase-averaging 5 chu kỳ, tạo ảnh I-V 64×64).
- Suy luận AI bằng CNN 2D int8 (~327 KB weights).
- Báo trạng thái qua LED PB2 (nháy khi JAM, sáng khi đo).
- Gửi dữ liệu nhị phân + AI status ra USB CDC-ACM để log trên PC.

## Phần cứng

- MCU: STM32H562RGTX (Cortex-M33, 250 MHz)
- IC đo: BL0940 (SPI, 20-bit signed waveform)
- Giao tiếp: SPI1 + DMA, USB FS, UART1 debug
- LED chỉ thị: PB2 (JAM / Measuring)

## Nhánh repo

Nhánh `Firmware` chứa toàn bộ project STM32CubeIDE, bao gồm driver, AI preprocess, USBX, ThreadX và model CubeAI.
