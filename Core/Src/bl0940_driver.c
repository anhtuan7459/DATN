/*
 * bl0940_driver.c
 *
 *  Created on: Jun 9, 2025
 *      Author: LENOVO
 */

#include "bl0940_driver.h"
#include "ai_preprocess.h"
#include "jam_detect.h"
#include "jam_led.h"
#include "jam_ai.h"
#include "main.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "ux_device_cdc_acm.h"


#define FRAME_LEN 6
#define SENSOR_BUFFER_SIZE 128
#define USB_BINARY_BUFFER_SIZE 4096
#define USB_FRAME_HEADER_0 0xAA
#define USB_FRAME_HEADER_1 0x55

typedef struct {
    uint8_t current[6];
    uint8_t voltage[6];
    uint8_t power[6];
    uint8_t phase[6];
    uint8_t current_wave[6];
    uint8_t voltage_wave[6];
} SensorData6B;

SensorData6B samples[SENSOR_BUFFER_SIZE];


extern RTC_HandleTypeDef hrtc;
extern SPI_HandleTypeDef hspi1;
extern UART_HandleTypeDef huart1;
extern TIM_HandleTypeDef htim1;
extern UX_SLAVE_CLASS_CDC_ACM *cdc_acm;
extern TX_SEMAPHORE semaphore;

static uint8_t spi_tx_buffer[6];
static uint8_t spi_rx_buffer[6];

typedef enum {
    STEP_READ_CURRENT = 0,   // 0x04
    STEP_READ_VOLTAGE,       // 0x06
    STEP_READ_POWER,         // 0x08
    STEP_READ_PHASE,         // 0x0C
    STEP_READ_CUR_WAVE,      // 0x01
    STEP_READ_VOL_WAVE,      // 0x03
    STEP_DONE
} BL0940_ReadStep;

static volatile BL0940_ReadStep read_step = STEP_DONE;
static volatile uint8_t read_in_progress = 0;
static uint32_t raw_current, raw_voltage, raw_power, raw_phase, raw_current_wave, raw_voltage_wave;
static SensorBuffer sensor_buffers[TRIPLE_BUFFER_COUNT];
static volatile uint8_t current_buffer_index = 0;
static volatile uint8_t record_index = 0;
static UCHAR usb_binary_buffer[USB_BINARY_BUFFER_SIZE];

float bl0940_convert_current_wave(uint32_t raw_wave)
{
    int32_t raw20 = (int32_t)(raw_wave & 0x000FFFFFu);
    if ((raw20 & 0x00080000u) != 0u) {
        raw20 |= (int32_t)0xFFF00000u;
    }
    return (float)raw20 * 1.218f * 50.4f / ((324004.0f * 3.3f * 1000.0f) / 2000.0f);
}

float bl0940_convert_voltage_wave(uint32_t raw_wave)
{
    int32_t raw20 = (int32_t)(raw_wave & 0x000FFFFFu);
    if ((raw20 & 0x00080000u) != 0u) {
        raw20 |= (int32_t)0xFFF00000u;
    }
    return (float)raw20 * 1.218f * 100.0f * 49.3f / (79931.0f * 24.0f);
}

static uint8_t calculateChecksum(uint8_t *rxData, uint8_t state, uint8_t address) {
    uint8_t checksum = state + address + rxData[2] + rxData[3] + rxData[4];
    return ~checksum;
}

static void WriteRegister(uint8_t addr, uint32_t data) {
    uint8_t unlock[6] = {0xA8, 0x1A, 0, 0, 0x55, 0xE8};
    HAL_SPI_Transmit(&hspi1, unlock, sizeof(unlock), 100);

    uint8_t tx[6] = {0xA8, addr, data >> 16, data >> 8, data, 0};
    tx[5] = calculateChecksum(tx, 0xA8, addr);
    HAL_SPI_Transmit(&hspi1, tx, sizeof(tx), 100);
}

void SetNoLoadThreshold(uint8_t value) {
    WriteRegister(0x17, value);
}

void BL0940_SetFrequency(uint32_t hz) {
    uint8_t addr = 0x18;
    uint8_t tx[6] = {0x58, addr, 0, 0, 0, 0};
    uint8_t rx[6] = {0};
    HAL_SPI_TransmitReceive(&hspi1, tx, rx, sizeof(tx), 100);
    uint32_t val = ((uint32_t)rx[2] << 16) | ((uint32_t)rx[3] << 8) | rx[4];

    if (hz == 50) val &= ~(1 << 9); else val |= (1 << 9);
    WriteRegister(addr, val);
}

void BL0940_SetUpdateRate(uint32_t rate) {
    uint8_t addr = 0x18;
    uint8_t tx[6] = {0x58, addr, 0, 0, 0, 0};
    uint8_t rx[6] = {0};
    HAL_SPI_TransmitReceive(&hspi1, tx, rx, sizeof(tx), 100);
    uint32_t val = ((uint32_t)rx[2] << 16) | ((uint32_t)rx[3] << 8) | rx[4];

    if (rate == 400) val &= ~(1 << 8); else val |= (1 << 8);
    WriteRegister(addr, val);
}

void BL0940_CheckConfig(void) {
    uint8_t addr = 0x18;
    uint8_t tx[6] = {0x58, addr, 0, 0, 0, 0};
    uint8_t rx[6] = {0};
    HAL_SPI_TransmitReceive(&hspi1, tx, rx, sizeof(tx), 100);
    uint32_t data = ((uint32_t)rx[2] << 16) | ((uint32_t)rx[3] << 8) | rx[4];
    uint8_t freq = (data & (1 << 9)) ? 60 : 50;
    uint16_t rate = (data & (1 << 8)) ? 800 : 400;
    char msg[64];
    snprintf(msg, sizeof(msg), "Freq=%uHz, UpdateRate=%uHz\r\n", freq, rate);
    HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
}

void BL0940_Reset(void) {
    WriteRegister(0x19, 0x5A5A5A);
}

void BL0940_TimerInit(void) {
    HAL_TIM_Base_Start_IT(&htim1);  // TIM1 tạo ngắt 1kHz
}

void SensorBuffer_SendToUSB_Binary(void) {
    ULONG actual_length;

    for (int i = 0; i < TRIPLE_BUFFER_COUNT; i++) {
        if (sensor_buffers[i].full) {
            uint8_t *ptr = usb_binary_buffer;

            for (int j = 0; j < RECORD_COUNT; j++) {
                SensorRecord *rec = &sensor_buffers[i].records[j];

                float voltage = rec->voltage * 1.218f * 100.0f / (79931.0f * 24.0f);
                float current = rec->current * 1.218f / ((324004.0f * 3.3f * 1000.0f) / 2000.0f);
                float power   = rec->power * 1.218f * 1.218f * 100.0f / (4046.0f * (3.3f * 1000.0f / 2000.0f) * 24.0f);
                float phase   = 2.0f * 3.1415926535f * rec->phase * (50.0f / 1000000.0f);
                float i_wave  = bl0940_convert_current_wave(rec->current_wave);
                float u_wave  = bl0940_convert_voltage_wave(rec->voltage_wave);

                memcpy(ptr, &voltage, sizeof(float)); ptr += 4;
                memcpy(ptr, &current, sizeof(float)); ptr += 4;
                memcpy(ptr, &power,   sizeof(float)); ptr += 4;
                memcpy(ptr, &phase,   sizeof(float)); ptr += 4;
                memcpy(ptr, &i_wave,  sizeof(float)); ptr += 4;
                memcpy(ptr, &u_wave,  sizeof(float)); ptr += 4;
            }

            // Mỗi record = 24 byte, 64 record = 1536 byte
            _ux_device_class_cdc_acm_write(cdc_acm, usb_binary_buffer, RECORD_COUNT * 29, &actual_length);

            sensor_buffers[i].full = 0;
        }
    }
}

void SensorBuffer_SendToUSB_Binary_Time(void) {
    ULONG actual_length;
    static float energy_accum_wh = 0.0f;

    for (int i = 0; i < TRIPLE_BUFFER_COUNT; i++) {
        if (sensor_buffers[i].full) {
            uint8_t *ptr = (uint8_t *)usb_binary_buffer;
            const uint32_t payload_len = RECORD_COUNT * 36U;

            // Frame: AA55 + payload + CRC(sum(payload)&0xFF)
            *ptr++ = USB_FRAME_HEADER_0;
            *ptr++ = USB_FRAME_HEADER_1;
            uint8_t *payload_start = ptr;

            for (int j = 0; j < RECORD_COUNT; j++) {
                SensorRecord *rec = &sensor_buffers[i].records[j];

                float voltage = rec->voltage * 1.218f * 100.0f / (79931.0f * 24.0f);
                float current = rec->current * 1.218f / ((324004.0f * 3.3f * 1000.0f) / 2000.0f);

                int32_t rawPower = (int32_t)rec->power;
                if (rawPower < 0) {
                    rawPower = -rawPower;
                }
                float power = (float)rawPower * 1.218f * 1.218f * 100.0f / (4046.0f * (3.3f * 1000.0f / 2000.0f) * 24.0f);
                float phase   = 2.0f * 3.1415926535f * rec->phase * (50.0f / 1000000.0f);
                float i_wave  = bl0940_convert_current_wave(rec->current_wave);
                float u_wave  = bl0940_convert_voltage_wave(rec->voltage_wave);
                // Approximate cumulative energy in Wh at ~1 kHz sampling.
                energy_accum_wh += power * (1.0f / 3600.0f) * 0.001f;

                // 7 floats = 28 bytes (voltage,current,power,energy,phase,i_wave,u_wave)
                memcpy(ptr, &voltage, sizeof(float)); ptr += 4;
                memcpy(ptr, &current, sizeof(float)); ptr += 4;
                memcpy(ptr, &power,   sizeof(float)); ptr += 4;
                memcpy(ptr, &energy_accum_wh, sizeof(float)); ptr += 4;
                memcpy(ptr, &phase,   sizeof(float)); ptr += 4;
                memcpy(ptr, &i_wave,  sizeof(float)); ptr += 4;
                memcpy(ptr, &u_wave,  sizeof(float)); ptr += 4;

                // Ghi timestamp (5 byte)
                *ptr++ = rec->hour;
                *ptr++ = rec->minute;
                *ptr++ = rec->second;
                memcpy(ptr, &rec->ms, sizeof(uint16_t)); ptr += 2;

                // Pad 3 byte -> mỗi record 36 byte
                memset(ptr, 0, 3); ptr += 3;
            }

            /* Append 12-byte AI status block */
            *ptr++ = g_jam_ai_status.ready;
            *ptr++ = g_jam_ai_status.last_jam;
            *ptr++ = g_jam_ai_status.last_output_u8;
            *ptr++ = (uint8_t)(g_jam_ai_status.run_count & 0xFFu);
            float ai_score = g_jam_ai_status.last_score;
            memcpy(ptr, &ai_score, sizeof(float)); ptr += 4;
            uint32_t ai_us = g_jam_ai_status.last_time_us;
            memcpy(ptr, &ai_us, sizeof(uint32_t)); ptr += 4;

            const uint32_t total_payload = payload_len + 12U;
            uint8_t crc = 0;
            for (uint32_t k = 0; k < total_payload; k++) {
                crc = (uint8_t)(crc + payload_start[k]);
            }
            *ptr++ = crc;

            _ux_device_class_cdc_acm_write(cdc_acm, usb_binary_buffer, (ULONG)(2U + total_payload + 1U), &actual_length);

            sensor_buffers[i].full = 0;
        }
    }
}


void BL0940_Request(uint8_t reg_addr) {
    spi_tx_buffer[0] = 0x58;              // State byte
    spi_tx_buffer[1] = reg_addr;          // Address byte
    memset(&spi_tx_buffer[2], 0, 4);      // Clear remaining bytes
    HAL_SPI_TransmitReceive_DMA(&hspi1, spi_tx_buffer, spi_rx_buffer, 6);
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM1 && !read_in_progress) {
        read_in_progress = 1;
        read_step = STEP_READ_CURRENT;
        BL0940_Request(0x04);  // Current
    }

    if (htim->Instance == TIM2) {
        HAL_IncTick();
    }
}

void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi) {
    if (hspi->Instance != SPI1) return;
    uint8_t cs = ~(0x58 + spi_tx_buffer[1] + spi_rx_buffer[2] + spi_rx_buffer[3] + spi_rx_buffer[4]);
    if (cs != spi_rx_buffer[5]) {
        read_in_progress = 0;
        read_step = STEP_DONE;
        return;
    }

    uint32_t raw = ((uint32_t)spi_rx_buffer[2] << 16) |
                   ((uint32_t)spi_rx_buffer[3] << 8) |
                   spi_rx_buffer[4];

    switch (read_step) {
        case STEP_READ_CURRENT:
            raw_current = raw;
            read_step = STEP_READ_VOLTAGE;
            BL0940_Request(0x06);
            break;

        case STEP_READ_VOLTAGE:
            raw_voltage = raw;
            read_step = STEP_READ_POWER;
            BL0940_Request(0x08);
            break;

        case STEP_READ_POWER:
//            raw_power = raw;
        	raw_power = (int32_t)(raw << 8) >> 8;
            read_step = STEP_READ_PHASE;
            BL0940_Request(0x0C);
            break;

        case STEP_READ_PHASE:
            raw_phase = raw;
            read_step = STEP_READ_CUR_WAVE;
            BL0940_Request(0x01);
            break;

        case STEP_READ_CUR_WAVE:
            raw_current_wave = raw;
            read_step = STEP_READ_VOL_WAVE;
            BL0940_Request(0x03);
            break;

        case STEP_READ_VOL_WAVE:
            raw_voltage_wave = raw;
            read_step = STEP_DONE;
            read_in_progress = 0;

            SensorBuffer *buf = &sensor_buffers[current_buffer_index];
            buf->records[record_index].current = raw_current;
            buf->records[record_index].voltage = raw_voltage;
            buf->records[record_index].power   = raw_power;
            buf->records[record_index].phase   = raw_phase;
            buf->records[record_index].current_wave   = raw_current_wave;
            buf->records[record_index].voltage_wave   = raw_voltage_wave;

            // 👉 Lấy thời gian RTC và lưu vào record
            RTC_TimeTypeDef sTime;
            RTC_DateTypeDef sDate;
            HAL_RTC_GetTime(&hrtc, &sTime, RTC_FORMAT_BIN);
            HAL_RTC_GetDate(&hrtc, &sDate, RTC_FORMAT_BIN);
            uint16_t ms = (uint16_t)((sTime.SubSeconds * 1000) / (hrtc.Init.SynchPrediv + 1));

            buf->records[record_index].hour   = sTime.Hours;
            buf->records[record_index].minute = sTime.Minutes;
            buf->records[record_index].second = sTime.Seconds;
            buf->records[record_index].ms     = ms;

            JamDetect_OnNewRecord(&buf->records[record_index]);

            record_index++;
            if (record_index >= RECORD_COUNT) {
                buf->full = 1;
                tx_semaphore_put(&semaphore);
                current_buffer_index = (current_buffer_index + 1) % TRIPLE_BUFFER_COUNT;
                record_index = 0;

                if (sensor_buffers[current_buffer_index].full) {
                    sensor_buffers[current_buffer_index].full = 0; // Overwrite
                }
            }
            break;

        default:
            read_step = STEP_DONE;
            read_in_progress = 0;
            break;
    }
}
void BL0940_Init(void) {
    BL0940_Reset();
    BL0940_SetFrequency(50);
    BL0940_SetUpdateRate(400);
    SetNoLoadThreshold(0x1D);
    BL0940_CheckConfig();
    BL0940_TimerInit();
    memset(sensor_buffers, 0, sizeof(sensor_buffers));
    AiPreprocess_Init();
    JamLed_SetMeasuring(1u);
}

int getActivePowerRaw(float *powerRaw) {
    if (powerRaw == NULL) return 0;

    uint8_t tx[6] = { 0x58, 0x08, 0, 0, 0, 0 };
    uint8_t rx[6] = {0};

    if (HAL_SPI_TransmitReceive(&hspi1, tx, rx, 6, 100) != HAL_OK) {
        return 0;  // lỗi truyền SPI
    }

    // Ghép 3 byte thành 24-bit
    uint32_t data = ((uint32_t)rx[2] << 16) | ((uint32_t)rx[3] << 8) | (uint32_t)rx[4];

    // Sign-extend từ 24-bit signed → 32-bit signed
    int32_t signedPower = (int32_t)(data << 8) >> 8;

    // Gán kết quả (dương hoặc âm) ra ngoài nếu cần dùng
    *powerRaw = (float)signedPower;

    // Kiểm tra chiều dòng
    if (signedPower < 0) {
        return -1;  // công suất âm → ngược chiều
    } else {
        return 1;   // công suất dương → đúng chiều
    }
}

int BL0940_CheckSensorDirection(void) {
    const int totalChecks = 7;
    int reversedCount = 0;
    int successCount = 0;

    for (int i = 0; i < totalChecks; i++) {
        float power = 0.0f;
        int status = getActivePowerRaw(&power);

        if (status == 0) {
            continue;
        }

        successCount++;

        if (status == -1) {
            reversedCount++;
        }

        HAL_Delay(50);
    }

    if (successCount == 0) {
        return 0;
    }

    if (reversedCount >= (successCount / 2 + 1)) {
        return -1;
    }
    return 1;
}
