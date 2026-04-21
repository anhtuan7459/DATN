/*
 * bl0940_driver.h
 *
 *  Created on: Jun 9, 2025
 *      Author: LENOVO
 */

#ifndef INC_BL0940_DRIVER_H_
#define INC_BL0940_DRIVER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


#define RECORD_COUNT 64
#define TRIPLE_BUFFER_COUNT 3

typedef struct {
    uint32_t current;
    uint32_t voltage;
    uint32_t power;
    uint32_t phase;
    uint32_t current_wave;
    uint32_t voltage_wave;

    uint8_t hour;
    uint8_t minute;
    uint8_t second;
    uint16_t ms;
} SensorRecord;

typedef struct {
    SensorRecord records[RECORD_COUNT];
    uint8_t full; // 1 = đầy, 0 = trống
} SensorBuffer;


void BL0940_Init(void);
void BL0940_Request(uint8_t reg_addr);
void SensorBuffer_SendToUSB_Binary(void);
void SensorBuffer_SendToUSB_Binary_Time(void);
int BL0940_CheckSensorDirection(void);

float bl0940_convert_current_wave(uint32_t raw_wave);
float bl0940_convert_voltage_wave(uint32_t raw_wave);

#ifdef __cplusplus
}
#endif

#endif /* INC_BL0940_DRIVER_H_ */
