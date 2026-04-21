#ifndef JAM_LED_H_
#define JAM_LED_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LED trạng thái đo / kẹt: GPIOB Pin 2 (cùng chân đã cấu hình trong Cube). */
void JamLed_SetMeasuring(uint8_t active);
void JamLed_SetJam(uint8_t jam);
void JamLed_Service(void);

#ifdef __cplusplus
}
#endif

#endif /* JAM_LED_H_ */
