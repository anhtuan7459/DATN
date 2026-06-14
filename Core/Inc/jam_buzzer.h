#ifndef JAM_BUZZER_H_
#define JAM_BUZZER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Còi báo kẹt: GPIO output nối module còi chip 3.5V.
 * Thay thế nhấp nháy LED trong trạng thái JAM. */
void JamBuzzer_SetMeasuring(uint8_t active);
void JamBuzzer_SetJam(uint8_t jam);
void JamBuzzer_Service(void);
void JamBuzzer_StartupBeep(void);

#ifdef __cplusplus
}
#endif

#endif /* JAM_BUZZER_H_ */
