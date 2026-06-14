#include "jam_buzzer.h"
#include "stm32h5xx_hal.h"
#include "tx_api.h"

#define JAM_BUZZER_PORT  GPIOC
#define JAM_BUZZER_PIN   GPIO_PIN_2

static volatile uint8_t s_jam;

void JamBuzzer_SetMeasuring(uint8_t active)
{
  (void)active;  /* active buzzer: không cần trạng thái measuring */
}

void JamBuzzer_SetJam(uint8_t jam)
{
  s_jam = jam ? 1u : 0u;
}

void JamBuzzer_StartupBeep(void)
{
  /* Passive buzzer: toggle at ~500Hz for 200ms, gap 100ms, repeat 3x */
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 100; j++) {
      HAL_GPIO_WritePin(JAM_BUZZER_PORT, JAM_BUZZER_PIN, GPIO_PIN_SET);
      tx_thread_sleep(1);
      HAL_GPIO_WritePin(JAM_BUZZER_PORT, JAM_BUZZER_PIN, GPIO_PIN_RESET);
      tx_thread_sleep(1);
    }
    tx_thread_sleep(100);
  }
}

void JamBuzzer_Service(void)
{
  if (s_jam) {
    HAL_GPIO_TogglePin(JAM_BUZZER_PORT, JAM_BUZZER_PIN);
  } else {
    HAL_GPIO_WritePin(JAM_BUZZER_PORT, JAM_BUZZER_PIN, GPIO_PIN_RESET);
  }
}
