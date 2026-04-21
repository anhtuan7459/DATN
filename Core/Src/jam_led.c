#include "jam_led.h"
#include "stm32h5xx_hal.h"

#define JAM_LED_PORT GPIOB
#define JAM_LED_PIN  GPIO_PIN_2

static volatile uint8_t s_measuring;
static volatile uint8_t s_jam;

void JamLed_SetMeasuring(uint8_t active)
{
  s_measuring = active ? 1u : 0u;
}

void JamLed_SetJam(uint8_t jam)
{
  s_jam = jam ? 1u : 0u;
}

void JamLed_Service(void)
{
  if (s_jam) {
    HAL_GPIO_TogglePin(JAM_LED_PORT, JAM_LED_PIN);
  } else if (s_measuring) {
    HAL_GPIO_WritePin(JAM_LED_PORT, JAM_LED_PIN, GPIO_PIN_SET);
  } else {
    HAL_GPIO_WritePin(JAM_LED_PORT, JAM_LED_PIN, GPIO_PIN_RESET);
  }
}
