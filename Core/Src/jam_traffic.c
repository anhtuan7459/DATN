#include "jam_traffic.h"
#include "stm32h5xx_hal.h"

/* Kênh đỏ  (PC0): active HIGH — PC0=HIGH bật relay, PC0=LOW tắt relay
 * Kênh vàng (PA0): active LOW  — PA0=LOW  bật relay, PA0=HIGH tắt relay
 */
#define TRAFFIC_RED_PORT    GPIOC
#define TRAFFIC_RED         GPIO_PIN_0
#define TRAFFIC_YELLOW_PORT GPIOA
#define TRAFFIC_YELLOW      GPIO_PIN_0

void JamTraffic_Init(void)
{
  HAL_GPIO_WritePin(TRAFFIC_RED_PORT,    TRAFFIC_RED,    GPIO_PIN_RESET); /* PC0=LOW  → đỏ OFF  */
  HAL_GPIO_WritePin(TRAFFIC_YELLOW_PORT, TRAFFIC_YELLOW, GPIO_PIN_RESET); /* PA0=LOW  → vàng ON */
}

void JamTraffic_SetJam(uint8_t jam)
{
  if (jam) {
    HAL_GPIO_WritePin(TRAFFIC_YELLOW_PORT, TRAFFIC_YELLOW, GPIO_PIN_SET);   /* PA0=HIGH → vàng OFF */
    HAL_GPIO_WritePin(TRAFFIC_RED_PORT,    TRAFFIC_RED,    GPIO_PIN_SET);   /* PC0=HIGH → đỏ ON   */
  } else {
    HAL_GPIO_WritePin(TRAFFIC_RED_PORT,    TRAFFIC_RED,    GPIO_PIN_RESET); /* PC0=LOW  → đỏ OFF  */
    HAL_GPIO_WritePin(TRAFFIC_YELLOW_PORT, TRAFFIC_YELLOW, GPIO_PIN_RESET); /* PA0=LOW  → vàng ON */
  }
}
