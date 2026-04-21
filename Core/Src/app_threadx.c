/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    app_threadx.c
  * @author  MCD Application Team
  * @brief   ThreadX applicative file
  ******************************************************************************
    * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "app_threadx.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ai_preprocess.h"
#include "jam_ai.h"
#include "main.h"
#include "jam_led.h"
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define THREAD_STACK_SIZE 1024
#define AI_PREPROCESS_THREAD_STACK_SIZE 1024u
#define AI_PREPROCESS_THREAD_PRIO      18u
#define JAM_AI_THREAD_STACK_SIZE       2048u
#define JAM_AI_THREAD_PRIO             19u
#define JAM_LED_THREAD_STACK_SIZE 512u
#define JAM_LED_THREAD_PRIO       20u
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
uint8_t thread_stack1[THREAD_STACK_SIZE];
TX_THREAD thread_ptr1;
static UCHAR ai_preprocess_thread_stack[AI_PREPROCESS_THREAD_STACK_SIZE];
static TX_THREAD ai_preprocess_thread;
static UCHAR jam_ai_thread_stack[JAM_AI_THREAD_STACK_SIZE];
static TX_THREAD jam_ai_thread;
static UCHAR jam_led_thread_stack[JAM_LED_THREAD_STACK_SIZE];
static TX_THREAD jam_led_thread;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */
VOID SPI_thread_entry(ULONG initial_input);
static VOID AiPreprocess_thread_entry(ULONG initial_input);
static VOID JamAi_thread_entry(ULONG initial_input);
static VOID JamLed_thread_entry(ULONG initial_input);
/* USER CODE END PFP */

/**
  * @brief  Application ThreadX Initialization.
  * @param memory_ptr: memory pointer
  * @retval int
  */
UINT App_ThreadX_Init(VOID *memory_ptr)
{
  UINT ret = TX_SUCCESS;
  /* USER CODE BEGIN App_ThreadX_MEM_POOL */
//  tx_thread_create(&thread_ptr1, "SPI_thread", SPI_thread_entry, 1, thread_stack1, THREAD_STACK_SIZE, 9, 8, TX_NO_TIME_SLICE, TX_AUTO_START);
  /* USER CODE END App_ThreadX_MEM_POOL */
  /* USER CODE BEGIN App_ThreadX_Init */
  ret = tx_thread_create(&ai_preprocess_thread, "ai_preprocess", AiPreprocess_thread_entry, 0U,
                         ai_preprocess_thread_stack, AI_PREPROCESS_THREAD_STACK_SIZE,
                         AI_PREPROCESS_THREAD_PRIO, AI_PREPROCESS_THREAD_PRIO,
                         TX_NO_TIME_SLICE, TX_AUTO_START);
  if (ret == TX_SUCCESS) {
    ret = tx_thread_create(&jam_ai_thread, "jam_ai", JamAi_thread_entry, 0U,
                           jam_ai_thread_stack, JAM_AI_THREAD_STACK_SIZE,
                           JAM_AI_THREAD_PRIO, JAM_AI_THREAD_PRIO,
                           TX_NO_TIME_SLICE, TX_AUTO_START);
  }
  if (ret == TX_SUCCESS) {
    ret = tx_thread_create(&jam_led_thread, "jam_led", JamLed_thread_entry, 0U,
                           jam_led_thread_stack, JAM_LED_THREAD_STACK_SIZE,
                           JAM_LED_THREAD_PRIO, JAM_LED_THREAD_PRIO, TX_NO_TIME_SLICE,
                           TX_AUTO_START);
  }
  /* USER CODE END App_ThreadX_Init */

  return ret;
}

  /**
  * @brief  Function that implements the kernel's initialization.
  * @param  None
  * @retval None
  */
void MX_ThreadX_Init(void)
{
  /* USER CODE BEGIN Before_Kernel_Start */

  /* USER CODE END Before_Kernel_Start */

  tx_kernel_enter();

  /* USER CODE BEGIN Kernel_Start_Error */

  /* USER CODE END Kernel_Start_Error */
}

/* USER CODE BEGIN 1 */
VOID SPI_thread_entry(ULONG initial_input){
	while(1){
//		HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_2);
		tx_thread_sleep(1);
	}
}

static VOID AiPreprocess_thread_entry(ULONG initial_input)
{
  TX_PARAMETER_NOT_USED(initial_input);
  for (;;) {
    AiPreprocess_Service();
    tx_thread_sleep(2);
  }
}

static VOID JamAi_thread_entry(ULONG initial_input)
{
  TX_PARAMETER_NOT_USED(initial_input);
  JamAi_Init();
  uint32_t last_print_count = 0u;
  for (;;) {
    JamAi_Service();
    if (g_jam_ai_status.run_count != last_print_count) {
      last_print_count = g_jam_ai_status.run_count;
      printf("[AI] #%lu  out=%u  score=%.3f  jam=%u  cycles=%lu  us=%lu\r\n",
             (unsigned long)g_jam_ai_status.run_count,
             g_jam_ai_status.last_output_u8,
             (double)g_jam_ai_status.last_score,
             g_jam_ai_status.last_jam,
             (unsigned long)g_jam_ai_status.last_cycles,
             (unsigned long)g_jam_ai_status.last_time_us);
    }
    tx_thread_sleep(2);
  }
}

static VOID JamLed_thread_entry(ULONG initial_input)
{
  TX_PARAMETER_NOT_USED(initial_input);
  for (;;) {
    JamLed_Service();
    tx_thread_sleep(20);
  }
}
/* USER CODE END 1 */
