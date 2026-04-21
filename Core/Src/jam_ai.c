#include "jam_ai.h"

#include "ai_preprocess.h"
#include "jam_led.h"
#include "main.h"
#include "network.h"
#include "network_data.h"

#include <string.h>

#define JAM_AI_INPUT_SCALE        (255.0f)
#define JAM_AI_OUTPUT_SCALE       (1.0f / 256.0f)
#define JAM_AI_JAM_THRESHOLD_U8   (128u)

AI_ALIGNED(4)
static ai_u8 s_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

static ai_handle s_network = AI_HANDLE_NULL;
static ai_buffer *s_inputs;
static ai_buffer *s_outputs;
static ai_u8 s_input_buffer[AI_NETWORK_IN_1_SIZE_BYTES];
static ai_u8 s_output_buffer[AI_NETWORK_OUT_1_SIZE_BYTES];
static uint8_t s_initialized;

volatile JamAiStatus g_jam_ai_status;

static volatile uint8_t s_dwt_available;

static void dwt_counter_enable(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  __DSB();
  __ISB();

  if ((DWT->CTRL & DWT_CTRL_NOCYCCNT_Msk) != 0u) {
    s_dwt_available = 0u;
    return;
  }

  DWT->CYCCNT = 0u;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  s_dwt_available = 1u;
}

static uint32_t cycles_to_us(uint32_t cycles)
{
  const uint32_t hclk_hz = HAL_RCC_GetHCLKFreq();
  if (hclk_hz == 0u) {
    return 0u;
  }

  return (uint32_t)((((uint64_t)cycles) * 1000000ull) / hclk_hz);
}

static uint8_t quantize_pixel(float value)
{
  float scaled;

  if (value <= 0.0f) {
    return 0u;
  }
  if (value >= 1.0f) {
    return 255u;
  }

  scaled = (value * JAM_AI_INPUT_SCALE) + 0.5f;
  if (scaled >= 255.0f) {
    return 255u;
  }

  return (uint8_t)scaled;
}

static void quantize_frame(const float *frame, ai_u8 *dst)
{
  uint32_t idx;

  for (idx = 0u; idx < AI_NETWORK_IN_1_SIZE; ++idx) {
    dst[idx] = quantize_pixel(frame[idx]);
  }
}

void JamAi_Init(void)
{
  ai_error err;
  static const ai_handle weights[] = {
      AI_HANDLE_PTR(s_network_weights_array_u64)
  };
  const ai_handle activations[] = {
      AI_HANDLE_PTR(s_activations)
  };

  if (s_initialized != 0u) {
    return;
  }

  g_jam_ai_status.reserved = 0x10u;
  dwt_counter_enable();
  memset(s_input_buffer, 0, sizeof(s_input_buffer));
  memset(s_output_buffer, 0, sizeof(s_output_buffer));

  g_jam_ai_status.reserved = 0x20u;
  err = ai_network_create_and_init(&s_network, activations, weights);
  if (err.type != AI_ERROR_NONE) {
    s_network = AI_HANDLE_NULL;
    g_jam_ai_status.reserved = 0x80u | (uint8_t)err.type;
    return;
  }
  g_jam_ai_status.reserved = 0x00u;

  s_inputs = ai_network_inputs_get(s_network, NULL);
  s_outputs = ai_network_outputs_get(s_network, NULL);
  s_inputs[0].data = AI_HANDLE_PTR(s_input_buffer);
  s_outputs[0].data = AI_HANDLE_PTR(s_output_buffer);
  s_initialized = 1u;
  g_jam_ai_status.ready = 1u;
}

void JamAi_Service(void)
{
  const float *frame;
  uint32_t frame_id;
  uint32_t start_cycles;
  ai_i32 batches;

  if (s_initialized == 0u) {
    JamAi_Init();
    if (s_initialized == 0u) {
      return;
    }
  }

  if (AiPreprocess_TakeLatestFrame(&frame, &frame_id) == 0u) {
    return;
  }

  (void)frame_id;
  quantize_frame(frame, s_input_buffer);

  if (s_dwt_available) {
    DWT->CYCCNT = 0u;
    start_cycles = DWT->CYCCNT;
  } else {
    start_cycles = 0u;
  }
  batches = ai_network_run(s_network, s_inputs, s_outputs);
  if (s_dwt_available) {
    g_jam_ai_status.last_cycles = DWT->CYCCNT - start_cycles;
    g_jam_ai_status.last_time_us = cycles_to_us(g_jam_ai_status.last_cycles);
  }

  if (batches != 1) {
    return;
  }

  g_jam_ai_status.last_frame_id = frame_id;
  g_jam_ai_status.last_output_u8 = s_output_buffer[0];
  g_jam_ai_status.last_score = ((float)g_jam_ai_status.last_output_u8) * JAM_AI_OUTPUT_SCALE;
  g_jam_ai_status.last_jam = (g_jam_ai_status.last_output_u8 >= JAM_AI_JAM_THRESHOLD_U8) ? 1u : 0u;
  g_jam_ai_status.run_count++;

  JamLed_SetJam(g_jam_ai_status.last_jam);
}

uint8_t JamAi_IsReady(void)
{
  return s_initialized;
}

float JamAi_GetLastScore(void)
{
  return g_jam_ai_status.last_score;
}

uint8_t JamAi_GetLastOutputU8(void)
{
  return g_jam_ai_status.last_output_u8;
}

uint32_t JamAi_GetLastCycles(void)
{
  return g_jam_ai_status.last_cycles;
}

uint32_t JamAi_GetLastTimeUs(void)
{
  return g_jam_ai_status.last_time_us;
}

uint32_t JamAi_GetRunCount(void)
{
  return g_jam_ai_status.run_count;
}
