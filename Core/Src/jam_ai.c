#include "jam_ai.h"

#include "ai_preprocess.h"
#include "jam_led.h"
#include "jam_traffic.h"
#include "main.h"
#include "network.h"
#include "network_data.h"

#include "tx_api.h"

#include <string.h>

extern TX_SEMAPHORE semaphore;

#define JAM_AI_INPUT_SCALE        (255.0f)
#define JAM_AI_OUTPUT_SCALE       (1.0f / 256.0f)
#define JAM_AI_JAM_THRESHOLD_U8   (223u)
#define JAM_AI_DECISION_GROUP_SIZE (5u)

/* FIFO: mỗi lần chốt vote nhóm → một snapshot gửi USB (không ghi đè nếu USB chậm). */
#define JAM_AI_USB_Q_CAP 48u

typedef struct {
  uint8_t ready;
  uint8_t last_jam;
  uint8_t last_output_u8;
  uint8_t reserved;
  uint8_t jam_votes;
  uint8_t normal_votes;
  uint8_t decision_group_size;
  float last_score;
  uint32_t last_time_us;
  uint32_t decision_count;
} JamAiUsbDecisionSnap;

static JamAiUsbDecisionSnap s_usb_q[JAM_AI_USB_Q_CAP];
static uint8_t s_usb_q_head;
static uint8_t s_usb_q_tail;
static uint8_t s_usb_q_count;

AI_ALIGNED(4)
static ai_u8 s_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];

static ai_handle s_network = AI_HANDLE_NULL;
static ai_buffer *s_inputs;
static ai_buffer *s_outputs;
static ai_u8 s_input_buffer[AI_NETWORK_IN_1_SIZE_BYTES];
static ai_u8 s_output_buffer[AI_NETWORK_OUT_1_SIZE_BYTES];
static uint8_t s_initialized;
static uint8_t s_pending_votes;
static uint8_t s_pending_jam_votes;
static float s_pending_score_sum;
static uint8_t s_stable_jam;
static uint8_t s_transition_target_jam;
static uint8_t s_transition_count;

volatile JamAiStatus g_jam_ai_status;

static volatile uint8_t s_dwt_available;

static uint8_t usb_q_try_push_from_status(void)
{
  uint32_t primask;

  primask = __get_PRIMASK();
  __disable_irq();
  if (s_usb_q_count >= JAM_AI_USB_Q_CAP) {
    __set_PRIMASK(primask);
    return 0u;
  }
  {
    JamAiUsbDecisionSnap *d = &s_usb_q[s_usb_q_tail];
    d->ready = g_jam_ai_status.ready;
    d->last_jam = g_jam_ai_status.last_jam;
    d->last_output_u8 = g_jam_ai_status.last_output_u8;
    d->reserved = 0u;
    d->jam_votes = g_jam_ai_status.jam_votes;
    d->normal_votes = g_jam_ai_status.normal_votes;
    d->decision_group_size = g_jam_ai_status.decision_group_size;
    d->last_score = g_jam_ai_status.last_score;
    d->last_time_us = g_jam_ai_status.last_time_us;
    d->decision_count = g_jam_ai_status.decision_count;
    s_usb_q_tail = (uint8_t)((s_usb_q_tail + 1u) % JAM_AI_USB_Q_CAP);
    s_usb_q_count++;
  }
  __set_PRIMASK(primask);
  return 1u;
}

uint8_t JamAi_UsbQueuePop(uint8_t *out)
{
  uint32_t primask;
  JamAiUsbDecisionSnap *d;
  uint8_t ok = 0u;

  if (out == NULL) {
    return 0u;
  }

  primask = __get_PRIMASK();
  __disable_irq();
  if (s_usb_q_count == 0u) {
    __set_PRIMASK(primask);
    return 0u;
  }
  d = &s_usb_q[s_usb_q_head];
  out[0] = d->ready;
  out[1] = d->last_jam;
  out[2] = d->last_output_u8;
  out[3] = d->reserved;
  out[4] = d->jam_votes;
  out[5] = d->normal_votes;
  out[6] = d->decision_group_size;
  memcpy(&out[7], &d->last_score, sizeof(float));
  memcpy(&out[11], &d->last_time_us, sizeof(uint32_t));
  memcpy(&out[15], &d->decision_count, sizeof(uint32_t));
  out[19] = 1u;
  s_usb_q_head = (uint8_t)((s_usb_q_head + 1u) % JAM_AI_USB_Q_CAP);
  s_usb_q_count--;
  ok = 1u;
  __set_PRIMASK(primask);
  return ok;
}

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
  s_pending_votes = 0u;
  s_pending_jam_votes = 0u;
  s_pending_score_sum = 0.0f;
  s_stable_jam = 0u;
  s_transition_target_jam = 0u;
  s_transition_count = 0u;
  g_jam_ai_status.jam_votes = 0u;
  g_jam_ai_status.normal_votes = 0u;
  g_jam_ai_status.last_raw_output_u8 = 0u;
  g_jam_ai_status.last_raw_jam = 0u;
  g_jam_ai_status.last_raw_score = 0.0f;
  g_jam_ai_status.decision_group_size = JAM_AI_DECISION_GROUP_SIZE;
  g_jam_ai_status.decision_count = 0u;
  s_usb_q_head = 0u;
  s_usb_q_tail = 0u;
  s_usb_q_count = 0u;

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
  }
  batches = ai_network_run(s_network, s_inputs, s_outputs);
  if (s_dwt_available) {
    g_jam_ai_status.last_cycles = DWT->CYCCNT;
    g_jam_ai_status.last_time_us = cycles_to_us(g_jam_ai_status.last_cycles);
  }

  if (batches != 1) {
    return;
  }

  g_jam_ai_status.last_frame_id = frame_id;
  g_jam_ai_status.run_count++;

  {
    uint8_t raw_out_u8 = s_output_buffer[0];
    float raw_score = ((float)raw_out_u8) * JAM_AI_OUTPUT_SCALE;
    uint8_t raw_jam = (raw_out_u8 >= JAM_AI_JAM_THRESHOLD_U8) ? 1u : 0u;

    g_jam_ai_status.last_raw_output_u8 = raw_out_u8;
    g_jam_ai_status.last_raw_jam = raw_jam;
    g_jam_ai_status.last_raw_score = raw_score;

    s_pending_votes++;
    s_pending_score_sum += raw_score;
    if (raw_jam != 0u) {
      s_pending_jam_votes++;
    }

    if (s_pending_votes >= JAM_AI_DECISION_GROUP_SIZE) {
      uint8_t jam_votes = s_pending_jam_votes;
      uint8_t normal_votes = (uint8_t)(JAM_AI_DECISION_GROUP_SIZE - s_pending_jam_votes);
      /* Candidate from current 5-vote group. */
      uint8_t candidate_jam = (jam_votes >= 4u) ? 1u : 0u;
      uint8_t final_jam = s_stable_jam;
      float final_score = s_pending_score_sum / (float)JAM_AI_DECISION_GROUP_SIZE;
      float out_scaled = (final_score * 256.0f) + 0.5f;

      if (out_scaled < 0.0f) {
        out_scaled = 0.0f;
      } else if (out_scaled > 255.0f) {
        out_scaled = 255.0f;
      }

      /* Hysteresis: switch state only after 2 consecutive opposite decisions. */
      if (candidate_jam == s_stable_jam) {
        s_transition_count = 0u;
        s_transition_target_jam = s_stable_jam;
      } else {
        if (s_transition_target_jam != candidate_jam) {
          s_transition_target_jam = candidate_jam;
          s_transition_count = 1u;
        } else {
          s_transition_count++;
        }
        if (s_transition_count >= 2u) {
          s_stable_jam = candidate_jam;
          s_transition_count = 0u;
        }
      }
      final_jam = s_stable_jam;

      g_jam_ai_status.last_output_u8 = (uint8_t)out_scaled;
      g_jam_ai_status.last_score = final_score;
      g_jam_ai_status.last_jam = final_jam;
      g_jam_ai_status.jam_votes = jam_votes;
      g_jam_ai_status.normal_votes = normal_votes;
      g_jam_ai_status.decision_count++;

      JamLed_SetJam(g_jam_ai_status.last_jam);
      JamTraffic_SetJam(g_jam_ai_status.last_jam);

      s_pending_votes = 0u;
      s_pending_jam_votes = 0u;
      s_pending_score_sum = 0.0f;

      while (usb_q_try_push_from_status() == 0u) {
        tx_thread_sleep(1);
      }
      (void)tx_semaphore_put(&semaphore);
    }
  }
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
