#include "ai_preprocess.h"

#include "main.h"

#include <string.h>

#define AI_PREPROCESS_PENDING_SAMPLES    256u
#define AI_PREPROCESS_MAX_CYCLE_SAMPLES  128u
#define AI_PREPROCESS_MIN_CYCLE_SAMPLES  10u
#define AI_PREPROCESS_HIST_MIN          (-1.1f)
#define AI_PREPROCESS_HIST_MAX           (1.1f)

typedef struct {
  float u_wave;
  float i_wave;
} AiWaveSample;

static AiWaveSample s_pending_samples[AI_PREPROCESS_PENDING_SAMPLES];
static uint16_t s_pending_read_idx;
static uint16_t s_pending_write_idx;
static uint16_t s_pending_count;

static AiWaveSample s_prev_sample;
static uint8_t s_has_prev_sample;
static uint8_t s_capture_started;

static float s_cycle_u[AI_PREPROCESS_MAX_CYCLE_SAMPLES];
static float s_cycle_i[AI_PREPROCESS_MAX_CYCLE_SAMPLES];
static uint16_t s_cycle_len;

static float s_window_u[AI_PREPROCESS_WINDOW_SIZE][AI_PREPROCESS_NUM_POINTS];
static float s_window_i[AI_PREPROCESS_WINDOW_SIZE][AI_PREPROCESS_NUM_POINTS];
static uint8_t s_window_fill;
static uint8_t s_window_write_idx;

static float s_latest_frame[AI_PREPROCESS_IMAGE_SIZE];
static uint8_t s_frame_ready;
static uint32_t s_frame_counter;

static float absf_local(float value)
{
  return (value < 0.0f) ? -value : value;
}

static void enter_critical(uint32_t *primask)
{
  *primask = __get_PRIMASK();
  __disable_irq();
}

static void exit_critical(uint32_t primask)
{
  __set_PRIMASK(primask);
}

static void resample_cycle(const float *src, uint16_t src_len, float *dst)
{
  uint32_t out_idx;

  if (src_len == 0u) {
    memset(dst, 0, AI_PREPROCESS_NUM_POINTS * sizeof(float));
    return;
  }

  if (src_len == 1u) {
    for (out_idx = 0u; out_idx < AI_PREPROCESS_NUM_POINTS; ++out_idx) {
      dst[out_idx] = src[0];
    }
    return;
  }

  for (out_idx = 0u; out_idx < AI_PREPROCESS_NUM_POINTS; ++out_idx) {
    float pos = ((float)out_idx * (float)(src_len - 1u)) /
                (float)(AI_PREPROCESS_NUM_POINTS - 1u);
    uint16_t left = (uint16_t)pos;
    uint16_t right = left + 1u;
    float frac = pos - (float)left;
    float left_val = src[left];
    float right_val = src[(right < src_len) ? right : left];
    dst[out_idx] = left_val + ((right_val - left_val) * frac);
  }
}

static void build_latest_frame(void)
{
  float mean_u[AI_PREPROCESS_NUM_POINTS];
  float mean_i[AI_PREPROCESS_NUM_POINTS];
  float max_abs_u = 0.0f;
  float max_abs_i = 0.0f;
  float max_count = 0.0f;
  float bin_width = (AI_PREPROCESS_HIST_MAX - AI_PREPROCESS_HIST_MIN) /
                    (float)AI_PREPROCESS_GRID_SIZE;
  uint32_t point_idx;
  uint32_t order_idx;

  memset(mean_u, 0, sizeof(mean_u));
  memset(mean_i, 0, sizeof(mean_i));
  memset(s_latest_frame, 0, sizeof(s_latest_frame));

  for (order_idx = 0u; order_idx < AI_PREPROCESS_WINDOW_SIZE; ++order_idx) {
    uint8_t slot = (uint8_t)((s_window_write_idx + order_idx) % AI_PREPROCESS_WINDOW_SIZE);
    for (point_idx = 0u; point_idx < AI_PREPROCESS_NUM_POINTS; ++point_idx) {
      mean_u[point_idx] += s_window_u[slot][point_idx];
      mean_i[point_idx] += s_window_i[slot][point_idx];
    }
  }

  for (point_idx = 0u; point_idx < AI_PREPROCESS_NUM_POINTS; ++point_idx) {
    mean_u[point_idx] /= (float)AI_PREPROCESS_WINDOW_SIZE;
    mean_i[point_idx] /= (float)AI_PREPROCESS_WINDOW_SIZE;

    if (absf_local(mean_u[point_idx]) > max_abs_u) {
      max_abs_u = absf_local(mean_u[point_idx]);
    }
    if (absf_local(mean_i[point_idx]) > max_abs_i) {
      max_abs_i = absf_local(mean_i[point_idx]);
    }
  }

  if (max_abs_u < 1.0e-9f) {
    max_abs_u = 1.0f;
  }
  if (max_abs_i < 1.0e-9f) {
    max_abs_i = 1.0f;
  }

  for (point_idx = 0u; point_idx < AI_PREPROCESS_NUM_POINTS; ++point_idx) {
    float u_norm = mean_u[point_idx] / max_abs_u;
    float i_norm = mean_i[point_idx] / max_abs_i;

    if ((u_norm < AI_PREPROCESS_HIST_MIN) || (u_norm > AI_PREPROCESS_HIST_MAX) ||
        (i_norm < AI_PREPROCESS_HIST_MIN) || (i_norm > AI_PREPROCESS_HIST_MAX)) {
      continue;
    }

    int32_t u_bin = (int32_t)((u_norm - AI_PREPROCESS_HIST_MIN) / bin_width);
    int32_t i_bin = (int32_t)((i_norm - AI_PREPROCESS_HIST_MIN) / bin_width);

    if (u_bin >= (int32_t)AI_PREPROCESS_GRID_SIZE) {
      u_bin = (int32_t)AI_PREPROCESS_GRID_SIZE - 1;
    }
    if (i_bin >= (int32_t)AI_PREPROCESS_GRID_SIZE) {
      i_bin = (int32_t)AI_PREPROCESS_GRID_SIZE - 1;
    }
    if ((u_bin < 0) || (i_bin < 0)) {
      continue;
    }

    s_latest_frame[(u_bin * AI_PREPROCESS_GRID_SIZE) + i_bin] += 1.0f;
  }

  for (point_idx = 0u; point_idx < AI_PREPROCESS_IMAGE_SIZE; ++point_idx) {
    if (s_latest_frame[point_idx] > max_count) {
      max_count = s_latest_frame[point_idx];
    }
  }

  if (max_count < 1.0e-9f) {
    max_count = 1.0f;
  }

  for (point_idx = 0u; point_idx < AI_PREPROCESS_IMAGE_SIZE; ++point_idx) {
    s_latest_frame[point_idx] /= max_count;
  }

  s_frame_ready = 1u;
  s_frame_counter++;
}

static void store_resampled_cycle(void)
{
  float resampled_u[AI_PREPROCESS_NUM_POINTS];
  float resampled_i[AI_PREPROCESS_NUM_POINTS];

  if (s_cycle_len < AI_PREPROCESS_MIN_CYCLE_SAMPLES) {
    return;
  }

  resample_cycle(s_cycle_u, s_cycle_len, resampled_u);
  resample_cycle(s_cycle_i, s_cycle_len, resampled_i);

  memcpy(s_window_u[s_window_write_idx], resampled_u, sizeof(resampled_u));
  memcpy(s_window_i[s_window_write_idx], resampled_i, sizeof(resampled_i));

  s_window_write_idx = (uint8_t)((s_window_write_idx + 1u) % AI_PREPROCESS_WINDOW_SIZE);
  if (s_window_fill < AI_PREPROCESS_WINDOW_SIZE) {
    s_window_fill++;
  }

  if (s_window_fill == AI_PREPROCESS_WINDOW_SIZE) {
    build_latest_frame();
    s_window_fill = 0u;
    s_window_write_idx = 0u;
  }
}

static void append_cycle_sample(float u_wave, float i_wave)
{
  if (s_cycle_len >= AI_PREPROCESS_MAX_CYCLE_SAMPLES) {
    return;
  }

  s_cycle_u[s_cycle_len] = u_wave;
  s_cycle_i[s_cycle_len] = i_wave;
  s_cycle_len++;
}

static void process_sample(const AiWaveSample *sample)
{
  uint8_t zero_cross;

  if (sample == NULL) {
    return;
  }

  if (s_has_prev_sample == 0u) {
    s_prev_sample = *sample;
    s_has_prev_sample = 1u;
    return;
  }

  zero_cross = ((s_prev_sample.u_wave < 0) && (sample->u_wave >= 0)) ? 1u : 0u;

  if (s_capture_started == 0u) {
    if (zero_cross != 0u) {
      s_capture_started = 1u;
      s_cycle_len = 0u;
      append_cycle_sample(s_prev_sample.u_wave, s_prev_sample.i_wave);
    }
  } else if (zero_cross != 0u) {
    store_resampled_cycle();
    s_cycle_len = 0u;
    append_cycle_sample(s_prev_sample.u_wave, s_prev_sample.i_wave);
  } else {
    append_cycle_sample(s_prev_sample.u_wave, s_prev_sample.i_wave);
  }

  s_prev_sample = *sample;
}

void AiPreprocess_Init(void)
{
  uint32_t primask;

  enter_critical(&primask);
  memset(s_pending_samples, 0, sizeof(s_pending_samples));
  s_pending_read_idx = 0u;
  s_pending_write_idx = 0u;
  s_pending_count = 0u;
  s_has_prev_sample = 0u;
  s_capture_started = 0u;
  s_cycle_len = 0u;
  s_window_fill = 0u;
  s_window_write_idx = 0u;
  s_frame_ready = 0u;
  s_frame_counter = 0u;
  memset(&s_prev_sample, 0, sizeof(s_prev_sample));
  memset(s_cycle_u, 0, sizeof(s_cycle_u));
  memset(s_cycle_i, 0, sizeof(s_cycle_i));
  memset(s_window_u, 0, sizeof(s_window_u));
  memset(s_window_i, 0, sizeof(s_window_i));
  memset(s_latest_frame, 0, sizeof(s_latest_frame));
  exit_critical(primask);
}

void AiPreprocess_PushRecord(const SensorRecord *rec)
{
  AiWaveSample sample;
  uint32_t primask;

  if (rec == NULL) {
    return;
  }

  sample.u_wave = bl0940_convert_voltage_wave(rec->voltage_wave);
  sample.i_wave = bl0940_convert_current_wave(rec->current_wave);

  enter_critical(&primask);
  s_pending_samples[s_pending_write_idx] = sample;
  s_pending_write_idx = (uint16_t)((s_pending_write_idx + 1u) % AI_PREPROCESS_PENDING_SAMPLES);

  if (s_pending_count < AI_PREPROCESS_PENDING_SAMPLES) {
    s_pending_count++;
  } else {
    s_pending_read_idx = (uint16_t)((s_pending_read_idx + 1u) % AI_PREPROCESS_PENDING_SAMPLES);
  }
  exit_critical(primask);
}

void AiPreprocess_Service(void)
{
  AiWaveSample sample;
  uint8_t has_sample;
  uint32_t primask;

  do {
    enter_critical(&primask);
    has_sample = (s_pending_count > 0u) ? 1u : 0u;
    if (has_sample != 0u) {
      sample = s_pending_samples[s_pending_read_idx];
      s_pending_read_idx = (uint16_t)((s_pending_read_idx + 1u) % AI_PREPROCESS_PENDING_SAMPLES);
      s_pending_count--;
    }
    exit_critical(primask);

    if (has_sample != 0u) {
      process_sample(&sample);
    }
  } while (has_sample != 0u);
}

uint8_t AiPreprocess_TakeLatestFrame(const float **image_ptr, uint32_t *frame_id)
{
  uint32_t primask;
  uint8_t ready;

  enter_critical(&primask);
  ready = s_frame_ready;
  if (ready != 0u) {
    if (image_ptr != NULL) {
      *image_ptr = s_latest_frame;
    }
    if (frame_id != NULL) {
      *frame_id = s_frame_counter;
    }
    s_frame_ready = 0u;
  }
  exit_critical(primask);

  return ready;
}

uint32_t AiPreprocess_GetFrameCount(void)
{
  uint32_t primask;
  uint32_t frame_count;

  enter_critical(&primask);
  frame_count = s_frame_counter;
  exit_critical(primask);
  return frame_count;
}
