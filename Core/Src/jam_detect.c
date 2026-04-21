#include "jam_detect.h"
#include "ai_preprocess.h"
#include "jam_led.h"

/*
 * Đặt thành 1 để test LED: coi là kẹt khi current_wave vượt ngưỡng (chỉ demo phần cứng).
 * Logic thật: thay bằng TFLite / mô hình đã train.
 */
#ifndef JAM_DETECT_THRESHOLD_DEMO
#define JAM_DETECT_THRESHOLD_DEMO 0
#endif

#if JAM_DETECT_THRESHOLD_DEMO
#ifndef JAM_DETECT_WAVE_THRESHOLD
#define JAM_DETECT_WAVE_THRESHOLD 500000u
#endif
#endif

void JamDetect_OnNewRecord(const SensorRecord *rec)
{
  AiPreprocess_PushRecord(rec);
#if JAM_DETECT_THRESHOLD_DEMO
  if (rec->current_wave > JAM_DETECT_WAVE_THRESHOLD) {
    JamLed_SetJam(1u);
  } else {
    JamLed_SetJam(0u);
  }
#else
  (void)rec;
#endif
}
