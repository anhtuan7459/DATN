#ifndef AI_PREPROCESS_H_
#define AI_PREPROCESS_H_

#include <stdint.h>
#include "bl0940_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AI_PREPROCESS_NUM_POINTS       50u
#define AI_PREPROCESS_WINDOW_SIZE      5u
#define AI_PREPROCESS_GRID_SIZE        64u
#define AI_PREPROCESS_IMAGE_SIZE       (AI_PREPROCESS_GRID_SIZE * AI_PREPROCESS_GRID_SIZE)

void AiPreprocess_Init(void);
void AiPreprocess_PushRecord(const SensorRecord *rec);
void AiPreprocess_Service(void);
uint8_t AiPreprocess_TakeLatestFrame(const float **image_ptr, uint32_t *frame_id);
uint32_t AiPreprocess_GetFrameCount(void);

#ifdef __cplusplus
}
#endif

#endif /* AI_PREPROCESS_H_ */
