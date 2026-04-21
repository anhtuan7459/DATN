#ifndef JAM_AI_H_
#define JAM_AI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JAM_AI_MODEL_MACC              7490548u
#define JAM_AI_MODEL_WEIGHTS_BYTES     327572u
#define JAM_AI_MODEL_ACTIVATIONS_BYTES 23824u
#define JAM_AI_MODEL_FLASH_BYTES       360098u
#define JAM_AI_MODEL_RAM_BYTES         27848u

typedef struct
{
  uint8_t ready;
  uint8_t last_output_u8;
  uint8_t last_jam;
  uint8_t reserved;
  float last_score;
  uint32_t last_frame_id;
  uint32_t last_cycles;
  uint32_t last_time_us;
  uint32_t run_count;
} JamAiStatus;

extern volatile JamAiStatus g_jam_ai_status;

void JamAi_Init(void);
void JamAi_Service(void);
uint8_t JamAi_IsReady(void);
float JamAi_GetLastScore(void);
uint8_t JamAi_GetLastOutputU8(void);
uint32_t JamAi_GetLastCycles(void);
uint32_t JamAi_GetLastTimeUs(void);
uint32_t JamAi_GetRunCount(void);

#ifdef __cplusplus
}
#endif

#endif /* JAM_AI_H_ */
