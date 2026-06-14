#ifndef JAM_AI_H_
#define JAM_AI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JAM_AI_MODEL_MACC              27574916u
#define JAM_AI_MODEL_WEIGHTS_BYTES     243300u
#define JAM_AI_MODEL_ACTIVATIONS_BYTES 42528u
#define JAM_AI_MODEL_FLASH_BYTES       275898u
#define JAM_AI_MODEL_RAM_BYTES         46552u

/** USB AI tail payload size (after AA 55), includes DecisionEvent byte. */
#define JAM_AI_USB_PAYLOAD_BYTES       20u

typedef struct
{
  uint8_t ready;
  uint8_t last_output_u8;
  uint8_t last_jam;
  uint8_t last_raw_output_u8;
  uint8_t last_raw_jam;
  float last_raw_score;
  uint8_t jam_votes;
  uint8_t normal_votes;
  uint8_t decision_group_size;
  uint8_t reserved;
  float last_score;
  uint32_t last_frame_id;
  uint32_t last_cycles;
  uint32_t last_time_us;
  uint32_t run_count;
  uint32_t decision_count;
} JamAiStatus;

extern volatile JamAiStatus g_jam_ai_status;

void JamAi_Init(void);
void JamAi_Service(void);
/** Pop one queued group-decision frame for USB (FIFO). Returns 1 if out[0..19] filled. */
uint8_t JamAi_UsbQueuePop(uint8_t *out);
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
