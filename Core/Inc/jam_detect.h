#ifndef JAM_DETECT_H_
#define JAM_DETECT_H_

#include "bl0940_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gọi sau khi đã ghi đủ một SensorRecord (trước khi tăng record_index). */
void JamDetect_OnNewRecord(const SensorRecord *rec);

#ifdef __cplusplus
}
#endif

#endif /* JAM_DETECT_H_ */
