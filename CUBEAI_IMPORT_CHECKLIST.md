# CubeAI Import Checklist

Project firmware:

`d:\Acer\Documents\BL0940_H563_JamLed\BL0940_JamLed`

Model to import:

`d:\Acer\Documents\tflite_cnn2d_jam80\cnn2d_jam80_int8.tflite`

## 1. Open CubeMX / STM32CubeIDE

1. Open `STM32h5_Bl0940.ioc`
2. In the middleware / software packs panel, add `X-CUBE-AI`
3. Open the X-CUBE-AI configuration page

## 2. Import model

1. Choose `Analyze`
2. Import `cnn2d_jam80_int8.tflite`
3. Confirm the network input is `1 x 64 x 64 x 1`
4. Confirm the network output is `1 x 1`
5. Generate code

Expected model quantization seen from TFLite:

- input dtype: `uint8`
- input quantization: `scale=0.003921568859368563`, `zero_point=0`
- output dtype: `uint8`
- output quantization: `scale=0.00390625`, `zero_point=0`

## 3. Save the metrics from CubeAI

Record these values from the CubeAI report:

- `MACC`
- `Weights (Flash)`
- `Activations / RAM`
- `Runtime RAM total`
- `Code / Flash total`

These are the values needed for the report.

## 4. Validation on target

If CubeAI offers validation:

1. Run validation on desktop
2. Run validation on target if available
3. Save:
   - inference time
   - cycle count if shown
   - RAM / Flash report screenshot

## 5. What is already ready in firmware

The firmware already has preprocessing close to `Trainuytin.py`:

- sign-extend waveform samples as signed 20-bit
- zero-crossing on `Uwave`
- resample each cycle to 50 points
- group 5 cycles
- average by phase
- build `64 x 64` histogram frame

This preprocessing lives in:

- `Core/Src/ai_preprocess.c`
- `Core/Inc/ai_preprocess.h`

## 6. What to do after CubeAI code generation

After CubeAI generates the files, the next firmware work is:

1. Read latest frame from `AiPreprocess_TakeLatestFrame(...)`
2. Copy / quantize it into CubeAI input buffer
3. Run inference
4. Read output score
5. Set `JamLed_SetJam(...)`
6. Measure inference time around the inference call

## 7. Inference timing to add after generation

Use DWT cycle counter around the generated inference call:

1. read `start_cycles`
2. run CubeAI inference
3. read `end_cycles`
4. `delta_cycles = end_cycles - start_cycles`
5. convert to microseconds using CPU clock (`250 MHz` in this project)

Formula:

`time_us = delta_cycles / 250.0`

## 8. Current limitation

This project does not yet contain generated CubeAI files, so real inference is not wired yet.
Once CubeAI code is generated, it can be connected to the existing preprocessing thread.
