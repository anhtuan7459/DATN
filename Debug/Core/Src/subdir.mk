################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Core/Src/ai_linker_wrap.c \
../Core/Src/ai_preprocess.c \
../Core/Src/app_threadx.c \
../Core/Src/bl0940_driver.c \
../Core/Src/jam_ai.c \
../Core/Src/jam_buzzer.c \
../Core/Src/jam_detect.c \
../Core/Src/jam_led.c \
../Core/Src/jam_traffic.c \
../Core/Src/linked_list.c \
../Core/Src/main.c \
../Core/Src/stm32h5xx_hal_msp.c \
../Core/Src/stm32h5xx_hal_timebase_tim.c \
../Core/Src/stm32h5xx_it.c \
../Core/Src/sysmem.c \
../Core/Src/system_stm32h5xx.c 

S_UPPER_SRCS += \
../Core/Src/tx_initialize_low_level.S 

OBJS += \
./Core/Src/ai_linker_wrap.o \
./Core/Src/ai_preprocess.o \
./Core/Src/app_threadx.o \
./Core/Src/bl0940_driver.o \
./Core/Src/jam_ai.o \
./Core/Src/jam_buzzer.o \
./Core/Src/jam_detect.o \
./Core/Src/jam_led.o \
./Core/Src/jam_traffic.o \
./Core/Src/linked_list.o \
./Core/Src/main.o \
./Core/Src/stm32h5xx_hal_msp.o \
./Core/Src/stm32h5xx_hal_timebase_tim.o \
./Core/Src/stm32h5xx_it.o \
./Core/Src/sysmem.o \
./Core/Src/system_stm32h5xx.o \
./Core/Src/tx_initialize_low_level.o 

S_UPPER_DEPS += \
./Core/Src/tx_initialize_low_level.d 

C_DEPS += \
./Core/Src/ai_linker_wrap.d \
./Core/Src/ai_preprocess.d \
./Core/Src/app_threadx.d \
./Core/Src/bl0940_driver.d \
./Core/Src/jam_ai.d \
./Core/Src/jam_buzzer.d \
./Core/Src/jam_detect.d \
./Core/Src/jam_led.d \
./Core/Src/jam_traffic.d \
./Core/Src/linked_list.d \
./Core/Src/main.d \
./Core/Src/stm32h5xx_hal_msp.d \
./Core/Src/stm32h5xx_hal_timebase_tim.d \
./Core/Src/stm32h5xx_it.d \
./Core/Src/sysmem.d \
./Core/Src/system_stm32h5xx.d 


# Each subdirectory must supply rules for building sources it contributes
Core/Src/%.o Core/Src/%.su Core/Src/%.cyclo: ../Core/Src/%.c Core/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m33 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32H562xx -DTX_INCLUDE_USER_DEFINE_FILE -DTX_SINGLE_MODE_NON_SECURE=1 -DUX_INCLUDE_USER_DEFINE_FILE -c -I../Core/Inc -I../Drivers/STM32H5xx_HAL_Driver/Inc -I../Drivers/STM32H5xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32H5xx/Include -I../Drivers/CMSIS/Include -I../AZURE_RTOS/App -I../USBX/App -I../USBX/Target -I../Middlewares/ST/threadx/common/inc -I../Middlewares/ST/threadx/ports/cortex_m33/gnu/inc -I../Middlewares/ST/usbx/common/core/inc -I../Middlewares/ST/usbx/ports/generic/inc -I../Middlewares/ST/usbx/common/usbx_stm32_device_controllers -I../Middlewares/ST/usbx/common/usbx_device_classes/inc -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Middlewares/ST/AI/Inc -Os -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
Core/Src/%.o: ../Core/Src/%.S Core/Src/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m33 -g3 -DDEBUG -DTX_SINGLE_MODE_NON_SECURE=1 -c -I../Core/Inc -I../AZURE_RTOS/App -I../USBX/App -I../USBX/Target -I../Drivers/STM32H5xx_HAL_Driver/Inc -I../Drivers/STM32H5xx_HAL_Driver/Inc/Legacy -I../Middlewares/ST/threadx/common/inc -I../Drivers/CMSIS/Device/ST/STM32H5xx/Include -I../Middlewares/ST/threadx/ports/cortex_m33/gnu/inc -I../Middlewares/ST/usbx/common/core/inc -I../Middlewares/ST/usbx/ports/generic/inc -I../Middlewares/ST/usbx/common/usbx_stm32_device_controllers -I../Middlewares/ST/usbx/common/usbx_device_classes/inc -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Middlewares/ST/AI/Inc -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-Core-2f-Src

clean-Core-2f-Src:
	-$(RM) ./Core/Src/ai_linker_wrap.cyclo ./Core/Src/ai_linker_wrap.d ./Core/Src/ai_linker_wrap.o ./Core/Src/ai_linker_wrap.su ./Core/Src/ai_preprocess.cyclo ./Core/Src/ai_preprocess.d ./Core/Src/ai_preprocess.o ./Core/Src/ai_preprocess.su ./Core/Src/app_threadx.cyclo ./Core/Src/app_threadx.d ./Core/Src/app_threadx.o ./Core/Src/app_threadx.su ./Core/Src/bl0940_driver.cyclo ./Core/Src/bl0940_driver.d ./Core/Src/bl0940_driver.o ./Core/Src/bl0940_driver.su ./Core/Src/jam_ai.cyclo ./Core/Src/jam_ai.d ./Core/Src/jam_ai.o ./Core/Src/jam_ai.su ./Core/Src/jam_buzzer.cyclo ./Core/Src/jam_buzzer.d ./Core/Src/jam_buzzer.o ./Core/Src/jam_buzzer.su ./Core/Src/jam_detect.cyclo ./Core/Src/jam_detect.d ./Core/Src/jam_detect.o ./Core/Src/jam_detect.su ./Core/Src/jam_led.cyclo ./Core/Src/jam_led.d ./Core/Src/jam_led.o ./Core/Src/jam_led.su ./Core/Src/jam_traffic.cyclo ./Core/Src/jam_traffic.d ./Core/Src/jam_traffic.o ./Core/Src/jam_traffic.su ./Core/Src/linked_list.cyclo ./Core/Src/linked_list.d ./Core/Src/linked_list.o ./Core/Src/linked_list.su ./Core/Src/main.cyclo ./Core/Src/main.d ./Core/Src/main.o ./Core/Src/main.su ./Core/Src/stm32h5xx_hal_msp.cyclo ./Core/Src/stm32h5xx_hal_msp.d ./Core/Src/stm32h5xx_hal_msp.o ./Core/Src/stm32h5xx_hal_msp.su ./Core/Src/stm32h5xx_hal_timebase_tim.cyclo ./Core/Src/stm32h5xx_hal_timebase_tim.d ./Core/Src/stm32h5xx_hal_timebase_tim.o ./Core/Src/stm32h5xx_hal_timebase_tim.su ./Core/Src/stm32h5xx_it.cyclo ./Core/Src/stm32h5xx_it.d ./Core/Src/stm32h5xx_it.o ./Core/Src/stm32h5xx_it.su ./Core/Src/sysmem.cyclo ./Core/Src/sysmem.d ./Core/Src/sysmem.o ./Core/Src/sysmem.su ./Core/Src/system_stm32h5xx.cyclo ./Core/Src/system_stm32h5xx.d ./Core/Src/system_stm32h5xx.o ./Core/Src/system_stm32h5xx.su ./Core/Src/tx_initialize_low_level.d ./Core/Src/tx_initialize_low_level.o

.PHONY: clean-Core-2f-Src

