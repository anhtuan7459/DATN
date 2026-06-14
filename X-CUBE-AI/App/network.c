/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-19T19:14:52+0700
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x4c241159843bfc98b60ccc547fa061da"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-05-19T19:14:52+0700"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_V_I_Image0_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 4096, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conversion_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 30752, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 12544, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  nl_14_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conversion_15_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 288, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 147456, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1060, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3968, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7168, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3712, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3072, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 4768, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  gemm_12_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 352, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  gemm_13_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 64, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019689665641635656f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0019689665641635656f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_1_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0017393057933077216f, 0.0021313352044671774f, 0.0011061228578910232f, 0.001870958600193262f, 0.0021339214872568846f, 0.0023440418299287558f, 0.0016800584271550179f, 0.0020201385486871004f, 0.0014943613205105066f, 0.0019620724488049746f, 0.0017479363596066833f, 0.0017923711566254497f, 0.0018830531043931842f, 0.0022037557791918516f, 0.0017565598245710135f, 0.0021835914812982082f, 0.0013170818565413356f, 0.0024239567574113607f, 0.0016647812444716692f, 0.0013717805268242955f, 0.0014930308097973466f, 0.00202822289429605f, 0.002344110980629921f, 0.0017755829030647874f, 0.0016612051986157894f, 0.002149605890735984f, 0.0018121531466022134f, 0.0013602368999272585f, 0.0017374971648678184f, 0.0015502112219110131f, 0.001742564607411623f, 0.0013630433240905404f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_3_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005579546559602022f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_3_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005579546559602022f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_3_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008696180302649736f, 0.0007090225699357688f, 0.0008072011405602098f, 0.0005172952078282833f, 0.0007607038132846355f, 0.0007489143754355609f, 0.0007753648096695542f, 0.000692369940225035f, 0.0007865555235184729f, 0.0007037452305667102f, 0.0007049966952763498f, 0.0006991843692958355f, 0.0008627085480839014f, 0.0009456253028474748f, 0.0006832986837252975f, 0.0006414339877665043f, 0.0007603339618071914f, 0.0007429748075082898f, 0.0006989677203819156f, 0.0008707882952876389f, 0.0006537961889989674f, 0.0005662342300638556f, 0.0007423117058351636f, 0.0006128812674432993f, 0.0007080041687004268f, 0.0007996595231816173f, 0.000991218606941402f, 0.0006780228577554226f, 0.0007648194441571832f, 0.0006446231273002923f, 0.000877282174769789f, 0.00093777448637411f, 0.0007708353805355728f, 0.0007257282850332558f, 0.0005523909348994493f, 0.0010172267211601138f, 0.0007829668466001749f, 0.0009613231522962451f, 0.0006696413038298488f, 0.0008488901657983661f, 0.0009338488453067839f, 0.0008366424590349197f, 0.0007916470058262348f, 0.0008082091226242483f, 0.0007442947826348245f, 0.0009462430607527494f, 0.0009749612654559314f, 0.0006511260871775448f, 0.0006879506981931627f, 0.0009323694976046681f, 0.0007447657990269363f, 0.0007342348108068109f, 0.0008334295707754791f, 0.0007318599964492023f, 0.0007225971203297377f, 0.0007170725730247796f, 0.0008819318027235568f, 0.0007119219517335296f, 0.0006863354938104749f, 0.000520269968546927f, 0.0008128553745336831f, 0.0009365883888676763f, 0.0008454067283309996f, 0.0007830225513316691f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010710864327847958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010710864327847958f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 128,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00045476845116354525f, 0.0004328954382799566f, 0.0004694121889770031f, 0.000610142364166677f, 0.000448416278231889f, 0.00047856627497822046f, 0.00045524685992859304f, 0.0006810782360844314f, 0.0005542324506677687f, 0.0006654852186329663f, 0.00046987514360807836f, 0.00048670085379853845f, 0.0004970152513124049f, 0.0005261806072667241f, 0.00044217309914529324f, 0.00048091032658703625f, 0.00041082038660533726f, 0.0004793544067069888f, 0.0004563900292851031f, 0.0005088916514068842f, 0.00039287167601287365f, 0.0005690092802979052f, 0.0004853210411965847f, 0.0005788428825326264f, 0.00036792695755138993f, 0.0004846772935707122f, 0.000651899550575763f, 0.00048774859169498086f, 0.0003841367142740637f, 0.000464285199996084f, 0.0005073737702332437f, 0.000481186289107427f, 0.0004004144575446844f, 0.0005152362864464521f, 0.00039868499152362347f, 0.0003927379730157554f, 0.000450634746812284f, 0.000459642440546304f, 0.000374255032511428f, 0.00037973004509694874f, 0.00041500746738165617f, 0.00041560520185157657f, 0.0005191084346733987f, 0.0005361766088753939f, 0.0004973343457095325f, 0.0005488850874826312f, 0.0006153343711048365f, 0.0005327468970790505f, 0.0005176780978217721f, 0.0005477085942402482f, 0.000475306180305779f, 0.00044211733620613813f, 0.0005575189716182649f, 0.0005615883274003863f, 0.0004180106916464865f, 0.000621468061581254f, 0.0005650981911458075f, 0.00034338823752477765f, 0.0004268226330168545f, 0.000609366805292666f, 0.0004921650397591293f, 0.0004130286106374115f, 0.00047425556113012135f, 0.00033910127240233123f, 0.0005312560824677348f, 0.00040529342368245125f, 0.0003675880143418908f, 0.0004836043226532638f, 0.0005392964230850339f, 0.00040974299190565944f, 0.000424992322223261f, 0.00070628133835271f, 0.00044543444528244436f, 0.00047314196126535535f, 0.0005881531978957355f, 0.00048463314305990934f, 0.0005256764125078917f, 0.0006451097433455288f, 0.00048443328705616295f, 0.0004530843871179968f, 0.00033476040698587894f, 0.00040676724165678024f, 0.0005694983410649002f, 0.000537651649210602f, 0.000404673715820536f, 0.0004503038653638214f, 0.00032887596171349287f, 0.0006778729148209095f, 0.00046250715968199074f, 0.00048082825378514826f, 0.00046140598715282977f, 0.0004565983545035124f, 0.0004374318814370781f, 0.0003828219778370112f, 0.0005481764092110097f, 0.0005027916631661355f, 0.0004752103704959154f, 0.0005619196454063058f, 0.0005993445520289242f, 0.0004127772990614176f, 0.0005187247879803181f, 0.0005095288506709039f, 0.0004617457161657512f, 0.0003696940839290619f, 0.000472799118142575f, 0.0003239350044168532f, 0.0007643568678759038f, 0.0005782615626230836f, 0.0006057280115783215f, 0.00041764971683733165f, 0.0005782650550827384f, 0.0005437114159576595f, 0.00044552146573551f, 0.0006487506325356662f, 0.00037241712561808527f, 0.0005533986259251833f, 8.702170539720555e-09f, 0.0005535485106520355f, 0.000338183919666335f, 0.000400651158997789f, 0.0006607711547985673f, 0.0005146397743374109f, 0.0005335027817636728f, 0.0003407821641303599f, 0.000515997875481844f, 0.0004821075708605349f, 0.0004934758762829006f, 0.0004379207966849208f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conversion_15_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_UINTQ_ZP(0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015499460510909557f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0006347890011966228f, 0.0005394552717916667f, 3.937008052901092e-09f, 1.1783765330619644e-05f, 0.0005804795655421913f, 1.7599915736354887e-05f, 2.1745306355569483e-07f, 2.343029109397321e-06f, 0.00012086274364264682f, 3.75323070329614e-05f, 1.310962716161157e-06f, 0.00020438648061826825f, 0.0006219035713002086f, 3.937008052901092e-09f, 0.0005889409803785384f, 0.000580638472456485f, 9.128236706601456e-05f, 7.309427746804431e-05f, 0.0005886077415198088f, 2.466585328875226e-06f, 0.000641816237475723f, 0.0006207990809343755f, 9.695572225609794e-05f, 0.000636301701888442f, 0.0005807952838949859f, 8.245996468758676e-06f, 0.0004029637493658811f, 0.0005982466391287744f, 9.599031045581796e-07f, 7.279276360350195e-06f, 6.718439544783905e-05f, 0.0005756167811341584f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_12_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008337842300534248f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_12_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.001487896079197526f, 0.0018695821054279804f, 0.0021294071339070797f, 0.002135583432391286f, 0.0011953365756198764f, 0.0021327172871679068f, 0.0020096751395612955f, 0.0019374846015125513f, 0.0015174185391515493f, 0.0016355561092495918f, 0.00019867673108819872f, 0.002044942229986191f, 0.001934243948198855f, 0.0015568032395094633f, 0.0004823378985747695f, 0.0017697046278044581f, 0.0014647772768512368f, 0.0017408606363460422f, 0.002294871723279357f, 0.001783824060112238f, 0.0017669032095000148f, 0.0017893724143505096f, 0.002166059333831072f, 0.0018976100254803896f, 0.0024398656096309423f, 0.0010888511314988136f, 0.0012539791641756892f, 0.0021516394335776567f, 0.001266506384126842f, 0.0010924914386123419f, 0.0012842850992456079f, 0.0020551327615976334f, 0.0019110612338408828f, 0.0005698651075363159f, 0.0011278835590928793f, 0.0019651774782687426f, 0.0018416591919958591f, 0.0017378488555550575f, 0.0014089231844991446f, 0.0020661968737840652f, 0.0007184775313362479f, 0.0013955066679045558f, 0.002027977490797639f, 0.001962828217074275f, 0.0022934875451028347f, 0.002155672525987029f, 0.002008972689509392f, 0.001971164485439658f, 0.00020144831796642393f, 0.001990171615034342f, 0.0009720525122247636f, 0.0013124299002811313f, 0.001956252148374915f, 0.002023013075813651f, 0.002368657384067774f, 0.002369079040363431f, 0.0015663951635360718f, 0.001568591338582337f, 0.0007798433653078973f, 0.0023263958282768726f, 0.0018573771230876446f, 0.001863512210547924f, 0.0013562912354245782f, 0.001719469204545021f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05025213584303856f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_13_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0029964556451886892f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_14_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_V_I_Image0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_UINTQ_ZP(0)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_1_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 31, 31), AI_STRIDE_INIT(4, 1, 1, 32, 992),
  1, &conv2d_1_output_array, &conv2d_1_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1060, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1060, 1060),
  1, &conv2d_1_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch1, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 62, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1984),
  1, &conv2d_1_scratch1_array, &conv2d_1_scratch1_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 32), AI_STRIDE_INIT(4, 1, 1, 32, 96),
  1, &conv2d_1_weights_array, &conv2d_1_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_3_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 14, 14), AI_STRIDE_INIT(4, 1, 1, 64, 896),
  1, &conv2d_3_output_array, &conv2d_3_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 7168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7168, 7168),
  1, &conv2d_3_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch1, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 29, 2), AI_STRIDE_INIT(4, 1, 1, 64, 1856),
  1, &conv2d_3_scratch1_array, &conv2d_3_scratch1_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 6144),
  1, &conv2d_3_weights_array, &conv2d_3_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_5_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 6, 6), AI_STRIDE_INIT(4, 1, 1, 128, 768),
  1, &conv2d_5_output_array, &conv2d_5_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output0, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 4608, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4608, 4608),
  1, &conv2d_5_output_array, &conv2d_5_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 9216, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9216, 9216),
  1, &conv2d_5_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch1, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 12, 2), AI_STRIDE_INIT(4, 1, 1, 128, 1536),
  1, &conv2d_5_scratch1_array, &conv2d_5_scratch1_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 24576),
  1, &conv2d_5_weights_array, &conv2d_5_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  conversion_0_output, AI_STATIC,
  16, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 64), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &conversion_0_output_array, &conversion_0_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conversion_15_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &conversion_15_output_array, &conversion_15_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &gemm_11_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_output, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &gemm_11_output_array, &gemm_11_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_scratch0, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 4768, 1, 1), AI_STRIDE_INIT(4, 2, 2, 9536, 9536),
  1, &gemm_11_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_weights, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 4608, 32, 1, 1), AI_STRIDE_INIT(4, 1, 4608, 147456, 147456),
  1, &gemm_11_weights_array, &gemm_11_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_12_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_output, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_12_output_array, &gemm_12_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_scratch0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 352, 1, 1), AI_STRIDE_INIT(4, 2, 2, 704, 704),
  1, &gemm_12_scratch0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  gemm_12_weights, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 32, 64, 1, 1), AI_STRIDE_INIT(4, 1, 32, 2048, 2048),
  1, &gemm_12_weights_array, &gemm_12_weights_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gemm_13_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_output, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &gemm_13_output_array, &gemm_13_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_scratch0, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 2, 2, 128, 128),
  1, &gemm_13_scratch0_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  gemm_13_weights, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 64, 1, 1, 1), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &gemm_13_weights_array, &gemm_13_weights_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  nl_14_output, AI_STATIC,
  30, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1, 1),
  1, &nl_14_output_array, &nl_14_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_V_I_Image0_output, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 64), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &serving_default_V_I_Image0_output_array, &serving_default_V_I_Image0_output_array_intq)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_15_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_15_layer, 15,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_15_chain,
  NULL, &conversion_15_layer, AI_STATIC, 
)


AI_STATIC_CONST ai_i8 nl_14_nl_params_data[] = { -128, -128, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -126, -125, -125, -125, -125, -125, -125, -124, -124, -124, -124, -124, -123, -123, -123, -123, -123, -122, -122, -122, -121, -121, -121, -120, -120, -119, -119, -119, -118, -118, -117, -117, -116, -115, -115, -114, -114, -113, -112, -111, -111, -110, -109, -108, -107, -106, -105, -104, -103, -102, -100, -99, -98, -96, -95, -93, -92, -90, -89, -87, -85, -83, -82, -80, -78, -76, -73, -71, -69, -67, -64, -62, -59, -57, -54, -52, -49, -46, -43, -40, -37, -35, -32, -28, -25, -22, -19, -16, -13, -10, -6, -3, 0, 3, 6, 10, 13, 16, 19, 22, 25, 28, 32, 35, 37, 40, 43, 46, 49, 52, 54, 57, 59, 62, 64, 67, 69, 71, 73, 76, 78, 80, 82, 83, 85, 87, 89, 90, 92, 93, 95, 96, 98, 99, 100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111, 112, 113, 114, 114, 115, 115, 116, 117, 117, 118, 118, 119, 119, 119, 120, 120, 121, 121, 121, 122, 122, 122, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    nl_14_nl_params, AI_ARRAY_FORMAT_S8,
    nl_14_nl_params_data, nl_14_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_14_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_14_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &nl_14_chain,
  NULL, &conversion_15_layer, AI_STATIC, 
  .nl_params = &nl_14_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_13_weights, &gemm_13_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_13_layer, 13,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA,
  &gemm_13_chain,
  NULL, &nl_14_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_12_weights, &gemm_12_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_12_layer, 12,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_12_chain,
  NULL, &gemm_13_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_11_weights, &gemm_11_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_11_layer, 11,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_11_chain,
  NULL, &gemm_12_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_5_scratch0, &conv2d_5_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 6,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &conv2d_5_chain,
  NULL, &gemm_11_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_3_scratch0, &conv2d_3_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 4,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &conv2d_3_chain,
  NULL, &conv2d_5_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_1_scratch0, &conv2d_1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 2,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &conv2d_1_chain,
  NULL, &conv2d_3_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conversion_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_V_I_Image0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conversion_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conversion_0_layer, 0,
  NL_TYPE, 0x0, NULL,
  nl, node_convert_integer,
  &conversion_0_chain,
  NULL, &conv2d_1_layer, AI_STATIC, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 243300, 1, 1),
    243300, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 42528, 1, 1),
    42528, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_V_I_Image0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &conversion_15_output),
  &conversion_0_layer, 0x65164010, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 243300, 1, 1),
      243300, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 42528, 1, 1),
      42528, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_V_I_Image0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &conversion_15_output),
  &conversion_0_layer, 0x65164010, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_V_I_Image0_output_array.data = AI_PTR(g_network_activations_map[0] + 28672);
    serving_default_V_I_Image0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28672);
    conversion_0_output_array.data = AI_PTR(g_network_activations_map[0] + 28672);
    conversion_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28672);
    conv2d_1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 32768);
    conv2d_1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 32768);
    conv2d_1_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 33828);
    conv2d_1_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 33828);
    conv2d_1_output_array.data = AI_PTR(g_network_activations_map[0] + 896);
    conv2d_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 896);
    conv2d_3_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 31648);
    conv2d_3_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 31648);
    conv2d_3_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 38816);
    conv2d_3_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 38816);
    conv2d_3_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_5_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 12544);
    conv2d_5_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 12544);
    conv2d_5_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 21760);
    conv2d_5_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 21760);
    conv2d_5_output_array.data = AI_PTR(g_network_activations_map[0] + 24832);
    conv2d_5_output_array.data_start = AI_PTR(g_network_activations_map[0] + 24832);
    gemm_11_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_11_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_11_output_array.data = AI_PTR(g_network_activations_map[0] + 9536);
    gemm_11_output_array.data_start = AI_PTR(g_network_activations_map[0] + 9536);
    gemm_12_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_12_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_12_output_array.data = AI_PTR(g_network_activations_map[0] + 704);
    gemm_12_output_array.data_start = AI_PTR(g_network_activations_map[0] + 704);
    gemm_13_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_13_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_13_output_array.data = AI_PTR(g_network_activations_map[0] + 128);
    gemm_13_output_array.data_start = AI_PTR(g_network_activations_map[0] + 128);
    nl_14_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_14_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conversion_15_output_array.data = AI_PTR(g_network_activations_map[0] + 4);
    conversion_15_output_array.data_start = AI_PTR(g_network_activations_map[0] + 4);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(g_network_weights_map[0] + 288);
    conv2d_1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 288);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(g_network_weights_map[0] + 416);
    conv2d_3_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 416);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(g_network_weights_map[0] + 18848);
    conv2d_3_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 18848);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(g_network_weights_map[0] + 19104);
    conv2d_5_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 19104);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(g_network_weights_map[0] + 92832);
    conv2d_5_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 92832);
    gemm_11_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_11_weights_array.data = AI_PTR(g_network_weights_map[0] + 93344);
    gemm_11_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 93344);
    gemm_11_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_11_bias_array.data = AI_PTR(g_network_weights_map[0] + 240800);
    gemm_11_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 240800);
    gemm_12_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_12_weights_array.data = AI_PTR(g_network_weights_map[0] + 240928);
    gemm_12_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 240928);
    gemm_12_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_12_bias_array.data = AI_PTR(g_network_weights_map[0] + 242976);
    gemm_12_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 242976);
    gemm_13_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_13_weights_array.data = AI_PTR(g_network_weights_map[0] + 243232);
    gemm_13_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 243232);
    gemm_13_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_13_bias_array.data = AI_PTR(g_network_weights_map[0] + 243296);
    gemm_13_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 243296);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 27574916,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x65164010,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 27574916,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x65164010,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

