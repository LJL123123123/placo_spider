#pragma once
#include <stdint.h>
#include "robot_msgs_c_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== View types: message layout for C API ==========
 * Rules:
 * - sequence<T>  => (T* data, int len)
 * - cap is NOT part of view; cap is an API parameter.
 */

/* PdState: ts + sequence<PdMotorState> motors */
typedef struct {
    int64_t ts;
    robot_msgs_PdMotorState_C* motors;
    int motors_len;
} robot_msgs_PdState_View;

/* PdCmd: ts + sequence<PdMotorCmd> motors */
typedef struct {
    int64_t ts;
    robot_msgs_PdMotorCmd_C* motors;
    int motors_len;
} robot_msgs_PdCmd_View;

/* Joystick: ts + sequence<float> axes + sequence<int32> buttons */
typedef struct {
    int64_t ts;
    float* axes;
    int axes_len;
    int32_t* buttons;
    int buttons_len;
} robot_msgs_Joystick_View;

/* ServoState: ts + sequence<Cia402State> motors */
typedef struct {
    int64_t ts;
    robot_msgs_Cia402State_C* motors;
    int motors_len;
} robot_msgs_ServoState_View;

/* ServoCmd: ts + sequence<Cia402Cmd> motors */
typedef struct {
    int64_t ts;
    robot_msgs_Cia402Cmd_C* motors;
    int motors_len;
} robot_msgs_ServoCmd_View;

/* CalData: ts + calState + sequence<float> offset */
typedef struct {
    int64_t ts;
    int16_t calState;
    float* offset;
    int offset_len;
} robot_msgs_CalData_View;

#ifdef __cplusplus
}
#endif
