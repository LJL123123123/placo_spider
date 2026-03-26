#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== POD types: mirror of IDL structs ========== */

typedef struct {
    float q;     /* rad */
    float qd;    /* rad/s */
    float tau;
    float kp;
    float kd;
    float kt;
    float temp;
    int32_t error;
    int32_t sw;  /* opmode + sw */
} robot_msgs_PdMotorState_C;

typedef struct {
    float q;     /* rad */
    float qd;    /* rad/s */
    float tau;
    float kp;
    float kd;
    float kt;
    int32_t mode;
} robot_msgs_PdMotorCmd_C;

typedef struct {
    int64_t ts;
    float acc[3];
    float gry[3];
    float mag[3];
    float euler[3];
    float quat[4];
} robot_msgs_IMU_C;

typedef struct {
    int32_t sw;
    int32_t pos;
    int32_t vel;
    int32_t tor;
    float   temp;
    int32_t error;
    int32_t opMode;
} robot_msgs_Cia402State_C;

typedef struct {
    int32_t cw;
    int32_t tPos;
    int32_t tVel;
    int32_t tTor;
    int32_t opMode;
} robot_msgs_Cia402Cmd_C;

typedef struct {
    int64_t ts;
    int16_t calState; /* IDL short */
    /* offset 是 sequence<float>，在 View 里表达 */
} robot_msgs_CalData_C_meta; /* 仅保留非 sequence 字段 */

#ifdef __cplusplus
}
#endif
