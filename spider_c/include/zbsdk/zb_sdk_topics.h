#pragma once
#include <stdint.h>
#include "robot_msgs_c_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * zb_sdk : Backend-agnostic Robot Messaging SDK
 * ============================================================ */

/* ================== SDK init / shutdown ================== */

typedef enum {
    ZB_BACKEND_DDS = 0,
    ZB_BACKEND_SHM,
    ZB_BACKEND_UDP,
    ZB_BACKEND_TCP,
    ZB_BACKEND_MOCK,
} zb_backend_t;

/* Global SDK initialization
 * - mode: backend selection
 * - global_config_json: backend-wide configuration (may be NULL)
 */
int  zb_sdk_init(zb_backend_t mode, const char* global_config_json);

/* Shutdown SDK and release all backend resources */
void zb_sdk_shutdown(void);


/* ============================================================
 * Topic lifecycle rules
 *
 * - Each topic has explicit reader / writer initialization
 * - topic_name is mandatory
 * - No per-topic config at this stage
 * - try_read / write do NOT create resources
 * ============================================================ */


/* ================== PdState ================== */

/* Reader / Writer init */
int zb_pdstate_reader_init(const char* topic_name);
int zb_pdstate_writer_init(const char* topic_name);

/* Reader API */
int zb_pdstate_try_read(robot_msgs_PdState_View* out, int motor_cap);
int zb_pdstate_has_new(void);
int zb_pdstate_get_notify_fd(void);

/* Writer API */
int zb_pdstate_write(const robot_msgs_PdState_View* in);


/* ================== PdCmd ================== */

int zb_pdcmd_reader_init(const char* topic_name);
int zb_pdcmd_writer_init(const char* topic_name);

int zb_pdcmd_try_read(robot_msgs_PdCmd_View* out, int motor_cap);
int zb_pdcmd_has_new(void);
int zb_pdcmd_get_notify_fd(void);

int zb_pdcmd_write(const robot_msgs_PdCmd_View* in);


/* ================== IMU ================== */
/* Fixed-size struct, no cap needed */

int zb_imu_reader_init(const char* topic_name);
int zb_imu_writer_init(const char* topic_name);

int zb_imu_try_read(robot_msgs_IMU_C* out);
int zb_imu_has_new(void);
int zb_imu_get_notify_fd(void);

int zb_imu_write(const robot_msgs_IMU_C* in);


/* ================== Joystick ================== */

int zb_joystick_reader_init(const char* topic_name);
int zb_joystick_writer_init(const char* topic_name);

int zb_joystick_try_read(robot_msgs_Joystick_View* out,
                         int axes_cap,
                         int buttons_cap);
int zb_joystick_has_new(void);
int zb_joystick_get_notify_fd(void);

int zb_joystick_write(const robot_msgs_Joystick_View* in);


/* ================== ServoState ================== */

int zb_servostate_reader_init(const char* topic_name);
int zb_servostate_writer_init(const char* topic_name);

int zb_servostate_try_read(robot_msgs_ServoState_View* out, int motor_cap);
int zb_servostate_has_new(void);
int zb_servostate_get_notify_fd(void);

int zb_servostate_write(const robot_msgs_ServoState_View* in);


/* ================== ServoCmd ================== */

int zb_servocmd_reader_init(const char* topic_name);
int zb_servocmd_writer_init(const char* topic_name);

int zb_servocmd_try_read(robot_msgs_ServoCmd_View* out, int motor_cap);
int zb_servocmd_has_new(void);
int zb_servocmd_get_notify_fd(void);

int zb_servocmd_write(const robot_msgs_ServoCmd_View* in);


/* ================== CalData ================== */

int zb_caldata_reader_init(const char* topic_name);
int zb_caldata_writer_init(const char* topic_name);

int zb_caldata_try_read(robot_msgs_CalData_View* out, int offset_cap);
int zb_caldata_has_new(void);
int zb_caldata_get_notify_fd(void);

int zb_caldata_write(const robot_msgs_CalData_View* in);


#ifdef __cplusplus
}
#endif
