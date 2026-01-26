#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPG + 全身运动学 QP Demo with Shared Memory Communication

说明:
- 使用 `FootTrajectoryCPG` 生成四足的期望足端位置 (相对于机体中心)
- 使用 placo 的 `KinematicsSolver` 将这些足端目标作为位置任务，进行全身逆运动学/QP 求解
- 通过共享内存与 MuJoCo 仿真通信 (100Hz)

使用方法:
  # Terminal 1: Start CPG planner
  python3 spider_cpg_qp.py
  
  # Terminal 2: Start MuJoCo sim
  python3 mujoco_sim.py

注意: 该 demo 假设仓库中已有 `placo`、`ischedule` 及示例项目依赖，且路径与已有示例一致。
"""
import pinocchio
import placo
import numpy as np
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, line_viz, point_viz, frame_viz, robot_frame_viz
from placo_utils.tf import tf
import pandas as pd
import time
import math
import sys
import termios
import tty
import select
import atexit

import csv

import sys
import os
import numpy as np
import time

# Import shared memory data structures
from shared_sim_data import SimToCPGData, CPGToSimData, cleanup_shared_memory

def quat_to_euler(quat):
    """
    四元数转换为欧拉角 (roll, pitch, yaw)
    Args:
        quat: 四元数 (x, y, z, w)
    Returns:
        roll, pitch, yaw: 欧拉角 (弧度)
    """
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    euler = (roll_x, pitch_y, yaw_z)

    return euler

try:
    from ischedule  import schedule, run_loop
except Exception:
    # 如果没有 ischedule，提供一个非常小的替代实现（仅用于脚本直接运行）
    def schedule(interval=0.01):
        def deco(f):
            return f
        return deco
    def run_loop():
        pass

try:
    # placo 相关可视化/机器人接口（和 quadruped_targets-CoM-tra.py 保持一致）
    import placo
    from placo_utils.visualization import robot_viz, point_viz
except Exception as e:
    print("警告: 未检测到 placo/placo_utils，可视化可能不可用:", e)
    placo = None
    robot_viz = None
    point_viz = None

# 将工作区内的本地 src 路径加入 sys.path，以便优先导入工作区代码而不是已安装的包
# 之前的实现计算两级父目录导致添加了 `/home/src`，无法找到本地仓库的 src。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 优先插入本地 cpg_go1_simulation 的 src（常见仓库布局）
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'cpg_go1_simulation', 'src'))
# 作为后备，也尝试插入仓库顶层的 src 目录（若存在）
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'src'))

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

v_vector = [1,0,0]
face_vector = [1,0,0]

def main():
    # Initialize shared memory (CPG is consumer of sim data, producer of commands)
    # Wait for MuJoCo to create both shared memory segments first
    print("CPG Planner: Waiting for MuJoCo to start...", flush=True)
    sim_to_cpg = None
    cpg_to_sim = None
    
    for attempt in range(30):  # Wait up to 3 seconds
        try:
            sim_to_cpg = SimToCPGData(create=False)
            cpg_to_sim = CPGToSimData(create=False)
            break
        except FileNotFoundError:
            time.sleep(0.1)
    
    if sim_to_cpg is None or cpg_to_sim is None:
        print("ERROR: MuJoCo sim not found. Start mujoco_sim.py first!", flush=True)
        return
    
    # # Register cleanup on exit
    # atexit.register(lambda: sim_to_cpg.close())
    # atexit.register(lambda: cpg_to_sim.close())
    
    print("CPG Planner: Shared memory connected", flush=True)
    
    # CPG 配置
    total_time = 5.0
    toc = 2.5
    dt = 0.01  # 100Hz for CPG planner
    step_length = 0.8
    body_height = 0.2
    # 创建 walk 类型的 FootTrajectoryCPG，开启一个小的 break_time
    cpg = FootTrajectoryCPG(
        before_ftype=1,
        after_ftype=1,
        total_time=total_time,
        toc=toc,
        break_time=0.1,
        step_height=0.2,
        step_length=step_length / 4,
        body_height=body_height,
        foot_spacing=0.3
    )

    # 创建机器人与求解器（尽量与 quadruped_targets-CoM-tra.py 保持一致）
    if placo is None:
        print("placo 不可用，退出 demo")
        return

    robot = placo.RobotWrapper("/home/placo_cpg/spider_sldasm/urdf", placo.Flags.ignore_collisions)
    robot.set_velocity_limits(1.)
    solver = placo.KinematicsSolver(robot)

    # 创建任务：四个足端和机身
    # 假设 leg 名称为 leg1..leg4，与 quadruped_targets 示例一致
    leg_foot_name_map = {
        "LH" : "Link3-6",
        "RH" : "Link4-6",
        "RF" : "Link1-6",
        "LF" : "Link2-6",
    }
    leg_tasks = {
        "LH": solver.add_position_task("Link3-6", np.array([-0.378168, 0.371678, 0.0])),
        "LF": solver.add_position_task("Link2-6", np.array([0.372385, 0.371678, 0.0])),
        "RF": solver.add_position_task("Link1-6", np.array([0.371678, -0.372385, 0.0])),
        "RH": solver.add_position_task("Link4-6", np.array([-0.361251, -0.360544, 0.0])),
    }
    leg_swim_tasks = {
        "LH": solver.add_relative_position_task("base_link","Link3-6", np.array([-0.378168, 0.371678, -body_height])),
        "LF": solver.add_relative_position_task("base_link","Link2-6", np.array([0.372385, 0.371678, -body_height])),
        "RF": solver.add_relative_position_task("base_link","Link1-6", np.array([0.371678, -0.372385, -body_height])),
        "RH": solver.add_relative_position_task("base_link","Link4-6", np.array([-0.361251, -0.360544, -body_height])),
    }

    # base_link task
    body_task = solver.add_position_task("base_link", np.array([0.0, 0.0, body_height]))
    body_init_task = solver.add_frame_task("base_link", tf.translation_matrix([0.0, 0.0, body_height]))
    body_init_task.configure("base_link", "soft", 1.0, 1e6)
    body_task.configure("base_link"
                    , "soft"                    
                    , 1e6
                    )

    polygon = np.array([
                    np.array([0.372385, 0.371678]),
                    np.array([0.371678, -0.372385]),
                    np.array([-0.361251, -0.360544])
                ])
    com = solver.add_com_polygon_constraint(polygon, 0.5)
    com.polygon = polygon
    com.configure("com_constraint", "soft", 1e3)
    # 可视化器
    viz = robot_viz(robot) if robot_viz is not None else None
    # --- set up MuJoCo simulation helper (mujoco_sim.py) ---
    use_mujoco = False
    sim_helper = None

    # 运行仿真循环：将 CPG 输出的足端位置作为任务目标
    t = 0.0
    cpg_dt = dt
    duration = total_time

    # 假设 CPG 输出的坐标系是机体坐标系，直接作为 leg target 的偏移
    # 映射说明：CPG 的 ['LF','RF','LH','RH'] 映射到上面 leg_tasks 的键
    # 这里我们选择 LF->leg1, LH->Link2-6, RF->Link3-6, RH->leg4（与上面创建一致）
    foot_pos = {
            "LF": np.array([0.0, 0.0, 0.0]),
            "LH": np.array([0.0, 0.0, 0.0]),
            "RF": np.array([0.0, 0.0, 0.0]),
            "RH": np.array([0.0, 0.0, 0.0]),
        }
    foot_swim_pos = {
            "LF": np.array([0.0, 0.0, 0.0]),
            "LH": np.array([0.0, 0.0, 0.0]),
            "RF": np.array([0.0, 0.0, 0.0]),
            "RH": np.array([0.0, 0.0, 0.0]),
        }

    q_init = np.array([0, 0., body_height, 0., 0., 0., 1.,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0])
    # while(time.time() % 1 < 0.5):
    #     sim_helper.step_target(q_init, kp=1.25, kd=0.0025 * 1.25, steps=1)

    robot.set_velocity_limits(1.)

    @schedule(interval=dt)
    def loop():
        nonlocal t
        t += 1.0*dt
        
        # Read current simulation state from shared memory
        try:
            sim_qpos, sim_ctrl, sim_timestamp = sim_to_cpg.read()
            # Update robot state with simulation data (for better consistency)
            # robot.state.q = sim_qpos  # Optionally use sim state
        except Exception as e:
            print(f"Warning: Failed to read sim state: {e}", flush=True)
            sim_qpos = robot.state.q
            sim_ctrl = np.zeros(12)
        
        swim_foot = cpg.get_all_foot_phases(t)
        # 更新 base_link 期望（简单前进或保持不动）
        body_pos = np.array([0.0 + step_length * (t / duration), 0.0, body_height])
        if swim_foot["LF"]["is_stance"] == True and swim_foot["RF"]["is_stance"] == True and swim_foot["LH"]["is_stance"] == True and swim_foot["RH"]["is_stance"] == True:
            body_task.target_world = body_pos
        # print(" body_task.target_world", body_task.target_world)
        # print(f"{swim_foot['LF']['is_stance']}, {swim_foot['RF']['is_stance']}, {swim_foot['LH']['is_stance']}, {swim_foot['RH']['is_stance']}")

        body_csv_path = '/home/placo_cpg/debug/body_data.csv'
        # 初始化（仅第一次循环写入表头）
        if not hasattr(loop, "body_csv_initialized"):
            os.makedirs(os.path.dirname(body_csv_path), exist_ok=True)
            with open(body_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t', 'body_x', 'body_y', 'body_z'])
            loop.body_csv_initialized = True
        body_arr = np.asarray(body_pos).flatten()
        row = [float(t), float(body_arr[0]), float(body_arr[1]), float(body_arr[2])]
        # 追加到 CSV
        with open(body_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 为每个足端取 CPG 输出并设置为对应 leg 的目标
        support_polygon =  []
        
        # print(f'time: {t:.2f}, foot phases: {swim_foot["LF"]["readyswim"]}, {swim_foot["RF"]["readyswim"]}, {swim_foot["LH"]["readyswim"]}, {swim_foot["RH"]["readyswim"]}')
        for foot in cpg.foot_names:
            # CPG 生成的足端位置（相对于机体中心）


            # 将机体位置加到足端偏移，得到世界坐标目标（简化假设）
            foot_pos[foot] = robot.get_T_world_frame(leg_foot_name_map[foot])[:3, 3]
            # foot_pos[2] = 0.0
            foot_swim_pos[foot] = cpg.generate_foot_position(foot, t) + np.array([0.0, 0.0, 0.35 - body_height])

            
            # 设置对应的 leg 任务目标
            if foot in leg_tasks:
                if swim_foot[foot]['is_stance'] == False:
                    leg_swim_tasks[foot].target = foot_swim_pos[foot]
                    leg_swim_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 1e5)
                    leg_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 0)
                else:
                    leg_tasks[foot].target_world = foot_pos[foot]
                    leg_tasks[foot].configure(leg_foot_name_map[foot]
                        #    , "soft"
                        #    , 1e6
                           ,"hard"
                           )
                    leg_swim_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 0)

                if swim_foot[foot]['readyswim'] == True:
                    support_polygon.append(foot_pos[foot][:2])

                # 可视化目标点
                if point_viz is not None:
                    point_viz(f"target_{foot}", foot_pos[foot], color=0x00FF00)
                    
        com.polygon = support_polygon
        com_world = robot.com_world()
        com_world[2] = 0.0
        point_viz("com", com_world, color=0xFF0000)
        # print(f'LF : {foot_pos["LF"]}')
        for k in range(support_polygon.__len__()):
            line_from = support_polygon[k]
            line_to = support_polygon[(k + 1) % support_polygon.__len__()]
            line_viz(f"support_{k}", np.array([line_from, line_to]), color=0xFFAA00)

        # print(support_polygon.__len__())
        # 求解并更新机器人状态
        solver.solve(True)
        robot.update_kinematics()

        q = robot.state.q
        qd = robot.state.qd
        qdd = robot.state.qdd
#         q = [0. ,  0. ,  0.19287 , 1.,  0. ,   0.  ,   0.  ,    0.   ,   0.,
#  0.  ,    0.   ,   0.   ,   0.   ,   0.   ,   0.     , 0.   ,   0.  ,    0.,
#  0.     ]

        # Send desired state and PD gains to MuJoCo via shared memory
        # Define PD gains (tune these for your robot)
        KP_HIP = 100.
        KP_ANKLE = 0.5 * KP_HIP
        KP_KNEE = 0.8 * KP_HIP
        KD_HIP = 0.1 * math.sqrt(KP_HIP)
        KD_ANKLE = 0.1 * KP_ANKLE
        KD_KNEE = 0.1 * KP_KNEE
        kp_gains = [KP_HIP, KP_ANKLE, KP_KNEE
                    ,KP_HIP, KP_ANKLE, KP_KNEE
                    ,KP_HIP, KP_ANKLE, KP_KNEE
                    ,KP_HIP, KP_ANKLE, KP_KNEE]  # Proportional gain for each actuator
        kd_gains = [KD_HIP, KD_ANKLE, KD_KNEE
                    ,KD_HIP, KD_ANKLE, KD_KNEE
                    ,KD_HIP, KD_ANKLE, KD_KNEE
                    ,KD_HIP, KD_ANKLE, KD_KNEE]   # Derivative gain for each actuator

        try:
            cpg_to_sim.write(q, kp_gains, kd_gains)
        except Exception as e:
            print(f"Warning: Failed to write to shared memory: {e}", flush=True)
        
        # Print status every 1 second (100 iterations at 100Hz)
        # if int(t / dt) % 100 == 0:
        #     print(f'CPG t={t:.2f}s, body_pos=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}], sim_z={sim_qpos[2]:.3f}', flush=True)
        
        # Optional: Log to CSV
        robot_csv_path = '/home/placo_cpg/debug/robot_data.csv'

        # 初始化（仅第一次循环写入表头）
        if not hasattr(loop, "robot_csv_initialized"):
            os.makedirs(os.path.dirname(robot_csv_path), exist_ok=True)
            with open(robot_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t', 'q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18'])
            loop.robot_csv_initialized = True

        # 准备要写入的行（确保 q 至少有 19 个元素）
        q_arr = np.asarray(q).flatten()
        if q_arr.size < 19:
            q_arr = np.pad(q_arr, (0, 19 - q_arr.size), 'constant', constant_values=0.0)
        row = [float(t), float(q_arr[0]), float(q_arr[1]), float(q_arr[2]) , float(q_arr[3]), float(q_arr[4]), float(q_arr[5]), float(q_arr[6]),
               float(q_arr[7]), float(q_arr[8]), float(q_arr[9]), float(q_arr[10]), float(q_arr[11]), float(q_arr[12]), float(q_arr[13]),
               float(q_arr[14]), float(q_arr[15]), float(q_arr[16]), float(q_arr[17]), float(q_arr[18])]

        # 追加到 CSV
        with open(robot_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        error_csv_path = '/home/placo_cpg/debug/error_data.csv'
        # 初始化（仅第一次循环写入表头）
        if not hasattr(loop, "error_csv_initialized"):
            os.makedirs(os.path.dirname(error_csv_path), exist_ok=True)
            with open(error_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['t', 'body_x_error', 'body_y_error', 'body_z_error'])
            loop.error_csv_initialized = True
        error = body_pos - q[:3]
        error_arr = np.asarray(error).flatten()
        row = [float(t), float(error_arr[0]), float(error_arr[1]), float(error_arr[2])]
        # 追加到 CSV
        with open(error_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        #     # 采样 IMU 并将数据传入 CPG（如果可用）
            try:
                imu = None
                
                imu_quat = sim_qpos[3:7]
                # print(f'IMU data: {imu_quat}', flush=True)
                imu_euler = None
                if imu_quat is not None:
                    imu_euler = quat_to_euler(imu_quat)
                # print(f'IMU Euler angles: {imu_euler}', flush=True)
                # print(f'IMU data: {imu}', flush=True)
                if imu_euler is not None and hasattr(cpg, 'set_imu_feedback'):
                    # FootTrajectoryCPG.set_imu_feedback(pitch, roll, accel=None, alpha=None)
                    try:
                        None
                        # cpg.set_imu_feedback(float(imu_euler[1]), float(imu_euler[0]))
                    except Exception:
                        print('Warning: CPG set_imu_feedback failed', flush=True)
                        # 若调用失败，不影响仿真循环
                        pass
                    try:
                        None
                        # robot.state.q[3] = float(imu_quat[0])
                        # robot.state.q[4] = float(imu_quat[1])
                        # robot.state.q[5] = float(imu_quat[2])
                        # robot.state.q[6] = float(imu_quat[3])
                    except Exception:
                        print('Warning: updating robot.state.q orientation from IMU failed', flush=True)
                        pass

            except Exception:
                # 保护性捕获，避免 IMU 引发循环中断
                pass

        viz.display(robot.state.q)

        # else:
        #     if viz is not None:
        #         viz.display(robot.state.q)

        # # 结束条件
        # if t >= duration:
        #     # 停止调度循环
        #     # ischedule 的 run_loop 会结束于程序退出，这里仅做提示
        #     print("Demo 运行完毕。")

    # 运行调度循环（如果存在 ischedule 的 run_loop）
    try:
        run_loop()
    except KeyboardInterrupt:
        None
        return
    except Exception:
        # 如果 ischedule 不可用，手动运行简化循环
        print("使用降级循环运行 demo（没有 ischedule）", flush=True)
        steps = int(duration / dt)
        for _ in range(steps):
            loop()


if __name__ == '__main__':
    main()
