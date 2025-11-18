#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPG + 全身运动学 QP Demo

说明:
- 使用 `FootTrajectoryCPG` 生成四足的期望足端位置 (相对于机体中心)
- 使用 placo 的 `KinematicsSolver` 将这些足端目标作为位置任务，进行全身逆运动学/QP 求解
- 将求解结果通过可视化工具展示出来

使用方法:
  python3 examples/demo_cpg_qp.py

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

import sys
import os
import numpy as np

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

# 将 src 路径加入 sys.path，以便导入本地 CPG 实现
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, 'src'))

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

v_vector = [1,0,0]
face_vector = [1,0,0]

def main():
    # CPG 配置
    total_time = 5.0
    toc = 2.5
    dt = 0.02
    step_length = 0.4
    body_height = 0.05
    # 创建 walk 类型的 FootTrajectoryCPG，开启一个小的 break_time
    cpg = FootTrajectoryCPG(
        before_ftype=1,
        after_ftype=1,
        total_time=total_time,
        toc=toc,
        break_time=0.1,
        step_height=0.05,
        step_length=step_length / 4,
        body_height=body_height,
        foot_spacing=0.1
    )

    # 创建机器人与求解器（尽量与 quadruped_targets-CoM-tra.py 保持一致）
    if placo is None:
        print("placo 不可用，退出 demo")
        return
    else:
        print("placo 可用，创建机器人与求解器")

    robot = placo.RobotWrapper("/home/placo-examples/models/quadruped", placo.Flags.ignore_collisions)
    solver = placo.KinematicsSolver(robot)

    # 创建任务：四个足端和机身
    # 假设 leg 名称为 leg1..leg4，与 quadruped_targets 示例一致
    leg_foot_name_map = {
        "LH" : "leg1",
        "RH" : "leg2",
        "RF" : "leg3",
        "LF" : "leg4",
    }
    leg_tasks = {
        "LH": solver.add_position_task("leg1", np.array([-0.15, 0.0, 0.0])),
        "LF": solver.add_position_task("leg4", np.array([0.02, 0.15, 0.0])),
        "RF": solver.add_position_task("leg3", np.array([0.15, 0.0, 0.0])),
        "RH": solver.add_position_task("leg2", np.array([0.02, -0.15, 0.0])),
    }
    leg_swim_tasks = {
        "LH": solver.add_relative_position_task("body","leg1", np.array([-0.15, 0.0, -body_height])),
        "LF": solver.add_relative_position_task("body","leg4", np.array([0.02, 0.15, -body_height])),
        "RF": solver.add_relative_position_task("body","leg3", np.array([0.15, 0.0, -body_height])),
        "RH": solver.add_relative_position_task("body","leg2", np.array([0.02, -0.15, -body_height])),
    }

    # body task
    body_task = solver.add_position_task("body", np.array([0.0, 0.0, body_height]))
    body_init_task = solver.add_frame_task("body", tf.translation_matrix([0.0, 0.0, body_height]))
    body_init_task.configure("body", "soft", 1.0, 1e6)
    body_task.configure("body"
                    , "soft"                    
                    , 1e6
                    )

    # 可视化器
    viz = robot_viz(robot) if robot_viz is not None else None
    # --- set up MuJoCo simulation helper (mujoco_sim.py) ---
    use_mujoco = False
    sim_helper = None
    
    try:
        print('prepare for mujoco sim helper')
        from mujoco_sim import MuJoCoSim
        print('MuJoCo helper available.')
        sim_helper = MuJoCoSim('/home/placo_cpg/models/quadruped/scene.xml')
        use_mujoco = bool(getattr(sim_helper, 'available', False))
        if not use_mujoco:
            sim_helper = None
            print('MuJoCo helper unavailable, falling back to previous viz (if any).')
        else:
            # print actuator -> joint mapping to help verify motor ordering
            print('Actuator mapping (actuator_index -> (qposadr, dofadr, joint_name)):', flush=True)
            for i, m in enumerate(sim_helper.act_map):
                print(f'  {i}: {m}', flush=True)
            # start viewer explicitly (do not auto-launch on import)
            try:
                started = sim_helper.start_viewer()
                print(f'sim_helper.start_viewer() -> {started}', flush=True)
            except Exception as e:
                print('sim_helper.start_viewer() raised:', e, flush=True)
    except Exception as e:
        print('MuJoCo helper unavailable, falling back to previous viz (if any):', e)

    # 运行仿真循环：将 CPG 输出的足端位置作为任务目标
    t = 0.0
    cpg_dt = dt
    duration = total_time

    # 假设 CPG 输出的坐标系是机体坐标系，直接作为 leg target 的偏移
    # 映射说明：CPG 的 ['LF','RF','LH','RH'] 映射到上面 leg_tasks 的键
    # 这里我们选择 LF->leg1, LH->leg2, RF->leg3, RH->leg4（与上面创建一致）
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

    
    @schedule(interval=dt)
    def loop():
        nonlocal t
        t += dt

        # 更新 body 期望（简单前进或保持不动）
        body_pos = np.array([0.0 + step_length * (t / duration), 0.0, body_height])
        body_task.target_world = body_pos

        # 为每个足端取 CPG 输出并设置为对应 leg 的目标
        for foot in cpg.foot_names:
            # CPG 生成的足端位置（相对于机体中心）


            # 将机体位置加到足端偏移，得到世界坐标目标（简化假设）
            foot_pos[foot] = robot.get_T_world_frame(leg_foot_name_map[foot])[:3, 3]
            foot_pos[2] = 0.0
            foot_swim_pos[foot] = cpg.generate_foot_position(foot, t) + np.array([0.0, 0.0, 0.35 - body_height])
            

            # 设置对应的 leg 任务目标
            if foot in leg_tasks:
                if foot_swim_pos[foot][2] > -0.05:
                    leg_swim_tasks[foot].target = foot_swim_pos[foot]
                    leg_swim_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 1)
                    leg_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 0)
                else:
                    leg_tasks[foot].target_world = foot_pos[foot]
                    leg_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 1)
                    leg_swim_tasks[foot].configure(leg_foot_name_map[foot]
                           , "soft"
                           , 0)

                # 可视化目标点
                if point_viz is not None:
                    point_viz(f"target_{foot}", foot_pos[foot], color=0x00FF00)

        # 求解并更新机器人状态
        solver.solve(True)
        robot.update_kinematics()

        q = robot.state.q
        qd = robot.state.qd
        qdd = robot.state.qdd
        # print( f"q: {q}" )
        # print( f"dq: {qd}" )
        # print( f"ddq: {qdd}" )

        # 可视化机器人: 使用 MuJoCo 仿真替代原有 viz
        if use_mujoco and sim_helper is not None:
            # target state q: try to use robot.state.q, otherwise fallback to example vector
            try:
                q_target = np.asarray(robot.state.q)
                
            except Exception:
                q_target = np.array([0.02879997, 0., 0.05, 0., 0., 0., 1.,
                                     0.70191237, -0.51864883, 1.48555772,
                                     0.97934966, -0.70745988, 2.24408594,
                                     1.06774713, -0.62819484, 1.96654732,
                                     1.25845303, -0.43142328, 1.48984719])
            # Ensure q_target has 7 (base) + 12 (joints) = 19 elements
            if q_target.size < 19:
                qt = np.zeros(19)
                qt[:min(q_target.size, 19)] = q_target[:min(q_target.size, 19)]
                q_target = qt

            # PD gains (tweak if robot is too aggressive)
            KP = 0.1
            KD = 0.01

            # compute number of physics steps per control step based on model timestep
            try:
                if getattr(sim_helper, '_use_new', False):
                    timestep = float(sim_helper.model.opt.timestep)
                else:
                    timestep = float(sim_helper.sim.model.opt.timestep)
            except Exception:
                timestep = 0.001
            steps = max(1, int(round(dt / timestep)))

            # print debug info immediately (avoid buffering when viewer/server runs)
            print(f'q_target size={q_target.size}, timestep={timestep}, steps per control={steps}', flush=True)
            sim_helper.step_target(q_target, kp=KP, kd=KD, steps=steps)
        else:
            if viz is not None:
                viz.display(robot.state.q)

        # # 结束条件
        # if t >= duration:
        #     # 停止调度循环
        #     # ischedule 的 run_loop 会结束于程序退出，这里仅做提示
        #     print("Demo 运行完毕。")

    # 运行调度循环（如果存在 ischedule 的 run_loop）
    try:
        run_loop()
    except KeyboardInterrupt:
        # allow graceful shutdown of viewer/server started by mujoco
        print("Simulation interrupted by user", flush=True)
        try:
            if sim_helper is not None and getattr(sim_helper, 'viewer', None) is not None:
                try:
                    sim_helper.viewer.close()
                except Exception:
                    try:
                        sim_helper.viewer.finish()
                    except Exception:
                        pass
        except Exception:
            pass
        return
    except Exception:
        # 如果 ischedule 不可用，手动运行简化循环
        print("使用降级循环运行 demo（没有 ischedule）", flush=True)
        steps = int(duration / dt)
        for _ in range(steps):
            loop()


if __name__ == '__main__':
    main()
