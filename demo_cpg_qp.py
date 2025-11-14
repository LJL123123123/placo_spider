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
    
    # 配置足端任务优先级
    leg_tasks["RF"].configure("leg3"
                           , "soft"
                           , 0)
    leg_tasks["LF"].configure("leg4"
                           , "soft"
                           , 0)
    leg_tasks["LH"].configure("leg1"
                           , "soft"
                           , 0)
    leg_tasks["RH"].configure("leg2"
                           , "soft"
                           , 0)

    # 配置足端任务优先级
    leg_swim_tasks["RF"].configure("leg3"
                           , "soft"
                           , 1e1)
    leg_swim_tasks["LF"].configure("leg4"
                           , "soft"
                           , 1e1)
    leg_swim_tasks["LH"].configure("leg1"
                           , "soft"
                           , 1e1)
    leg_swim_tasks["RH"].configure("leg2"
                           , "soft"
                           , 1e1)

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
            # foot_pos = cpg.generate_foot_position(foot, t) + np.array([0.0, 0.0, 0.50])

            # 将机体位置加到足端偏移，得到世界坐标目标（简化假设）
            foot_pos[foot] = cpg.generate_foot_position(foot, t) + np.array([0.0, 0.0, 0.35])
            foot_swim_pos[foot] = cpg.generate_foot_position(foot, t) + np.array([0.0, 0.0, 0.35 - body_height])
            # if foot == "LF":
            #     print(f"t={t:.2f} LF foot pos: {foot_pos[foot]} foot_swim_pos: {foot_swim_pos[foot]}")
            

            # 设置对应的 leg 任务目标
            if foot in leg_tasks:
                # if foot_pos[foot][2] > 0.50:
                #     leg_swim_task[foot].target_world = foot_pos[foot]
                #     leg_tasks[foot].configure("leg3"
                #            , "soft"
                #            , 1)
                leg_swim_tasks[foot].target = foot_swim_pos[foot]
                # if foot == "LF":
                #     # leg_swim_tasks[foot].configure(leg_foot_name_map[foot]
                #     #        , "soft"
                #     #        , 1)
                #     print(f"LF foot pos: {foot_swim_pos[foot]} leg_swim_tasks[foot].target_world: {leg_swim_tasks[foot].target_world}")
                # else:
                # leg_swim_tasks[foot].configure(leg_foot_name_map[foot]
                #         , "soft"
                #         , 0)
                # 可视化目标点
                if point_viz is not None:
                    point_viz(f"target_{foot}", foot_pos[foot], color=0x00FF00)

        # 求解并更新机器人状态
        solver.solve(True)
        robot.update_kinematics()

        # 可视化机器人
        if viz is not None:
            viz.display(robot.state.q)

        # 结束条件
        if t >= duration:
            # 停止调度循环
            # ischedule 的 run_loop 会结束于程序退出，这里仅做提示
            print("Demo 运行完毕。")

    # 运行调度循环（如果存在 ischedule 的 run_loop）
    try:
        run_loop()
    except Exception:
        # 如果 ischedule 不可用，手动运行简化循环
        print("使用降级循环运行 demo（没有 ischedule）")
        steps = int(duration / dt)
        for _ in range(steps):
            loop()


if __name__ == '__main__':
    main()
