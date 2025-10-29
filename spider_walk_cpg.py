#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spider_walk_cpg.py

将原始的 `spider_walk2.py` 与 `FootTrajectoryCPG` 融合：
- 使用 `FootTrajectoryCPG(..., break_time=0.1)` 为每个摆动腿生成足端轨迹
- 在主循环中根据 gait 切换摆动腿，并将 CPG 产生的足端位置映射到世界坐标后下发给位置任务

说明/假设:
- 将 CPG 的 foot order ["LF","RF","LH","RH"] 映射到机器人上的链接顺序
  Link1 -> LF, Link2 -> RF, Link3 -> LH, Link4 -> RH (与原 `spider_walk2.py` 的 leg1..leg4 一致)
- 这里把 CPG 输出视为相对于机体中心的坐标 (与 `FootTrajectoryCPG.foot_base_positions` 一致)，
  所以将其变换到世界坐标时用机体变换 + 基坐标偏移。
"""

import time
import numpy as np
import pandas as pd
import pinocchio
import placo
from ischedule import schedule, run_loop
from placo_utils.visualization import robot_viz, line_viz, point_viz, frame_viz, robot_frame_viz

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG


# ---- robot / solver 初始化 (基于 spider_walk2.py) ----
robot = placo.RobotWrapper("/home/placo_cpg/spider_sldasm/urdf", placo.Flags.ignore_collisions)

for leg in ["Joint1", "Joint2", "Joint3", "Joint4"]:
    robot.set_joint_limits(leg + "-1", -np.pi/2, np.pi/2)
    robot.set_joint_limits(leg + "-3", 0.0, np.pi)

solver = placo.KinematicsSolver(robot)

# 一些全局常量
high = np.array([0.0, 0.0, 0.15])  # 机体偏高

# 创建 CPG —— 使用 break_time=0.1
cpg = FootTrajectoryCPG(
    before_ftype=1,
    after_ftype=1,
    total_time=5.0,
    toc=6.0,
    break_time=0.1,  # 要求：0.1s 的 break
)

# 把 CPG 的脚名映射到 robot 的 Link 名
# 假设: LF->Link1-6, RF->Link2-6, LH->Link3-6, RH->Link4-6
CPG_TO_LINK = {
    "LF": "Link1-6",
    "RF": "Link2-6",
    "LH": "Link3-6",
    "RH": "Link4-6",
}

# 读取初始世界变换
T_world_body = robot.get_T_world_frame("base_link")

# 位置任务：把四条腿初始化到当前末端位姿的上方一点（加上 high）
T_world_leg1 = robot.get_T_world_frame("Link1-6")
T_world_leg2 = robot.get_T_world_frame("Link2-6")
T_world_leg3 = robot.get_T_world_frame("Link3-6")
T_world_leg4 = robot.get_T_world_frame("Link4-6")

position_leg1 = T_world_leg1[:3, 3] + high
position_leg2 = T_world_leg2[:3, 3] + high
position_leg3 = T_world_leg3[:3, 3] + high
position_leg4 = T_world_leg4[:3, 3] + high

leg1 = solver.add_position_task("Link1-6", position_leg1)
leg2 = solver.add_position_task("Link2-6", position_leg2)
leg3 = solver.add_position_task("Link3-6", position_leg3)
leg4 = solver.add_position_task("Link4-6", position_leg4)

leg1.configure("Link1-6", "soft", 1.0)
leg2.configure("Link2-6", "soft", 1.0)
leg3.configure("Link3-6", "soft", 1.0)
leg4.configure("Link4-6", "soft", 1.0)

# 一些初始化求解步
for _ in range(16):
    robot.update_kinematics()
    solver.solve(True)

robot.set_velocity_limits(1.0)
solver.enable_velocity_limits(True)
viz = robot_viz(robot)


# Helper: 根据 leg index (1..4) 找到对应的 task 对象
LEG_TASKS = {1: leg1, 2: leg2, 3: leg3, 4: leg4}
LEG_LINKS = {1: "Link1-6", 2: "Link2-6", 3: "Link3-6", 4: "Link4-6"}
CPG_ORDER = ["LF", "RF", "LH", "RH"]


# gait timing 控制（复用 spider_walk2 的简单时序）
flying_leg = 1

def gait(t):
    global flying_leg
    if t % 12 < 3:
        flying_leg = 1
    elif t % 12 < 6:
        flying_leg = 2
    elif t % 12 < 9:
        flying_leg = 3
    else:
        flying_leg = 4


# 主循环参数
t = 0.0
dt = 0.01
solver.dt = dt


@schedule(interval=dt)
def loop():
    global t, flying_leg
    t += dt
    gait(t)

    # 更新机体及四足世界坐标
    T_world_body = robot.get_T_world_frame("base_link")
    R_world_body = T_world_body[:3, :3]
    p_world_body = T_world_body[:3, 3]

    # 对每条腿：如果为摆动腿 -> 从 CPG 获取目标（相对于机体），然后变换到世界
    for idx, cpg_name in enumerate(CPG_ORDER, start=1):
        task = LEG_TASKS[idx]
        link = LEG_LINKS[idx]

        if idx == flying_leg:
            # 使用 CPG 输出作为摆动目标
            foot_pos_body = cpg.generate_foot_position(cpg_name, t)

            # 将 CPG 输出（相对于机体）转换到世界坐标
            foot_pos_world = R_world_body.dot(foot_pos_body) + p_world_body

            # 可视化目标点
            point_viz(f"cpg_target_{idx}", foot_pos_world, color=0x00FF00)

            # 设置为该足位置任务的目标（world）
            try:
                task.target_world = foot_pos_world
            except Exception:
                # 有些 task 实现可能要求使用不同属性名，退回到 configure + set
                task.configure(link, "soft", 1.0)
                task.target_world = foot_pos_world

            # 让摆动腿软约束，优先级较低
            task.configure(link, "soft", 0.0)
        else:
            # 支撑腿：保持当前位置为目标，加入一定刚度
            T_leg_world = robot.get_T_world_frame(link)
            leg_world_pos = T_leg_world[:3, 3]
            task.target_world = leg_world_pos
            task.configure(link, "soft", 1e6)

            # 可视化支撑边
            point_viz(f"support_{idx}", leg_world_pos, color=0xFFAA00)

    # 显示机体与COM
    robot_frame_viz(robot, "base_link")
    com_world = robot.com_world()
    com_world_vis = com_world.copy(); com_world_vis[2] = 0.0
    point_viz("com", com_world_vis, color=0xFF0000)

    solver.solve(True)
    robot.update_kinematics()
    viz.display(robot.state.q)


def main():
    print("Starting spider_walk_cpg with FootTrajectoryCPG (break_time=0.1)")
    run_loop()


if __name__ == '__main__':
    main()
