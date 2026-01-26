#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""spider_gait_manager_qp.py

用 NumPy CPU 版本 gait manager（`gait_manager.py`）替代 CPG，生成 COM+四足目标，
并通过 placo QP 求解关节角，再通过共享内存发送给 MuJoCo。

运行方式（两个终端）：
- Terminal 1: python3 spider_gait_manager_qp.py
- Terminal 2: python3 mujoco_sim.py

说明：
- 这个脚本尽量复用 `spider_cpg_qp.py` 的结构（shared memory + placo solver）。
- 速度指令这里先给一个简单的“示例指令发生器”（可改成键盘/上层控制）。
"""

import time
import math
import os
import sys
from typing import Dict
import select
import termios
import tty

import numpy as np

# Import shared memory data structures
from shared_sim_data import SimToCPGData, CPGToSimData

try:
    import placo
    from placo_utils.visualization import robot_viz, point_viz
    from placo_utils.tf import tf
except Exception as e:
    placo = None
    robot_viz = None
    point_viz = None
    tf = None
    print("警告: 未检测到 placo/placo_utils，可视化与QP可能不可用:", e)

from gait_manager import GaitCycleManager, GaitParams


ALL_LEGS = ['LF', 'RF', 'LH', 'RH']


class _KeyboardController:
    """非阻塞键盘控制器（Linux/TTY）。

    键位（按住会重复触发，属于“速度增量”式控制）：
    - W/S: vx +/-(前/后)
    - A/D: vy +/-（左/右）
    - Q/E: yaw_rate +/-（左/右转）
    - R/F: vz +/-（升/降）
    - Space: 急停（vx/vy/vz/yaw_rate 置 0）
    - X: 退出

    说明：
    - 指令会带一个衰减（松手后会缓慢回到 0），避免“卡住”。
    """

    def __init__(
        self,
        vx_step: float = 0.03,
        vy_step: float = 0.03,
        vz_step: float = 0.02,
        yaw_step: float = 0.15,
        decay: float = 0.95,
        vxy_max: float = 0.4,
        vz_max: float = 0.2,
        yaw_max: float = 1.5,
    ):
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw = 0.0

        self.vx_step = float(vx_step)
        self.vy_step = float(vy_step)
        self.vz_step = float(vz_step)
        self.yaw_step = float(yaw_step)
        self.decay = float(decay)

        self.vxy_max = float(vxy_max)
        self.vz_max = float(vz_max)
        self.yaw_max = float(yaw_max)

        self._fd = sys.stdin.fileno()
        self._old_term = None

    def __enter__(self):
        # 保存并切到 raw 模式
        try:
            self._old_term = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self._old_term = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        if self._old_term is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
            except Exception:
                pass
            self._old_term = None

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _apply_decay(self):
        self.vx *= self.decay
        self.vy *= self.decay
        self.vz *= self.decay
        self.yaw *= self.decay

        # 很小的数直接归零，避免漂
        for k in ("vx", "vy", "vz", "yaw"):
            if abs(getattr(self, k)) < 1e-4:
                setattr(self, k, 0.0)

    def poll(self):
        """读取当前所有已输入字符（非阻塞），更新内部速度，返回是否需要退出。"""
        should_exit = False
        # 每个周期先做衰减
        self._apply_decay()

        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                c = ch.lower()

                if c == 'w':
                    self.vx += self.vx_step
                elif c == 's':
                    self.vx -= self.vx_step
                elif c == 'a':
                    self.vy += self.vy_step
                elif c == 'd':
                    self.vy -= self.vy_step
                elif c == 'q':
                    self.yaw += self.yaw_step
                elif c == 'e':
                    self.yaw -= self.yaw_step
                elif c == 'r':
                    self.vz += self.vz_step
                elif c == 'f':
                    self.vz -= self.vz_step
                elif c == ' ':  # space
                    self.vx = self.vy = self.vz = self.yaw = 0.0
                elif c == 'x':
                    should_exit = True

        except Exception:
            # stdin 不可用时不让程序崩
            pass

        # 限幅
        self.vx = self._clip(self.vx, -self.vxy_max, self.vxy_max)
        self.vy = self._clip(self.vy, -self.vxy_max, self.vxy_max)
        self.vz = self._clip(self.vz, -self.vz_max, self.vz_max)
        self.yaw = self._clip(self.yaw, -self.yaw_max, self.yaw_max)

        return should_exit

    def get_command(self):
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64), float(self.yaw)


def _Rz(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _world_from_body(com_pos: np.ndarray, com_R: np.ndarray, p_body: np.ndarray) -> np.ndarray:
    return com_pos + com_R @ p_body


def build_initial_targets(body_height: float = 0.26) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
    """构造一个简单的站立初始目标（世界系下com与四足位置）。"""
    com = np.array([0.0, 0.0, body_height], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    # 这里沿用 spider_cpg_qp.py 里的大致站立足端布局（世界系）
    feet = {
        'LH': np.array([-0.378168, 0.371678, 0.0], dtype=np.float64),
        'LF': np.array([0.372385, 0.371678, 0.0], dtype=np.float64),
        'RF': np.array([0.371678, -0.372385, 0.0], dtype=np.float64),
        'RH': np.array([-0.361251, -0.360544, 0.0], dtype=np.float64),
    }

    target_pos = {'com': com, **feet}
    target_ori = {'com': R}
    return target_pos, target_ori

def main():
    # --- shared memory handshake ---
    print("GaitManager QP: Waiting for MuJoCo to start...", flush=True)
    # sim_to_cpg = None
    # cpg_to_sim = None
    # for _ in range(30):
    #     try:
    #         sim_to_cpg = SimToCPGData(create=False)
    #         cpg_to_sim = CPGToSimData(create=False)
    #         break
    #     except FileNotFoundError:
    #         time.sleep(0.1)

    # if sim_to_cpg is None or cpg_to_sim is None:
    #     # print("ERROR: MuJoCo sim not found. Start mujoco_sim.py first!", flush=True)
    #     return

    print("GaitManager QP: Shared memory connected", flush=True)

    if placo is None:
        print("placo 不可用，退出 demo")
        return

    # --- robot + qp solver ---
    # 这里沿用 spider_cpg_qp.py 的 URDF 路径风格（注意：你的环境里可能是 placo_cpg_1/2 不同前缀）
    # 先尽量用现有路径：/home/placo_cpg_1/... 如果不存在再尝试 /home/placo_cpg_2/...
    urdf_candidates = [
        "/home/placo_cpg/spider_sldasm/urdf",
    ]
    urdf_root = None
    for p in urdf_candidates:
        if os.path.exists(p):
            urdf_root = p
            break
    if urdf_root is None:
        # 最后兜底：仓库相对路径
        rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unitree_model", "robots", "go1_description", "urdf")
        if os.path.exists(rel):
            urdf_root = rel

    if urdf_root is None:
        print("ERROR: 找不到 go1 urdf 目录，无法创建 RobotWrapper", flush=True)
        return

    robot = placo.RobotWrapper(urdf_root, placo.Flags.ignore_collisions)
    robot.set_velocity_limits(1.0)
    solver = placo.KinematicsSolver(robot)

    leg_foot_name_map = {
        "LH" : "Link3-6",
        "RH" : "Link4-6",
        "RF" : "Link1-6",
        "LF" : "Link2-6",
    }

    # 初始站立目标
    target_pos, target_ori = build_initial_targets(body_height=0.20)
    target_init_pos = target_pos

    # 任务：足端位置（世界系）+ base_link 位置
    leg_tasks = {
        leg: solver.add_position_task(leg_foot_name_map[leg], np.array(target_pos[leg]))
        for leg in ALL_LEGS
    }

    body_pos_task = solver.add_position_task("base_link", np.array(target_pos['com']))
    body_pos_task.configure("base_link", "soft", 1e6)
    body_ori_task = solver.add_orientation_task("base_link", np.array(target_ori['com']))
    body_ori_task.configure("base_link", "soft", 1e6)

    if tf is not None:
        body_init_task = solver.add_frame_task("base_link", tf.translation_matrix(target_pos['com'].tolist()))
        body_init_task.configure("base_link", "soft", 1.0, 1.0)

    viz = robot_viz(robot) if robot_viz is not None else None

    # --- gait manager ---
    params = GaitParams(
        cycle_period=1.0,
        swing_height=0.08,
        lookahead=1.0,
        cmd_epsilon=1e-4,
        cmd_timeout=2.0,
        stand_transition_duration=0.4,
    )
    gm = GaitCycleManager(params=params, dtype=np.float64)
    gm.set_stand_targets(target_pos, target_ori)

    # --- control loop ---
    dt = 0.01
    t = 0.0

    # PD gains（沿用 spider_cpg_qp.py）
    KP_HIP = 100.0
    KP_ANKLE = 0.5 * KP_HIP
    KP_KNEE = 0.8 * KP_HIP
    KD_HIP = 0.1 * math.sqrt(KP_HIP)
    KD_ANKLE = 0.1 * KP_ANKLE
    KD_KNEE = 0.1 * KP_KNEE
    kp_gains = [
        KP_HIP, KP_ANKLE, KP_KNEE,
        KP_HIP, KP_ANKLE, KP_KNEE,
        KP_HIP, KP_ANKLE, KP_KNEE,
        KP_HIP, KP_ANKLE, KP_KNEE,
    ]
    kd_gains = [
        KD_HIP, KD_ANKLE, KD_KNEE,
        KD_HIP, KD_ANKLE, KD_KNEE,
        KD_HIP, KD_ANKLE, KD_KNEE,
        KD_HIP, KD_ANKLE, KD_KNEE,
    ]

    print(
        "GaitManager QP: running (keyboard control enabled)\n"
        "  W/S: vx +/-   A/D: vy +/-   Q/E: yaw +/-   R/F: vz +/-\n"
        "  Space: stop   X: exit\n",
        flush=True,
    )

    with _KeyboardController() as kb:
        last_print = 0.0
        while True:
            t += dt

            if kb.poll():
                print("Exit requested.")
                break

            # 读 sim 状态（当前脚本不强依赖，但保留以便你后续做闭环）
            try:
                sim_qpos, sim_ctrl, sim_ts = sim_to_cpg.read()
            except Exception:
                sim_qpos = None

            cmd_vxyz, cmd_yaw = kb.get_command()

            # 低频打印一下当前指令，便于确认按键生效
            if (t - last_print) > 0.25:
                last_print = t
                # print(
                #     f"cmd: vx={cmd_vxyz[0]:+.3f} vy={cmd_vxyz[1]:+.3f} vz={cmd_vxyz[2]:+.3f} yaw={cmd_yaw:+.3f}",
                #     flush=True,
                # )

            plan = gm.update(
                t=t,
                target_pos=target_pos,
                target_ori=target_ori,
                cmd_vxyz=cmd_vxyz,
                cmd_yaw_rate=cmd_yaw,
                dt=dt,
            )

            target_pos = plan.target_pos
            target_ori = plan.target_ori

            # 设置 QP 目标
            body_pos_task.target_world = target_pos['com']
            # 注意：placo 的 OrientationTask 使用 R_world_frame 作为目标旋转矩阵
            body_ori_task.R_world_frame = np.asarray(target_ori['com'], dtype=float)
            for leg in ALL_LEGS:
                leg_tasks[leg].target_world = target_pos[leg]
                # leg_tasks[leg].target_world = target_ori['com'] @ np.array(target_init_pos[leg], dtype=float)

            solver.solve(True)
            robot.update_kinematics()

            q = robot.state.q
            # 发送到 MuJoCo
            try:
                cpg_to_sim.write(qpos_desired=q, kp=kp_gains, kd=kd_gains)
            except Exception as e:
                # print("Warning: Failed to write to shared memory:", e, flush=True)
                pass

            # 可视化（robot_viz/point_viz 依赖 placo_utils；若不可用则不会弹窗）
            if viz is not None:
                viz.display(robot.state.q)
            if point_viz is not None:
                com_world = robot.com_world()
                com_world[2] = 0.0
                point_viz("com", com_world, color=0xFF0000)

            # 100Hz
            time.sleep(dt)


if __name__ == '__main__':
    main()
