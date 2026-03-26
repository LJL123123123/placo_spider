"""run_spider.py

对应 run_wbc.py：提供一个可直接运行的主程序。

- 使用键盘输入生成 cmd_vxyz / yaw_rate
- 调用 SpiderIK.step() 完成规划+QP+通信+logger+可视化

运行：
    python3 run_spider.py

注意：
- 需要先启动 mujoco_sim.py（它会创建 shared memory）。
"""

from __future__ import annotations

import time
import sys
import select
import termios
import tty
import atexit
import numpy as np

from spider_ik import SpiderIK, SpiderIkConfig,SpiderIkData
from typing import Dict, Optional, Tuple, List
from shared_sim_data import SimToCPGData, CPGToSimData
from dataclasses import dataclass, field
import argparse

# Constants
QPOS_SIZE = 31  # 7 (free joint) + 12 (actuated joints) + 12 (fixed joints) = 31, but we only use 19 for qpos (free + actuated)
CTRL_SIZE = 24  # 12 actuators + 12 (fixed joints) 
KP_SIZE = 24 # 12 actuators + 12 (fixed joints)
KD_SIZE = 24 # 12 actuators + 12 (fixed joints)

# Joint gains: 4 legs × 6 joints per leg
# First 3 joints per leg: hip_yaw, hip_pitch, knee (kp=600, kd=6)
# Last 3 joints per leg: ankle_yaw, ankle_roll, wheel (kp=200, kd=2)
JOINT_KP = np.array([
    600, 600, 600, 200, 200, 200,  # Leg 0
    600, 600, 600, 200, 200, 200,  # Leg 1
    600, 600, 600, 200, 200, 200,  # Leg 2
    600, 600, 600, 200, 200, 200,  # Leg 3
], dtype=np.float64)

JOINT_KD = np.array([
    6, 6, 6, 2, 2, 2,  # Leg 0
    6, 6, 6, 2, 2, 2,  # Leg 1
    6, 6, 6, 2, 2, 2,  # Leg 2
    6, 6, 6, 2, 2, 2,  # Leg 3
], dtype=np.float64)
@dataclass
class SHM2DDSData:
    q: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    qdd: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(CTRL_SIZE, dtype=np.float64))
    kp: np.ndarray = field(default_factory=lambda: np.zeros(KP_SIZE, dtype=np.float64))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(KD_SIZE, dtype=np.float64))

def SpiderIkData2SHM2DDSData(data: SpiderIkData) -> SHM2DDSData:
    # SpiderIkData:
    #   q/qd/qdd: [base(7), leg(12)]
    # SHM2DDSData:
    #   [leg0(6), leg1(6), leg2(6), leg3(6)]
    # 其中每条腿先映射 [hip_yaw, hip_pitch, knee]，其余 [ankle_yaw, ankle_roll, wheel] 置零。

    q_src = np.asarray(data.q, dtype=np.float64).reshape(-1)
    qd_src = np.asarray(data.qd, dtype=np.float64).reshape(-1)
    qdd_src = np.asarray(data.qdd, dtype=np.float64).reshape(-1)
    ctrl_src = np.asarray(data.ctrl, dtype=np.float64).reshape(-1)

    q_leg12 = q_src[7:19] if q_src.size >= 19 else np.zeros(12, dtype=np.float64)
    qd_leg12 = qd_src[7:19] if qd_src.size >= 19 else np.zeros(12, dtype=np.float64)
    qdd_leg12 = qdd_src[7:19] if qdd_src.size >= 19 else np.zeros(12, dtype=np.float64)

    q_out = np.zeros(CTRL_SIZE, dtype=np.float64)
    qd_out = np.zeros(CTRL_SIZE, dtype=np.float64)
    qdd_out = np.zeros(CTRL_SIZE, dtype=np.float64)
    ctrl_out = np.zeros(CTRL_SIZE, dtype=np.float64)

    # 每条腿: src 3维 -> dst 前3维, dst 后3维保持0
    for leg_i in range(4):
        src_base = leg_i * 3
        dst_base = leg_i * 6
        q_out[dst_base:dst_base + 3] = q_leg12[src_base:src_base + 3]
        qd_out[dst_base:dst_base + 3] = qd_leg12[src_base:src_base + 3]
        qdd_out[dst_base:dst_base + 3] = qdd_leg12[src_base:src_base + 3]

        if ctrl_src.size >= src_base + 3:
            ctrl_out[dst_base:dst_base + 3] = ctrl_src[src_base:src_base + 3]
    q_out[4] = -1.57
    q_out[10] = -1.57
    q_out[16] = 1.57
    q_out[22] = 1.57

    return SHM2DDSData(
        q=q_out,
        qd=qd_out,
        qdd=qdd_out,
        ctrl=ctrl_out,
        kp=np.ones(KP_SIZE, dtype=np.float64),
        kd=np.zeros(KD_SIZE, dtype=np.float64),
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--enable-shm",
        action="store_true",
        help="Enable shared memory communication with MuJoCo.",
    )

    return p.parse_args()

from XboxController import XboxController
try:
    rc = XboxController()
    rc.start()
except Exception as e:
    print("Xbox Controller not found or init failed, fallback to keyboard:", e)
print("Using RC:", type(rc).__name__)
def main():
    args = parse_args()
    data = SHM2DDSData(
        q =  np.zeros(CTRL_SIZE, dtype=np.float64),
        qd = np.zeros(CTRL_SIZE, dtype=np.float64),
        qdd = np.zeros(CTRL_SIZE, dtype=np.float64),
        ctrl = np.zeros(CTRL_SIZE, dtype=np.float64),
        kp = JOINT_KP.copy(),
        kd = JOINT_KD.copy(),
    )
    # --- shared memory ---
    sim_to_cpg: Optional[SimToCPGData] = None
    cpg_to_sim: Optional[CPGToSimData] = None
    if args.enable_shm:
        # attach (MuJoCo creates)
        sim_to_cpg = SimToCPGData(create=True)
        atexit.register(lambda: sim_to_cpg.close())
        atexit.register(lambda: sim_to_cpg.unlink())
        cpg_to_sim = CPGToSimData(create=True)
        atexit.register(lambda: cpg_to_sim.close())
        atexit.register(lambda: cpg_to_sim.unlink())
    cfg = SpiderIkConfig(
        dt=0.001,
        gait_mode='quasi_static',

    )

    spider = SpiderIK(cfg)

    print(
        "SpiderIK running (keyboard)\n"
        "  W/S: vx  A/D: vy  Q/E: yaw  R/F: vz  Ctrl-C to quit\n",
        flush=True,
    )
    speed_forward, speed_lateral, yaw_speed, height_speed = 0.1, 0.1, 0.2, 0.05
    try:
        while True:
            cmd = rc.get_cmd()
            cmd_vxyz = cmd[:3]
            cmd_vxyz[0] *= speed_forward
            cmd_vxyz[1] *= speed_lateral
            cmd_yaw = cmd[2] * yaw_speed
            cmd_vxyz[2] = 0
            ikdata = spider.step(cmd_vxyz, cmd_yaw)
            data = SpiderIkData2SHM2DDSData(ikdata)
            time.sleep(cfg.dt)

            # send to MuJoCo
            if args.enable_shm and cpg_to_sim is not None:
                try:
                    cpg_to_sim.write(qpos_desired=data.q, ctrl_desired=data.ctrl, kp=JOINT_KP, kd=JOINT_KD)
                except Exception:
                    pass
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
