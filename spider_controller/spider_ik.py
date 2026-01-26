"""spider_ik.py

类 WBIK 结构的 Spider IK/QP 运行体（对应你給的 wbc.py）。

组成：
- gait 管理：使用本仓库的 `gait_manager.py::GaitCycleManager`
- QP/IK：使用 placo.KinematicsSolver
- 通信：shared_sim_data (SimToCPGData / CPGToSimData)
- logger：spider_logger.SpiderCsvLogger
- 可视化：spider_visual.SpiderVisualizer

目标：
- 尽量把变量都变成 self.xxx，便于外部打印/调试/读取
- 把原 `spider_gait_manager_qp.py` 的流程拆成可复用类
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
import math
import time
import numpy as np

from shared_sim_data import SimToCPGData, CPGToSimData

from gait_manager import GaitCycleManager, GaitParams
from spider_logger import SpiderCsvLogger
from spider_visual import SpiderVisualizer


ALL_LEGS = ['LF', 'RF', 'LH', 'RH']


def build_initial_targets(body_height: float = 0.20) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    com = np.array([0.0, 0.0, body_height], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    feet = {
        'LH': np.array([-0.378168, 0.371678, 0.0], dtype=np.float64),
        'LF': np.array([0.372385, 0.371678, 0.0], dtype=np.float64),
        'RF': np.array([0.371678, -0.372385, 0.0], dtype=np.float64),
        'RH': np.array([-0.361251, -0.360544, 0.0], dtype=np.float64),
    }

    target_pos = {'com': com, **feet}
    target_ori = {'com': R}
    return target_pos, target_ori


@dataclass
class SpiderIkConfig:
    dt: float = 0.01
    gait_mode: str = 'quasi_static'

    # runtime toggles
    enable_shm: bool = False
    enable_visual: bool = True
    enable_logger: bool = True

    # gait params
    cycle_period: float = 1.0
    swing_height: float = 0.08
    lookahead: float = 1.0
    cmd_epsilon: float = 1e-4
    cmd_timeout: float = 2.0
    stand_transition_duration: float = 0.1

    # com polygon constraint
    polygon_margin: float = 0.1

    # solver weights
    leg_task_weight: float = 1e3
    body_task_weight: float = 1e2

    # URDF root candidates
    urdf_candidates: Tuple[str, ...] = ("/home/placo_cpg/spider_sldasm/urdf",)


class SpiderIK:
    def __init__(self, cfg: SpiderIkConfig):
        self.cfg = cfg
        self.enable_visual = bool(cfg.enable_visual)
        self.enable_shm = bool(cfg.enable_shm)
        self.enable_logger = bool(cfg.enable_logger)

        # --- runtime time ---
        self.t = 0.0

        # --- shared memory ---
        self.sim_to_cpg: Optional[SimToCPGData] = None
        self.cpg_to_sim: Optional[CPGToSimData] = None
        if self.enable_shm:
            # attach (MuJoCo creates)
            self.sim_to_cpg = SimToCPGData(create=False)
            self.cpg_to_sim = CPGToSimData(create=False)

        # --- placo ---
        try:
            import placo
            from placo_utils.tf import tf
        except Exception as e:
            raise RuntimeError(f"placo/placo_utils 不可用: {e}")

        self.placo = placo
        self.tf = tf

        # find urdf root
        self.urdf_root = None
        for p in self.cfg.urdf_candidates:
            if os.path.exists(p):
                self.urdf_root = p
                break
        if self.urdf_root is None:
            raise FileNotFoundError(f"URDF root not found in candidates: {self.cfg.urdf_candidates}")

        self.robot = self.placo.RobotWrapper(self.urdf_root, self.placo.Flags.ignore_collisions)
        self.robot.set_velocity_limits(1.0)
        self.solver = self.placo.KinematicsSolver(self.robot)

        # frames mapping
        self.leg_foot_name_map = {
            'LH': 'Link3-6',
            'RH': 'Link4-6',
            'RF': 'Link1-6',
            'LF': 'Link2-6',
        }

        # --- initial targets ---
        self.target_pos, self.target_ori = build_initial_targets(body_height=0.20)
        self.target_init_pos = {k: np.asarray(v).copy() for k, v in self.target_pos.items()}

        # --- tasks ---
        self.leg_tasks = {
            leg: self.solver.add_position_task(self.leg_foot_name_map[leg], np.array(self.target_pos[leg]))
            for leg in ALL_LEGS
        }
        for leg in ALL_LEGS:
            self.leg_tasks[leg].configure(self.leg_foot_name_map[leg], 'soft', float(self.cfg.leg_task_weight))

        # base tasks
        self.body_pos_task = self.solver.add_position_task('base_link', np.array(self.target_pos['com']))
        self.body_pos_task.configure('base_link', 'soft', float(self.cfg.body_task_weight))

        self.body_ori_task = self.solver.add_orientation_task('base_link', np.array(self.target_ori['com']))
        self.body_ori_task.configure('base_link', 'soft', float(self.cfg.body_task_weight))

        # com polygon constraint (placeholder init polygon, will update each loop)
        init_polygon = np.array([
            np.array([0.372385, 0.371678]),
            np.array([0.371678, -0.372385]),
            np.array([-0.361251, -0.360544]),
        ])
        self.com_constraint = self.solver.add_com_polygon_constraint(init_polygon, float(self.cfg.polygon_margin))
        self.com_constraint.polygon = init_polygon
        self.com_constraint.configure('com_constraint', 'soft', 1.0)

        # --- gait manager ---
        self.gait_params = GaitParams(
            gait_mode=self.cfg.gait_mode,
            cycle_period=float(self.cfg.cycle_period),
            swing_height=float(self.cfg.swing_height),
            lookahead=float(self.cfg.lookahead),
            cmd_epsilon=float(self.cfg.cmd_epsilon),
            cmd_timeout=float(self.cfg.cmd_timeout),
            stand_transition_duration=float(self.cfg.stand_transition_duration),
        )
        self.gait = GaitCycleManager(params=self.gait_params, dtype=np.float64)
        self.gait.set_stand_targets(self.target_pos, self.target_ori)

        # --- PD gains ---
        self.KP_HIP = 100.0
        self.KP_ANKLE = 0.5 * self.KP_HIP
        self.KP_KNEE = 0.8 * self.KP_HIP
        self.KD_HIP = 0.1 * math.sqrt(self.KP_HIP)
        self.KD_ANKLE = 0.1 * self.KP_ANKLE
        self.KD_KNEE = 0.1 * self.KP_KNEE
        self.kp_gains = [
            self.KP_HIP, self.KP_ANKLE, self.KP_KNEE,
            self.KP_HIP, self.KP_ANKLE, self.KP_KNEE,
            self.KP_HIP, self.KP_ANKLE, self.KP_KNEE,
            self.KP_HIP, self.KP_ANKLE, self.KP_KNEE,
        ]
        self.kd_gains = [
            self.KD_HIP, self.KD_ANKLE, self.KD_KNEE,
            self.KD_HIP, self.KD_ANKLE, self.KD_KNEE,
            self.KD_HIP, self.KD_ANKLE, self.KD_KNEE,
            self.KD_HIP, self.KD_ANKLE, self.KD_KNEE,
        ]

        # --- logger & visual ---
        self.logger = SpiderCsvLogger(base_dir='./debug') if self.enable_logger else None
        self.visual = SpiderVisualizer(self.robot, enabled=self.enable_visual)

        # last plan snapshot
        self.last_plan = None

    def step(self, cmd_vxyz: np.ndarray, cmd_yaw_rate: float):
        """One control step: plan -> update tasks -> solve -> send shm -> log."""
        self.t += float(self.cfg.dt)

        cmd_vxyz = np.asarray(cmd_vxyz, dtype=np.float64).reshape(3)
        cmd_yaw_rate = float(cmd_yaw_rate)

        # log command
        if self.logger is not None:
            self.logger.write_row(
                name='cmd',
                filename='cmd_data.csv',
                header=['t', 'vx', 'vy', 'vz', 'yaw_rate'],
                row=[self.t, float(cmd_vxyz[0]), float(cmd_vxyz[1]), float(cmd_vxyz[2]), float(cmd_yaw_rate)],
            )

        # plan
        plan = self.gait.update(
            t=self.t,
            target_pos=self.target_pos,
            target_ori=self.target_ori,
            cmd_vxyz=cmd_vxyz,
            cmd_yaw_rate=cmd_yaw_rate,
            dt=float(self.cfg.dt),
        )
        self.last_plan = plan

        # update support polygon from contact_state (use planned positions)
        contacts = [leg for leg, c in plan.contact_state.items() if bool(c)]
        support_polygon = [np.asarray(plan.target_pos[leg][0:2], dtype=float) for leg in contacts]
        if len(support_polygon) >= 3:
            self.com_constraint.polygon = np.array(support_polygon, dtype=float)

        # log support polygon
        sp = [np.asarray(x, dtype=float).reshape(2) for x in support_polygon]
        while len(sp) < 4:
            sp.append(np.array([np.nan, np.nan], dtype=float))
        if self.logger is not None:
            self.logger.write_row(
                name='support_polygon',
                filename='support_polygon_data.csv',
                header=['t'] + [f'p{i}_x' for i in range(4)] + [f'p{i}_y' for i in range(4)],
                row=[
                    self.t,
                    float(sp[0][0]), float(sp[1][0]), float(sp[2][0]), float(sp[3][0]),
                    float(sp[0][1]), float(sp[1][1]), float(sp[2][1]), float(sp[3][1]),
                ],
            )

        # log targets
        if self.logger is not None:
            self.logger.write_row(
                name='com_target',
                filename='com_target_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(plan.target_pos['com'][0]), float(plan.target_pos['com'][1]), float(plan.target_pos['com'][2])],
            )
            for leg in ALL_LEGS:
                p = plan.target_pos[leg]
                self.logger.write_row(
                    name=f'{leg.lower()}_target',
                    filename=f'{leg}_target_data.csv',
                    header=['t', 'x', 'y', 'z'],
                    row=[self.t, float(p[0]), float(p[1]), float(p[2])],
                )

            self.logger.write_row(
                name='contact_state',
                filename='contact_state_data.csv',
                header=['t'] + [f'{leg}_contact' for leg in ALL_LEGS],
                row=[self.t] + [1.0 if bool(plan.contact_state.get(leg, True)) else 0.0 for leg in ALL_LEGS],
            )

        # commit targets
        self.target_pos = plan.target_pos
        self.target_ori = plan.target_ori

        # set QP targets
        self.body_pos_task.target_world = self.target_pos['com']
        self.body_ori_task.R_world_frame = np.asarray(self.target_ori['com'], dtype=float)
        for leg in ALL_LEGS:
            self.leg_tasks[leg].target_world = self.target_pos[leg]

        # solve
        self.solver.solve(True)
        self.robot.update_kinematics()

        q = self.robot.state.q

        # send to MuJoCo
        if self.enable_shm and self.cpg_to_sim is not None:
            try:
                self.cpg_to_sim.write(qpos_desired=q, kp=self.kp_gains, kd=self.kd_gains)
            except Exception:
                pass

        # visualization
        self.visual.display_robot(q)
        try:
            com_world = self.robot.com_world()
            self.visual.display_com_xy(com_world, name='com')
        except Exception:
            pass
        self.visual.display_support_polygon(support_polygon)

        return q
