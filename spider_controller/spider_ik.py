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

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import os
import math
import time
import numpy as np


from gait_manager import GaitCycleManager, GaitParams
from spider_logger import SpiderCsvLogger
from spider_visual import SpiderVisualizer
# from compensation import Compensation
from spider_comp import SpiderCompensation

ALL_LEGS = ['LF', 'RF', 'LH', 'RH']


def build_initial_targets(body_height: float = 0.20) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    com = np.array([0.0, 0.0, body_height], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    feet = {
        'LH': np.array([-0.37, 0.37, 0.0], dtype=np.float64),
        'LF': np.array([0.37, 0.37, 0.0], dtype=np.float64),
        'RF': np.array([0.37, -0.37, 0.0], dtype=np.float64),
        'RH': np.array([-0.37, -0.37, 0.0], dtype=np.float64),
    }

    target_pos = {'com': com, **feet}
    target_ori = {'com': R}
    return target_pos, target_ori


@dataclass
class SpiderIkConfig:
    dt: float = 0.01
    gait_mode: str = 'quasi_static'

    # runtime toggles
    enable_visual: bool = True
    enable_logger: bool = True

    # gait params
    cycle_period: float = 2.0
    swing_height: float = 0.12
    lookahead: float = 1.0
    cmd_epsilon: float = 1e-4
    cmd_timeout: float = 2.0
    stand_transition_duration: float = 0.5

    qs_swing_duty: float = 0.5

    # com polygon constraint
    polygon_margin: float = 0.05

    # solver weights
    leg_task_weight: float = 1e3
    leg_task_ori_weight: float = 1e1+50
    body_task_weight: float = 1e1
    body_task_ori_weight: float = 1e2
    com_constraint_weight: float = 1e2+50

    alpha_max: float = 0.1  # 弧度，约 28.6 度

    # URDF root candidates
    body_height: float = 0.26
    urdf_candidates: Tuple[str, ...] = ("../sqr_description/sqr_a1_description/urdf/",)
    # urdf_candidates: Tuple[str, ...] = ("../spider_sldasm/urdf",)
    
    # control filter parameters
    enable_ctrl_filter: bool = True
    ctrl_filter_alpha: float = 0.8  # 滤波系数，0-1之间，越大越平滑
    ctrl_max: float = 50.0
    leg_init_state: np.ndarray = field(default_factory=lambda: np.array([
            0,0,0.26,
            0,0,0,1,
            0.7899997871565971,0.01658366118250419,-0.004088938263042,
            -0.7867905831961373,-0.016585433996785828,0.004100463013502682,
            -0.7867905831961349,0.016585433996786143,-0.00410046301350216,
            0.7899997871565964,-0.016583661182505084,0.004088938263039494]))

@dataclass
class SpiderIkData:
    q: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
    qdd: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64))

@dataclass
class MeasuredData:
    q: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
    qdd: np.ndarray = field(default_factory=lambda: np.zeros(19, dtype=np.float64))
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64))

class SpiderIK:
    def __init__(self, cfg: SpiderIkConfig):
        self.cfg = cfg
        self.data = SpiderIkData()
        self.enable_visual = bool(cfg.enable_visual)
        self.enable_logger = bool(cfg.enable_logger)

        # --- runtime time ---
        self.t = 0.0

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
        self.robot.state.q = self.cfg.leg_init_state.copy()
        # print joint order for debugging / inspection
        joint_names = None
        try:
            if hasattr(self.robot.model, 'jointNames'):
                joint_names = list(self.robot.model.jointNames)
            elif hasattr(self.robot.model, 'joint_names'):
                joint_names = list(self.robot.model.joint_names)
            elif hasattr(self.robot.model, 'names'):
                joint_names = list(self.robot.model.names)
            elif hasattr(self.robot.model, 'getJointNames'):
                joint_names = list(self.robot.model.getJointNames())
            elif hasattr(self.robot, 'joint_names'):
                joint_names = list(self.robot.joint_names)
            else:
                # fallback: try to collect names from frames (best-effort)
                joint_names = []
                if hasattr(self.robot.model, 'frames'):
                    for f in getattr(self.robot.model, 'frames'):
                        name = getattr(f, 'name', None)
                        if name and name not in joint_names:
                            joint_names.append(name)
        except Exception:
            joint_names = None

        if not joint_names:
            try:
                qlen = len(self.robot.state.q)
            except Exception:
                qlen = None
            print("SpiderIK: 无法从 robot.model 获取关节名称。q 长度:", qlen)
        else:
            # normalize bytes -> str
            joint_names = [n.decode() if isinstance(n, bytes) else str(n) for n in joint_names]
            print(f"SpiderIK: joint names (total {len(joint_names)}):")
            for idx, name in enumerate(joint_names):
                print(f"  [{idx:2d}] {name}")

            # 如果 state.q 包含浮动基 + 关节，打印 actuated joints 映射（浮动基假设前 6 个）
            try:
                qlen = len(self.robot.state.q)
                # assume actuated joints = qlen - 6 if a 6-DOF floating base exists, else use full length if names match
                if len(joint_names) >= qlen:
                    base_offset = 0 if len(joint_names) == qlen else max(0, qlen - 12)  # best-effort
                else:
                    base_offset = max(0, qlen - len(joint_names))
                print("Actuated joints mapping (q index -> name):")
                for i in range(qlen):
                    name = joint_names[i] if i < len(joint_names) else "<unknown>"
                    marker = " (actuated)" if i >= base_offset else " (base)"
                    print(f"  q[{i:2d}] -> {name}{marker}")
            except Exception:
                pass
        self.robot.set_velocity_limits(1.)

        self.solver = self.placo.KinematicsSolver(self.robot);print("self.robot.model.velocityLimit",self.robot.model.velocityLimit)
        self.solver.enable_velocity_limits(True)
        self.solver.dt = 0.001

        # frames mapping
        self.leg_foot_name_map = {
            'LH': 'LR_wheel_link',
            'RH': 'RR_wheel_link',
            'RF': 'RF_wheel_link',
            'LF': 'LF_wheel_link',
        }
        
        self.leg_foot_joint_map = {
            'LH': [6,7,8],
            'RH': [9,10,11],
            'RF': [0,1,2],
            'LF': [3,4,5],
        }
        self.compensation = SpiderCompensation(self.robot, self.leg_foot_name_map)

        # --- initial targets ---
        self.target_pos, self.target_ori = build_initial_targets(body_height=self.cfg.body_height)
        self.target_init_pos = {k: np.asarray(v).copy() for k, v in self.target_pos.items()}

        # --- tasks ---
        self.leg_tasks = {
            leg: self.solver.add_position_task(self.leg_foot_name_map[leg], np.array(self.target_pos[leg]))
            for leg in ALL_LEGS
        }
        self.leg_orien_tasks = {
            'LH': self.solver.add_axisalign_task(self.leg_foot_name_map['LH'], np.array([0, 1, 0]), np.array([0, 0, -1])),
            'RH': self.solver.add_axisalign_task(self.leg_foot_name_map['RH'], np.array([0, -1, 0]), np.array([0, 0, -1])),
            'RF': self.solver.add_axisalign_task(self.leg_foot_name_map['RF'], np.array([0, -1, 0]), np.array([0, 0, -1])),
            'LF': self.solver.add_axisalign_task(self.leg_foot_name_map['LF'], np.array([0, 1, 0]), np.array([0, 0, -1])),
        }
        for leg in ALL_LEGS:
            self.leg_tasks[leg].configure(self.leg_foot_name_map[leg], 'soft', float(self.cfg.leg_task_weight))
            self.leg_orien_tasks[leg].configure(self.leg_foot_name_map[leg], 'soft', float(self.cfg.leg_task_ori_weight))

        # base tasks
        self.body_pos_task = self.solver.add_position_task('base_link', np.array(self.target_pos['com']))
        self.body_pos_task.configure('base_link', 'soft', float(self.cfg.body_task_weight))

        self.body_ori_task = self.solver.add_orientation_task('base_link', np.array(self.target_ori['com']))
        self.body_ori_task.configure('base_link', 'soft', float(self.cfg.body_task_ori_weight))

        # com polygon constraint (placeholder init polygon, will update each loop)
        init_polygon = np.array([
            np.array([0.372385, 0.371678]),
            np.array([0.371678, -0.372385]),
            np.array([-0.361251, -0.360544]),
        ])
        self.regularization_task = self.solver.add_regularization_task(float(self.cfg.com_constraint_weight))
        self.com_constraint = self.solver.add_com_polygon_constraint(init_polygon, float(self.cfg.polygon_margin))
        self.com_constraint.polygon = init_polygon
        self.com_constraint.configure('com_constraint', 'soft', float(self.cfg.com_constraint_weight))

        # --- gait manager ---
        self.gait_params = GaitParams(
            gait_mode=self.cfg.gait_mode,
            cycle_period=float(self.cfg.cycle_period),
            swing_height=float(self.cfg.swing_height),
            lookahead=float(self.cfg.lookahead),
            cmd_epsilon=float(self.cfg.cmd_epsilon),
            cmd_timeout=float(self.cfg.cmd_timeout),
            stand_transition_duration=float(self.cfg.stand_transition_duration),
            qs_cycle_period = float(self.cfg.cycle_period),  # use same cycle period
            qs_swing_duty = float(self.cfg.qs_swing_duty),  # 60% of each leg phase is swing, 40% is stance (more conservative for stability)
        )
        self.gait = GaitCycleManager(params=self.gait_params, dtype=np.float64)
        self.gait.set_stand_targets(self.target_pos, self.target_ori)

        # --- PD gains ---
        # self.comp = Compensation(self.cfg.urdf_candidates)
        self.KP_HIP = 100.0
        self.KP_ANKLE = 1.2 * self.KP_HIP
        self.KP_KNEE = 1.1 * self.KP_HIP
        self.KD_HIP = 0.05 * self.KP_HIP
        self.KD_ANKLE = 0.05 * self.KP_ANKLE
        self.KD_KNEE = 0.05 * self.KP_KNEE
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

        # --- control filter ---
        self.prev_ctrl = np.zeros(12, dtype=np.float64)  # 上一次的控制指令，用于滤波
        self.filter_initialized = False  # 滤波器是否已初始化

        # last plan snapshot
        self.last_plan = None

    @staticmethod
    def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def _polygon_signed_area(polygon: np.ndarray) -> float:
        poly = np.asarray(polygon, dtype=np.float64)
        if poly.shape[0] < 3:
            return 0.0
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def _ensure_ccw_polygon(self, polygon: np.ndarray) -> np.ndarray:
        poly = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        if poly.shape[0] < 3:
            return poly.copy()
        if self._polygon_signed_area(poly) < 0.0:
            poly = poly[::-1].copy()
        return poly

    def _compute_support_polygon(self, plan) -> Tuple[List[str], np.ndarray]:
        contacts = [leg for leg, c in plan.contact_state.items() if bool(c)]
        support_polygon = np.array(
            [np.asarray(plan.target_pos[leg][0:2], dtype=np.float64) for leg in contacts],
            dtype=np.float64,
        ) if len(contacts) > 0 else np.zeros((0, 2), dtype=np.float64)
        return contacts, self._ensure_ccw_polygon(support_polygon)

    def _project_point_to_segment(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            return a.copy()
        t = float(np.dot(p - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        return a + t * ab

    def _is_inside_convex_polygon(self, point: np.ndarray, polygon: np.ndarray, tol: float = 1e-9) -> bool:
        poly = self._ensure_ccw_polygon(polygon)
        n = poly.shape[0]
        if n < 3:
            return False
        p = np.asarray(point, dtype=np.float64)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            if self._cross2d(b - a, p - a) < -tol:
                return False
        return True

    def _shrink_support_polygon(self, polygon: np.ndarray, margin: float) -> np.ndarray:
        poly = self._ensure_ccw_polygon(polygon)
        n = poly.shape[0]
        if n < 3 or margin <= 0.0:
            return poly.copy()

        shrunk = []
        for i in range(n):
            prev_pt = poly[(i - 1) % n]
            curr_pt = poly[i]
            next_pt = poly[(i + 1) % n]

            e_prev = curr_pt - prev_pt
            e_next = next_pt - curr_pt
            len_prev = float(np.linalg.norm(e_prev))
            len_next = float(np.linalg.norm(e_next))
            if len_prev <= 1e-9 or len_next <= 1e-9:
                return poly.copy()

            t_prev = e_prev / len_prev
            t_next = e_next / len_next
            n_prev = np.array([-t_prev[1], t_prev[0]], dtype=np.float64)
            n_next = np.array([-t_next[1], t_next[0]], dtype=np.float64)

            p1 = curr_pt + margin * n_prev
            p2 = curr_pt + margin * n_next
            A = np.column_stack((t_prev, -t_next))
            rhs = p2 - p1
            try:
                sol = np.linalg.solve(A, rhs)
                new_pt = p1 + sol[0] * t_prev
            except np.linalg.LinAlgError:
                return poly.copy()
            shrunk.append(new_pt)

        shrunk = np.asarray(shrunk, dtype=np.float64)
        if abs(self._polygon_signed_area(shrunk)) < 1e-10:
            return poly.copy()
        return shrunk

    def _project_point_to_convex_polygon(self, point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        poly = self._ensure_ccw_polygon(polygon)
        p = np.asarray(point, dtype=np.float64)
        n = poly.shape[0]
        if n == 0:
            return p.copy()
        if n == 1:
            return poly[0].copy()
        if n == 2:
            return self._project_point_to_segment(p, poly[0], poly[1])
        if self._is_inside_convex_polygon(p, poly):
            return p.copy()

        best = None
        best_dist2 = float('inf')
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cand = self._project_point_to_segment(p, a, b)
            dist2 = float(np.sum((cand - p) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best = cand
        return best if best is not None else p.copy()

    def _compute_projected_com_target(self, com_nominal: np.ndarray, support_polygon: np.ndarray) -> np.ndarray:
        com_exec = np.asarray(com_nominal, dtype=np.float64).copy()
        if support_polygon.shape[0] < 3:
            return com_exec

        shrunk_polygon = self._shrink_support_polygon(support_polygon, float(self.cfg.polygon_margin))
        if shrunk_polygon.shape[0] >= 3 and abs(self._polygon_signed_area(shrunk_polygon)) > 1e-10:
            polygon_for_projection = shrunk_polygon
        else:
            polygon_for_projection = support_polygon

        com_exec[:2] = self._project_point_to_convex_polygon(com_exec[:2], polygon_for_projection)
        return com_exec

    def apply_ctrl_filter(self, ctrl_raw: np.ndarray) -> np.ndarray:
        """
        对控制指令应用低通滤波
        使用简单的指数移动平均滤波器：
        filtered_ctrl = alpha * prev_ctrl + (1 - alpha) * current_ctrl
        """
        if not self.cfg.enable_ctrl_filter:
            return ctrl_raw
        
        if not self.filter_initialized:
            # 第一次调用，直接使用原始值
            self.prev_ctrl = ctrl_raw.copy()
            self.filter_initialized = True
            return ctrl_raw
        
        # 应用指数移动平均滤波
        alpha = self.cfg.ctrl_filter_alpha
        filtered_ctrl = alpha * self.prev_ctrl + (1.0 - alpha) * ctrl_raw
        
        # 更新历史值
        self.prev_ctrl = filtered_ctrl.copy()
        
        return filtered_ctrl

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
        contacts, support_polygon = self._compute_support_polygon(plan)

        contacts_bool = []
        for leg in ALL_LEGS:
            contacts_bool.append(1.0 if leg in contacts else 0.0)
        # print("Contacts bool:", contacts_bool)
        if support_polygon.shape[0] >= 3:
            self.com_constraint.polygon = np.array(support_polygon, dtype=float)

        # log targets
        if self.logger is not None:
            com_exec_for_log = self._compute_projected_com_target(plan.target_pos['com'], support_polygon)
            self.logger.write_row(
                name='com_target',
                filename='com_target_data.csv',
                header=['t', 'x', 'y', 'z', 'x_exec', 'y_exec', 'z_exec'],
                row=[
                    self.t,
                    float(plan.target_pos['com'][0]),
                    float(plan.target_pos['com'][1]),
                    float(plan.target_pos['com'][2]),
                    float(com_exec_for_log[0]),
                    float(com_exec_for_log[1]),
                    float(com_exec_for_log[2]),
                ],
            )

            self.logger.write_row(
                name='robot_state',
                filename='robot_state_data.csv',
                header=['t'] +[f'q_{i}' for i in range(len(self.data.q))],
                row=[self.t] + [float(q) for q in self.data.q],
            )

            p = plan.target_pos["LH"]
            self.logger.write_row(
                name='LH_target',
                filename='LH_target_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(p[0]), float(p[1]), float(p[2])],
            )

            self.logger.write_row(
                name='contact_state',
                filename='contact_state_data.csv',
                header=['t'] + [f'{leg}_contact' for leg in ALL_LEGS],
                row=[self.t] + [1.0 if bool(plan.contact_state.get(leg, True)) else 0.0 for leg in ALL_LEGS],
            )

            com_world = self.robot.com_world()
            self.logger.write_row(
                name='com_world',
                filename='com_world_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(com_world[0]), float(com_world[1]), float(com_world[2])],
            )

            LH_foot_world = self.robot.get_T_world_frame(self.leg_foot_name_map['LH'])
            self.logger.write_row(
                name='LH_foot_world',
                filename='LH_foot_world_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(LH_foot_world[0, 3]), float(LH_foot_world[1, 3]), float(LH_foot_world[2, 3])],
            )
            LF_foot_world = self.robot.get_T_world_frame(self.leg_foot_name_map['LF'])
            self.logger.write_row(
                name='LF_foot_world',
                filename='LF_foot_world_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(LF_foot_world[0, 3]), float(LF_foot_world[1, 3]), float(LF_foot_world[2, 3])],
            )
            RF_foot_world = self.robot.get_T_world_frame(self.leg_foot_name_map['RF'])
            self.logger.write_row(
                name='RF_foot_world',
                filename='RF_foot_world_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(RF_foot_world[0, 3]), float(RF_foot_world[1, 3]), float(RF_foot_world[2, 3])],
            )
            RH_foot_world = self.robot.get_T_world_frame(self.leg_foot_name_map['RH'])
            self.logger.write_row(
                name='RH_foot_world',
                filename='RH_foot_world_data.csv',
                header=['t', 'x', 'y', 'z'],
                row=[self.t, float(RH_foot_world[0, 3]), float(RH_foot_world[1, 3]), float(RH_foot_world[2, 3])],
            )
            tauff = self.data.ctrl
            self.logger.write_row(
                name='tauff',
                filename='tauff_data.csv',
                header=['t'] + [f'tau_{i}' for i in range(len(tauff))],
                row=[self.t] + [float(tauff[i]) for i in range(len(tauff))],
            )

        # commit targets
        self.target_pos = plan.target_pos
        self.target_ori = plan.target_ori

        # set QP targets
        body_pos_exec = self._compute_projected_com_target(self.target_pos['com'], support_polygon)
        self.body_pos_task.target_world = body_pos_exec
        self.body_ori_task.R_world_frame = np.asarray(self.target_ori['com'], dtype=float)
        for leg in ALL_LEGS:
            self.leg_tasks[leg].target_world = self.target_pos[leg]

        # solve
        self.solver.solve(True)
        self.robot.update_kinematics()
        tau_ff, contact_forces = self.compensation.compute(contacts)

        gravity_torques = self.robot.generalized_gravity()
        # print("gravity_torques",gravity_torques)
        self.data.q = self.robot.state.q
        self.data.ctrl = gravity_torques[6:]  # gravity compensation for all joints
        
        # 只为接触的足端分配tau_ff值
        tau_idx = 0
        for leg in ALL_LEGS:
            if leg in contacts:
                joint_indices = self.leg_foot_joint_map[leg]
                for i, joint_idx in enumerate(joint_indices):
                    if tau_idx < len(tau_ff):
                        self.data.ctrl[joint_idx] = tau_ff[tau_idx]
                        tau_idx += 1
        
        # 力矩限幅
        for i in range(len(self.data.ctrl)):
            if self.data.ctrl[i] > self.cfg.ctrl_max:
                self.data.ctrl[i] = self.cfg.ctrl_max
            elif self.data.ctrl[i] < -self.cfg.ctrl_max:
                self.data.ctrl[i] = -self.cfg.ctrl_max
        
        # 应用滤波器
        self.data.ctrl = self.apply_ctrl_filter(self.data.ctrl)

        # visualization
        self.visual.display_robot(self.data.q)
        try:
            com_world = self.robot.com_world()
            self.visual.display_com_xy(com_world, name='com')
        except Exception:
            pass
        self.visual.display_support_polygon(support_polygon.tolist() if isinstance(support_polygon, np.ndarray) else support_polygon)

        return self.data
