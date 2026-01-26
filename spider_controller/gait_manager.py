"""gait_manager.py

CPU (NumPy) gait cycle manager for this repo.

Design goals:
- Accept velocity commands (vx, vy, vz, yaw_rate) and generate target COM + feet.
- When no command:
    - Keep current targets (robot holds steady).
    - If no command for >2s: exit walking cycle and smoothly transition to stand.
- When command present:
    - If not in cycle: initialize and enter a cycle.
    - If in cycle: manage trot gait and generate swing trajectories.

Notes:
- This is a kinematic target generator. It does not require Pinocchio.
- All math is NumPy-based and runs on CPU.

Expected target dict keys: 'com', 'LF', 'RF', 'LH', 'RH'
Expected target_ori['com']: (3,3) rotation matrix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Literal
import time
import numpy as np

PAIR1 = ['LF', 'RH']
PAIR2 = ['RF', 'LH']
ALL_LEGS = ['LF', 'RF', 'LH', 'RH']

GaitMode = Literal["trot", "quasi_static"]


@dataclass
class GaitParams:
    cycle_period: float = 1.0
    swing_height: float = 0.06
    lookahead: float = 1.0
    cmd_epsilon: float = 1e-4
    cmd_timeout: float = 2.0
    stand_transition_duration: float = 0.4

    # Gait mode selection
    gait_mode: GaitMode = "trot"

    # Quasi-static (crawl) parameters
    # A full crawl cycle is split into 4 equal phases; in each phase
    # only one leg is swinging (3 legs in stance).
    qs_cycle_period: float = 1.0
    # Swing duty inside each single-leg phase. Must be in (0,1).
    # Example: 0.5 means first half of the phase is swing, second half is stance.
    qs_swing_duty: float = 0.6
    # Swing order for crawl gait (typical: LF -> RH -> RF -> LH)
    qs_order: tuple = ("LF", "RF", "RH", "LH")


@dataclass
class GaitPlan:
    in_cycle: bool
    stance_legs: List[str]
    swing_legs: List[str]
    # Contact state in world (True = in contact/stance, False = swing)
    contact_state: Dict[str, bool]
    target_pos: Dict[str, np.ndarray]
    target_ori: Dict[str, np.ndarray]


class GaitCycleManager:
    """NumPy CPU gait cycle manager.

    Public API intentionally mirrors the previous torch/CUDA implementation.
    """

    def __init__(self, params: Optional[GaitParams] = None, dtype=np.float64):
        self.dtype = dtype
        self.params = params or GaitParams()

        self.in_cycle = False
        self._t0 = 0.0
        self._last_cmd_wall: Optional[float] = None

        # Stand targets set on first call (or explicit call)
        self._stand_pos: Optional[Dict[str, np.ndarray]] = None
        self._stand_ori: Optional[Dict[str, np.ndarray]] = None
        # Stand foot COM-relative offsets in body frame (x/y).
        self._stand_foot_offset_b: Dict[str, np.ndarray] = {}
        # Cycle foot COM-relative offsets in body frame (x/y).
        self._cycle_foot_offset_b: Dict[str, np.ndarray] = {}


        # Persistent foot state in world
        self._feet_world: Dict[str, np.ndarray] = {}
        self._swing_active = {k: False for k in ALL_LEGS}
        self._swing_start: Dict[str, np.ndarray] = {}
        self._swing_goal: Dict[str, np.ndarray] = {}

        # Stand transition
        self._stand_trans_active = False
        self._stand_trans_t0 = 0.0
        self._stand_trans_p0: Optional[Dict[str, np.ndarray]] = None
        # Two transition modes:
        # - "normal": feet start from world contact points and immediately align
        #             to COM-relative stand offsets (old behavior).
        # - "timeout": when exiting a gait due to cmd timeout, start from the
        #              CURRENT foot targets (may be mid-swing) and smoothly
        #              interpolate feet XY to COM-relative stand offsets.
        self._stand_trans_mode: str = "normal"

        # When transitioning to stand after command timeout, we don't want to
        # pull COM x/y back to the original stand pose, and we don't want to
        # pull yaw back either. We freeze the reference x/y/yaw when the
        # transition starts.
        self._freeze_com_xy: Optional[np.ndarray] = None  # shape (2,)
        self._freeze_yaw: Optional[float] = None          # scalar

        # Track previous half-cycle to make phase boundary handling explicit.
        # This is critical for world-fixed stance + continuous touchdown.
        self._prev_first_half = None  # type: Optional[bool]

        # Quasi-static phase tracking (for explicit boundary handling)
        self._prev_qs_swing_leg: Optional[str] = None

    def set_gait_mode(self, mode: GaitMode):
        """Set gait mode: 'trot' or 'quasi_static'.

        This does not immediately force a reset, but by default we reset phase
        bookkeeping to avoid half-cycle artifacts when switching.
        """
        if mode not in ("trot", "quasi_static"):
            raise ValueError(f"Unsupported gait mode: {mode}")
        self.params.gait_mode = mode
        # Reset phase boundary trackers
        self._prev_first_half = None
        self._prev_qs_swing_leg = None

    def set_stand_targets(self, target_pos: Dict[str, np.ndarray], target_ori: Dict[str, np.ndarray]):
        self._stand_pos = {k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_pos.items()}
        self._stand_ori = {k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_ori.items()}

        # Cache nominal feet COM-frame offsets at stand.
        # Store them in the *body frame* (x/y). This makes yaw handling consistent:
        # desired world XY = com_xy + Rz(yaw) @ off_body.
        if 'com' in self._stand_pos:
            com_xy = self._stand_pos['com'][0:2]
            R0 = self._stand_ori['com'] if (self._stand_ori is not None and 'com' in self._stand_ori) else np.eye(3, dtype=self.dtype)
            yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
            cy0 = float(np.cos(yaw0))
            sy0 = float(np.sin(yaw0))
            RzT0 = np.array([[cy0, sy0], [-sy0, cy0]], dtype=self.dtype)  # Rz(-yaw0)
            self._stand_foot_offset_b = {
                leg: (RzT0 @ (self._stand_pos[leg][0:2] - com_xy)).copy()
                for leg in ALL_LEGS
            }

    def _now(self) -> float:
        return time.time()

    def _cmd_active(self, vxyz: np.ndarray, yaw_rate: float) -> bool:
        eps = self.params.cmd_epsilon
        return (np.linalg.norm(vxyz) > eps) or (abs(float(yaw_rate)) > eps)

    def _cmd_walk_active(self, vxyz: np.ndarray, yaw_rate: float) -> bool:
        """Whether the command should trigger/maintain a walking gait.

        Height-only commands (non-zero z with ~zero xy and yaw) should NOT enter
        a walking cycle.
        """
        eps = self.params.cmd_epsilon
        vxy = float(np.linalg.norm(vxyz[0:2]))
        return (vxy > eps) or (abs(float(yaw_rate)) > eps)

    def _cubic(self, p0: np.ndarray, pf: np.ndarray, s: float) -> np.ndarray:
        s = float(max(0.0, min(1.0, s)))
        a = 3.0 * s * s - 2.0 * s * s * s
        return p0 + (pf - p0) * a

    def _swing_traj(self, p0: np.ndarray, pf: np.ndarray, s: float) -> np.ndarray:
        p = self._cubic(p0, pf, s)
        z_bump = 4.0 * self.params.swing_height * float(s) * (1.0 - float(s))
        p2 = p.copy()
        p2[2] = p2[2] + z_bump
        return p2

    def update(
        self,
        t: float,
        target_pos: Dict[str, np.ndarray],
        target_ori: Dict[str, np.ndarray],
        cmd_vxyz: np.ndarray,
        cmd_yaw_rate: float,
        dt: float,
        *,
        meas_com_pos: Optional[np.ndarray] = None,
        meas_com_R: Optional[np.ndarray] = None,
    ) -> GaitPlan:
        """Update and return planned targets.

        Inputs:
        - t: controller time (seconds)
        - target_pos/target_ori: current targets (will not be mutated)
        - cmd_vxyz: desired body-frame velocity (vx,vy,vz)
        - cmd_yaw_rate: desired yaw rate (omega_z)
        - dt: controller step
        """

        cmd_vxyz = np.asarray(cmd_vxyz, dtype=self.dtype).reshape(3)
        cmd_yaw_rate = float(np.asarray(cmd_yaw_rate).reshape(()))

        # Measured trunk/COM pose feedback (world frame).
        # If not provided, fall back to the planner's previous targets.
        meas_com_pos = np.asarray(meas_com_pos, dtype=self.dtype).reshape(3) if meas_com_pos is not None else None
        meas_com_R = np.asarray(meas_com_R, dtype=self.dtype).reshape(3, 3) if meas_com_R is not None else None

        # Use measurement for planning when possible.
        base_com_pos = meas_com_pos if meas_com_pos is not None else np.asarray(target_pos['com'], dtype=self.dtype)
        base_com_R = meas_com_R if meas_com_R is not None else np.asarray(target_ori['com'], dtype=self.dtype)

        # initialize stand targets once
        if self._stand_pos is None:
            self.set_stand_targets(target_pos, target_ori)

        active = self._cmd_active(cmd_vxyz, cmd_yaw_rate)
        walk_active = self._cmd_walk_active(cmd_vxyz, cmd_yaw_rate)
        # yaw-only spin in place (no commanded vxy). Used to keep stance feet
        # fixed in world while swing feet stay approximately fixed in body frame.
        eps = self.params.cmd_epsilon
        spin_in_place = (float(np.linalg.norm(cmd_vxyz[0:2])) <= eps) and (abs(float(cmd_yaw_rate)) > eps)
        now = self._now()
        if active:
            self._last_cmd_wall = now

        # timeout -> exit cycle
        # IMPORTANT: do not abruptly flip to stand targets, otherwise feet/com
        # targets can jump (stance feet are world-fixed in _feet_world).
        # Instead, trigger a stand transition starting from the LAST planned
        # world contact points.
        if (not active) and (self._last_cmd_wall is not None):
            if now - self._last_cmd_wall > self.params.cmd_timeout:
                if self.in_cycle:
                    # Start stand transition from a continuous pose.
                    if not self._stand_trans_active:
                        self._stand_trans_active = True
                        self._stand_trans_t0 = t
                        # timeout exit: start from CURRENT targets (may be mid-swing)
                        self._stand_trans_mode = "timeout"
                        p0 = {k: np.asarray(target_pos[k], dtype=self.dtype).copy() for k in ['com'] + ALL_LEGS}
                        self._stand_trans_p0 = p0

                        com0 = p0['com']
                        self._freeze_com_xy = com0[0:2].copy()
                        R0 = np.asarray(target_ori['com'], dtype=self.dtype)
                        yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
                        self._freeze_yaw = yaw0

                    # Exit cycle; subsequent not-active branch will execute stand transition.
                    self.in_cycle = False
                else:
                    self.in_cycle = False

        # enter cycle if needed (walking-triggered only)
        if walk_active and (not self.in_cycle):
            self.in_cycle = True
            # Start phase from 0 at the moment we enter the cycle.
            # Using (t - dt) makes the first update land at a small positive phase
            # instead of potentially skipping the very beginning due to caller timing.
            self._t0 = float(t) - float(dt)
            # capture current COM-relative foot offsets in the *body frame* (x/y).
            # This is critical when the robot starts walking with a non-zero yaw, and it
            # also makes yaw rotation during gait consistent.
            com_xy0 = np.asarray(base_com_pos, dtype=self.dtype)[0:2].copy()
            yaw0 = float(np.arctan2(base_com_R[1, 0], base_com_R[0, 0]))
            cy0 = float(np.cos(yaw0))
            sy0 = float(np.sin(yaw0))
            RzT0 = np.array([[cy0, sy0], [-sy0, cy0]], dtype=self.dtype)  # Rz(-yaw0)
            self._cycle_foot_offset_b = {
                leg: (RzT0 @ (np.asarray(target_pos[leg], dtype=self.dtype)[0:2].copy() - com_xy0)).copy()
                for leg in ALL_LEGS
            }
            # init feet world from current targets
            for leg in ALL_LEGS:
                self._feet_world[leg] = np.asarray(target_pos[leg], dtype=self.dtype).copy()
                self._swing_active[leg] = False
                self._swing_start[leg] = self._feet_world[leg].copy()
                self._swing_goal[leg] = self._feet_world[leg].copy()
            # cancel any stand transition
            self._stand_trans_active = False
            self._prev_first_half = None

        # Height-only command: no walking. Keep all feet as stance and only move COM z.
        if active and (not walk_active):
            new_pos = {k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_pos.items()}
            new_ori = {k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_ori.items()}

            com_next = new_pos['com'].copy()
            com_next[2] = com_next[2] + cmd_vxyz[2] * float(dt)
            z_hi = float(self._stand_pos['com'][2].item()) if self._stand_pos is not None else 0.26
            z_lo = 0.0
            com_next[2] = float(np.clip(com_next[2], z_lo, z_hi))
            new_pos['com'] = com_next

            # ensure no gait cycle
            self.in_cycle = False
            self._stand_trans_active = False
            self._freeze_com_xy = None
            self._freeze_yaw = None

            return GaitPlan(
                in_cycle=self.in_cycle,
                stance_legs=ALL_LEGS,
                swing_legs=[],
                contact_state={k: True for k in ALL_LEGS},
                target_pos=new_pos,
                target_ori=new_ori,
            )

        # no command: hold pose (do not inject drift)
        if not active:
            if self.in_cycle:
                # in-cycle but cmd absent: keep current targets steady (hold)
                return GaitPlan(
                    in_cycle=self.in_cycle,
                    stance_legs=ALL_LEGS,
                    swing_legs=[],
                    contact_state={k: True for k in ALL_LEGS},
                    target_pos={k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_pos.items()},
                    target_ori={k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_ori.items()},
                )

            # not in cycle: smoothly go back to stand
            if not self._stand_trans_active:
                self._stand_trans_active = True
                self._stand_trans_t0 = t
                self._stand_trans_p0 = {k: np.asarray(target_pos[k], dtype=self.dtype).copy() for k in ['com'] + ALL_LEGS}
                self._stand_trans_mode = "normal"
                # Ensure feet start from world-fixed contact points if available.
                for leg in ALL_LEGS:
                    if leg in self._feet_world:
                        self._stand_trans_p0[leg] = self._feet_world[leg].copy()

                # freeze COM x/y and current yaw at the moment we start the stand transition
                com0 = self._stand_trans_p0['com']
                self._freeze_com_xy = com0[0:2].copy()
                # compute yaw from current com rotation matrix
                R0 = np.asarray(target_ori['com'], dtype=self.dtype)
                yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
                self._freeze_yaw = yaw0

            s = (t - self._stand_trans_t0) / max(1e-6, self.params.stand_transition_duration)
            s = max(0.0, min(1.0, s))
            new_pos = dict(target_pos)
            for k in ['com'] + ALL_LEGS:
                new_pos[k] = self._cubic(self._stand_trans_p0[k], self._stand_pos[k], s)

            # Override COM behavior during stand transition:
            # - keep x/y frozen
            # - keep z as-is (maintain height-control result)
            com = np.asarray(new_pos['com'], dtype=self.dtype).copy()
            if self._freeze_com_xy is not None:
                com[0:2] = self._freeze_com_xy
            # keep current z (do not pull back to stand height)
            com[2] = self._stand_trans_p0['com'][2]
            new_pos['com'] = com

            # Override feet behavior during stand transition.
            # We always want to end at the COM-relative stand offsets with frozen yaw,
            # but:
            # - normal mode: immediately enforce COM-relative XY (old behavior)
            # - timeout mode: interpolate XY from current feet to those offsets
            if self._freeze_com_xy is not None:
                yaw = float(self._freeze_yaw) if (self._freeze_yaw is not None) else 0.0
                cy = float(np.cos(yaw))
                sy = float(np.sin(yaw))
                Rz2 = np.array([[cy, -sy], [sy, cy]], dtype=self.dtype)
                for leg in ALL_LEGS:
                    # fall back to stand absolute pose if offsets missing
                    if not hasattr(self, '_stand_foot_offset_b'):
                        continue
                    if leg not in self._stand_foot_offset_b:
                        continue
                    off_xy = self._stand_foot_offset_b[leg]
                    leg_xy = self._freeze_com_xy + (Rz2 @ off_xy)
                    if getattr(self, '_stand_trans_mode', 'normal') == 'timeout':
                        # smooth XY: interpolate from p0.xy to desired COM-relative xy
                        leg_p = np.asarray(new_pos[leg], dtype=self.dtype).copy()
                        p0_xy = self._stand_trans_p0[leg][0:2]
                        leg_p[0:2] = self._cubic(p0_xy, leg_xy, s)
                        new_pos[leg] = leg_p
                    else:
                        # normal: snap XY to desired COM-relative xy
                        leg_p = np.asarray(new_pos[leg], dtype=self.dtype).copy()
                        leg_p[0:2] = leg_xy
                        new_pos[leg] = leg_p

            new_ori = dict(target_ori)

            # Orientation behavior during stand transition:
            # - roll/pitch -> 0
            # - yaw kept frozen
            # Build R = Rz(yaw) * Ry(0) * Rx(0) = Rz(yaw)
            yaw = float(self._freeze_yaw) if (self._freeze_yaw is not None) else 0.0
            cy = float(np.cos(yaw))
            sy = float(np.sin(yaw))
            new_ori['com'] = np.array(
                [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
                dtype=self.dtype,
            )

            if s >= 1.0:
                self._stand_trans_active = False
                self._freeze_com_xy = None
                self._freeze_yaw = None
                self._stand_trans_mode = "normal"

            return GaitPlan(
                in_cycle=self.in_cycle,
                stance_legs=ALL_LEGS,
                swing_legs=[],
                contact_state={k: True for k in ALL_LEGS},
                target_pos=new_pos,
                target_ori=new_ori,
            )

    # active command & in cycle: integrate COM, yaw, and gait feet
        # integrate COM in world: v_world = R_com @ v_body
        # integrate COM in world: v_world = R_meas @ v_body
        # Using measured orientation avoids drift/lag when yaw is controlled by the robot.
        Rcom = base_com_R
        v_world = Rcom @ cmd_vxyz

        new_pos = {k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_pos.items()}
        new_ori = {k: np.asarray(v, dtype=self.dtype).copy() for k, v in target_ori.items()}

        # COM integration: use x/y from rotated command, and z from cmd_vxyz[2]
        # so height control is decoupled from body yaw (z is world-up).
        # Start integration from measured (or last planned) COM pose.
        com_next = np.asarray(base_com_pos, dtype=self.dtype).copy() + v_world * float(dt)
        com_next[2] = float(base_com_pos[2]) + float(cmd_vxyz[2]) * float(dt)

        # Clamp height to stand limits (use stand com z as upper bound).
        z_hi = float(self._stand_pos['com'][2]) if self._stand_pos is not None else 0.26
        z_lo = 0.0
        com_next[2] = float(np.clip(com_next[2], z_lo, z_hi))
        new_pos['com'] = com_next

        # yaw integration around z axis
        # yaw integration around z axis: apply on the measured (or last planned) orientation
        omega = float(cmd_yaw_rate)
        if abs(omega) > 1e-12:
            ang = omega * float(dt)
            c = float(np.cos(ang))
            s = float(np.sin(ang))
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=self.dtype)
            new_ori['com'] = Rz @ base_com_R
        else:
            new_ori['com'] = base_com_R.copy()

        # Extract current yaw from planned COM orientation
        yaw = float(np.arctan2(new_ori['com'][1, 0], new_ori['com'][0, 0]))
        cy = float(np.cos(yaw))
        sy = float(np.sin(yaw))
        Rz2 = np.array([[cy, -sy], [sy, cy]], dtype=self.dtype)

        # helper: get desired COM-relative XY in world from rotated offset
        def com_relative_xy_for_leg(leg: str) -> np.ndarray:
            if hasattr(self, '_cycle_foot_offset_b') and (leg in self._cycle_foot_offset_b):
                off = self._cycle_foot_offset_b[leg]
            elif hasattr(self, '_stand_foot_offset_b') and (leg in self._stand_foot_offset_b):
                off = self._stand_foot_offset_b[leg]
            else:
                return self._stand_pos[leg][0:2].copy()
            # Use NEW planned com_next and yaw for upcoming landing placement
            return com_next[0:2] + (Rz2 @ off)

        mode = getattr(self.params, "gait_mode", "trot")
        if mode == "quasi_static":
            # --- Quasi-static crawl gait (single-leg swing) ---
            T = float(getattr(self.params, "qs_cycle_period", max(self.params.cycle_period, 2.0)))
            duty = float(getattr(self.params, "qs_swing_duty", 0.6))
            duty = float(np.clip(duty, 1e-3, 1.0 - 1e-3))
            order = list(getattr(self.params, "qs_order", ("LF", "RH", "RF", "LH")))
            if len(order) != 4 or any(l not in ALL_LEGS for l in order):
                order = ["LF", "RH", "RF", "LH"]

            phase = ((t - self._t0) % T) / T  # [0,1)
            phase4 = phase * 4.0
            idx = int(np.floor(phase4)) % 4
            u = float(phase4 - np.floor(phase4))  # [0,1) within this leg phase
            swing_leg = str(order[idx])

            # inside the leg phase: first 'duty' part is swing, remainder is stance
            swing_active = (u < duty)
            s_phase = (u / duty) if swing_active else 1.0
            s_phase = max(0.0, min(1.0, float(s_phase)))

            swing_legs = [swing_leg] if swing_active else []
            stance_legs = [l for l in ALL_LEGS if l not in swing_legs]

            # Boundary handling: if previous phase's swing leg changed, force touchdown
            # for the previous swing leg to avoid snapping back to old _feet_world.
            if self._prev_qs_swing_leg is not None and swing_leg != self._prev_qs_swing_leg:
                prev = self._prev_qs_swing_leg
                pf_prev = self._swing_goal[prev].copy() if prev in self._swing_goal else self._feet_world[prev].copy()
                self._feet_world[prev] = pf_prev.copy()
                new_pos[prev] = pf_prev.copy()
                self._swing_active[prev] = False
                self._swing_start[prev] = pf_prev.copy()
                self._swing_goal[prev] = pf_prev.copy()

            self._prev_qs_swing_leg = swing_leg

            # Quasi-static step placement: smaller lookahead (more conservative)
            mid_time = 0.5 * (T / 4.0)  # half of a single-leg phase
            delta_xy = np.array([v_world[0], v_world[1]], dtype=self.dtype) * mid_time

            # Stance legs: world-fixed
            for leg in stance_legs:
                self._swing_active[leg] = False
                new_pos[leg] = self._feet_world[leg].copy()

            # Swing leg: if active, generate swing trajectory; otherwise keep world-fixed
            if swing_active:
                if not self._swing_active[swing_leg]:
                    self._swing_active[swing_leg] = True
                    self._swing_start[swing_leg] = self._feet_world[swing_leg].copy()

                p0 = self._swing_start[swing_leg]
                pf = p0.copy()
                pf[0:2] = com_relative_xy_for_leg(swing_leg) + delta_xy
                pf[2] = float(self._stand_pos[swing_leg][2]) if (self._stand_pos is not None and swing_leg in self._stand_pos) else 0.0
                self._swing_goal[swing_leg] = pf

                new_pos[swing_leg] = self._swing_traj(p0, pf, s_phase)
                if s_phase >= 1.0 - 1e-6:
                    self._feet_world[swing_leg] = pf.copy()
                    self._swing_active[swing_leg] = False
            else:
                new_pos[swing_leg] = self._feet_world[swing_leg].copy()

            contact_state = {leg: (leg in stance_legs) for leg in ALL_LEGS}
            return GaitPlan(
                in_cycle=self.in_cycle,
                stance_legs=stance_legs,
                swing_legs=swing_legs,
                contact_state=contact_state,
                target_pos=new_pos,
                target_ori=new_ori,
            )

        # --- Default: trot gait (two-leg swing) ---
        phase = ((t - self._t0) % self.params.cycle_period) / self.params.cycle_period
        first_half = phase < 0.5
        swing_legs = PAIR1 if first_half else PAIR2
        stance_legs = [l for l in ALL_LEGS if l not in swing_legs]
        s_phase = (phase / 0.5) if first_half else ((phase - 0.5) / 0.5)
        s_phase = max(0.0, min(1.0, float(s_phase)))

        # Detect half-cycle switch and force touchdown for previous swing legs
        half_changed = (self._prev_first_half is not None) and (first_half != self._prev_first_half)
        if half_changed:
            prev_swing = PAIR1 if self._prev_first_half else PAIR2
            for leg in prev_swing:
                pf = self._swing_goal[leg].copy() if leg in self._swing_goal else self._feet_world[leg].copy()
                self._feet_world[leg] = pf.copy()
                new_pos[leg] = pf.copy()
                self._swing_active[leg] = False
                self._swing_start[leg] = pf.copy()
                self._swing_goal[leg] = pf.copy()

            prev_stance = [l for l in ALL_LEGS if l not in prev_swing]
            for leg in prev_stance:
                new_pos[leg] = self._feet_world[leg].copy()

        self._prev_first_half = first_half
        self._prev_qs_swing_leg = None

        mid_time = 0.25 * float(self.params.cycle_period)
        delta_xy = np.array([v_world[0], v_world[1]], dtype=self.dtype) * mid_time

        for leg in swing_legs:
            if not self._swing_active[leg]:
                self._swing_active[leg] = True
                self._swing_start[leg] = self._feet_world[leg].copy()

        for leg in stance_legs:
            self._swing_active[leg] = False
            new_pos[leg] = self._feet_world[leg].copy()

        for leg in swing_legs:
            p0 = self._swing_start[leg]
            pf = p0.copy()
            pf[0:2] = com_relative_xy_for_leg(leg) + delta_xy
            pf[2] = float(self._stand_pos[leg][2]) if (self._stand_pos is not None and leg in self._stand_pos) else 0.0
            self._swing_goal[leg] = pf

            new_pos[leg] = self._swing_traj(p0, pf, s_phase)
            if s_phase >= 1.0 - 1e-6:
                self._feet_world[leg] = pf.copy()
                self._swing_active[leg] = False

        contact_state = {leg: (leg in stance_legs) for leg in ALL_LEGS}
        return GaitPlan(
            in_cycle=self.in_cycle,
            stance_legs=stance_legs,
            swing_legs=swing_legs,
            contact_state=contact_state,
            target_pos=new_pos,
            target_ori=new_ori,
        )