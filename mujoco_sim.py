"""Lightweight MuJoCo simulation helper used by demos.
This version prefers the new `mujoco` Python bindings (mujoco>=2.3), and falls
back to `mujoco_py` when the new bindings are unavailable.

The helper exposes MuJoCoSim which loads an XML, creates a model/data and a
viewer, and provides `step_target(...)` to PD-control actuator motors toward a
desired q (base pos+quat, then joint targets). It builds a robust mapping from
actuator -> joint qpos/dof using either the model metadata or actuator name
convention (actuator names like `a_jointname`).
"""

from typing import Optional
import numpy as np

_HAS_NEW_MJ = False
_HAS_OLD_MJ = False
# To avoid importing `mujoco` at module import time (some builds start a
# viewer or create GL contexts on import), prefer the older `mujoco_py` if
# available. Import of the new `mujoco` bindings is done lazily inside the
# class only when explicitly requested.
try:
    from mujoco_py import load_model_from_path, MjSim, MjViewer
    _HAS_OLD_MJ = True
except Exception:
    _HAS_OLD_MJ = False

# Lazy holders for the new bindings if/when requested
mj = None
mjviewer = None


class MuJoCoSim:
    def __init__(self, xml_path: str, prefer_new: bool = False):
        self.available = False
        self._use_new = False
        self.model = None
        self.data = None
        self.sim = None
        self.viewer = None
        # actuator -> (qposadr, dofadr, joint_name)
        self.act_map = []
        # cached values to compute accelerations for a simple IMU
        self._prev_base_vel = None  # previous linear velocity (world frame)
        self._prev_time = None
        self._last_imu = None
        # If prefer_new is True, attempt to lazily import the new `mujoco` binding.
        if prefer_new:
            try:
                import mujoco as mj_local
                try:
                    from mujoco import viewer as mjviewer_local
                except Exception:
                    mjviewer_local = None
                # confirm the API looks correct
                if hasattr(mj_local, 'MjModel') and hasattr(mj_local, 'MjData'):
                    # bind to module-level variables for use in other methods
                    global mj, mjviewer
                    mj = mj_local
                    mjviewer = mjviewer_local
                    self._use_new = True
                    # load model & data
                    self.model = mj.MjModel.from_xml_path(xml_path)
                    self.data = mj.MjData(self.model)
                    self.viewer = None

                    # build actuator mapping
                    nu = self.model.nu
                    for a in range(nu):
                        try:
                            trnid = self.model.actuator_trnid[a]
                            joint_id = int(trnid[1])
                            qpos_adr = int(self.model.jnt_qposadr[joint_id])
                            dof_adr = int(self.model.jnt_dofadr[joint_id])
                            joint_name = ''
                        except Exception:
                            try:
                                act_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, a).decode()
                            except Exception:
                                act_name = ''
                            joint_name = act_name[2:] if act_name.startswith('a_') else act_name
                            try:
                                joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
                                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                                dof_adr = int(self.model.jnt_dofadr[joint_id])
                            except Exception:
                                qpos_adr = -1
                                dof_adr = -1

                        self.act_map.append((qpos_adr, dof_adr, joint_name))

                    self.available = True
                    return
            except Exception:
                # lazy new-binding init failed; fall back to mujoco_py
                self._use_new = False

        # Prefer mujoco_py (imported at module load) unless forced to use new binding
        if _HAS_OLD_MJ:
            try:
                model = load_model_from_path(xml_path)
                sim = MjSim(model)
            except Exception:
                # model load failed for old backend; try new binding as last resort
                if not prefer_new:
                    # try new binding once more
                    return self.__init__(xml_path, prefer_new=True)
                self.available = False
                return

            viewer = None
            try:
                viewer = MjViewer(sim)
            except Exception:
                viewer = None

            self.model = model
            self.sim = sim
            self.viewer = viewer

            # build actuator mapping using actuator_trnid or name convention
            nu = sim.model.nu
            for a in range(nu):
                try:
                    trn = sim.model.actuator_trnid[a]
                    # actuator_trnid stores target id(s); in mujoco_py the first
                    # element is the joint id for motor actuators (second may be -1).
                    joint_id = int(trn[0])
                    if joint_id >= 0:
                        qpos_adr = int(sim.model.jnt_qposadr[joint_id])
                        dof_adr = int(sim.model.jnt_dofadr[joint_id])
                        joint_name = sim.model.joint_names[joint_id]
                    else:
                        raise Exception('invalid joint id')
                except Exception:
                    try:
                        act_name = sim.model.actuator_names[a]
                    except Exception:
                        act_name = ''
                    joint_name = act_name[2:] if act_name.startswith('a_') else act_name
                    try:
                        joint_id = sim.model.joint_names.index(joint_name)
                        qpos_adr = int(sim.model.jnt_qposadr[joint_id])
                        dof_adr = int(sim.model.jnt_dofadr[joint_id])
                    except Exception:
                        qpos_adr = -1
                        dof_adr = -1

                self.act_map.append((qpos_adr, dof_adr, joint_name))

            self.sim = sim
            self.available = True
            return

        # neither backend available
        self.available = False

    def step_target(self, q_target: np.ndarray, kp: float = 40.0, kd: float = 2.0, steps: int = 1):
        """Drive actuators to follow q_target (base pos+quat followed by joints).

        This method works with either the new `mujoco` binding or `mujoco_py`.
        """
        if not self.available:
            return

        q_target = np.asarray(q_target).flatten()

        if self._use_new:
            model = self.model
            data = self.data
            # desired qpos
            desired_qpos = np.zeros(model.nq)
            n_base = min(7, q_target.size)
            desired_qpos[0:n_base] = q_target[0:n_base]
            if q_target.size > n_base:
                n_j = min(model.nq - n_base, q_target.size - n_base)
                desired_qpos[n_base:n_base + n_j] = q_target[n_base:n_base + n_j]

            # compute torques per actuator
            for a, (qpos_adr, dof_adr, joint_name) in enumerate(self.act_map):
                if qpos_adr >= 0 and dof_adr >= 0 and dof_adr < model.nv:
                    qpos_val = float(data.qpos[qpos_adr])
                    qvel_val = float(data.qvel[dof_adr])
                    # desired joint q is at desired_qpos[qpos_adr]
                    q_des = float(desired_qpos[qpos_adr])
                    torque = kp * (q_des - qpos_val) - kd * qvel_val 
                    data.ctrl[a] = float(torque)
                else:
                    data.ctrl[a] = 0.0

            data.ctrl += data.qfrc_gravcomp  # add gravity compensation
            # step
            for _ in range(steps):
                mj.mj_step(model, data)

            # render / sync viewer if available
            try:
                if self.viewer is not None:
                    self.viewer.sync()
            except Exception:
                pass

        else:
            # mujoco_py backend
            sim = self.sim
            model = sim.model
            data = sim.data
            desired_qpos = np.zeros(model.nq)
            n_base = min(7, q_target.size)
            desired_qpos[0:n_base] = q_target[0:n_base]
            if q_target.size > n_base:
                n_j = min(model.nq - n_base, q_target.size - n_base)
                desired_qpos[n_base:n_base + n_j] = q_target[n_base:n_base + n_j]

            nu = model.nu
            ctrl = np.zeros(nu, dtype=np.float64)
            for a, (qpos_adr, dof_adr, joint_name) in enumerate(self.act_map):
                if qpos_adr >= 0 and dof_adr >= 0 and dof_adr < model.nv:
                    qpos_val = float(data.qpos[qpos_adr])
                    qvel_val = float(data.qvel[dof_adr])
                    q_des = float(desired_qpos[qpos_adr])
                    ctrl[a] = kp * (q_des - qpos_val) - kd * qvel_val
                else:
                    ctrl[a] = 0.0

            try:
                data.ctrl[:] = ctrl
            except Exception:
                for i in range(len(ctrl)):
                    data.ctrl[i] = float(ctrl[i])

            for _ in range(steps):
                sim.step()

            try:
                if self.viewer is not None:
                    self.viewer.render()
            except Exception:
                pass

    def get_qpos(self):
        if not self.available:
            return None

        current_qpos = np.zeros(12)

        if self._use_new:
            model = self.model
            data = self.data
            for a, (qpos_adr, dof_adr, joint_name) in enumerate(self.act_map):
                if qpos_adr >= 0 and dof_adr >= 0 and dof_adr < model.nv:
                    current_qpos[a] = float(data.qpos[a])
                else:
                    current_qpos[a] = 0.0

        else:
            # mujoco_py backend
            sim = self.sim
            model = sim.model
            data = sim.data
            for a, (qpos_adr, dof_adr, joint_name) in enumerate(self.act_map):
                if qpos_adr >= 0 and dof_adr >= 0 and dof_adr < model.nv:
                    current_qpos[a] = float(data.qpos[a])
                else:
                    current_qpos[a] = 0.0

        return current_qpos

    # ------------------ IMU / base-state helpers ------------------
    def get_base_pos_quat(self):
        """Return base position (3,) and quaternion (4,) in MuJoCo qpos ordering.

        Returns (pos, quat) or (None, None) if unavailable.
        """
        if not self.available:
            return None, None

        if self._use_new:
            data = self.data
            pos = np.array(data.qpos[0:3], dtype=float)
            quat = np.array(data.qpos[3:7], dtype=float)
        else:
            sim = self.sim
            data = sim.data
            pos = np.array(data.qpos[0:3], dtype=float)
            quat = np.array(data.qpos[3:7], dtype=float)

        return pos, quat

    def get_base_velocity(self):
        """Return base linear velocity (3,) and angular velocity (3,).

        Linear velocity is in world frame. Angular velocity is in body/world
        consistent frame provided by MuJoCo (returned as-is from qvel).
        """
        if not self.available:
            return None, None

        if self._use_new:
            data = self.data
            lin = np.array(data.qvel[0:3], dtype=float)
            ang = np.array(data.qvel[3:6], dtype=float)
        else:
            data = self.sim.data
            lin = np.array(data.qvel[0:3], dtype=float)
            ang = np.array(data.qvel[3:6], dtype=float)

        return lin, ang

    def _quat_to_euler(self, q):
        """Convert quaternion (w,x,y,z) to roll, pitch, yaw (radians).

        Uses the standard Tait-Bryan ZYX convention (yaw-pitch-roll).
        """
        w, x, y, z = q
        # roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch (y-axis)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * (np.pi / 2)
        else:
            pitch = np.arcsin(sinp)

        # yaw (z-axis)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _quat_to_rotmat(self, q):
        """Convert quaternion (w,x,y,z) to 3x3 rotation matrix from body->world.
        """
        w, x, y, z = q
        # normalize
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n == 0:
            return np.eye(3)
        w /= n; x /= n; y /= n; z /= n
        R = np.zeros((3, 3), dtype=float)
        R[0, 0] = 1 - 2 * (y * y + z * z)
        R[0, 1] = 2 * (x * y - z * w)
        R[0, 2] = 2 * (x * z + y * w)
        R[1, 0] = 2 * (x * y + z * w)
        R[1, 1] = 1 - 2 * (x * x + z * z)
        R[1, 2] = 2 * (y * z - x * w)
        R[2, 0] = 2 * (x * z - y * w)
        R[2, 1] = 2 * (y * z + x * w)
        R[2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def sample_imu(self):
        """Compute a simple IMU reading from the simulator state.

        Returns a dict with keys:
          - 'accel': 3-array, linear acceleration in body frame (m/s^2), gravity compensated
          - 'gyro': 3-array, angular velocity (rad/s)
          - 'pitch', 'roll', 'yaw'

        This uses finite differences on base linear velocity to estimate
        acceleration. It's a lightweight approximation suitable for demos.
        """
        if not self.available:
            return None

        # get time, vel, quat
        if self._use_new:
            data = self.data
        else:
            data = self.sim.data

        t = float(getattr(data, 'time', 0.0))
        lin_vel, ang_vel = self.get_base_velocity()
        pos, quat = self.get_base_pos_quat()

        # compute dt
        if self._prev_time is None:
            dt = None
        else:
            dt = max(1e-8, t - self._prev_time)

        # estimate linear acceleration in world frame
        if dt is None or self._prev_base_vel is None:
            lin_acc_world = np.zeros(3, dtype=float)
        else:
            lin_acc_world = (lin_vel - self._prev_base_vel) / dt

        # gravity in world frame
        g_world = np.array([0.0, 0.0, -9.81], dtype=float)

        # rotation matrix body->world
        R_bw = self._quat_to_rotmat(quat)
        # specific force measured by IMU (body frame): R^T * (lin_acc_world - g_world)
        specific_force_body = R_bw.T.dot(lin_acc_world - g_world)

        # cache values
        self._prev_base_vel = lin_vel.copy()
        self._prev_time = t
        roll, pitch, yaw = self._quat_to_euler(quat)
        # convert roll,pitch,yaw (ZYX) back to quaternion and store as x,y,z,w
        # roll = phi, pitch = theta, yaw = psi
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w_q = cr * cp * cy + sr * sp * sy
        x_q = sr * cp * cy - cr * sp * sy
        y_q = cr * sp * cy + sr * cp * sy
        z_q = cr * cp * sy - sr * sp * cy

        imu = {
            'accel': specific_force_body,
            'gyro': np.array(ang_vel, dtype=float),
            'pitch': float(pitch),
            'roll': float(roll),
            'yaw': float(yaw),
            'time': t,
            'x': float(x_q),
            'y': float(y_q),
            'z': float(z_q),
            'w': float(w_q),
        }
        self._last_imu = imu
        return imu

    def get_imu(self):
        """Return last IMU sample; if not sampled yet, produce one.
        """
        if self._last_imu is None:
            return self.sample_imu()
        return self._last_imu

    def start_viewer(self) -> bool:
        """Start or attach a viewer for the current backend. Returns True if a
        viewer was successfully started/attached.
        """
        # new mujoco bindings
        if self._use_new and mj is not None:
            if mjviewer is None:
                return False
            try:
                # launch passive viewer (non-blocking)
                self.viewer = mjviewer.launch_passive(self.model, self.data)
                return True
            except Exception:
                return False

        # mujoco_py backend
        if not self._use_new and getattr(self, 'sim', None) is not None:
            try:
                # create a viewer if not present
                if self.viewer is None:
                    self.viewer = MjViewer(self.sim)
                return True
            except Exception:
                return False

        return False

    def stop_viewer(self):
        """Attempt to cleanly stop/close any started viewer."""
        try:
            if self.viewer is None:
                return
            # try common close methods
            try:
                self.viewer.close()
                return
            except Exception:
                pass
            try:
                self.viewer.finish()
                return
            except Exception:
                pass
        except Exception:
            pass


if __name__ == '__main__':
    print('MuJoCoSim loaded. new mujoco:', _HAS_NEW_MJ, 'mujoco_py:', _HAS_OLD_MJ)
