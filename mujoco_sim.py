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
