import mujoco
import mujoco_viewer
import time
import traceback
import numpy as np
import atexit

# Import shared memory data structures
from shared_sim_data import SimToCPGData, CPGToSimData, cleanup_shared_memory

# module-level caches for storing Python-side metadata about C-backed MjModel
_MODEL_ACT_TO_DOF = {}
_MODEL_GRAV_ERRORS = {}
_MODEL_OUTER_WARNED = set()

try:
    from mujoco_py import load_model_from_path, MjSim, MjViewer
    _HAS_OLD_MJ = True
except Exception:
    _HAS_OLD_MJ = False

xml_path = '/home/placo_cpg/spider_sldasm/urdf/scene.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initialize shared memory (MuJoCo is the producer of sim data, consumer of CPG commands)
sim_to_cpg = SimToCPGData(create=True)
cpg_to_sim = CPGToSimData(create=True)  # Create both in mujoco_sim for simplicity

# Register cleanup on exit
atexit.register(lambda: sim_to_cpg.close())
atexit.register(lambda: sim_to_cpg.unlink())
atexit.register(lambda: cpg_to_sim.close())
# Don't unlink cpg_to_sim here, let CPG do it

print("MuJoCo Sim: Shared memory initialized", flush=True)

# Store initial joint positions for PD control
mujoco.mj_forward(model, data)  # compute forward kinematics to get initial state
initial_qpos = data.qpos.copy()
print(f"Initial qpos: {initial_qpos}", flush=True)

# implement actuator-bias callback: fill per-actuator bias from joint-level qfrc_bias
def _py_actuator_bias(m, d, act_bias):
    """MuJoCo callback to fill actuator-level bias array (act_bias).

    act_bias is a writable array of length m.nu provided by MuJoCo.
    We'll map each actuator to a joint DOF (using the same mapping logic) and
    copy the joint-level bias (d.qfrc_bias[dof]) into act_bias[a].
    """
    mid = id(m)
    # ensure we have an actuator->dof mapping cached
    if mid not in _MODEL_ACT_TO_DOF:
        nu = m.nu
        act_to_dof = [-1] * nu
        for a in range(nu):
            try:
                trnid = m.actuator_trnid[a]
                joint_id = -1
                try:
                    for v in trnid:
                        if int(v) >= 0:
                            joint_id = int(v)
                            break
                except Exception:
                    try:
                        if int(trnid) >= 0:
                            joint_id = int(trnid)
                    except Exception:
                        joint_id = -1

                if joint_id >= 0:
                    dof_adr = int(m.jnt_dofadr[joint_id])
                    act_to_dof[a] = dof_adr
                else:
                    raise Exception('no valid joint id in actuator_trnid')
            except Exception:
                try:
                    act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a).decode()
                except Exception:
                    act_name = ''
                joint_name = act_name[2:] if act_name.startswith('a_') else act_name
                try:
                    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    dof_adr = int(m.jnt_dofadr[joint_id])
                    act_to_dof[a] = dof_adr
                except Exception:
                    act_to_dof[a] = -1

        _MODEL_ACT_TO_DOF[mid] = act_to_dof

    act_map = _MODEL_ACT_TO_DOF.get(mid, [])
    # fill act_bias array
    for a, dof in enumerate(act_map):
        try:
            if dof is not None and 0 <= dof < m.nv:
                act_bias[a] = float(d.qfrc_bias[dof])
            else:
                act_bias[a] = 0.0
        except Exception:
            # if anything goes wrong, set zero
            try:
                act_bias[a] = 0.0
            except Exception:
                pass

# register the callback so MuJoCo calls it during mj_step
mujoco.set_mjcb_act_bias(_py_actuator_bias)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

viewer.add_line_to_fig(line_name="root-pos-x", fig_idx=0)
viewer.add_line_to_fig(line_name="root-pos-z", fig_idx=0)
# register lines for actuator controls (must add line names before adding data)
for a in range(model.nu):
    viewer.add_line_to_fig(line_name=f"data.ctrl[{a}]", fig_idx=1)

# user has access to mjvFigure
fig = viewer.figs[0]
fig.title = "Root Position"
fig.flg_legend = True
fig.xlabel = "Timesteps"
fig.figurergba[0] = 0.2
fig.figurergba[3] = 0.2
fig.gridsize[0] = 5
fig.gridsize[1] = 5

fig = viewer.figs[1]
fig.title = "Joint position"
fig.flg_legend = True
fig.figurergba[0] = 0.2
fig.figurergba[3] = 0.2

# simulate and render
dt = 1.0 / 500.0  # 500 Hz
next_time = time.perf_counter()

while True:
    # sleep until the next period (sleep most of the time, then busy-wait)
    now = time.perf_counter()
    if next_time > now:
        remaining = next_time - now
        if remaining > 0.001:
            time.sleep(remaining - 0.0005)
        while time.perf_counter() < next_time:
            pass

    viewer.add_data_to_line(line_name="root-pos-x",
                            line_data=data.qpos[0], fig_idx=0)
    viewer.add_data_to_line(line_name="root-pos-z",
                            line_data=data.qpos[2], fig_idx=0)

    # Read desired state and PD gains from CPG planner
    try:
        qpos_desired, kp, kd, cpg_timestamp = cpg_to_sim.read()
    except Exception as e:
        # If CPG not ready yet, use initial position with default gains
        qpos_desired = initial_qpos
        kp = np.ones(12) * 0.0
        kd = np.ones(12) * 0.0
        cpg_timestamp = 0.0

    # Get actuator->dof mapping (built by callback on first iteration)
    mid = id(model)
    act_map = _MODEL_ACT_TO_DOF.get(mid, [])
    
    # Build mapping if not yet available (first iteration)
    if not act_map:
        # Trigger the callback to build the mapping
        dummy_bias = np.zeros(model.nu)
        _py_actuator_bias(model, data, dummy_bias)
        act_map = _MODEL_ACT_TO_DOF.get(mid, [])

    # PD Controller: compute desired torques for actuated joints
    # data.ctrl = kp * (qpos_desired - qpos) - kd * qvel
    for a in range(model.nu):
        dof = act_map[a] if a < len(act_map) else -1
        # Only control actuated joints (dof >= 6), not the free joint (dof 0-5)
        if dof is not None and 6 <= dof < model.nv:
            q_err = qpos_desired[dof] - data.qpos[dof]
            qd = data.qvel[dof]
            data.ctrl[a] = kp[a] * q_err - kd[a] * qd
        else:
            data.ctrl[a] = 0.0

    # Write current state to shared memory for CPG to read
    try:
        sim_to_cpg.write(data.qpos, data.ctrl)
    except Exception as e:
        print(f"Warning: Failed to write to shared memory: {e}", flush=True)

    try:
        for a in range(model.nu):
            viewer.add_data_to_line(line_name=f"data.ctrl[{a}]",
                            line_data=data.ctrl[a], fig_idx=1)
    except Exception:
        pass

    # Note: Gravity compensation is now handled by the actuator-bias callback
    # (_py_actuator_bias) registered via mujoco.set_mjcb_act_bias().
    # MuJoCo calls this callback internally during mj_step to populate actuator bias.
    # The final actuator force = ctrl + bias (where bias comes from our callback).

    # step the simulation
    mujoco.mj_step(model, data)
    viewer.render()
    if not viewer.is_alive:
        break
    # Print status every 100 steps (0.2 seconds at 500Hz)
    if int(data.time / dt) % 100 == 0:
        print(f't={data.time:.2f}s, qpos[2]={data.qpos[2]:.4f}, ctrl_rms={np.sqrt(np.mean(data.ctrl**2)):.2f}', flush=True)
    next_time += dt

# close
viewer.close()
