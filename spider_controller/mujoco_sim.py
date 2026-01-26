import mujoco
import mujoco_viewer
import time
import traceback
import numpy as np
import atexit

# Import shared memory data structures
from shared_sim_data import SimToCPGData, CPGToSimData, cleanup_shared_memory
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

mujoco.mj_forward(model, data)  # compute forward kinematics to get initial state

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
while True:
    if viewer.is_alive:
        viewer.add_data_to_line(line_name="root-pos-x",
                            line_data=data.qpos[0], fig_idx=0)
        viewer.add_data_to_line(line_name="root-pos-z",
                            line_data=data.qpos[2], fig_idx=0)
        try:
            for a in range(model.nu):
                viewer.add_data_to_line(line_name=f"data.ctrl[{a}]",
                                line_data=data.ctrl[a], fig_idx=1)
        except Exception as e:
            print(f"Warning: Failed to add data to line: {e}", flush=True)

        try:
            qpos_desired, kp, kd, cpg_timestamp = cpg_to_sim.read()
        except Exception as e:
            # If CPG not ready yet, use initial position with default gains
            qpos_desired = initial_qpos
            kp = np.ones(12) * 0.0
            kd = np.ones(12) * 0.0
            cpg_timestamp = 0.0

        for a in range(model.nu):
            data.ctrl[a] = kp[a] * (qpos_desired[model.nq - model.nu + a] - data.qpos[model.nq - model.nu + a]) - kd[a] * data.qvel[model.nv - model.nu + a]
        mujoco.mj_step(model, data)
        viewer.render()

        try:
            sim_to_cpg.write(data.qpos, data.ctrl)
        except Exception as e:
            print(f"Warning: Failed to write to shared memory: {e}", flush=True)
    else:
        break

# close
viewer.close()