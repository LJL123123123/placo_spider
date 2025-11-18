#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os
import math
import numpy as np

model = load_model_from_path("/home/placo_cpg/models/quadruped/scene.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
# TextureModder requires model materials/textures; some converted XMLs have no
# <material>/<texture> entries so model.mat_texid can be None which triggers
# TypeError inside TextureModder. Create modder defensively and skip texture
# operations when unavailable.
modder = None
try:
    modder = TextureModder(sim)
except Exception as e:
    # Don't fail the whole program for missing materials/textures.
    print('TextureModder unavailable, continuing without texture randomization:', e)

t = 0
# Simple physics + joint control loop
# This will step the physics and write control signals to all actuators
# Currently uses a sinusoidal open-loop torque/force signal per actuator.
dt = sim.model.opt.timestep if hasattr(sim.model.opt, 'timestep') else 0.001
# number of control inputs (actuators)
nu = sim.model.nu

# PD control parameters (can be tuned)
KP = 40.0
KD = 2.0
amp = 0.3  # amplitude for small reference motion around initial pose (rad)
freq = 0.6

# Build actuator -> joint / qpos/qvel mapping where possible
act_to_qpos = [-1] * nu
act_to_dof = [-1] * nu
initial_qpos = [0.0] * nu
for a in range(nu):
    try:
        # actuator_trnid gives (type, id) where id is the target id (joint id in many cases)
        trn = sim.model.actuator_trnid[a]
        # second entry is the id
        joint_id = int(trn[1])
        # qpos and dof addresses for the joint
        qpos_adr = int(sim.model.jnt_qposadr[joint_id])
        dof_adr = int(sim.model.jnt_dofadr[joint_id])
        # check validity
        if qpos_adr >= 0:
            act_to_qpos[a] = qpos_adr
            act_to_dof[a] = dof_adr
            # read initial pose for that qpos adr (store as reference)
            try:
                initial_qpos[a] = float(sim.data.qpos[qpos_adr])
            except Exception:
                initial_qpos[a] = 0.0
    except Exception:
        # mapping unavailable: leave -1 so we fallback to open-loop
        act_to_qpos[a] = -1
        act_to_dof[a] = -1

try:
    while True:
        # advance simulation time for control
        t += dt

        # prepare control array
        ctrl = np.zeros(nu, dtype=np.float64)

        for a in range(nu):
            if act_to_qpos[a] >= 0 and act_to_dof[a] >= 0:
                qpos_val = float(sim.data.qpos[act_to_qpos[a]])
                qvel_val = float(sim.data.qvel[act_to_dof[a]]) if act_to_dof[a] >= 0 else 0.0
                # small sinusoidal reference around initial position
                ref = initial_qpos[a] + amp * math.sin(2.0 * math.pi * freq * t + a * 0.3)
                err = ref - qpos_val
                # PD torque (kp * error + kd * (ref_vel - qvel)) ; ref_vel assumed 0 for simple tracking
                torque = KP * err - KD * qvel_val
                ctrl[a] = torque
            else:
                # fallback: keep previous open-loop sinusoid if mapping missing
                ctrl[a] = amp * math.sin(2.0 * math.pi * freq * t + a * 0.3)

        # write controls and step
        try:
            sim.data.ctrl[:] = ctrl
        except Exception:
            # elementwise fallback
            for i in range(len(ctrl)):
                sim.data.ctrl[i] = float(ctrl[i])

        sim.step()

        # render when possible
        try:
            viewer.render()
        except Exception:
            pass

        # quick exit for automated tests
        if t > 0.1 and os.getenv('TESTING') is not None:
            break

except KeyboardInterrupt:
    print('Simulation interrupted by user')