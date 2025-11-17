#!/usr/bin/env python3
"""
Displays robot fetch at a disco party.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os

model = load_model_from_path("/home/placo_cpg/models/quadruped/robot_mjcf.xml")
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

while True:
    # Randomize textures only if modder was successfully created.
    # for name in sim.model.geom_names:
    #     if modder is not None:
    #         modder.rand_all(name)

    viewer.render()
    t += 1
    if t > 100 and os.getenv('TESTING') is not None:
        break