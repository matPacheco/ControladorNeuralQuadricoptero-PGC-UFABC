import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["GALLIUM_DRIVER"] = "llvmpipe"

import pybullet as p
p.connect(p.GUI)
p.loadURDF("plane.urdf")
input("Pressione Enter para sair...")