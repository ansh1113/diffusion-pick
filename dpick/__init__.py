"""diffusion-pick: imitation learning for 'put the red cube in the bowl' with a Franka Panda."""
import os

# Headless rendering by default (must be set before mujoco is imported).
os.environ.setdefault("MUJOCO_GL", "egl")
