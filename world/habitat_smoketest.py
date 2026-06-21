#!/usr/bin/env python3
"""Habitat-Sim headless smoke test on sm_70 — M1 / ADR-0001.

Confirms habitat-sim initializes its EGL/CUDA renderer on a V100 and produces
sensor frames. Uses a test scene if available; otherwise an empty stage. Saves an
RGB frame. Exit 0 = the simulator runs headless on this card.
"""
import os
import sys

import numpy as np

try:
    import habitat_sim
except Exception as e:
    print("IMPORT_FAIL:", e); sys.exit(2)


def make_cfg(scene):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    rgb.position = [0.0, 1.5, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main():
    # "NONE" stage = empty world; always available, proves the renderer inits.
    scene = os.environ.get("HABITAT_TEST_SCENE", "NONE")
    try:
        sim = habitat_sim.Simulator(make_cfg(scene))
    except Exception as e:
        import traceback; traceback.print_exc()
        print("SIM_INIT_FAIL:", e); sys.exit(3)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])
    print(f"HABITAT_OK scene={scene} rgb={rgb.shape} dtype={rgb.dtype} "
          f"backend_ok={rgb.size > 0}")
    try:
        from PIL import Image
        Image.fromarray(rgb[..., :3]).save("/work/output/habitat_rgb.png")
        print("saved /work/output/habitat_rgb.png")
    except Exception:
        pass
    sim.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
