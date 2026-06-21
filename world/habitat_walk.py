#!/usr/bin/env python3
"""M1+: a skinned humanoid WALKS across a ReplicaCAD scene, recorded to mp4.

Uses habitat-lab's KinematicHumanoid + HumanoidRearrangeController (the same path
as test_humanoid_controller) to generate a walk cycle toward a navmesh target,
captured from a fixed camera. Proves the navigate-to action primitive the M2 mind
loop will call. Headless on V100.
"""
import argparse
import os
import sys
import numpy as np

try:
    import habitat_sim
    import magnum as mn
    from omegaconf import DictConfig
    from habitat.articulated_agents.humanoids import kinematic_humanoid
    from habitat.articulated_agent_controllers.humanoid_rearrange_controller import (
        HumanoidRearrangeController,
    )
except Exception as e:
    print("IMPORT_FAIL:", e); sys.exit(2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dataset", required=True)
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--frames-dir", default="/work/output/frames_walk_hab")
    ap.add_argument("--res", type=int, nargs=2, default=[720, 960])
    ap.add_argument("--max-steps", type=int, default=180)
    return ap.parse_args()


def make_cfg(a):
    bk = habitat_sim.SimulatorConfiguration()
    bk.gpu_device_id = 0
    bk.scene_dataset_config_file = a.scene_dataset
    bk.scene_id = a.scene
    bk.enable_physics = True
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = a.res; rgb.position = [0.0, 0.0, 0.0]
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [rgb]
    return habitat_sim.Configuration(bk, [ag])


def main():
    a = parse_args()
    os.makedirs(a.frames_dir, exist_ok=True)
    from PIL import Image
    from habitat_sim.utils.common import quat_from_two_vectors

    import habnav
    sim = habitat_sim.Simulator(make_cfg(a))
    pf = habnav.setup_navmesh(sim)            # navmesh that respects furniture

    # humanoid via habitat-lab wrapper (skinned via ao_config)
    agent_cfg = DictConfig({"articulated_agent_urdf": a.urdf, "motion_data_path": a.motion})
    hum = kinematic_humanoid.KinematicHumanoid(agent_cfg, sim)
    hum.reconfigure(); hum.update()

    # pick start + a target ~2.5 m away, both navigable
    start = np.array(pf.snap_point(pf.get_random_navigable_point()))
    target = None
    for _ in range(200):
        c = np.array(pf.get_random_navigable_point())
        if 2.0 < np.linalg.norm((c - start)[[0, 2]]) < 3.5 and abs(c[1] - start[1]) < 0.2:
            target = np.array(pf.snap_point(c)); break
    if target is None:
        target = start + np.array([2.0, 0, 0])
    hum.base_pos = mn.Vector3(*start.tolist())

    # camera vantage: a NAVIGABLE floor point ~3 m from the walk midpoint (floor
    # points are clear of furniture), raised to eye height. Re-aimed each frame to
    # track the walking humanoid so it stays framed regardless of layout.
    mid = (start + target) / 2.0
    eye = None
    best = -1
    for _ in range(300):
        c = np.array(pf.get_random_navigable_point())
        if abs(c[1] - mid[1]) > 0.2:
            continue
        dist = np.linalg.norm((c - mid)[[0, 2]])
        if 2.5 < dist < 4.0 and dist > best:
            best = dist; eye = c
    if eye is None:
        eye = mid + np.array([3.0, 0, 0])
    eye = eye + np.array([0, 1.5, 0])

    def aim_at(p):
        st = sim.get_agent(0).get_state()
        st.position = eye.astype(np.float32)
        st.rotation = habnav.look_at_rot(eye, p)       # roll-free
        sim.get_agent(0).set_state(st)

    ctrl = HumanoidRearrangeController(a.motion)
    fc = [0]

    def on_frame():
        aim_at(np.array(hum.base_pos) + np.array([0, 0.85, 0]))   # track torso
        rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
        Image.fromarray(rgb).save(os.path.join(a.frames_dir, f"frame_{fc[0]:04d}.png"))
        fc[0] += 1

    # walk ALONG the navmesh path (routes around furniture), not straight at target
    n = habnav.walk_path(hum, ctrl, pf, target, on_frame=on_frame, max_steps=a.max_steps)

    print(f"WALK_OK steps={n} start={start.round(2).tolist()} target={target.round(2).tolist()} "
          f"end={np.array(hum.base_pos).round(2).tolist()} frames={a.frames_dir}")
    sim.close()


if __name__ == "__main__":
    main()
