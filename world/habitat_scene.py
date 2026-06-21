#!/usr/bin/env python3
"""M1: load a ReplicaCAD scene in Habitat, drop a humanoid on the navmesh, point a
camera at it, save an RGB frame. Proves a humanoid renders inside a real scene on
the V100 (headless EGL). ADR-0001 / issue #1.

Run in lifeworld-habitat with --gpus and /data mounted:
    python3 world/habitat_scene.py \
        --scene-dataset /data/.../replicaCAD.scene_dataset_config.json \
        --scene apt_0 [--humanoid-urdf /data/.../female_0/female_0.urdf] \
        --out /work/output/habitat_scene.png
"""
import argparse
import sys
import numpy as np

try:
    import habitat_sim
    import magnum as mn
except Exception as e:
    print("IMPORT_FAIL:", e); sys.exit(2)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dataset", required=True)
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--humanoid-urdf", default=None)
    ap.add_argument("--out", default="/work/output/habitat_scene.png")
    ap.add_argument("--res", type=int, nargs=2, default=[720, 960])
    return ap.parse_args()


def make_cfg(a):
    bk = habitat_sim.SimulatorConfiguration()
    bk.gpu_device_id = 0
    bk.scene_dataset_config_file = a.scene_dataset
    bk.scene_id = a.scene
    bk.enable_physics = True
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = a.res; rgb.position = [0.0, 1.4, 0.0]
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [rgb]
    return habitat_sim.Configuration(bk, [ag])


def look_at(eye, target):
    """agent rotation (about Y) so the sensor faces target, + state position."""
    d = np.array(target) - np.array(eye)
    yaw = np.arctan2(-d[0], -d[2])     # habitat: -Z forward
    return yaw


def main():
    a = parse_args()
    try:
        sim = habitat_sim.Simulator(make_cfg(a))
    except Exception as e:
        import traceback; traceback.print_exc(); print("SCENE_FAIL:", e); sys.exit(3)

    pf = sim.pathfinder
    if not pf.is_loaded:
        print("WARN: no navmesh; using origin")
        spot = np.array([0.0, 0.0, 0.0])
    else:
        spot = np.array(pf.get_random_navigable_point())

    placed = "none"
    if a.humanoid_urdf:
        try:
            aom = sim.get_articulated_object_manager()
            hum = aom.add_articulated_object_from_urdf(a.humanoid_urdf, fixed_base=True)
            hum.translation = mn.Vector3(float(spot[0]), float(spot[1]), float(spot[2]))
            placed = f"urdf links={hum.num_links}"
        except Exception as e:
            placed = f"FAILED ({e})"

    # camera 2.6 m from the spot, eye height 1.5, looking at chest height
    eye = spot + np.array([2.6, 1.5, 0.0])
    yaw = look_at(eye, spot + np.array([0, 0.9, 0]))
    agent = sim.get_agent(0)
    st = agent.get_state()
    st.position = eye.astype(np.float32)
    st.rotation = mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3(0, 1, 0))
    agent.set_state(st)

    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["rgb"])[..., :3]
    try:
        from PIL import Image
        Image.fromarray(rgb).save(a.out)
    except Exception as e:
        print("save warn:", e)
    print(f"SCENE_OK scene={a.scene} spot={spot.round(2).tolist()} humanoid={placed} "
          f"rgb={rgb.shape} -> {a.out}")
    sim.close()


if __name__ == "__main__":
    main()
