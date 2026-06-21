#!/usr/bin/env python3
"""M3 (embodied): TWO humanoids stand together in ReplicaCAD and have a logged
conversation. Bridges the text society (mind/society.py) to the embodied world:
two NPCs co-present in a real scene, dialogue driven by DeepSeek and written to
Neo4j, rendered as a two-shot. Headless on V100.

Run in lifeworld-habitat: --network host (Neo4j+DeepSeek) --gpus device=N
-e DEEPSEEK_API_KEY -e NEO4J_PASSWORD, /data + repo mounted.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, "/work")
import habitat_sim
import magnum as mn
from omegaconf import DictConfig
from habitat_sim.utils.common import quat_from_two_vectors
from habitat.articulated_agents.humanoids import kinematic_humanoid
from habitat.articulated_agent_controllers.humanoid_rearrange_controller import (
    HumanoidRearrangeController,
)
from mind.brain import decide
from memory.graph import Memory

CAST = [
    {"name": "Mara", "urdf": "female_0", "persona": "tidy, anxious, protective of her food"},
    {"name": "Theo", "urdf": "male_0", "persona": "easygoing, forgetful musician, conflict-averse"},
]
SCENE = "Mara just found her labeled leftovers missing from the fridge; Theo wanders in."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dataset", required=True)
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--humanoids-root", required=True)
    ap.add_argument("--out", default="/work/output/social_twoshot.png")
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--res", type=int, nargs=2, default=[600, 800])
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
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb]
    return habitat_sim.Configuration(bk, [ag])


def main():
    a = parse_args()
    from PIL import Image
    sim = habitat_sim.Simulator(make_cfg(a))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.3; ns.agent_height = 1.4
    sim.recompute_navmesh(sim.pathfinder, ns)
    pf = sim.pathfinder

    # two spots ~1.6 m apart at the same floor height
    p0 = np.array(pf.snap_point(pf.get_random_navigable_point()))
    p1 = None
    for _ in range(300):
        c = np.array(pf.get_random_navigable_point())
        if 1.3 < np.linalg.norm((c - p0)[[0, 2]]) < 1.9 and abs(c[1] - p0[1]) < 0.1:
            p1 = np.array(pf.snap_point(c)); break
    if p1 is None:
        p1 = p0 + np.array([1.6, 0, 0])

    def face_yaw(frm, to):
        d = to - frm
        return float(np.arctan2(d[0], d[2]))   # habitat base_rot about Y

    hums = []
    for cast, pos, look in ((CAST[0], p0, p1), (CAST[1], p1, p0)):
        urdf = f"{a.humanoids_root}/{cast['urdf']}/{cast['urdf']}.urdf"
        motion = f"{a.humanoids_root}/{cast['urdf']}/{cast['urdf']}_motion_data_smplx.pkl"
        cfg = DictConfig({"articulated_agent_urdf": urdf, "motion_data_path": motion})
        h = kinematic_humanoid.KinematicHumanoid(cfg, sim); h.reconfigure(); h.update()
        h.base_pos = mn.Vector3(*pos.tolist())
        try:
            h.base_rot = face_yaw(pos, look)
        except Exception as e:
            print("base_rot warn:", e)
        # nudge out of T-pose into a natural stance via one controller pose
        try:
            ctrl = HumanoidRearrangeController(motion)
            ctrl.reset(h.base_transformation)
            ctrl.calculate_walk_pose(mn.Vector3(*(look - pos).tolist()))
            pose = ctrl.get_pose()
            joints, base, off = pose[:-16], pose[-16:], pose[-32:-16]
            if np.array(off).sum() != 0:
                vb = [mn.Vector4(base[i*4:(i+1)*4]) for i in range(4)]
                vo = [mn.Vector4(off[i*4:(i+1)*4]) for i in range(4)]
                h.set_joint_transform(joints, mn.Matrix4(*vo), mn.Matrix4(*vb))
        except Exception as e:
            print("pose warn:", e)
        hums.append(h)

    # camera: perpendicular to the pair, framing both at chest height
    mid = (p0 + p1) / 2
    line = (p1 - p0); line[1] = 0; line /= (np.linalg.norm(line) + 1e-9)
    perp = np.array([-line[2], 0, line[0]])
    eye = None; best = -1
    for s in (1, -1):
        c = mid + perp * s * 2.6
        if pf.is_navigable(mn.Vector3(*c.tolist())):
            d = 2.6
            if d > best:
                best, eye = d, c
    if eye is None:
        eye = mid + perp * 2.6
    eye = eye + np.array([0, 1.45, 0])
    look = mid + np.array([0, 0.9, 0])
    dd = look - eye; dd /= (np.linalg.norm(dd) + 1e-9)
    st = sim.get_agent(0).get_state()
    st.position = eye.astype(np.float32)
    st.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), dd.astype(np.float64))
    sim.get_agent(0).set_state(st)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    Image.fromarray(rgb).save(a.out)
    print(f"TWOSHOT saved {a.out} p0={p0.round(2).tolist()} p1={p1.round(2).tolist()}")

    # dialogue between exactly these two, logged to Neo4j
    mem = Memory()
    for c in CAST:
        mem.ensure_agent(c["name"])
    with mem.drv.session() as s:
        s.run("MATCH (u:Utterance {scene:'social'}) DETACH DELETE u")
    transcript = []
    for t in range(a.turns):
        spk = CAST[t % 2]; other = CAST[(t + 1) % 2]
        convo = "\n".join(f"{x['who']}: {x['say']}" for x in transcript[-6:])
        out = decide(
            f"You are {spk['name']}: {spk['persona']}. You're face to face with "
            f"{other['name']} in your apartment. Brief, in-character. Reply ONLY JSON.",
            f"SCENE: {SCENE}\nSo far:\n{convo or '(start)'}\nYour turn. "
            '{"say":"<line>","do":"<action>","feeling":"<one word toward them>",'
            '"sentiment":"<positive|neutral|negative>"}',
            temperature=0.9)
        transcript.append({"who": spk["name"], "say": out.get("say", "")})
        mem.add_utterance(spk["name"], t, out.get("say", ""), out.get("do", ""), scene="social")
        mem.set_feeling(spk["name"], other["name"], out.get("feeling", ""),
                        out.get("sentiment", "neutral"))
        print(f"  {spk['name']}: \"{out.get('say','')}\"  [{out.get('do','')}]")
    mem.close()
    sim.close()
    print(f"SOCIAL_OK turns={a.turns}")


if __name__ == "__main__":
    main()
