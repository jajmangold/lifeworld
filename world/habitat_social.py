#!/usr/bin/env python3
"""M3 (embodied, multi-character): N humanoids stand in a coherent conversation
cluster in ReplicaCAD and talk to each other; dialogue logged to Neo4j, rendered
as a group shot. Each agent is placed on open floor facing the group's centre, so
the scene reads as people actually talking together (not scattered/clipping).

Run in lifeworld-habitat: --network host --gpus device=N
-e DEEPSEEK_API_KEY -e NEO4J_PASSWORD, /data + repo mounted.
"""
import argparse
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
import habnav
from mind.brain import decide
from memory.graph import Memory

CAST = [
    {"name": "Mara", "urdf": "female_0", "persona": "tidy, anxious, protective of her food"},
    {"name": "Theo", "urdf": "male_0", "persona": "easygoing, forgetful musician, conflict-averse"},
    {"name": "Priya", "urdf": "female_1", "persona": "blunt, funny, the peacemaker"},
]
SCENE = ("Saturday morning in the shared apartment. Mara just found her labeled "
         "leftovers gone from the fridge; the three roommates end up talking it out.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dataset", required=True)
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--humanoids-root", required=True)
    ap.add_argument("--out", default="/work/output/social_group.png")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--radius", type=float, default=0.95)
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
    rgb.resolution = a.res; rgb.position = [0.0, 0.0, 0.0]
    ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = [rgb]
    return habitat_sim.Configuration(bk, [ag])


def find_cluster(pf, n, radius):
    """A centre with `n` navigable, non-furniture member spots evenly around it."""
    angles = [2 * np.pi * i / n for i in range(n)]
    for _ in range(400):
        c = np.array(pf.snap_point(pf.get_random_navigable_point()))
        if not np.all(np.isfinite(c)):
            continue
        spots, ok = [], True
        for ang in angles:
            p = c + np.array([radius * np.cos(ang), 0, radius * np.sin(ang)])
            sp = np.array(pf.snap_point(mn.Vector3(*p.tolist())))
            if (not np.all(np.isfinite(sp)) or abs(sp[1] - c[1]) > 0.1
                    or np.linalg.norm((sp - p)[[0, 2]]) > 0.25):
                ok = False; break
            spots.append(sp)
        if ok:
            return c, spots
    return None, None


def main():
    a = parse_args()
    from PIL import Image
    sim = habitat_sim.Simulator(make_cfg(a))
    pf = habnav.setup_navmesh(sim)
    cast = CAST[:max(2, min(len(CAST), 3))]

    centre, spots = find_cluster(pf, len(cast), a.radius)
    if centre is None:
        print("CLUSTER_FAIL: no open space found"); sim.close(); return

    def yaw_to(frm, to):
        d = to - frm
        return float(np.arctan2(d[0], d[2]))

    for c, pos in zip(cast, spots):
        urdf = f"{a.humanoids_root}/{c['urdf']}/{c['urdf']}.urdf"
        motion = f"{a.humanoids_root}/{c['urdf']}/{c['urdf']}_motion_data_smplx.pkl"
        h = kinematic_humanoid.KinematicHumanoid(
            DictConfig({"articulated_agent_urdf": urdf, "motion_data_path": motion}), sim)
        h.reconfigure(); h.update()
        h.base_pos = mn.Vector3(*pos.tolist())
        try:
            h.base_rot = yaw_to(pos, centre)          # face the group centre
        except Exception as e:
            print("rot warn:", e)
        try:                                          # nudge out of T-pose
            ctrl = HumanoidRearrangeController(motion)
            ctrl.reset(h.base_transformation)
            ctrl.calculate_walk_pose(mn.Vector3(*((centre - pos) * 0.3).tolist()))
            pose = ctrl.get_pose()
            j, b, o = pose[:-16], pose[-16:], pose[-32:-16]
            if np.array(o).sum() != 0:
                vb = [mn.Vector4(b[i * 4:(i + 1) * 4]) for i in range(4)]
                vo = [mn.Vector4(o[i * 4:(i + 1) * 4]) for i in range(4)]
                h.set_joint_transform(j, mn.Matrix4(*vo), mn.Matrix4(*vb))
        except Exception as e:
            print("pose warn:", e)

    # camera: closest clear navigable spot ~2.2-3.0 m out, slight 3/4 downward, so the
    # group fills the frame and reads as a conversation
    eye = None; best = 1e9
    for _ in range(400):
        cand = np.array(pf.get_random_navigable_point())
        if abs(cand[1] - centre[1]) > 0.2:
            continue
        d = np.linalg.norm((cand - centre)[[0, 2]])
        if 2.2 < d < 3.0 and d < best:
            best, eye = d, cand
    if eye is None:
        eye = centre + np.array([2.5, 0, 0])
    eye = eye + np.array([0, 1.65, 0])                 # slightly above head height
    look = centre + np.array([0, 0.95, 0])
    dd = look - eye; dd /= (np.linalg.norm(dd) + 1e-9)
    st = sim.get_agent(0).get_state()
    st.position = eye.astype(np.float32)
    st.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), dd.astype(np.float64))
    sim.get_agent(0).set_state(st)
    rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    Image.fromarray(rgb).save(a.out)
    print(f"GROUP saved {a.out} centre={centre.round(2).tolist()} n={len(cast)}")

    # multi-party conversation, logged to Neo4j
    mem = Memory()
    for c in cast:
        mem.ensure_agent(c["name"])
    with mem.drv.session() as s:
        s.run("MATCH (u:Utterance {scene:'group'}) DETACH DELETE u")
        s.run("MATCH (:Agent)-[r:FEELS]->(:Agent) DELETE r")
    names = {c["name"] for c in cast}
    transcript = []; t = 0
    for _ in range(a.rounds):
        for c in cast:
            others = [x["name"] for x in cast if x["name"] != c["name"]]
            convo = "\n".join(f"{x['who']}: {x['say']}" for x in transcript[-8:])
            out = decide(
                f"You are {c['name']}: {c['persona']}. You're in a room with {', '.join(others)}. "
                "Brief, in-character, react to what was just said. Reply ONLY JSON.",
                f"SCENE: {SCENE}\nSo far:\n{convo or '(start)'}\nYour turn. "
                '{"say":"<line>","do":"<action>","toward":"<one name or empty>",'
                '"feeling":"<one word>","sentiment":"<positive|neutral|negative>"}',
                temperature=0.9)
            transcript.append({"who": c["name"], "say": out.get("say", "")})
            mem.add_utterance(c["name"], t, out.get("say", ""), out.get("do", ""), scene="group")
            tw = str(out.get("toward", "")).strip()
            if tw in names:
                mem.set_feeling(c["name"], tw, out.get("feeling", ""), out.get("sentiment", "neutral"))
            print(f"  {c['name']}: \"{out.get('say','')}\"  [{out.get('do','')}]")
            t += 1

    print("\n=== relationships (Neo4j) ===")
    for r in mem.relationships():
        print(f"  {r['a']} -> {r['b']}: {r['feeling']} ({r['sentiment']})")
    mem.close(); sim.close()
    print(f"SOCIAL_OK cast={len(cast)} turns={t}")


if __name__ == "__main__":
    main()
