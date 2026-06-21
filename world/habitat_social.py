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


def _visible_heads(sim, eye, heads):
    vis = 0
    for hd in heads:
        v = hd - eye; L = float(np.linalg.norm(v)); v = v / (L + 1e-9)
        ray = habitat_sim.geo.Ray(mn.Vector3(*eye.tolist()), mn.Vector3(*v.tolist()))
        res = sim.cast_ray(ray)
        if (not res.has_hits()) or res.hits[0].ray_distance >= L - 0.5:
            vis += 1
    return vis


def find_cluster(sim, pf, n, radius, fps):
    """Find a conversation spot AND a camera that can see everyone. Returns
    (centre, [member spots], camera_eye). Couples both searches so the group never
    lands where it can't be filmed (cramped corner -> mutual occlusion)."""
    angles = [2 * np.pi * i / n for i in range(n)]
    best = None  # (visible_count, centre, spots, eye) fallback if no perfect shot
    for _ in range(300):
        c = np.array(pf.snap_point(pf.get_random_navigable_point()))
        if not np.all(np.isfinite(c)) or not habnav.is_clear(c, fps):
            continue
        spots, ok = [], True
        for ang in angles:
            p = c + np.array([radius * np.cos(ang), 0, radius * np.sin(ang)])
            sp = np.array(pf.snap_point(mn.Vector3(*p.tolist())))
            if (not np.all(np.isfinite(sp)) or abs(sp[1] - c[1]) > 0.1
                    or np.linalg.norm((sp - p)[[0, 2]]) > 0.2
                    or not habnav.is_clear(sp, fps)):
                ok = False; break
            spots.append(sp)
        if not ok:
            continue
        heads = [sp + np.array([0, 1.45, 0]) for sp in spots]
        for _ in range(120):                       # look for a camera seeing all heads
            cand = np.array(pf.get_random_navigable_point())
            if abs(cand[1] - c[1]) > 0.2 or not habnav.is_clear(cand, fps):
                continue
            d = np.linalg.norm((cand - c)[[0, 2]])
            if not (2.0 < d < 3.6):
                continue
            eye = cand + np.array([0, 1.55, 0])
            vis = _visible_heads(sim, eye, heads)
            if vis == n:
                return c, spots, eye
            if best is None or vis > best[0]:
                best = (vis, c, spots, eye)
    if best is not None:
        return best[1], best[2], best[3]
    return None, None, None


def main():
    a = parse_args()
    from PIL import Image
    sim = habitat_sim.Simulator(make_cfg(a))
    pf = habnav.setup_navmesh(sim)
    cast = CAST[:max(2, min(len(CAST), 3))]

    fps = habnav.obstacle_footprints(sim)         # furniture body-band footprints
    print(f"[social] {len(fps)} furniture footprints to avoid")
    centre, spots, eye = find_cluster(sim, pf, len(cast), a.radius, fps)
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
    # camera (verified by find_cluster to see the group), aimed roll-free at centre
    look = centre + np.array([0, 1.0, 0])
    st = sim.get_agent(0).get_state()
    st.position = eye.astype(np.float32)
    st.rotation = habnav.look_at_rot(eye, look)
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
