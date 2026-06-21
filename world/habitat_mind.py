#!/usr/bin/env python3
"""M2 mind loop: an NPC lives a few minutes of life in ReplicaCAD.

Each tick: PERCEIVE (nearby objects from the scene) -> DECIDE (DeepSeek V4 Flash
picks where to go + why) -> ACT (humanoid walks there) -> REMEMBER (log the step
to Neo4j). The walk is recorded; the Neo4j graph becomes the agent's life-record.

Run in lifeworld-habitat with --network host (reach Neo4j :7688 + DeepSeek) and
-e DEEPSEEK_API_KEY -e NEO4J_PASSWORD, /data + repo mounted.
"""
import argparse
import os
import re
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

SYS = ("You are {name}, an ordinary person at home in your apartment. You wander "
       "room to room doing normal things. Be decisive and human. Reply ONLY JSON.")


def clean(handle):
    s = re.sub(r"_:\d+$", "", handle)
    s = re.sub(r"^(frl_apartment_|apt_|kitchen_|ai_)", "", s)
    return s.replace("_", " ").strip()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dataset", required=True)
    ap.add_argument("--scene", default="apt_0")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--name", default="Mara")
    ap.add_argument("--goal", default="explore your home, get a snack, then relax")
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--frames-dir", default="/work/output/frames_mind")
    ap.add_argument("--res", type=int, nargs=2, default=[540, 720])
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
    os.makedirs(a.frames_dir, exist_ok=True)
    from PIL import Image

    sim = habitat_sim.Simulator(make_cfg(a))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = 0.3; ns.agent_height = 1.4
    sim.recompute_navmesh(sim.pathfinder, ns)
    pf = sim.pathfinder

    # catalog of go-to objects from the scene's object managers: name -> positions
    catalog = {}
    for mgr in (sim.get_rigid_object_manager(), sim.get_articulated_object_manager()):
        for h in mgr.get_object_handles():
            try:
                o = mgr.get_object_by_handle(h)
                p = np.array(o.translation, dtype=np.float64)
            except Exception:
                continue
            nm = clean(h)
            if nm and len(nm) > 2:
                catalog.setdefault(nm, []).append(p)
    print(f"[mind] catalog: {len(catalog)} object types")

    agent_cfg = DictConfig({"articulated_agent_urdf": a.urdf, "motion_data_path": a.motion})
    hum = kinematic_humanoid.KinematicHumanoid(agent_cfg, sim)
    hum.reconfigure(); hum.update()
    hum.base_pos = mn.Vector3(*pf.snap_point(pf.get_random_navigable_point()))
    ctrl = HumanoidRearrangeController(a.motion)

    mem = Memory(); mem.reset_agent(a.name)
    fi = [0]
    visited = set()

    def cam_aim(eye, look):
        d = (look - eye); d /= (np.linalg.norm(d) + 1e-9)
        st = sim.get_agent(0).get_state()
        st.position = eye.astype(np.float32)
        st.rotation = quat_from_two_vectors(np.array([0.0, 0.0, -1.0]), d.astype(np.float64))
        sim.get_agent(0).set_state(st)

    def walk_to(tpos, max_steps=160):
        ctrl.reset(hum.base_transformation)
        # camera vantage: navigable point ~3m from the midpoint of the trip
        hp0 = np.array(hum.base_pos); mid = (hp0 + tpos) / 2
        eye, best = mid + np.array([3.0, 0, 0]), -1
        for _ in range(200):
            c = np.array(pf.get_random_navigable_point())
            if abs(c[1] - mid[1]) < 0.2:
                dd = np.linalg.norm((c - mid)[[0, 2]])
                if 2.0 < dd < 4.5 and dd > best:
                    best, eye = dd, c
        eye = eye + np.array([0, 1.5, 0])
        steps = 0
        while steps < max_steps:
            diff = mn.Vector3(*tpos.tolist()) - hum.base_pos
            if diff.length() < 0.4:
                break
            ctrl.calculate_walk_pose(diff)
            pose = ctrl.get_pose()
            joints, base, off = pose[:-16], pose[-16:], pose[-32:-16]
            if np.array(off).sum() != 0:
                vb = [mn.Vector4(base[i*4:(i+1)*4]) for i in range(4)]
                vo = [mn.Vector4(off[i*4:(i+1)*4]) for i in range(4)]
                hum.set_joint_transform(joints, mn.Matrix4(*vo), mn.Matrix4(*vb))
            cam_aim(eye, np.array(hum.base_pos) + np.array([0, 0.85, 0]))
            rgb = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
            Image.fromarray(rgb).save(os.path.join(a.frames_dir, f"frame_{fi[0]:04d}.png"))
            fi[0] += 1; steps += 1
        return steps

    for tick in range(a.ticks):
        hp = np.array(hum.base_pos)
        nearby = []
        for nm, ps in catalog.items():
            if nm in visited:
                continue
            dmin = min(np.linalg.norm((p - hp)[[0, 2]]) for p in ps)
            if 1.3 < dmin < 9.0:          # far enough to be worth walking to
                nearby.append((dmin, nm))
        nearby.sort()
        names = [nm for _, nm in nearby[:14]]
        if not names:                      # nothing new in reach -> wander
            tpos = np.array(pf.get_random_navigable_point())
            steps = walk_to(tpos)
            mem.log_step(a.name, tick, "(nothing new nearby)", "wandering", "elsewhere")
            print(f"[tick {tick}] nothing new -> wandered {steps} frames")
            continue
        out = decide(
            SYS.format(name=a.name),
            f"Goal: {a.goal}. You are standing at {hp.round(1).tolist()}. You have already "
            f"visited: {sorted(visited)}. Nearby NEW things you could walk to: {names}. "
            "Pick ONE to walk to now. "
            'Respond {"target":"<exact item from the list>","reason":"<short, first person>"}.')
        target = str(out.get("target", names[0] if names else "")).lower()
        reason = out.get("reason", "")
        # match target back to catalog (exact, then substring)
        match = next((nm for nm in names if nm == target), None) \
            or next((nm for nm in names if target in nm or nm in target), None) \
            or (names[0] if names else None)
        if match is None:
            break
        tpos = min(catalog[match], key=lambda p: np.linalg.norm((p - hp)[[0, 2]]))
        tpos = np.array(pf.snap_point(mn.Vector3(*tpos.tolist())))
        if not np.all(np.isfinite(tpos)):
            tpos = np.array(pf.get_random_navigable_point())
        steps = walk_to(tpos)
        visited.add(match)
        mem.log_step(a.name, tick, ", ".join(names[:8]), reason, match)
        print(f"[tick {tick}] sees={names[:6]} -> '{match}' ({reason}) walked {steps} frames")

    print("[mind] life so far:")
    for s in mem.life_story(a.name):
        print(f"   t{s['t']}: went to {s['target']} — {s['thought']}")
    mem.close(); sim.close()
    print(f"MIND_OK ticks={a.ticks} frames={fi[0]} dir={a.frames_dir}")


if __name__ == "__main__":
    main()
