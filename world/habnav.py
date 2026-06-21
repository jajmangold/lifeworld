#!/usr/bin/env python3
"""Shared Habitat navigation: a navmesh that respects furniture, and path-following
walking so humanoids route AROUND obstacles instead of clipping through cabinets.

The bug this fixes: a default recomputed navmesh ignores scene objects, and walking
straight at the target cuts through furniture. We (1) recompute with
include_static_objects=True so cabinets/tables become obstacles, and (2) walk along
the geodesic path's waypoints — the navmesh funnel guarantees the straight segment
between consecutive waypoints stays on walkable floor.
"""
import habitat_sim
import magnum as mn
import numpy as np


def setup_navmesh(sim, radius=0.3, height=1.4):
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = radius
    ns.agent_height = height
    ns.include_static_objects = True          # <-- furniture becomes navmesh obstacles
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim.pathfinder


def geodesic(pf, start, end):
    """Waypoints routing around obstacles. Returns (found, [np.array,...])."""
    sp = habitat_sim.ShortestPath()
    sp.requested_start = mn.Vector3(*np.asarray(start, dtype=float).tolist())
    sp.requested_end = mn.Vector3(*np.asarray(end, dtype=float).tolist())
    found = pf.find_path(sp)
    pts = [np.array(p, dtype=float) for p in sp.points] if found else [np.asarray(end, float)]
    return found, pts


def walk_path(hum, ctrl, pf, target, on_frame=None, max_steps=400, reach=0.3):
    """Walk `hum` to `target` along the navmesh path. on_frame() called per rendered
    step. Returns frames stepped."""
    target = np.array(pf.snap_point(mn.Vector3(*np.asarray(target, float).tolist())), float)
    _, waypoints = geodesic(pf, np.array(hum.base_pos), target)
    ctrl.reset(hum.base_transformation)
    steps = 0
    for wp in (waypoints[1:] if len(waypoints) > 1 else waypoints):
        guard = 0
        while steps < max_steps and guard < 220:
            base = np.array(hum.base_pos)
            diff = wp - base
            if np.linalg.norm(diff[[0, 2]]) < reach:
                break
            ctrl.calculate_walk_pose(mn.Vector3(*diff.tolist()))
            pose = ctrl.get_pose()
            j, b, o = pose[:-16], pose[-16:], pose[-32:-16]
            if np.array(o).sum() != 0:
                vb = [mn.Vector4(b[i * 4:(i + 1) * 4]) for i in range(4)]
                vo = [mn.Vector4(o[i * 4:(i + 1) * 4]) for i in range(4)]
                hum.set_joint_transform(j, mn.Matrix4(*vo), mn.Matrix4(*vb))
            if on_frame is not None:
                on_frame()
            steps += 1
            guard += 1
    return steps
