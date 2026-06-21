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
from habitat_sim.utils.common import quat_from_magnum


def look_at_rot(eye, target, up=(0.0, 1.0, 0.0)):
    """Roll-free camera rotation: orients the sensor's -Z at `target` with world up.
    (quat_from_two_vectors aligns only the view dir and lets roll drift -> tilted shots.)"""
    m = mn.Matrix4.look_at(
        mn.Vector3(*np.asarray(eye, float).tolist()),
        mn.Vector3(*np.asarray(target, float).tolist()),
        mn.Vector3(*up))
    return quat_from_magnum(mn.Quaternion.from_matrix(m.rotation()))


def setup_navmesh(sim, radius=0.3, height=1.4):
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = radius
    ns.agent_height = height
    ns.include_static_objects = True          # <-- furniture becomes navmesh obstacles
    sim.recompute_navmesh(sim.pathfinder, ns)
    return sim.pathfinder


def obstacle_footprints(sim, y_lo=0.25, y_hi=1.6, min_area=0.10):
    """XZ footprints of furniture that occupies the torso/head band. The navmesh only
    clears furniture *legs* (foot level); a table TOP overhangs floor the navmesh still
    calls walkable, so a standing body clips it. These footprints let us also keep
    placements out from UNDER overhangs."""
    fps = []
    for mgr in (sim.get_rigid_object_manager(), sim.get_articulated_object_manager()):
        for h in mgr.get_object_handles():
            try:
                node = mgr.get_object_by_handle(h).root_scene_node
                wbb = habitat_sim.geo.get_transformed_bb(
                    node.cumulative_bb, node.absolute_transformation())
                lo, hi = wbb.min, wbb.max
            except Exception:
                continue
            if hi[1] < y_lo or lo[1] > y_hi:          # not in body band
                continue
            if (hi[0] - lo[0]) * (hi[2] - lo[2]) < min_area:   # ignore small clutter
                continue
            fps.append((float(lo[0]), float(hi[0]), float(lo[2]), float(hi[2])))
    return fps


def is_clear(p, fps, margin=0.22):
    x, z = float(p[0]), float(p[2])
    for x0, x1, z0, z1 in fps:
        if x0 - margin <= x <= x1 + margin and z0 - margin <= z <= z1 + margin:
            return False
    return True


def clear_navpoint(pf, fps, near=None, lo=0.0, hi=99.0, tries=400):
    """A navigable point that is also clear of furniture footprints (no body clipping)."""
    for _ in range(tries):
        c = np.array(pf.snap_point(pf.get_random_navigable_point()), float)
        if not np.all(np.isfinite(c)) or not is_clear(c, fps):
            continue
        if near is not None:
            d = np.linalg.norm((c - np.asarray(near, float))[[0, 2]])
            if not (lo <= d <= hi):
                continue
        return c
    return None


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
