#!/usr/bin/env python3
"""Minimal Genesis physics smoke test for sm_70 (V100/CMP) — ADR-0005/0007.

Validates the open question: does Genesis's GPU physics backend run on Volta
(no RT cores)? We DON'T touch the ray-traced renderer (that's the RTX-only part);
we only step rigid-body physics headless and confirm a dropped sphere falls.
Exit 0 = physics works on this card.
"""
import sys

try:
    import genesis as gs
except Exception as e:
    print("IMPORT_FAIL:", e); sys.exit(2)

try:
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    ball = scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.2))
    scene.build()
    z0 = float(ball.get_pos()[2])
    for _ in range(120):
        scene.step()
    z1 = float(ball.get_pos()[2])
    print(f"PHYSICS_OK backend=gpu z0={z0:.3f} z1={z1:.3f} fell={z0 - z1:.3f}")
    sys.exit(0 if z1 < z0 - 0.1 else 3)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("RUN_FAIL:", e); sys.exit(4)
