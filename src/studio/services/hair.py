"""Hair service — resolve a talent's HAAR groom (generate via the resident `hairgen` container if not cached)
and stage it to the render box. Thin wrapper over render/gen_hair.py; idempotent (spec -> path).

A talent's `hair` spec: {"groom": "<name>", "prompt": "<optional text>", "melanin": 0.4, "face_slim": 0.35}.
If the named groom isn't in assets/grooms/, and a prompt is given, HAAR generates it. Returns the render-box
path to pass as NEWS_HAIR_GROOM, plus melanin/slim for NEWS_HAIR_MEL / NEWS_FACE_SLIM.
"""
import os, subprocess

BOT = "/srv/nvme-data/containers/live/studio"
GROOMS = os.path.join(BOT, "assets", "grooms")
RTX0 = "josh@rtx0:/mnt/datadisk/containers/sampl"

def resolve(hair):
    """hair: talent['hair'] dict or None. Returns dict {groom_remote, melanin, face_slim} or {} if no hair."""
    if not hair or not hair.get("groom"):
        return {}
    name = hair["groom"]
    local = os.path.join(GROOMS, f"{name}.ply")
    if not os.path.exists(local):
        if not hair.get("prompt"):
            raise FileNotFoundError(f"groom '{name}' not cached and no prompt to generate it")
        subprocess.run(["python3", os.path.join(BOT, "render", "gen_hair.py"),
                        "--prompt", hair["prompt"], "--name", name], check=True)
    # stage to the render box (idempotent scp)
    remote_name = f"groom_{name}.ply"
    subprocess.run(f"scp -q {local} {RTX0}/{remote_name}", shell=True, check=True)
    return {"groom_remote": f"/work/{remote_name}",
            "melanin": hair.get("melanin", 0.4),
            "face_slim": hair.get("face_slim", 0.0)}
