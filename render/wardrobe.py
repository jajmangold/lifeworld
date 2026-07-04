"""Wardrobe dept — the manager over catalog/wardrobe.json + the character/garment assets. Ties together the
loose ops (BlenderKit download, texture dump, ESRGAN upscale, Klein instruction-edit, Blender repack) into
one dept with a coherent CLI. The hair 'department' (HAAR grooms) lives in gen_hair.py + services/hair.py;
this covers characters + their textures (skin/clothing).

  python3 render/wardrobe.py list
  python3 render/wardrobe.py show <char_id>
  python3 render/wardrobe.py textures <char_id>                         # dump the char's base-color maps + sizes
  python3 render/wardrobe.py retexture <char_id> --method klein --maps suit \
        --instruction "finely-woven charcoal wool with a subtle navy pinstripe"
  python3 render/wardrobe.py retexture <char_id> --method esrgan --maps suit,skin   # 4x albedo upscale only

Retexture pipeline (crosses machines): Blender export base-color maps (rtx0 sampl) -> pull -> enhance locally
(Klein-4B instruction edit @ /v1/images/edits, then ESRGAN 4x to restore map resolution | or ESRGAN only)
-> push -> Blender reimport + repack -> HD .blend (rtx0) -> pull to assets/ + record hd_blend in the catalog.
Klein changes WHAT the fabric is; ESRGAN restores HOW sharp the map is. Only ALBEDO is touched (never
normal/roughness). Idempotent per (char, method, maps).
"""
import argparse, json, os, subprocess, sys, shutil, tempfile

BOT = "/srv/nvme-data/containers/live/studio"
CAT = os.path.join(BOT, "catalog", "wardrobe.json")
RTX0 = "josh@rtx0"
SAMPL = "/mnt/datadisk/containers/sampl"        # = /work in the sampl (Blender) container
GROOMS = os.path.join(BOT, "assets", "grooms")

def sh(cmd, **kw): return subprocess.run(cmd, shell=True, text=True, **kw)
def cap(cmd): return subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout
def load(): return json.load(open(CAT))
def save(d): json.dump(d, open(CAT, "w"), indent=2)

def _char(d, cid):
    c = d.get("characters", {}).get(cid)
    if not c: sys.exit(f"no character '{cid}' (have: {', '.join(d.get('characters', {}))})")
    return c

# ---------- read commands ----------
def cmd_list(d, a):
    ch, g = d.get("characters", {}), d.get("garments", {})
    print(f"=== CHARACTERS ({len(ch)}) ===")
    for k, v in ch.items():
        hd = " +HD" if v.get("hd_blend") else ""
        print(f"  {k:22} {v['gender']:6} {v['role']:8} {v['blend_format']:14} hair_ok={str(v.get('hair_ok')):5}"
              f" used_by={v.get('used_by') or '-'}{hd}  {','.join(v.get('tags', [])[:3])}")
    print(f"=== GARMENTS ({len(g)}) ===")
    for k, v in g.items():
        print(f"  {k:22} {v.get('kind','?'):10} {v.get('notes','')[:50]}")
    print(f"=== GROOMS (hair) ===  {', '.join(f[:-4] for f in sorted(os.listdir(GROOMS))) if os.path.isdir(GROOMS) else '(none)'}")

def cmd_show(d, a):
    print(json.dumps(_char(d, a.char), indent=2))

def cmd_textures(d, a):
    c = _char(d, a.char); blend = os.path.join(BOT, c["blend"])
    cb = f"wt_{a.char}.blend"
    sh(f"scp -q {blend} {RTX0}:{SAMPL}/{cb}")
    sh(f"scp -q {BOT}/render/dump_textures.py {RTX0}:{SAMPL}/")
    out = cap(f"ssh {RTX0} \"docker exec sampl bash -lc 'cd /work && /opt/blender/blender -b --python "
              f"dump_textures.py -- /work/{cb} 2>&1 | grep -aE \\\"IMAGES|Base Color|Normal|Roughness|x[0-9]\\\"'\"")
    print(out or "(no textures found)")

# ---------- retexture pipeline ----------
def _stage(a, blend):
    cb = f"retex_{a.char}.blend"
    rtexdir = f"/work/retex_{a.char}_tex"; rtexdir_host = f"{SAMPL}/retex_{a.char}_tex"
    sh(f"scp -q {blend} {RTX0}:{SAMPL}/{cb}")
    sh(f"scp -q {BOT}/render/retexture.py {BOT}/render/reproject_garment.py {RTX0}:{SAMPL}/")
    return cb, rtexdir, rtexdir_host

def _export_maps(a, cb, rtexdir, rtexdir_host, texdir):
    sh(f"ssh {RTX0} \"docker exec sampl bash -lc 'rm -rf {rtexdir}; cd /work && /opt/blender/blender -b "
       f"--python retexture.py -- export /work/{cb} {rtexdir} 2>&1 | grep -aE \\\"EXPORT|RETEX_EXPORT_DONE\\\"; chmod -R 777 {rtexdir}'\"")
    sh(f"scp -q -r {RTX0}:{rtexdir_host}/* {texdir}/ 2>/dev/null")
    return [f for f in os.listdir(texdir) if f.lower().endswith(".png") and not f.startswith("up_")]

def _reimport(a, cb, rtexdir, rtexdir_host, texdir):
    sh(f"ssh {RTX0} 'mkdir -p {rtexdir_host}'")
    sh(f"scp -q {texdir}/up_*.png {RTX0}:{rtexdir_host}/ 2>/dev/null")
    hd_rel = f"assets/blenderkit/{a.char}_hd.blend"; hd_abs = os.path.join(BOT, hd_rel)
    sh(f"ssh {RTX0} \"docker exec sampl bash -lc 'cd /work && /opt/blender/blender -b --python retexture.py -- "
       f"import /work/{cb} {rtexdir} /work/{a.char}_hd.blend 2>&1 | grep -aE \\\"IMPORT|RETEX_IMPORT_DONE\\\"'\"")
    sh(f"scp -q {RTX0}:{SAMPL}/{a.char}_hd.blend {hd_abs}")
    return hd_rel, hd_abs

def _finish(d, a, c, hd_rel, hd_abs, extra):
    if os.path.exists(hd_abs):
        c["hd_blend"] = hd_rel
        c.setdefault("textures", {})["retexture"] = {"method": a.method, "maps": a.maps or "all",
                                                     "instruction": a.instruction, **extra}
        save(d)
        print(f"RETEXTURE_OK -> {hd_rel} ({os.path.getsize(hd_abs)//1024//1024}MB); catalog updated")
    else:
        print("RETEXTURE_FAILED (no HD blend produced)")

def cmd_retexture(d, a):
    {"klein": _retex_klein, "esrgan": _retex_esrgan, "recolor": _retex_recolor}[a.method](d, a)

def _render_and_klein(a, cb, material, vlocal, only_front=False):
    """Shared: render flat ~albedo views on rtx0, then Klein-edit them (sd.cpp). Returns the view names."""
    rg = f"/work/rg_{a.char}"; rg_host = f"{SAMPL}/rg_{a.char}"
    sh(f"ssh {RTX0} \"docker exec sampl bash -lc 'rm -rf {rg}; cd /work && CUDA_VISIBLE_DEVICES=0 /opt/blender/blender "
       f"-b --python reproject_garment.py -- render /work/{cb} {material} {rg} 2>&1 | grep -aE \\\"rendered|RENDER_DONE|Error\\\"; chmod -R 777 {rg}'\"")
    sh(f"scp -q {RTX0}:{rg_host}/view_*.png {vlocal}/ 2>/dev/null")
    views = sorted(f[5:-4] for f in os.listdir(vlocal) if f.startswith("view_") and f.endswith(".png"))
    if not views: sys.exit("no views rendered (material not on a mesh?)")
    if only_front: views = [v for v in views if v == "front"] or views[:1]
    print(f"[wardrobe] rendered views: {views}")
    shutil.copy(os.path.join(BOT, "viverse_avatar", "klein_edit.py"), os.path.join(vlocal, "_klein_edit.py"))
    instr = f"{a.instruction}. Keep the person's pose, body, face and background identical; change only the clothing."
    for v in views:
        sh(f"docker run --rm --network lm-stack_ai -v {vlocal}:/t -w /t --entrypoint python3 klein-proxy:1.0 "
           f"/t/_klein_edit.py /t/view_{v}.png \"{instr}\" /t/edited_{v}.png",
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(f"{vlocal}/edited_{v}.png"):
            shutil.copy(f"{vlocal}/view_{v}.png", f"{vlocal}/edited_{v}.png")
        print(f"  klein {'OK' if os.path.exists(f'{vlocal}/edited_{v}.png') else 'FAIL'} {v}")
    return views, rg, rg_host

def _retex_recolor(d, a):
    # THE robust path for a COLOR change (grey suit -> navy/burgundy/charcoal): keep the original albedo's
    # luminance/weave/AO, shift only chroma toward the target. sd.cpp/Klein still DRIVES the color (sampled from
    # the edited render's changed region), so natural language works -- but the application is a clean per-texel
    # LAB transfer: no projection, no seams, no aliasing (unlike --method klein, which is for spatial PATTERNS).
    if not a.instruction: sys.exit("--method recolor needs --instruction (e.g. 'deep navy blue')")
    if not a.maps: sys.exit("--method recolor needs --maps <material substring> (e.g. outfit)")
    material = a.maps.split(",")[0].strip()
    c = _char(d, a.char); blend = os.path.join(BOT, c["blend"])
    vlocal = os.path.join(BOT, "output", f"retex_{a.char}_views")
    texdir = os.path.join(vlocal, "tex"); os.makedirs(texdir, exist_ok=True)
    cb, rtexdir, rtexdir_host = _stage(a, blend)
    print(f"[wardrobe] recolor material='{material}': render -> klein(pick color) -> LAB chroma transfer")
    _render_and_klein(a, cb, material, vlocal, only_front=True)
    pngs = _export_maps(a, cb, rtexdir, rtexdir_host, texdir)
    orig = next((p for p in pngs if material.lower() in p.lower()), None)
    if not orig: sys.exit(f"no '{material}' base-color map among {pngs}")
    shutil.copy(os.path.join(BOT, "render", "colorize_garment.py"), os.path.join(BOT, "output", "colorize_garment.py"))
    rel = f"retex_{a.char}_views"
    sh(f"docker exec swap-server python3 /o/colorize_garment.py /o/{rel}/tex/{orig} /o/{rel}/tex/up_{orig} "
       f"/o/{rel}/view_front.png /o/{rel}/edited_front.png")
    print(f"  colorize {'OK' if os.path.exists(f'{texdir}/up_{orig}') else 'FAIL'} -> up_{orig}")
    hd_rel, hd_abs = _reimport(a, cb, rtexdir, rtexdir_host, texdir)
    _finish(d, a, c, hd_rel, hd_abs, {"material": material, "mode": "lab_chroma_transfer"})

def _retex_esrgan(d, a):
    # albedo detail upscale of the UV atlas (sharpens; no semantic change)
    c = _char(d, a.char); blend = os.path.join(BOT, c["blend"])
    maps = [m.strip().lower() for m in a.maps.split(",")] if a.maps else None
    work = tempfile.mkdtemp(prefix=f"retex_{a.char}_"); texdir = os.path.join(work, "tex"); os.makedirs(texdir)
    cb, rtexdir, rtexdir_host = _stage(a, blend)
    pngs = _export_maps(a, cb, rtexdir, rtexdir_host, texdir)
    if not pngs: sys.exit("no base-color maps exported")
    targets = [p for p in pngs if (not maps or any(m in p.lower() for m in maps))]
    print(f"[wardrobe] esrgan 4x upscale {len(targets)} maps: {targets}")
    for p in targets:
        sh(f"python3 {BOT}/render/esrgan_tex.py {texdir} {p}")
        print(f"  {'OK ' if os.path.exists(os.path.join(texdir, 'up_' + p)) else 'FAIL'} {p}")
    hd_rel, hd_abs = _reimport(a, cb, rtexdir, rtexdir_host, texdir)
    _finish(d, a, c, hd_rel, hd_abs, {})
    shutil.rmtree(work, ignore_errors=True)

def _retex_klein(d, a):
    # render -> Klein-edit the RENDER (Klein sees the suit) -> reproject/bake back onto the UV, blended over
    # the original map. The correct way to actually restyle a garment (vs. editing the abstract UV atlas).
    if not a.instruction: sys.exit("--method klein needs --instruction")
    if not a.maps: sys.exit("--method klein needs --maps <material substring> (e.g. outfit)")
    material = a.maps.split(",")[0].strip()
    c = _char(d, a.char); blend = os.path.join(BOT, c["blend"])
    vlocal = os.path.join(BOT, "output", f"retex_{a.char}_views")   # UNDER output/ so swap-server (=/o) can combine
    texdir = os.path.join(vlocal, "tex"); os.makedirs(texdir, exist_ok=True)
    cb, rtexdir, rtexdir_host = _stage(a, blend)
    rg = f"/work/rg_{a.char}"; rg_host = f"{SAMPL}/rg_{a.char}"
    print(f"[wardrobe] klein reproject material='{material}': render -> edit -> reproject")

    # 1) render flat-lit ~albedo views on rtx0
    sh(f"ssh {RTX0} \"docker exec sampl bash -lc 'rm -rf {rg}; cd /work && CUDA_VISIBLE_DEVICES=0 /opt/blender/blender "
       f"-b --python reproject_garment.py -- render /work/{cb} {material} {rg} 2>&1 | grep -aE \\\"rendered|RENDER_DONE|Error\\\"; chmod -R 777 {rg}'\"")
    sh(f"scp -q {RTX0}:{rg_host}/view_*.png {vlocal}/ 2>/dev/null")
    views = sorted(f[5:-4] for f in os.listdir(vlocal) if f.startswith("view_") and f.endswith(".png"))
    if not views: sys.exit("no views rendered (material not on a mesh?)")
    print(f"[wardrobe] rendered views: {views}")

    # 2) Klein-edit each view (change only the outfit)
    shutil.copy(os.path.join(BOT, "viverse_avatar", "klein_edit.py"), os.path.join(vlocal, "_klein_edit.py"))
    instr = f"{a.instruction}. Keep the person's pose, body, face and background identical; change only the clothing."
    for v in views:
        sh(f"docker run --rm --network lm-stack_ai -v {vlocal}:/t -w /t --entrypoint python3 klein-proxy:1.0 "
           f"/t/_klein_edit.py /t/view_{v}.png \"{instr}\" /t/edited_{v}.png",
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not os.path.exists(f"{vlocal}/edited_{v}.png"):
            shutil.copy(f"{vlocal}/view_{v}.png", f"{vlocal}/edited_{v}.png")   # fallback so the bake still runs
        print(f"  klein {'OK' if os.path.exists(f'{vlocal}/edited_{v}.png') else 'FAIL'} {v}")
        # Camera-projection baking maps world position -> UV; on curved/foreshortened surfaces (sleeves,
        # collar, lapels) that mapping is many-to-one, so a fine pattern (pinstripes) aliases into camo-like
        # noise once baked. A mild blur pre-pass suppresses the offending high frequency before the bake while
        # keeping the overall color/coarse pattern (flatter regions like pants keep their visible pinstripe).
        from PIL import Image, ImageFilter
        p = f"{vlocal}/edited_{v}.png"; tmp = f"{p}.blur.png"
        Image.open(p).convert("RGB").filter(ImageFilter.GaussianBlur(radius=6)).save(tmp)
        os.replace(tmp, p)   # edited_*.png is root-owned (written by the klein-proxy container); replace, don't overwrite-in-place

    # 3) push edited views + bake the reprojection on rtx0
    sh(f"scp -q {vlocal}/edited_*.png {RTX0}:{rg_host}/ 2>/dev/null")
    sh(f"ssh {RTX0} \"docker exec sampl bash -lc 'cd /work && CUDA_VISIBLE_DEVICES=0 /opt/blender/blender -b "
       f"--python reproject_garment.py -- bake /work/{cb} {material} {rg} 2>&1 | grep -aE \\\"baked|BAKE_DONE|Error\\\"'\"")
    sh(f"scp -q {RTX0}:{rg_host}/bake_*.png {vlocal}/ 2>/dev/null")

    # 4) original map + combine (swap-server cv2/numpy) restyle-over-original
    pngs = _export_maps(a, cb, rtexdir, rtexdir_host, texdir)
    orig = next((p for p in pngs if material.lower() in p.lower()), None)
    if not orig: sys.exit(f"no '{material}' base-color map among {pngs}")
    shutil.copy(os.path.join(BOT, "render", "reproject_combine.py"), os.path.join(BOT, "output", "reproject_combine.py"))
    rel = f"retex_{a.char}_views"
    sh(f"docker exec swap-server python3 /o/reproject_combine.py /o/{rel} /o/{rel}/tex/{orig} "
       f"/o/{rel}/tex/up_{orig} {','.join(views)}", stdout=subprocess.DEVNULL)
    print(f"  combine {'OK' if os.path.exists(f'{texdir}/up_{orig}') else 'FAIL'} -> up_{orig}")

    # 5) reimport restyled map -> HD blend
    hd_rel, hd_abs = _reimport(a, cb, rtexdir, rtexdir_host, texdir)
    _finish(d, a, c, hd_rel, hd_abs, {"material": material, "views": views})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("show"); p.add_argument("char")
    p = sub.add_parser("textures"); p.add_argument("char")
    p = sub.add_parser("retexture"); p.add_argument("char")
    p.add_argument("--method", choices=["recolor", "klein", "esrgan"], default="recolor")
    p.add_argument("--maps", default="")           # comma substrings (e.g. suit,skin); empty = all albedo
    p.add_argument("--instruction", default="")    # klein: what to change the material into
    a = ap.parse_args()
    d = load()
    {"list": cmd_list, "show": cmd_show, "textures": cmd_textures, "retexture": cmd_retexture}[a.cmd](d, a)
