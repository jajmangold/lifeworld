"""Reporter package: a brief on-camera STANDUP (reporter on the field backdrop) that hands off to
VOICEOVER over b-roll — one continuous clip. Splits the reporter narration at a natural pause near
--standup-sec: the head goes to make_anchor (fullscreen_anchor + reporter face + field bg, premium),
gets an NNS bug + reporter/location lower-third; the remainder becomes VO-over-broll. Concats to 2560x1440.

Runs on the HOST (drives make_anchor.sh + the mp-extract container), like factory.py.
  python3 newscast/build_reporter_pkg.py --audio vo.wav --name "Elise Warren" --location "KANSAS CITY, KS"
        [--headline "..."] [--broll img_or_vid] [--standup-sec 8] [--face reporter_face.png]
        [--bg output/reporter_bg.png] [--music news_bed.mp3] [--premium] [--seed 11] --out out.mp4
"""
import sys, os, subprocess, re

BOT = "/srv/nvme-data/containers/projects/bot"
os.chdir(BOT)
def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
AUDIO = arg("--audio"); OUT = arg("--out", "output/reporter_pkg.mp4")
NAME = arg("--name", "NNS Correspondent"); LOC = arg("--location", "")
HEAD = arg("--headline", ""); BROLL = arg("--broll", "")
STANDUP = float(arg("--standup-sec", "8")); FACE = arg("--face", "reporter_face.png")
BG = arg("--bg", "output/reporter_bg.png"); MUSIC = arg("--music", "newscast/music/news_bed.mp3")
SEED = arg("--seed", "11"); PREMIUM = "--premium" in sys.argv
stem = os.path.splitext(os.path.basename(OUT))[0]

def sh(cmd, **k): return subprocess.run(cmd, cwd=BOT, **k)
def probe(p, k): return subprocess.run(["ffprobe","-v","error","-show_entries",k,"-of","csv=p=0",p],
                                       capture_output=True, text=True).stdout.strip()
DOCK = ["docker","run","--rm","-v",f"{BOT}:/io","-w","/io","mp-extract:1.0","python3"]

DUR = float(probe(AUDIO, "format=duration") or "0")
# choose the split at a real pause near STANDUP (window standup-2 .. standup+5); else hard cut
split = min(STANDUP, max(2.0, DUR - 1.0))
sd = subprocess.run(["ffmpeg","-i",AUDIO,"-af","silencedetect=noise=-35dB:d=0.30","-f","null","-"],
                    capture_output=True, text=True).stderr
sils = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", sd)]
cands = [s for s in sils if STANDUP - 2.0 <= s <= STANDUP + 5.0]
if cands: split = min(cands, key=lambda s: abs(s - STANDUP))
has_vo = DUR - split >= 2.0
print(f"[reporter] dur={DUR:.1f}s split={split:.1f}s vo={'yes' if has_vo else 'no'} premium={PREMIUM}")

su_wav = f"output/{stem}_su.wav"; vo_wav = f"output/{stem}_vo.wav"
sh(["ffmpeg","-y","-i",AUDIO,"-t",f"{split:.3f}",su_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if has_vo:
    sh(["ffmpeg","-y","-ss",f"{split:.3f}","-i",AUDIO,vo_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 1) STANDUP head (reporter face on field backdrop)
su_raw = f"output/{stem}_su_raw.mp4"
cmd = ["bash","make_anchor.sh","--audio",su_wav,"--out",su_raw,"--face",FACE,"--bg",BG,
       "--format","fullscreen_anchor","--headtex","","--seed",str(SEED)]
if PREMIUM: cmd += ["--premium"]
print("[reporter] standup render…"); r = sh(cmd)
if r.returncode != 0 or not os.path.exists(su_raw): sys.exit("standup render failed")

# 2) reporter lower-third (PIL in mp-extract)
lt = f"output/{stem}_lt.png"
sh(DOCK + ["newscast/reporter_lt.py","--name",NAME,"--location",LOC,"--out",lt], check=True)

# 3) standup finish: upscale to 2560x1440, NNS bug + reporter lower-third, music bed under the VO
su_fin = f"output/{stem}_su_final.mp4"
vf = ("[0:v]scale=2560:1440:force_original_aspect_ratio=increase,crop=2560:1440,fps=25[v0];"
      "[v0][1:v]overlay=W-overlay_w-34:34[v1];"
      "[v1][2:v]overlay=60:H-overlay_h-70:enable='gte(t,0.25)'[v]")
cmd = ["ffmpeg","-y","-i",su_raw,"-i","output/gfx_bug.png","-i",lt]
if MUSIC and os.path.exists(MUSIC):
    cmd += ["-stream_loop","-1","-i",MUSIC]
    af = ("[0:a]aformat=fltp:44100:stereo,volume=1.0[vv];[3:a]aformat=fltp:44100:stereo,volume=0.07[mm];"
          "[vv][mm]amix=inputs=2:duration=first[a]")
else:
    af = "[0:a]aformat=fltp:44100:stereo[a]"
cmd += ["-filter_complex", vf+";"+af, "-map","[v]","-map","[a]","-c:v","libx264","-pix_fmt","yuv420p",
        "-crf","18","-r","25","-c:a","aac","-b:a","192k","-shortest",su_fin]
sh(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if not has_vo:
    sh(["ffmpeg","-y","-i",su_fin,"-c","copy",OUT], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("REPORTER_PKG_OK (standup only) ->", OUT); sys.exit(0)

# 4) VO-over-broll for the remainder
vo_mp4 = f"output/{stem}_vopart.mp4"
cmd = DOCK + ["newscast/build_vo_broll.py","--audio",vo_wav,"--out",vo_mp4,"--music",MUSIC]
if BROLL: cmd += ["--broll",BROLL]
if HEAD: cmd += ["--headline",HEAD]
sh(cmd, check=True)

# 5) concat standup + VO (both 2560x1440@25, aac) — re-encode via concat filter for safety
sh(["ffmpeg","-y","-i",su_fin,"-i",vo_mp4,"-filter_complex",
    "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]","-map","[v]","-map","[a]",
    "-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-r","25","-c:a","aac","-b:a","192k",OUT],
   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("REPORTER_PKG_OK ->", OUT)
