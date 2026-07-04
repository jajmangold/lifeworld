"""Chroma-key a green anchor clip, composite BEHIND the newsroom desk, mux audio. Outputs 1440p by default.
  python3 ltx_matte_composite.py <anchor_green.mp4> <set_bg.png> <out.mp4> [audio.wav] [OW OH]"""
import sys, os, subprocess, tempfile, numpy as np
from PIL import Image, ImageFilter, ImageDraw
ANCHOR, BG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
AUDIO = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] not in ("-","") else None
OW = int(sys.argv[5]) if len(sys.argv) > 5 else 2560
OH = int(sys.argv[6]) if len(sys.argv) > 6 else 1440
td = tempfile.mkdtemp(); fr = os.path.join(td, "fr"); os.makedirs(fr)
subprocess.run(["ffmpeg","-y","-i",ANCHOR,os.path.join(td,"a%04d.png")],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
frames=sorted(f for f in os.listdir(td) if f.startswith("a") and f.endswith(".png"))
if BG.lower().endswith((".mp4",".mov")):
    subprocess.run(["ffmpeg","-y","-i",BG,"-vframes","1",os.path.join(td,"bg.png")],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); BG=os.path.join(td,"bg.png")
bg=Image.open(BG).convert("RGB"); s=max(OW/bg.width,OH/bg.height)
bg=bg.resize((int(bg.width*s+.5),int(bg.height*s+.5))); bg=bg.crop(((bg.width-OW)//2,(bg.height-OH)//2,(bg.width-OW)//2+OW,(bg.height-OH)//2+OH))
DESK_Y=int(0.72*OH)
def key(im):
    a=np.asarray(im.convert("RGB")).astype(np.float32); r,g,b=a[...,0],a[...,1],a[...,2]
    gr=g-np.maximum(r,b); al=np.clip(1.0-(gr-12)/55.0,0,1); al[gr<12]=1.0
    a[...,1]=np.where(g>np.maximum(r,b),np.maximum(r,b)+(g-np.maximum(r,b))*0.12,g)
    o=Image.fromarray(np.dstack([a,al*255]).astype(np.uint8),"RGBA")
    o.putalpha(Image.fromarray((al*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))); return o
def grade(rgba):
    # match the anchor to the cool newsroom key: gentle contrast + slight cool tint (alpha untouched)
    a=np.asarray(rgba).astype(np.float32); rgb=a[...,:3]/255.0
    rgb=(rgb-0.5)*1.06+0.5
    rgb[...,0]*=0.97; rgb[...,2]=np.clip(rgb[...,2]*1.05,0,1)   # -red +blue -> cooler, matches the set
    return Image.fromarray(np.dstack([np.clip(rgb,0,1)*255,a[...,3]]).astype(np.uint8),"RGBA")
for i,f in enumerate(frames):
    an=key(Image.open(os.path.join(td,f)))
    an=an.crop((0,0,an.width,int(an.height*0.72)))          # drop the anchor's own desk/hands
    sc=(DESK_Y+int(0.06*OH))/an.height; an=an.resize((int(an.width*sc),int(an.height*sc)))
    an=grade(an)
    ax=OW//2-an.width//2; ay=DESK_Y+int(0.06*OH)-an.height
    frame=bg.copy().convert("RGBA")
    # contact shadow: soft dark ellipse where the anchor meets the desk -> grounds him (kills the 'float')
    sh=Image.new("RGBA",(OW,OH),(0,0,0,0)); ImageDraw.Draw(sh).ellipse(
        [OW//2-int(an.width*0.42), DESK_Y-int(0.02*OH), OW//2+int(an.width*0.42), DESK_Y+int(0.05*OH)], fill=(0,0,0,150))
    frame=Image.alpha_composite(frame, sh.filter(ImageFilter.GaussianBlur(28)))
    frame.alpha_composite(an,(ax,ay))
    frame.convert("RGB").save(os.path.join(fr,f"c{i:04d}.png"))
cmd=["ffmpeg","-y","-framerate","25","-i",os.path.join(fr,"c%04d.png")]
if AUDIO and os.path.exists(AUDIO): cmd+=["-i",AUDIO,"-map","0:v","-map","1:a","-c:a","aac","-b:a","192k","-shortest"]
cmd+=["-c:v","libx264","-pix_fmt","yuv420p","-crf","16",OUT]
subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print("COMPOSITE_OK",OUT,f"{len(frames)}f {OW}x{OH}")
