import sys, os, cv2
from PIL import Image
sys.path.insert(0, "/champ/DWPose/ControlNet-v1-1-nightly")
os.chdir("/champ/DWPose/ControlNet-v1-1-nightly")     # for relative annotator/ckpts paths
from annotator.dwpose import DWposeDetector
det = DWposeDetector()
indir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
fs = sorted(f for f in os.listdir(indir) if f.endswith((".png", ".jpg")))
for f in fs:
    img = cv2.imread(os.path.join(indir, f))
    res = det(img)
    Image.fromarray(res).save(os.path.join(outdir, f))
print("DWPOSE_DONE", len(fs))
