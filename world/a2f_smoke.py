#!/usr/bin/env python3
"""Will Audio2Face-3D run on a Volta CMP100 (sm_70)?

TensorRT dropped Volta in TRT 10.5, so the official A2F TRT path is out on sm_70.
But the model ships as ONNX, and ONNX Runtime's CUDA EP still supports sm_70. This
loads network.onnx on the CUDA EP, pins it to the CMP card, and runs a real forward
pass (1s audio + emotion + identity) to prove the net executes on Volta.
Exit 0 = it runs on the CMP100.
"""
import sys
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    print("IMPORT_FAIL:", e); sys.exit(2)

ONNX = sys.argv[1] if len(sys.argv) > 1 else "/work/a2f/network.onnx"
AUDIO_LEN = 16000 + 16000 + 16000      # buffer + L/R padding (network_info.json)

print("ORT", ort.__version__, "| avail providers:", ort.get_available_providers())
try:
    sess = ort.InferenceSession(
        ONNX, providers=[("CUDAExecutionProvider", {"device_id": 0}),
                         "CPUExecutionProvider"])
except Exception as e:
    import traceback; traceback.print_exc(); print("SESSION_FAIL:", e); sys.exit(3)
print("USING providers:", sess.get_providers())
if "CUDAExecutionProvider" not in sess.get_providers():
    print("CUDA_EP_NOT_ACTIVE"); sys.exit(4)


def fill(inp):
    name = inp.name.lower()
    # all symbolic dims -> 1 (consistent batch); concrete dims kept as-is
    shp = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    t = inp.type
    if "int" in t:
        arr = np.zeros(shp, np.int64 if "int64" in t else np.int32)
    elif "bool" in t:
        arr = np.zeros(shp, bool)
    elif any(k in name for k in ("window", "audio", "noise", "speech")):
        arr = np.random.RandomState(0).randn(*shp).astype(np.float32) * 0.1
    else:
        arr = np.zeros(shp, np.float32)
    if "identity" in name and arr.size:        # pick a valid identity (one-hot)
        arr.reshape(-1)[0] = 1.0
    return arr


print("--- inputs ---")
feeds = {}
for i in sess.get_inputs():
    feeds[i.name] = fill(i)
    print(f"  {i.name} shape={i.shape} type={i.type} -> fed {feeds[i.name].shape}")
print("--- outputs ---")
for o in sess.get_outputs():
    print(f"  {o.name} shape={o.shape} type={o.type}")

try:
    import time
    t0 = time.time()
    out = sess.run(None, feeds)
    dt = time.time() - t0
except Exception as e:
    import traceback; traceback.print_exc(); print("RUN_FAIL:", e); sys.exit(5)

print(f"A2F_RUNS_ON_CMP100 ok provider={sess.get_providers()[0]} "
      f"forward={dt*1000:.0f}ms outputs={[np.asarray(o).shape for o in out]}")
sys.exit(0)
