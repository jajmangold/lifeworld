#!/usr/bin/env python3
"""Dead-simple web UI to tune the teeth/mouth and preview the bite.

Run on the host (needs docker to call teeth.sh):
    python3 render/teeth_ui.py            # then open http://<host>:8771
Edit the sliders -> Render -> see the open mouth (front + 3/4). Writes render/mouth_params.json,
which also drives the real cinematic (scene_cine.py --reuse).
"""
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOT = "/srv/nvme-data/containers/projects/bot"
PARAMS = os.path.join(BOT, "render/mouth_params.json")
IMG = os.path.join(BOT, "output/teeth.png")
PORT = 8771

PARTS = ["upper", "lower", "tongue", "cavity"]
GLOBALS = [("width", 0, 1, 0.01), ("y_drop", -0.03, 0.03, 0.001),
           ("anchor_up", -0.02, 0.02, 0.001), ("drop_scale", 0, 0.15, 0.005),
           ("gate", 0, 0.5, 0.01)]
FIELDS = [("y0", -0.03, 0.03, 0.0005), ("y1", -0.03, 0.03, 0.0005),
          ("z", -0.05, 0.01, 0.001), ("w", 0.2, 1.6, 0.01)]

PAGE = """<!doctype html><html><head><meta charset=utf8><title>teeth tuner</title>
<style>
 body{font:14px system-ui;margin:0;background:#1a1a1e;color:#ddd;display:flex;height:100vh}
 #ctrl{width:430px;overflow:auto;padding:16px;box-sizing:border-box;border-right:1px solid #333}
 #view{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;background:#0c0c0e}
 #view img{max-width:96%;border:1px solid #333;border-radius:6px}
 h3{margin:14px 0 6px;color:#9cf} .g{display:grid;grid-template-columns:70px 1fr 64px;gap:6px;align-items:center;margin:3px 0}
 input[type=range]{width:100%} .v{font-variant-numeric:tabular-nums;color:#aaa;text-align:right}
 .row{display:flex;gap:8px;align-items:center;margin:4px 0}
 button{background:#3a6;color:#fff;border:0;padding:10px 16px;border-radius:6px;font-size:15px;cursor:pointer;width:100%;margin-top:12px}
 button:disabled{background:#555} select{background:#222;color:#ddd;border:1px solid #444;padding:4px}
 .part{border:1px solid #333;border-radius:6px;padding:6px 10px;margin:8px 0} input[type=color]{width:48px;height:24px;border:0;background:none}
 #status{color:#9c9;height:18px;margin-top:6px}
</style></head><body>
<div id=ctrl>
 <div class=row><b>Teeth tuner</b><span style=flex:1></span>
   <label>body <select id=gender><option>female</option><option>male</option></select></label></div>
 <div id=globals></div><div id=parts></div>
 <button id=go onclick=render()>Render (~15s)</button>
 <div id=status></div>
</div>
<div id=view><img id=img src="/teeth.png?t=0"></div>
<script>
const P = __PARAMS__;
const GLOBALS = __GLOBALS__, FIELDS = __FIELDS__, PARTS = __PARTS__;
function hex(c){return '#'+c.map(x=>x.toString(16).padStart(2,'0')).join('')}
function rgb(h){return [1,3,5].map(i=>parseInt(h.slice(i,i+2),16))}
function slider(key,val,lo,hi,st,onin){
  const d=document.createElement('div');d.className='g';
  d.innerHTML=`<label>${key}</label><input type=range min=${lo} max=${hi} step=${st} value=${val}><span class=v>${(+val).toFixed(4)}</span>`;
  const r=d.querySelector('input'),v=d.querySelector('.v');
  r.oninput=()=>{v.textContent=(+r.value).toFixed(4);onin(+r.value)};return d;
}
const G=document.getElementById('globals');
G.innerHTML='<h3>global</h3>';
GLOBALS.forEach(([k,lo,hi,st])=>G.appendChild(slider(k,P[k],lo,hi,st,x=>P[k]=x)));
const PD=document.getElementById('parts');
PARTS.forEach(name=>{
  const box=document.createElement('div');box.className='part';
  box.innerHTML=`<h3>${name}</h3>`;
  FIELDS.forEach(([f,lo,hi,st])=>box.appendChild(slider(f,P[name][f],lo,hi,st,x=>P[name][f]=x)));
  const cr=document.createElement('div');cr.className='row';
  cr.innerHTML=`<label>color</label>`;
  const ci=document.createElement('input');ci.type='color';ci.value=hex(P[name].color);
  ci.oninput=()=>P[name].color=rgb(ci.value);cr.appendChild(ci);box.appendChild(cr);
  PD.appendChild(box);
});
async function render(){
  const b=document.getElementById('go'),s=document.getElementById('status');
  b.disabled=true;s.textContent='rendering...';
  P._gender=document.getElementById('gender').value;
  try{
    const r=await fetch('/render',{method:'POST',body:JSON.stringify(P)});
    const j=await r.json();
    if(j.ok){document.getElementById('img').src='/teeth.png?t='+Date.now();s.textContent='done in '+j.sec+'s';}
    else s.textContent='ERROR: '+j.err;
  }catch(e){s.textContent='ERROR: '+e}
  b.disabled=false;
}
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            cur = json.load(open(PARAMS))
            page = (PAGE.replace("__PARAMS__", json.dumps(cur))
                        .replace("__GLOBALS__", json.dumps(GLOBALS))
                        .replace("__FIELDS__", json.dumps(FIELDS))
                        .replace("__PARTS__", json.dumps(PARTS)))
            self._send(200, "text/html; charset=utf-8", page.encode())
        elif self.path.startswith("/teeth.png"):
            if os.path.exists(IMG):
                self._send(200, "image/png", open(IMG, "rb").read())
            else:
                self._send(404, "text/plain", b"no render yet")
        else:
            self._send(404, "text/plain", b"404")

    def do_POST(self):
        if self.path != "/render":
            return self._send(404, "text/plain", b"404")
        n = int(self.headers.get("Content-Length", 0))
        try:
            p = json.loads(self.rfile.read(n))
            gender = p.pop("_gender", "female")
            cur = json.load(open(PARAMS))
            for k, v in p.items():
                if isinstance(v, dict) and isinstance(cur.get(k), dict):
                    cur[k].update(v)
                else:
                    cur[k] = v
            json.dump(cur, open(PARAMS, "w"), indent=2)
            t = time.time()
            r = subprocess.run(["bash", "render/teeth.sh", "--gender", gender],
                               cwd=BOT, capture_output=True, text=True, timeout=180)
            if "TEETH_OK" not in r.stdout:
                return self._send(200, "application/json",
                                  json.dumps({"ok": False, "err": (r.stderr or r.stdout)[-300:]}).encode())
            self._send(200, "application/json",
                       json.dumps({"ok": True, "sec": round(time.time() - t, 1)}).encode())
        except Exception as e:
            self._send(200, "application/json", json.dumps({"ok": False, "err": str(e)}).encode())


if __name__ == "__main__":
    print(f"teeth tuner -> http://0.0.0.0:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
