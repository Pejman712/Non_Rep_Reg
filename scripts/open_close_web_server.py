#!/usr/bin/env python3
"""
open_close_web_server.py — standalone real-time OPEN/CLOSED (dense/sparse) +
DEGENERATE classifier for any dataset, decoupled from the LIO/mapping.

It reads a dataset bag DIRECTLY (rosbags, no ROS runtime, no registration, no
map), streams the scans over a WebSocket, and for each scan computes ONLY:
  * point DENSITY  — points per occupied cell (subsampled), smoothed over a
    window and compared to a self-calibrating baseline → CLOSED (dense) / OPEN
    (sparse) with hysteresis.
  * DEGENERACY — PCA eigenvalues of the scan → dominant-axis extent + planarity
    ratio → degenerate flag (geometry too concentrated / collapsed to a plane).

The web page shows a big CLOSED/OPEN + DEGENERATE status, live rolling plots of
the signals, and a top-down view of the current scan; all thresholds/window are
live sliders so you can tune the dense-vs-open distinction across datasets.

    python3 open_close_web_server.py            # serve on 0.0.0.0:8079
    python3 open_close_web_server.py --port 8090
"""
import argparse
import asyncio
import json
import os
from collections import deque

import numpy as np
from aiohttp import web, WSMsgType
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

TS = get_typestore(Stores.ROS2_HUMBLE)
DS_ROOT = "/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset"

DATASETS = {
    "Tier Avia": {"dir": f"{DS_ROOT}/Tier/Livox_avia", "topic": "/avia/livox/points",
                  "seqs": ["indoor1_avia", "indoor2_avia", "indoor3_avia", "indoor6_avia"]},
    "Tier Horizon": {"dir": f"{DS_ROOT}/Tier/Livox_horizen", "topic": "/livox/points",
                     "seqs": ["indoor1_horizen", "indoor2_horizen", "indoor3_horizen", "indoor6_horizen"]},
    "iilab Mid-360": {"dir": f"{DS_ROOT}/iilab_benchmark/livox_mid-360", "topic": "/eve/lidar3d",
                      "seqs": ["loop", "nav_a_diff", "slippage", "nav_a_omni"]},
    "CERN L1": {"dir": f"{DS_ROOT}/CERN/unitree_unilidar_L1", "topic": "/unilidar/cloud",
                "seqs": ["BA6", "BA51", "BA52", "BA801", "BA802", "BA803", "927full", "charm", "Dumparea"]},
}

DEFAULTS = dict(voxel=0.3, subsample=4000, window=60, alpha=0.02,
                open_ratio=0.85, close_ratio=1.15,
                degen_extent=0.5, degen_planar=0.01, degen_minpts=80, disp_pts=1200)


def find_bag(group, seq):
    d = DATASETS[group]["dir"]
    return f"{d}/{seq}/{seq}", DATASETS[group]["topic"]


def pick_lidar_conn(reader, preferred):
    """Prefer the configured topic; else the PointCloud2 topic with widest scans."""
    pc2 = [c for c in reader.connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
    for c in pc2:
        if c.topic == preferred:
            return c
    return max(pc2, key=lambda c: c.msgcount) if pc2 else None


def _safe_next(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


def decode_xyz(raw, msgtype):
    m = TS.deserialize_cdr(raw, msgtype)
    n = m.width * m.height
    if n < 10:
        return None
    b = np.frombuffer(m.data, dtype=np.uint8).reshape(n, m.point_step)
    fo = {f.name: f.offset for f in m.fields}
    if not all(k in fo for k in ("x", "y", "z")):
        return None

    def col(k):
        o = fo[k]
        return b[:, o:o + 4].copy().ravel().view(np.float32).astype(np.float64)

    xyz = np.column_stack([col("x"), col("y"), col("z")])
    r = np.linalg.norm(xyz, axis=1)
    return xyz[np.isfinite(r) & (r > 0.3) & (r < 100)]


def classify(xyz, st):
    p = st["params"]
    if xyz is None or len(xyz) < 20:
        return None
    sub = xyz if len(xyz) <= p["subsample"] else \
        xyz[np.linspace(0, len(xyz) - 1, int(p["subsample"])).astype(int)]
    # --- density: points per occupied cell ---
    q = np.floor(sub / p["voxel"]).astype(np.int64)
    n_occ = np.unique(q, axis=0).shape[0]
    dens = len(sub) / max(1, n_occ)
    win = st["win"]
    if win.maxlen != int(p["window"]):
        st["win"] = win = deque(win, maxlen=max(1, int(p["window"])))
    win.append(dens)
    dbar = float(np.mean(win))
    st["base"] = dbar if st["base"] is None else st["base"] + p["alpha"] * (dbar - st["base"])
    ratio = dbar / max(st["base"], 1e-6)
    if ratio < p["open_ratio"]:
        st["cls"] = "OPEN"
    elif ratio > p["close_ratio"]:
        st["cls"] = "CLOSED"
    # --- degeneracy: PCA ---
    c = sub - sub.mean(axis=0)
    w = np.clip(np.linalg.eigvalsh((c.T @ c) / len(sub)), 0.0, None)  # λ0≤λ1≤λ2
    extent = float(np.sqrt(w[2]))
    planar = float(w[0] / max(w[2], 1e-9))
    degen = (extent < p["degen_extent"]) or (planar < p["degen_planar"]) or (len(xyz) < p["degen_minpts"])
    medr = float(np.median(np.linalg.norm(xyz, axis=1)))
    # --- display cloud (top-down) ---
    disp = xyz if len(xyz) <= p["disp_pts"] else \
        xyz[np.linspace(0, len(xyz) - 1, int(p["disp_pts"])).astype(int)]
    dr = np.linalg.norm(disp, axis=1)
    return {"type": "scan", "idx": st["n"], "n": int(len(xyz)),
            "density": round(dens, 3), "dbar": round(dbar, 3), "base": round(st["base"], 3),
            "ratio": round(ratio, 3), "cls": st.get("cls", "CLOSED"),
            "extent": round(extent, 3), "planar": round(planar, 4), "degen": bool(degen),
            "medrange": round(medr, 2),
            "x": [round(float(v), 2) for v in disp[:, 0]],
            "y": [round(float(v), 2) for v in disp[:, 1]],
            "r": [round(float(v), 1) for v in dr]}


# ---- websocket: one playback engine per client -------------------------------
async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    loop = asyncio.get_event_loop()
    st = {"playing": False, "group": None, "seq": None, "rate": 1.0,
          "params": dict(DEFAULTS), "reader": None, "gen": None,
          "win": deque(maxlen=DEFAULTS["window"]), "base": None, "cls": "CLOSED", "n": 0}

    def reset_stream():
        if st["reader"] is not None:
            try:
                st["reader"].close()
            except Exception:
                pass
        st["reader"] = None; st["gen"] = None
        st["win"] = deque(maxlen=int(st["params"]["window"]))
        st["base"] = None; st["n"] = 0; st["cls"] = "CLOSED"

    def open_stream():
        bag, topic = find_bag(st["group"], st["seq"])
        r = Reader(bag); r.open()
        conn = pick_lidar_conn(r, topic)
        st["reader"] = r
        st["gen"] = ((c.msgtype, raw) for c, _, raw in r.messages(connections=[conn]))

    async def play():
        while not ws.closed:
            if not st["playing"] or not st["group"]:
                await asyncio.sleep(0.05); continue
            if st["gen"] is None:
                try:
                    await loop.run_in_executor(None, open_stream)
                except Exception as e:                                    # noqa: BLE001
                    await ws.send_json({"type": "error", "msg": str(e)})
                    st["playing"] = False; continue
            nxt = await loop.run_in_executor(None, _safe_next, st["gen"])
            if nxt is None:                                          # end of sequence
                await ws.send_json({"type": "end"}); st["playing"] = False
                reset_stream(); continue
            mt, raw = nxt
            xyz = await loop.run_in_executor(None, decode_xyz, raw, mt)
            res = classify(xyz, st)
            st["n"] += 1
            if res is not None:
                try:
                    await ws.send_json(res)
                except Exception:
                    break
            await asyncio.sleep(max(0.005, 0.1 / max(st["rate"], 0.01)))

    task = asyncio.ensure_future(play())
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            d = json.loads(msg.data); cmd = d.get("cmd")
            if cmd == "select":
                st["playing"] = False; reset_stream()
                st["group"] = d.get("group"); st["seq"] = d.get("seq")
            elif cmd == "play":
                st["playing"] = True
            elif cmd == "pause":
                st["playing"] = False
            elif cmd == "stop":
                st["playing"] = False; reset_stream()
            elif cmd == "rate":
                st["rate"] = float(d.get("rate", 1.0))
            elif cmd == "params":
                st["params"].update({k: float(v) for k, v in d.get("params", {}).items()})
    finally:
        task.cancel(); reset_stream()
    return ws


async def api_datasets(request):
    return web.json_response({"datasets": {g: v["seqs"] for g, v in DATASETS.items()},
                              "defaults": DEFAULTS})


PAGE = r"""<!doctype html><html><head><meta charset=utf-8><title>open / closed classifier</title>
<style>
 body{margin:0;font-family:ui-monospace,Menlo,Consolas,monospace;background:#0f1216;color:#d6dbe1}
 header{padding:12px 16px;background:#161b22;border-bottom:1px solid #2a313a;display:flex;
   gap:12px;align-items:center;flex-wrap:wrap}
 select,button,input{font:inherit;font-size:13px;background:#0d1117;color:#d6dbe1;
   border:1px solid #30363d;border-radius:5px;padding:5px 8px}
 button{cursor:pointer}#play{background:#2ea043;border:0;color:#fff}#pause{background:#b08800;border:0;color:#fff}
 #stop{background:#da3633;border:0;color:#fff}
 .wrap{display:flex;gap:14px;padding:14px}
 .col{display:flex;flex-direction:column;gap:12px}
 #status{font-size:34px;font-weight:700;text-align:center;padding:18px;border-radius:10px;letter-spacing:1px}
 .closed{background:#123d1e;color:#3fb950;border:2px solid #2ea043}
 .open{background:#0d2b40;color:#58a6ff;border:2px solid #1f6feb}
 #degbadge{display:inline-block;margin-top:8px;padding:4px 12px;border-radius:20px;font-size:14px;font-weight:600}
 .degon{background:#5a1e1e;color:#f85149;border:1px solid #da3633}
 .degoff{background:#21262d;color:#8b949e;border:1px solid #30363d}
 .kv{font-size:12.5px;line-height:1.9}.kv b{color:#8b949e;font-weight:400;display:inline-block;width:92px}
 canvas{background:#0b0e12;border:1px solid #2a313a;border-radius:6px;display:block}
 .sliders{background:#161b22;border:1px solid #2a313a;border-radius:8px;padding:10px 12px;font-size:12px}
 .sl{display:flex;align-items:center;gap:8px;margin:5px 0}.sl label{width:120px;color:#8b949e}
 .sl input[type=range]{flex:1}.sl span{width:52px;text-align:right}
 h3{margin:0 0 6px;font-size:12px;color:#8b949e;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
</style></head><body>
<header>
 <b>OPEN / CLOSED classifier</b>
 <select id=group></select><select id=seq></select>
 <button id=play>▶ Play</button><button id=pause>❚❚</button><button id=stop>■</button>
 <label style=color:#8b949e>rate <input id=rate type=number value=2 step=0.5 style=width:56px></label>
 <span id=info style=margin-left:auto;color:#8b949e;font-size:12px></span>
</header>
<div class=wrap>
 <div class=col style=width:300px>
   <div id=status class=closed>—</div>
   <div style=text-align:center><span id=degbadge class=degoff>geometry OK</span></div>
   <div class=kv id=readout></div>
   <div class=sliders id=sliders><h3>classifier — live</h3></div>
 </div>
 <div class=col>
   <h3>current scan (top-down, coloured by range)</h3>
   <canvas id=cloud width=440 height=440></canvas>
 </div>
 <div class=col style=flex:1>
   <div><h3>density (dbar) vs baseline · open↓ / closed↑</h3><canvas id=pden width=560 height=170></canvas></div>
   <div><h3>degeneracy — planarity ratio (log) · extent</h3><canvas id=pdeg width=560 height=170></canvas></div>
 </div>
</div>
<script>
let OPT, ws, dq=[], plq=[], exq=[], W=300;
const SLIDERS=[
 ["voxel","density cell [m]",0.1,1.0,0.05],["window","smooth window",5,120,5],
 ["alpha","baseline rate",0.005,0.1,0.005],["open_ratio","OPEN if ratio<",0.1,2.0,0.01],
 ["close_ratio","CLOSED if ratio>",0.5,2.0,0.01],["degen_extent","degen extent<[m]",0.1,2.0,0.05],
 ["degen_planar","degen planar<",0.001,0.1,0.001],["subsample","subsample",1000,8000,500]];
function el(id){return document.getElementById(id)}
async function boot(){
 OPT=await (await fetch('/api/datasets')).json();
 for(const g in OPT.datasets){const o=document.createElement('option');o.textContent=g;el('group').appendChild(o)}
 el('group').onchange=fillseq; fillseq();
 // sliders
 for(const[k,lab,mn,mx,stp] of SLIDERS){
   const d=OPT.defaults[k];const row=document.createElement('div');row.className='sl';
   row.innerHTML=`<label>${lab}</label><input type=range min=${mn} max=${mx} step=${stp} value=${d} data-k=${k}><span>${d}</span>`;
   el('sliders').appendChild(row);
   const inp=row.querySelector('input');inp.oninput=()=>{row.querySelector('span').textContent=(+inp.value).toString();sendParams()};
 }
 connect();
}
function fillseq(){el('seq').innerHTML='';for(const s of OPT.datasets[el('group').value]){const o=document.createElement('option');o.textContent=s;el('seq').appendChild(o)}}
function params(){const p={};document.querySelectorAll('#sliders input').forEach(i=>p[i.dataset.k]=+i.value);return p}
function connect(){
 ws=new WebSocket((location.protocol=='https:'?'wss':'ws')+'://'+location.host+'/ws');
 ws.onopen=()=>sendParams();
 ws.onmessage=e=>{const m=JSON.parse(e.data);if(m.type=='scan')render(m);else if(m.type=='end')el('info').textContent='— end of sequence —';else if(m.type=='error')el('info').textContent='error: '+m.msg};
 ws.onclose=()=>setTimeout(connect,1000);
}
function send(o){if(ws&&ws.readyState==1)ws.send(JSON.stringify(o))}
function sendParams(){send({cmd:'params',params:params()})}
el('play').onclick=()=>{send({cmd:'select',group:el('group').value,seq:el('seq').value});send({cmd:'rate',rate:+el('rate').value});sendParams();send({cmd:'play'});dq=[];plq=[];exq=[]};
el('pause').onclick=()=>send({cmd:'pause'});
el('stop').onclick=()=>{send({cmd:'stop'});el('status').textContent='—'};
el('rate').onchange=()=>send({cmd:'rate',rate:+el('rate').value});
el('group').addEventListener('change',fillseq);

function render(m){
 const s=el('status');s.textContent=m.cls;s.className=m.cls=='OPEN'?'open':'closed';
 const db=el('degbadge');db.textContent=m.degen?'DEGENERATE':'geometry OK';db.className=m.degen?'degon':'degoff';
 el('readout').innerHTML=
   `<div><b>scan</b> ${m.idx}  (${m.n} pts)</div>`+
   `<div><b>density</b> ${m.density}  (dbar ${m.dbar})</div>`+
   `<div><b>baseline</b> ${m.base}</div>`+
   `<div><b>ratio</b> ${m.ratio}  → ${m.cls}</div>`+
   `<div><b>med range</b> ${m.medrange} m</div>`+
   `<div><b>PCA extent</b> ${m.extent} m</div>`+
   `<div><b>planarity</b> ${m.planar}</div>`;
 drawCloud(m);
 dq.push(m.ratio);plq.push(m.planar);exq.push(m.extent);
 if(dq.length>W){dq.shift();plq.shift();exq.shift()}
 drawPlot('pden',[[dq,'#17becf',false]],[[+params().open_ratio,'#58a6ff'],[+params().close_ratio,'#3fb950'],[1,'#555']]);
 drawPlot('pdeg',[[plq.map(v=>Math.log10(Math.max(v,1e-4))),'#f85149',false],[exq.map(v=>v/5),'#8c564b',false]],
          [[Math.log10(+params().degen_planar),'#f85149']]);
}
function drawCloud(m){
 const cv=el('cloud'),g=cv.getContext('2d');g.clearRect(0,0,cv.width,cv.height);
 const R=15,cx=cv.width/2,cy=cv.height/2,sc=cv.width/(2*R);
 g.strokeStyle='#1b2027';for(let rr=5;rr<=R;rr+=5){g.beginPath();g.arc(cx,cy,rr*sc,0,7);g.stroke()}
 for(let i=0;i<m.x.length;i++){const px=cx+m.x[i]*sc,py=cy-m.y[i]*sc;const t=Math.min(1,m.r[i]/R);
   g.fillStyle=`hsl(${200*t+20},80%,${60-20*t}%)`;g.fillRect(px,py,2,2)}
 g.fillStyle='#d6dbe1';g.fillRect(cx-2,cy-2,4,4);
}
function drawPlot(id,seriesArr,hlines){
 const cv=el(id),g=cv.getContext('2d'),h=cv.height,w=cv.width;g.clearRect(0,0,w,h);
 let lo=1e9,hi=-1e9;for(const[s]of seriesArr)for(const v of s){if(v<lo)lo=v;if(v>hi)hi=v}
 for(const[y]of hlines){if(y<lo)lo=y;if(y>hi)hi=y}
 if(hi-lo<1e-6){hi+=0.5;lo-=0.5}const pad=(hi-lo)*0.1;lo-=pad;hi+=pad;
 const Y=v=>h-(v-lo)/(hi-lo)*h;
 g.strokeStyle='#30363d';g.setLineDash([3,3]);
 for(const[y,c]of hlines){g.strokeStyle=c;g.beginPath();g.moveTo(0,Y(y));g.lineTo(w,Y(y));g.stroke()}
 g.setLineDash([]);
 for(const[s,c]of seriesArr){g.strokeStyle=c;g.beginPath();
   for(let i=0;i<s.length;i++){const x=i/W*w,y=Y(s[i]);i?g.lineTo(x,y):g.moveTo(x,y)}g.stroke()}
}
boot();
</script></body></html>"""


async def index(request):
    return web.Response(text=PAGE, content_type="text/html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8079)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler),
                    web.get("/api/datasets", api_datasets)])
    host = os.uname().nodename
    print(f"open/closed classifier:  http://{host}:{args.port}   (Ctrl-C to stop)")
    print(f"   remote:  ssh -L {args.port}:localhost:{args.port} Habibip1@{host}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
