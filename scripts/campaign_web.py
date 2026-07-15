#!/usr/bin/env python3
"""
campaign_web.py — tiny WebSocket control panel for the nrlio parameter campaign.

A separate, dependency-light (aiohttp) web page on its OWN port (default 8078,
distinct from bench_web's 8077) with two buttons — Start and Stop — and a live
WebSocket stream of progress: running state, experiments done / total, elapsed,
and the tail of progress.log.  Start launches nrlio_campaign.py (resumable) with
ROS sourced; Stop kills the campaign and its bench-engine children.

    python3 campaign_web.py               # serve on 0.0.0.0:8078
    python3 campaign_web.py --port 8090
"""
import argparse
import asyncio
import importlib.util
import os
import signal
import subprocess
import time

from aiohttp import web, WSMsgType

HERE = os.path.dirname(os.path.abspath(__file__))
ROS_SETUP = "/opt/ros/humble/setup.bash"
WS_SETUP = "/u/97/habibip1/unix/ros2_ws/install/setup.bash"

# which campaign script this panel drives (set from --campaign; default convergence)
CAMPAIGN_PY = os.path.join(HERE, os.environ.get("CAMPAIGN_SCRIPT", "nrlio_campaign.py"))
camp = None


def load_campaign(script):
    """(Re)load the campaign module so the panel reads its paths/queue."""
    global CAMPAIGN_PY, camp
    CAMPAIGN_PY = script if os.path.isabs(script) else os.path.join(HERE, script)
    name = os.path.basename(CAMPAIGN_PY)[:-3]
    spec = importlib.util.spec_from_file_location(name, CAMPAIGN_PY)
    camp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(camp)
    return camp


load_campaign(CAMPAIGN_PY)

STATE = {"proc": None, "started_at": None}


# ---- process control ---------------------------------------------------------
def campaign_running():
    if STATE["proc"] is not None and STATE["proc"].poll() is None:
        return True
    # also catch an externally-started run of THIS campaign script
    return subprocess.run(["pgrep", "-f", os.path.basename(CAMPAIGN_PY)],
                          capture_output=True).returncode == 0


def start_campaign(budget_s, bag_rate):
    if campaign_running():
        return "already running"
    cmd = (f"source {ROS_SETUP} 2>/dev/null; source {WS_SETUP} 2>/dev/null; "
           f"cd {HERE}; exec python3 {os.path.basename(CAMPAIGN_PY)}")
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               CAMPAIGN_BUDGET_S=str(budget_s), CAMPAIGN_BAG_RATE=str(bag_rate))
    STATE["proc"] = subprocess.Popen(["bash", "-c", cmd], env=env,
                                     start_new_session=True)
    STATE["started_at"] = time.time()
    return "started"


def stop_campaign():
    p = STATE["proc"]
    if p is not None and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            pass
    for pat in (os.path.basename(CAMPAIGN_PY), "run_benchmarks", "lib/regnonrep",
                "ros2 bag play", "ros2 launch regnonrep", "imu_rescale", "odom_to_tum"):
        subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
    STATE["proc"] = None
    STATE["started_at"] = None
    return "stopped"


# ---- status snapshot ---------------------------------------------------------
def _total_experiments():
    try:
        return len(camp.build_static_queue()) + 2   # + up to 2 phase-3 confirmations
    except Exception:
        return 0


def status():
    running = campaign_running()
    # experiments fully recorded in master.csv
    done = 0
    queue = []
    try:
        queue = camp.build_static_queue()
        recorded = camp.load_done()   # keyed by (exp_id, rep) in the current campaign
        for exp in queue:
            have = recorded.get((exp["exp_id"], "0"), recorded.get(exp["exp_id"], set()))
            if have.issuperset(set(exp["seqs"])):
                done += 1
    except Exception:
        pass
    total = len(queue) + 2 if queue else _total_experiments()
    # progress.log tail
    log = ""
    try:
        with open(camp.PROG) as f:
            log = "".join(f.readlines()[-40:])
    except OSError:
        log = "(no progress yet)"
    elapsed = ""
    if STATE["started_at"]:
        s = int(time.time() - STATE["started_at"])
        elapsed = f"{s//3600}h{(s%3600)//60:02d}m"
    return {"running": running, "done": done, "total": total,
            "elapsed": elapsed, "log": log}


# ---- web ---------------------------------------------------------------------
PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>nrlio campaign</title><style>
 body{font-family:ui-monospace,Menlo,Consolas,monospace;margin:0;background:#0f1216;color:#d6dbe1}
 header{padding:14px 18px;background:#161b22;border-bottom:1px solid #2a313a;display:flex;
   align-items:center;gap:14px;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
 button{font:inherit;font-size:14px;padding:8px 18px;border:0;border-radius:6px;cursor:pointer;color:#fff}
 #start{background:#2ea043}#start:hover{background:#3fb950}
 #stop{background:#da3633}#stop:hover{background:#f85149}
 button:disabled{opacity:.4;cursor:not-allowed}
 #stat{margin-left:auto;font-size:13px}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:middle}
 .on{background:#3fb950;box-shadow:0 0 6px #3fb950}.off{background:#6e7681}
 #bar{height:6px;background:#21262d}#fill{height:100%;background:#2ea043;width:0;transition:width .4s}
 pre{margin:0;padding:14px 18px;font-size:12.5px;line-height:1.5;white-space:pre-wrap;
   overflow:auto;height:calc(100vh - 96px)}
 label{font-size:12px;color:#8b949e}input{width:52px;font:inherit;background:#0d1117;color:#d6dbe1;
   border:1px solid #30363d;border-radius:4px;padding:3px 5px}
</style></head><body>
<header>
 <h1>nrlio parameter campaign</h1>
 <button id=start>▶ Start</button>
 <button id=stop>■ Stop</button>
 <label>budget h <input id=budget value=14></label>
 <label>rate <input id=rate value=0.5></label>
 <span id=stat><span class=dot off id=dot></span><span id=txt>connecting…</span></span>
</header>
<div id=bar><div id=fill></div></div>
<pre id=log></pre>
<script>
let ws;
function connect(){
 ws=new WebSocket((location.protocol=='https:'?'wss':'ws')+'://'+location.host+'/ws');
 ws.onmessage=e=>{const s=JSON.parse(e.data);render(s);};
 ws.onclose=()=>{document.getElementById('txt').textContent='disconnected — retrying';
   setTimeout(connect,1500);};
}
function render(s){
 document.getElementById('dot').className='dot '+(s.running?'on':'off');
 const pct=s.total?Math.round(100*s.done/s.total):0;
 document.getElementById('txt').textContent=
   (s.running?'RUNNING':'idle')+' · '+s.done+'/'+s.total+' exps'+(pct?' ('+pct+'%)':'')
   +(s.elapsed?' · '+s.elapsed:'');
 document.getElementById('fill').style.width=pct+'%';
 document.getElementById('start').disabled=s.running;
 document.getElementById('stop').disabled=!s.running;
 const log=document.getElementById('log');const atBottom=log.scrollTop+log.clientHeight>=log.scrollHeight-30;
 log.textContent=s.log;if(atBottom)log.scrollTop=log.scrollHeight;
}
function send(cmd){const b=document.getElementById('budget').value,r=document.getElementById('rate').value;
 ws.send(JSON.stringify({cmd,budget:b,rate:r}));}
document.getElementById('start').onclick=()=>send('start');
document.getElementById('stop').onclick=()=>send('stop');
connect();
</script></body></html>"""


async def index(request):
    page = PAGE.replace("nrlio parameter campaign",
                        f"campaign: {os.path.basename(CAMPAIGN_PY)}")
    return web.Response(text=page, content_type="text/html")


async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async def pusher():
        while not ws.closed:
            try:
                await ws.send_json(status())
            except Exception:
                break
            await asyncio.sleep(1.5)

    task = asyncio.ensure_future(pusher())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                import json
                d = json.loads(msg.data)
                cmd = d.get("cmd")
                if cmd == "start":
                    budget = int(float(d.get("budget", 14)) * 3600)
                    start_campaign(budget, d.get("rate", "1.0"))
                elif cmd == "stop":
                    stop_campaign()
                await ws.send_json(status())
    finally:
        task.cancel()
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--campaign", default="nrlio_campaign.py",
                    help="campaign script this panel drives (e.g. nrlio_den_campaign.py)")
    args = ap.parse_args()
    load_campaign(args.campaign)
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler)])
    host = os.uname().nodename
    print(f"campaign control panel [{os.path.basename(CAMPAIGN_PY)}]:  "
          f"http://{host}:{args.port}   (Ctrl-C to stop)")
    print(f"   if remote, SSH-forward:  ssh -L {args.port}:localhost:{args.port} "
          f"Habibip1@{host}")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
