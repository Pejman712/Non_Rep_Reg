#!/usr/bin/env python3
"""
bench_web.py — local web UI for run_benchmarks.sh.

Pick methods, sequences/datasets, bag speed, duration, start-offset, post-flush
wait and dry-run from a browser; the server builds the run_benchmarks.sh command,
runs it, streams the log live, shows a progress bar + ETA, a parsed metrics table
(APE/RMSE per method×sequence), per-method route plots, overlay comparison plots
(all methods on one figure per sequence), and download links for the trajectories.

    python3 bench_web.py [--port 8077] [--host 0.0.0.0]
    then open  http://<this-host>:8077   (SSH-forward the port if remote)

Dependency-free (Python stdlib only).  Only one benchmark runs at a time.
"""
import argparse
import glob
import io
import json
import os
import re
import shlex
import subprocess
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.realpath(__file__))
SCRIPT = os.path.join(HERE, "run_benchmarks.sh")
PLOT = os.path.join(HERE, "plot_tum.py")

ROOT_TIER = "/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/Tier"
ROOT_IILAB = "/u/97/habibip1/unix/point_cloud_registeration_benchmark/dataset/iilab_benchmark"

# dataset → label, default rate, post-wait, root, sensor, {sequence: approx bag len [s]}
DATASETS = {
    "tier_avia": {"label": "Tier / Livox Avia", "default_rate": 0.3, "post": 30,
                  "root": ROOT_TIER, "sensor": "Livox_avia",
                  "seqs": {"indoor1_avia": 115, "indoor2_avia": 42, "indoor3_avia": 47}},
    "tier_horizen": {"label": "Tier / Livox Horizon", "default_rate": 0.3, "post": 30,
                     "root": ROOT_TIER, "sensor": "Livox_horizen",
                     "seqs": {"indoor1_horizen": 114, "indoor2_horizen": 42, "indoor3_horizen": 47}},
    "iilab": {"label": "iilab / livox_mid-360", "default_rate": 0.8, "post": 3,
              "root": ROOT_IILAB, "sensor": "livox_mid-360",
              "seqs": {"nav_a_diff": 757, "nav_a_omni": 388, "loop": 624, "slippage": 92}},
}


def list_methods():
    try:
        out = subprocess.check_output(["bash", SCRIPT, "--list-methods"], text=True)
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception as e:
        return [f"(error: {e})"]


# ── metrics parsing ───────────────────────────────────────────────────────────
HDR_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s*$")
STAT_RE = re.compile(r"^\s*(rmse|mean|median|max|min|std)\s+([-+0-9.eE]+)\s*$")
# extra standard metrics line emitted by run_benchmarks.sh extra_metrics()
METRICS_RE = re.compile(r"\[metrics\] ate_se3=(?P<ate_se3>\S+) "
                        r"are_deg=(?P<are_deg>\S+) rpe_1m=(?P<rpe_1m>\S+)")


def parse_metrics(buf):
    rows, order, cur = {}, [], None
    for line in buf.splitlines():
        h = HDR_RE.match(line)
        if h:
            cur = h.group(3).strip()
            if cur not in rows:
                rows[cur] = {"tag": cur, "status": "running",
                             "rmse": None, "mean": None, "median": None, "max": None,
                             "ate_se3": None, "are_deg": None, "rpe_1m": None}
                order.append(cur)
            continue
        if cur is None:
            continue
        r = rows[cur]
        m = STAT_RE.match(line)
        if m:
            r[m.group(1)] = float(m.group(2))
            if r["status"] in ("running", "ok"):
                r["status"] = "ok"
            continue
        mm = METRICS_RE.search(line)
        if mm:
            for k in ("ate_se3", "are_deg", "rpe_1m"):
                v = mm.group(k)
                if v and v.lower() != "nan":
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
            if r["status"] == "running":
                r["status"] = "ok"
            continue
        if "ERROR" in line and ("no TUM" in line or "not found" in line):
            r["status"] = "failed"
        elif "no ground truth" in line.lower() or "no ground" in line.lower():
            if r["status"] == "running":
                r["status"] = "no-gt"
        elif line.strip().startswith("saved → result") and r["status"] == "running":
            r["status"] = "ok"
        elif "Absolute Trajectory Error" in line:   # iilabs3d eval: ATE → rmse col
            im = re.search(r"([0-9]+\.[0-9]+)\s*m", line)
            if im:
                r["rmse"] = float(im.group(1))
                if r["status"] == "running":
                    r["status"] = "ok"
        elif "Relative Translation Error" in line:   # iilabs3d eval: RTE %
            im = re.search(r"([0-9]+\.[0-9]+)\s*%", line)
            if im:
                r["rte_pct"] = float(im.group(1))
    out = []
    for t in order:
        r = rows[t]
        p = t.split("/")
        r["method"] = p[0] if p else t
        r["dataset"] = p[1] if len(p) > 1 else ""
        r["seq"] = p[2] if len(p) > 2 else ""
        out.append(r)
    return out


# ── per-method diagnostics (parsed from the run logs) ─────────────────────────
# Each is (column, regex).  Counts occurrences across a method's .log + .node.log.
DIAG_PATTERNS = [
    ("dropped",    re.compile(r"sync failed|skip this scan|too old|discard(ed)?|drop(ped)? scan", re.I)),
    ("empty",      re.compile(r"no point|too few|empty (cloud|scan)", re.I)),
    ("degenerate", re.compile(r"degenerate|singular|not enough (corr|points)|chi2|umeyama|ill[- ]conditioned", re.I)),
    ("fallback",   re.compile(r"fall ?back|identity init|imu[- ]only|imu propagation|no ground", re.I)),
    ("reg_fail",   re.compile(r"registration fail|align(?:ment)? fail|gicp fail|effect num\s*:\s*0\b", re.I)),
    ("crash",      re.compile(r"malloc\(\)|segmentation|core dumped|\baborted\b|terminate called|what\(\):|bad_alloc", re.I)),
    ("no_tum",     re.compile(r"no tum written", re.I)),
    ("warn",       re.compile(r"\bwarn(ing)?\b|\[warn\]|W\d{4} ", re.I)),
    ("error",      re.compile(r"\berror\b|\[error\]|E\d{4} ", re.I)),
]

# per-scan processing time (ms): captures (mean, p95).  The "per-scan processing"
# line is the exact compute time from lio_base (regnonrep variants); the
# "inter-pose wall interval" line is the recorder's method-agnostic proxy.
PROC_SCAN_RE = re.compile(r"per-scan processing: mean=([0-9.]+) median=[0-9.]+ p95=([0-9.]+)")
PROC_NODE_RE = re.compile(r"node cpu: total=[0-9.]+s poses=\d+ mean=([0-9.]+) ms/scan")  # external exact
PROC_POSE_RE = re.compile(r"inter-pose wall interval: mean=([0-9.]+) median=[0-9.]+ p95=([0-9.]+)")
PROC_PERIODIC_RE = re.compile(r"proc_ms=[0-9.]+\(avg ([0-9.]+)\)")  # regnonrep periodic fallback


def parse_diagnostics(logdir):
    """Scan every <tag>.log / <tag>.node.log and count debug-relevant events."""
    if not logdir or not os.path.isdir(logdir):
        return []
    rows = {}
    for fn in sorted(os.listdir(logdir)):
        if fn.endswith(".node.log"):
            tag = fn[:-len(".node.log")]
        elif fn.endswith(".log"):
            tag = fn[:-len(".log")]
        else:
            continue
        try:
            txt = open(os.path.join(logdir, fn), errors="replace").read()
        except Exception:
            continue
        r = rows.setdefault(tag, {"tag": tag, "lines": 0, "files": [],
                                  "proc_ms": None, "proc_p95": None, "proc_src": "",
                                  **{k: 0 for k, _ in DIAG_PATTERNS}})
        r["lines"] += txt.count("\n")
        r["files"].append(fn)
        for name, rx in DIAG_PATTERNS:
            r[name] += len(rx.findall(txt))
        # per-scan processing time, best-source-wins:
        #   scan  = exact per-scan compute (regnonrep / lio_base)
        #   cpu   = exact CPU-time/scan of the external node (bag-independent)
        #   pose  = inter-pose wall-interval proxy (bag-confounded, last resort)
        m = PROC_SCAN_RE.search(txt)
        if m:
            r["proc_ms"], r["proc_p95"], r["proc_src"] = float(m.group(1)), float(m.group(2)), "scan"
        if r["proc_src"] != "scan":
            m = PROC_NODE_RE.search(txt)
            if m:
                r["proc_ms"], r["proc_p95"], r["proc_src"] = float(m.group(1)), None, "cpu"
        if r["proc_src"] == "":
            m = PROC_POSE_RE.search(txt)
            if m:
                r["proc_ms"], r["proc_p95"], r["proc_src"] = float(m.group(1)), float(m.group(2)), "pose"
        if r["proc_src"] == "":                   # no summary line (e.g. hard kill)
            pm = PROC_PERIODIC_RE.findall(txt)
            if pm:
                r["proc_ms"], r["proc_src"] = float(pm[-1]), "scan~"
    out = []
    for tag in sorted(rows):
        r = rows[tag]
        p = tag.split("__")
        r["method"] = p[0] if p else tag
        r["seq"] = p[1] if len(p) > 1 else ""
        r["ok"] = (r["crash"] == 0 and r["no_tum"] == 0)
        out.append(r)
    return out


def generate_overlays(logdir):
    """One comparison figure per sequence: GT + every method that produced a result."""
    tdir = os.path.join(logdir, "trajectories")
    pdir = os.path.join(logdir, "plots")
    if not os.path.isdir(tdir):
        return
    for ds, info in DATASETS.items():
        for seq in info["seqs"]:
            ests = sorted(glob.glob(os.path.join(tdir, f"*__{ds}_{seq}.tum")))
            if len(ests) < 1:
                continue
            gt = os.path.join(info["root"], info["sensor"], seq, seq + ".tum")
            out = os.path.join(pdir, f"zz_overlay__{ds}_{seq}.png")
            cmd = ["python3", PLOT, "--no-show"]
            if os.path.isfile(gt):
                cmd += ["--gt", gt]
            cmd += ests + [out]
            try:
                subprocess.run(cmd, check=False, capture_output=True, timeout=120)
            except Exception:
                pass


# ── shared run state ──────────────────────────────────────────────────────────
class Run:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.buf = ""
        self.running = False
        self.logdir = None
        self.cmd = ""
        self.rc = None
        self.current = 0
        self.total = 0
        self.started = 0.0
        self.dry = False

    def start(self, args, dry):
        with self.lock:
            if self.running:
                return False, "a benchmark is already running"
            self.buf, self.logdir, self.rc = "", None, None
            self.current, self.total, self.dry = 0, 0, dry
            self.running, self.started = True, time.time()
            self.cmd = "run_benchmarks.sh " + " ".join(shlex.quote(a) for a in args)
        cmdstr = ("source /opt/ros/humble/setup.bash 2>/dev/null; exec bash "
                  + shlex.quote(SCRIPT) + " " + " ".join(shlex.quote(a) for a in args))
        self.proc = subprocess.Popen(["bash", "-c", cmdstr], cwd=HERE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        return True, "started"

    def _reader(self):
        for line in self.proc.stdout:
            with self.lock:
                self.buf += line
                if self.logdir is None and "benchmark_results/" in line:
                    for w in line.split():
                        if "benchmark_results/" in w:
                            pre, post = w.split("benchmark_results/", 1)
                            self.logdir = os.path.realpath(pre + "benchmark_results/" + post.split("/")[0])
                            break
                h = HDR_RE.match(line)
                if h:
                    self.current, self.total = int(h.group(1)), int(h.group(2))
        self.proc.wait()
        if self.logdir and not self.dry:
            generate_overlays(self.logdir)
        with self.lock:
            self.rc = self.proc.returncode
            self.running = False

    def stop(self):
        p = self.proc
        if p and p.poll() is None:
            subprocess.run(["pkill", "-9", "-P", str(p.pid)], check=False)
            p.terminate()
        # regnonrep variants + the converter/recorder + every external SOTA node
        # (the external LIO nodes are grandchildren of the script, so pkill -P on
        # our pid does not reach them — kill them by executable name too).
        for pat in ("lib/regnonrep/", "ros2 bag play", "ros2 launch regnonrep",
                    "livox_to_velodyne", "imu_rescale", "odom_to_tum",
                    "fastlio_mapping", "run_mapping_online", "ig_lio_node",
                    "pointlio_mapping", "super_lio_node", "dlio_odom_node"):
            subprocess.run(["pkill", "-9", "-f", pat], check=False)

    def get_logdir(self):
        with self.lock:
            return self.logdir

    def snapshot(self, offset):
        with self.lock:
            return {"text": self.buf[offset:], "offset": len(self.buf),
                    "running": self.running, "rc": self.rc, "logdir": self.logdir,
                    "current": self.current, "total": self.total,
                    "elapsed": round(time.time() - self.started, 1) if self.started else 0}

    def metrics(self):
        with self.lock:
            return parse_metrics(self.buf)


RUN = Run()
METHODS = list_methods()
# external "state of the art" packages (vs the regnonrep lio_base variants)
SOTA = ["dlio", "fast_lio", "ig_lio", "point_lio", "super_lio"]  # faster_lio dropped (crashes)
VARIANTS = [m for m in METHODS if m not in SOTA]
SOTA_PRESENT = [m for m in METHODS if m in SOTA]


def build_args(p):
    args = []
    methods = p.get("methods") or []
    if methods and len(methods) < len(METHODS):
        args.append("--methods=" + ",".join(methods))
    if p.get("sequences"):
        args.append("--sequences=" + ",".join(p["sequences"]))
    if p.get("bag_rate"):
        args.append("--bag-rate=" + str(p["bag_rate"]))
    if int(p.get("duration") or 0) > 0:
        args.append("--duration=" + str(int(p["duration"])))
    try:
        so = float(p.get("start_offset"))
    except (TypeError, ValueError):
        so = 5.0
    if so != 5.0:                                  # 5 s is the script default
        args.append("--start-offset=" + repr(so))
    if p.get("post_wait"):
        args.append("--post-wait=" + str(p["post_wait"]))
    if p.get("dry_run"):
        args.append("--dry-run")
    return args


# ── HTTP handler ──────────────────────────────────────────────────────────────
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if u.path == "/api/options":
            return self._send(200, {"methods": METHODS, "variants": VARIANTS,
                                    "sota": SOTA_PRESENT, "datasets": DATASETS})
        if u.path == "/api/log":
            return self._send(200, RUN.snapshot(int(q.get("offset", ["0"])[0])))
        if u.path == "/api/metrics":
            return self._send(200, {"rows": RUN.metrics()})
        if u.path == "/api/diagnostics":
            return self._send(200, {"rows": parse_diagnostics(RUN.get_logdir()),
                                    "cols": [k for k, _ in DIAG_PATTERNS]})
        if u.path == "/api/results":
            return self._send(200, self._results())
        if u.path == "/plot":
            return self._serve(q.get("file", [""])[0], "plots", "image/png", ".png")
        if u.path == "/traj":
            return self._serve(q.get("file", [""])[0], "trajectories", "text/plain", ".tum")
        if u.path == "/logfile":
            return self._serve_log(q.get("file", [""])[0])
        if u.path == "/download_all":
            return self._zip_all()
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            p = json.loads(self.rfile.read(n).decode() if n else "{}")
        except Exception:
            p = {}
        if u.path == "/api/run":
            args = build_args(p)
            ok, msg = RUN.start(args, bool(p.get("dry_run")))
            return self._send(200 if ok else 409, {"ok": ok, "msg": msg,
                              "cmd": "run_benchmarks.sh " + " ".join(args)})
        if u.path == "/api/stop":
            RUN.stop()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})

    def _results(self):
        logdir = RUN.get_logdir()
        res = {"logdir": logdir, "overlays": [], "plots": [], "trajectories": []}
        if not logdir:
            return res
        pd = os.path.join(logdir, "plots")
        if os.path.isdir(pd):
            for n in sorted(os.listdir(pd)):
                if n.endswith(".png"):
                    item = {"name": n, "url": "/plot?file=" + n}
                    (res["overlays"] if n.startswith("zz_overlay__") else res["plots"]).append(item)
        td = os.path.join(logdir, "trajectories")
        if os.path.isdir(td):
            res["trajectories"] = [{"name": n, "url": "/traj?file=" + n}
                                   for n in sorted(os.listdir(td)) if n.endswith(".tum")]
        res["logs"] = [{"name": n, "url": "/logfile?file=" + n}
                       for n in sorted(os.listdir(logdir)) if n.endswith(".log")]
        return res

    def _zip_all(self):
        logdir = RUN.get_logdir()
        if not logdir or not os.path.isdir(logdir):
            return self._send(404, b"no run yet", "text/plain")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("plots", "trajectories"):
                d = os.path.join(logdir, sub)
                if os.path.isdir(d):
                    for n in sorted(os.listdir(d)):
                        z.write(os.path.join(d, n), arcname=f"{sub}/{n}")
            for n in sorted(os.listdir(logdir)):
                if n.endswith(".log"):
                    z.write(os.path.join(logdir, n), arcname=f"logs/{n}")
        data = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition",
                         f'attachment; filename="benchmark_{os.path.basename(logdir)}.zip"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve(self, fname, sub, ctype, ext):
        logdir = RUN.get_logdir()
        fname = os.path.basename(fname)  # path-traversal guard
        if not logdir:
            return self._send(404, b"no run", "text/plain")
        path = os.path.join(logdir, sub, fname)
        if not (fname.endswith(ext) and os.path.isfile(path)):
            return self._send(404, b"not found", "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype)

    def _serve_log(self, fname):
        logdir = RUN.get_logdir()
        fname = os.path.basename(fname)  # path-traversal guard
        if not logdir:
            return self._send(404, b"no run", "text/plain")
        path = os.path.join(logdir, fname)  # .log files live at the run root
        if not (fname.endswith(".log") and os.path.isfile(path)):
            return self._send(404, b"not found", "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), "text/plain; charset=utf-8")


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>regnonrep benchmark</title>
<style>
 :root{--bg:#0e1116;--panel:#171b22;--ac:#4f9cf9;--ok:#3fb950;--bad:#f85149;--mut:#8b949e}
 *{box-sizing:border-box} body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:#e6edf3}
 header{padding:12px 18px;background:#11151c;border-bottom:1px solid #232a33;font-weight:600}
 .wrap{display:flex;gap:14px;padding:14px;align-items:flex-start}
 .col{background:var(--panel);border:1px solid #232a33;border-radius:8px;padding:14px}
 .left{width:430px;flex:none} .right{flex:1;min-width:0}
 h3{margin:6px 0 8px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.5px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px}
 label.chk{display:flex;gap:6px;align-items:center;cursor:pointer;padding:2px}
 .dsblock{border:1px solid #232a33;border-radius:6px;padding:8px;margin-bottom:8px}
 .dshead{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-weight:600}
 input[type=number]{width:80px;background:#0e1116;color:#e6edf3;border:1px solid #2a313b;border-radius:5px;padding:4px}
 select{background:#0e1116;color:#e6edf3;border:1px solid #2a313b;border-radius:5px;padding:4px}
 .row{display:flex;gap:14px;align-items:center;margin:6px 0;flex-wrap:wrap}
 button{background:var(--ac);color:#fff;border:0;border-radius:6px;padding:9px 16px;font-weight:600;cursor:pointer}
 button.sec{background:#2a313b} button:disabled{opacity:.5;cursor:default} button.stop{background:var(--bad)}
 .est{font-size:13px;color:var(--mut)} .est b{color:#e6edf3}
 .pbar{height:10px;background:#0a0d12;border:1px solid #232a33;border-radius:6px;overflow:hidden;flex:1;min-width:160px}
 .pbar>div{height:100%;width:0;background:var(--ac);transition:width .4s}
 #log{height:34vh;overflow:auto;background:#0a0d12;border:1px solid #232a33;border-radius:6px;padding:10px;
      white-space:pre-wrap;font:12px/1.4 ui-monospace,monospace}
 .pill{font-size:12px;padding:2px 8px;border-radius:10px;background:#2a313b}
 .pill.run{background:#1f6feb} .pill.ok{background:var(--ok)} .pill.bad{background:var(--bad)}
 table{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
 th,td{border-bottom:1px solid #232a33;padding:4px 8px;text-align:right} th{color:var(--mut);text-align:right;cursor:pointer}
 th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
 td.s-ok{color:var(--ok)} td.s-failed{color:var(--bad)} td.s-running{color:var(--ac)} td.s-no-gt{color:var(--mut)}
 .gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;margin-top:10px}
 .gal figure{margin:0;background:#0a0d12;border:1px solid #232a33;border-radius:6px;padding:6px}
 .gal img{width:100%;border-radius:4px;cursor:zoom-in} .gal figcaption{font-size:11px;color:var(--mut);word-break:break-all}
 a.mini{color:var(--ac);font-size:12px;text-decoration:none} .muted{color:var(--mut)}
 details>summary{cursor:pointer;color:var(--mut);margin:10px 0 4px}
</style></head><body>
<header>regnonrep benchmark &nbsp;·&nbsp; web control</header>
<div class=wrap>
 <div class="col left">
   <h3>Methods (regnonrep)</h3>
   <div class=row><a class=mini href=# onclick="allMethods(1);return false">all</a>
        <a class=mini href=# onclick="allMethods(0);return false">none</a></div>
   <div id=methods class=grid></div>
   <h3 style=margin-top:14px>State-of-the-art methods</h3>
   <div class=row><a class=mini href=# onclick="allSota(1);return false">all</a>
        <a class=mini href=# onclick="allSota(0);return false">none</a></div>
   <div id=sota class=grid></div>
   <h3 style=margin-top:14px>Datasets &amp; sequences</h3>
   <div id=datasets></div>
   <h3 style=margin-top:14px>Run parameters</h3>
   <div class=row><label>Bag rate
        <select id=rate><option value="">per-dataset default</option>
        <option>0.1</option><option>0.2</option><option>0.3</option><option>0.5</option>
        <option>0.8</option><option>1.0</option><option>1.5</option><option>2.0</option></select></label></div>
   <div class=row>
     <label>Duration (s, 0=full) <input id=dur type=number value=0 min=0></label>
     <label>Start offset (s) <input id=off type=number value=5 min=0></label></div>
   <div class=row><label>Post-flush wait (s, blank=default) <input id=wait type=number min=0 placeholder=auto></label></div>
   <div class=row><label class=chk><input type=checkbox id=dry> dry-run (no launch)</label></div>
   <div class=row><div class=est id=est></div></div>
   <div class=row>
     <button id=runbtn onclick=startRun()>▶ Run benchmark</button>
     <button class="sec stop" id=stopbtn onclick=stopRun() disabled>■ Stop</button></div>
 </div>

 <div class="col right">
   <div class=row><h3 style=margin:0>Progress</h3><span id=status class=pill>idle</span>
       <div class=pbar><div id=pfill></div></div><span class=muted id=ptext></span></div>
   <div class=muted id=cmd style=font-size:12px></div>
   <details open><summary>Live log</summary><div id=log></div></details>

   <div class=row style=margin-top:6px><h3 style=margin:0>Metrics</h3>
       <span class=muted>(click a header to sort)</span></div>
   <div id=metrics><span class=muted>no metrics yet</span></div>

   <div class=row style=margin-top:12px><h3 style=margin:0>Diagnostics</h3>
       <span class=muted>(per-method event counts parsed from the logs)</span>
       <button class=sec onclick=loadDiagnostics()>refresh</button></div>
   <div id=diag><span class=muted>no diagnostics yet</span></div>

   <div class=row style=margin-top:12px><h3 style=margin:0>Results</h3>
       <button class=sec onclick=loadResults()>refresh</button>
       <a class=mini href="/download_all" download>⬇ download all (plots + tums + logs, .zip)</a></div>
   <div id=overlays class=gal></div>
   <details><summary>Per-method plots</summary><div id=gallery class=gal></div></details>
   <details><summary>Download trajectories (.tum)</summary><div id=trajs class=muted></div></details>
   <details><summary>Node / run logs (.log)</summary><div id=logs class=muted></div></details>
 </div>
</div>
<script>
let OPT=null, off=0, poll=null, sortKey='', sortAsc=true, lastRows=[];
async function init(){
  OPT=await (await fetch('/api/options')).json();
  document.getElementById('methods').innerHTML=(OPT.variants||OPT.methods).map(m=>
    `<label class=chk><input type=checkbox class="m mv" value="${m}" checked onchange=estimate()> ${m}</label>`).join('');
  document.getElementById('sota').innerHTML=(OPT.sota||[]).map(m=>
    `<label class=chk><input type=checkbox class="m ms" value="${m}" checked onchange=estimate()> ${m}</label>`).join('');
  let h='';
  for(const [ds,info] of Object.entries(OPT.datasets)){
    h+=`<div class=dsblock><div class=dshead><label class=chk><input type=checkbox class=dsall data-ds="${ds}" checked onchange=toggleDs("${ds}",this.checked)> ${info.label}</label>`
      +`<span class=muted>def ${info.default_rate}x</span></div><div class=grid>`;
    for(const s of Object.keys(info.seqs))
      h+=`<label class=chk><input type=checkbox class=s data-ds="${ds}" value="${s}" checked onchange=estimate()> ${s}</label>`;
    h+='</div></div>';
  }
  document.getElementById('datasets').innerHTML=h; estimate();
}
function allMethods(v){document.querySelectorAll('.mv').forEach(c=>c.checked=v);estimate();}
function allSota(v){document.querySelectorAll('.ms').forEach(c=>c.checked=v);estimate();}
function toggleDs(ds,v){document.querySelectorAll(`.s[data-ds="${ds}"]`).forEach(c=>c.checked=v);estimate();}
function selMethods(){return [...document.querySelectorAll('.m:checked')].map(c=>c.value);}
function selSeqs(){return [...document.querySelectorAll('.s:checked')].map(c=>c.value);}
function estimate(){
  const nm=selMethods().length, rate=document.getElementById('rate').value;
  const dur=+document.getElementById('dur').value, o=+document.getElementById('off').value;
  const wf=document.getElementById('wait').value; let per=0;
  for(const [ds,info] of Object.entries(OPT.datasets)){
    const r=rate?+rate:info.default_rate, wait=wf!==''?+wf:info.post;
    for(const c of document.querySelectorAll(`.s[data-ds="${ds}"]:checked`)){
      const bag=info.seqs[c.value], play=dur>0?Math.min(dur,(bag-o)/r):(bag-o)/r;
      per+=play+1+3+wait+15+3;}}
  const tot=nm*per, runs=nm*selSeqs().length;
  document.getElementById('est').innerHTML=
    `Selected: <b>${nm}</b> method(s) × <b>${selSeqs().length}</b> sequence(s) = <b>${runs}</b> run(s) · est. <b>${fmt(tot)}</b>`;
}
function fmt(s){s=Math.round(s);const h=Math.floor(s/3600),m=Math.round((s%3600)/60);return h?`${h}h ${m}m`:`${m}m ${s%60|0}s`.replace(/ 0s$/,'');}
['rate','dur','off','wait'].forEach(id=>document.getElementById(id).addEventListener('input',estimate));

async function startRun(){
  const body={methods:selMethods(),sequences:selSeqs(),
    bag_rate:document.getElementById('rate').value,duration:document.getElementById('dur').value,
    start_offset:document.getElementById('off').value,post_wait:document.getElementById('wait').value,
    dry_run:document.getElementById('dry').checked};
  if(!body.methods.length||!body.sequences.length){alert('pick at least one method and one sequence');return;}
  const r=await (await fetch('/api/run',{method:'POST',body:JSON.stringify(body)})).json();
  if(!r.ok){alert(r.msg);return;}
  document.getElementById('cmd').textContent=r.cmd;
  document.getElementById('log').textContent=''; off=0;
  document.getElementById('runbtn').disabled=true; document.getElementById('stopbtn').disabled=false;
  if(poll)clearInterval(poll); poll=setInterval(tick,1000); tick();
}
async function stopRun(){await fetch('/api/stop',{method:'POST'});}
async function tick(){
  const r=await (await fetch('/api/log?offset='+off)).json();
  if(r.text){const l=document.getElementById('log');const atBot=l.scrollHeight-l.scrollTop-l.clientHeight<40;
    l.textContent+=r.text; off=r.offset; if(atBot)l.scrollTop=l.scrollHeight;}
  // progress
  const pct=r.total?Math.round(100*(r.current-(r.running?1:0))/r.total):(r.rc===0?100:0);
  document.getElementById('pfill').style.width=Math.max(0,pct)+'%';
  if(r.total){const done=Math.max(0,r.current-1),eta=done>0?r.elapsed/done*(r.total-done):0;
    document.getElementById('ptext').textContent=`run ${r.current}/${r.total} · ${fmt(r.elapsed)} elapsed`+(eta>0&&r.running?` · ~${fmt(eta)} left`:'');}
  const st=document.getElementById('status');
  if(r.running){st.textContent='running';st.className='pill run';}
  loadMetrics(); loadDiagnostics();
  if(!r.running){clearInterval(poll);poll=null;
    document.getElementById('runbtn').disabled=false;document.getElementById('stopbtn').disabled=true;
    if(r.rc===0){st.textContent='done ✓';st.className='pill ok';document.getElementById('pfill').style.width='100%';}
    else if(r.rc===null){st.textContent='idle';st.className='pill';}
    else{st.textContent='exited rc='+r.rc;st.className='pill bad';}
    loadResults();}
}
async function loadMetrics(){
  const r=await (await fetch('/api/metrics')).json(); lastRows=r.rows||[]; renderMetrics();
}
function renderMetrics(){
  const el=document.getElementById('metrics');
  if(!lastRows.length){el.innerHTML='<span class=muted>no metrics yet</span>';return;}
  let rows=[...lastRows];
  if(sortKey){rows.sort((a,b)=>{let x=a[sortKey],y=b[sortKey];
    if(x==null)x=Infinity;if(y==null)y=Infinity;
    if(typeof x==='string')return sortAsc?x.localeCompare(y):y.localeCompare(x);
    return sortAsc?x-y:y-x;});}
  const cols=[['method','method'],['dataset','dataset'],['seq','seq'],['rmse','ATE·rmse (orig)'],['mean','mean'],['median','median'],['max','max'],['ate_se3','ATE·SE3'],['are_deg','rot°'],['rpe_1m','RPE/1m'],['status','status']];
  const num=v=>v==null?'–':(v<0.01?v.toExponential(2):v.toFixed(4));
  let h='<table><thead><tr>'+cols.map(c=>`<th onclick="setSort('${c[0]}')" title="${({rmse:'origin-aligned ATE RMSE [m] (raw drift, no Umeyama)',ate_se3:'SE(3)-aligned ATE RMSE [m] (Umeyama — published-comparison standard)',are_deg:'rotation ATE RMSE [deg]','rpe_1m':'translation RPE RMSE per 1 m [m] (relative drift)'})[c[0]]||''}">${c[1]}${sortKey===c[0]?(sortAsc?' ▲':' ▼'):''}</th>`).join('')+'</tr></thead><tbody>';
  for(const r of rows)h+=`<tr><td>${r.method}</td><td>${r.dataset}</td><td>${r.seq}</td>`
    +`<td>${num(r.rmse)}</td><td>${num(r.mean)}</td><td>${num(r.median)}</td><td>${num(r.max)}</td>`
    +`<td>${num(r.ate_se3)}</td><td>${num(r.are_deg)}</td><td>${num(r.rpe_1m)}</td>`
    +`<td class="s-${r.status}">${r.status}</td></tr>`;
  el.innerHTML=h+'</tbody></table>';
}
function setSort(k){if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true;}renderMetrics();}
async function loadDiagnostics(){
  const r=await (await fetch('/api/diagnostics')).json();
  const el=document.getElementById('diag'); const rows=r.rows||[];
  if(!rows.length){el.innerHTML='<span class=muted>no diagnostics yet</span>';return;}
  const cols=r.cols||['dropped','empty','degenerate','fallback','reg_fail','crash','no_tum','warn','error'];
  const hdr=['method','seq','proc ms (p95)','lines'].concat(cols);
  let h='<table><thead><tr>'+hdr.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  for(const d of rows){
    const bad = d.crash||d.no_tum;
    const srcInfo={scan:'exact per-scan compute (regnonrep)',cpu:'exact CPU-time per scan (external, bag-independent)',pose:'inter-pose wall interval — PROXY, bag-confounded','scan~':'periodic running-avg (fallback)'};
    const pm=(d.proc_ms==null)?'–':`${d.proc_ms.toFixed(1)}${d.proc_p95!=null?' ('+d.proc_p95.toFixed(1)+')':''}${d.proc_src==='pose'?'*':''}`;
    h+=`<tr${bad?' style="color:var(--bad)"':''}><td>${d.method}</td><td>${d.seq}</td><td title="${srcInfo[d.proc_src]||''}">${pm}</td><td>${d.lines}</td>`;
    for(const c of cols){const v=d[c]||0;
      const col=(v>0&&['crash','no_tum','reg_fail','error'].includes(c))?'color:var(--bad)':
                (v>0&&['dropped','empty','degenerate','fallback','warn'].includes(c))?'color:#d29922':'';
      h+=`<td style="${col}">${v}</td>`;}
    h+='</tr>';
  }
  el.innerHTML=h+'</tbody></table>';
}
async function loadResults(){
  const r=await (await fetch('/api/results')).json();
  const ov=document.getElementById('overlays');
  ov.innerHTML=(r.overlays&&r.overlays.length)?r.overlays.map(p=>
    `<figure><img src="${p.url}" onclick="window.open(this.src)"><figcaption>${p.name.replace('zz_overlay__','')}</figcaption></figure>`).join('')
    :'<span class=muted>no comparison plots yet</span>';
  const g=document.getElementById('gallery');
  g.innerHTML=(r.plots&&r.plots.length)?r.plots.map(p=>
    `<figure><img src="${p.url}" onclick="window.open(this.src)"><figcaption>${p.name}</figcaption></figure>`).join('')
    :'<span class=muted>none</span>';
  const t=document.getElementById('trajs');
  t.innerHTML=(r.trajectories&&r.trajectories.length)?r.trajectories.map(x=>
    `<a class=mini href="${x.url}" download>${x.name}</a>`).join('<br>'):'none';
  const lg=document.getElementById('logs');
  lg.innerHTML=(r.logs&&r.logs.length)?r.logs.map(x=>
    `<a class=mini href="${x.url}" target=_blank>${x.name}</a>`).join('<br>'):'none';
}
init();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    host = a.host if a.host != "0.0.0.0" else (os.uname().nodename or "localhost")
    print(f"benchmark web UI:  http://{host}:{a.port}   (Ctrl-C to stop)")
    print(f"   if remote, SSH-forward:  ssh -L {a.port}:localhost:{a.port} {os.getenv('USER','user')}@{host}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
