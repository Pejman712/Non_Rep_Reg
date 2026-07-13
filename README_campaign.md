# Campaign web panel — how to run

A simple Start/Stop web page (WebSocket) to run the parameter-optimization
campaigns, with live progress. Two campaigns are available:

| campaign script | what it tunes | default rate |
|---|---|---|
| `nrlio_den_campaign.py` | `nrlio_op_den` density-based p2p↔GICP switch (Tier) | 0.5× |
| `nrlio_campaign.py` | `nrlio_optimized` convergence params (Tier) | 0.5× |

---

## 1. Start the panel (in your terminal on `l23-0499`)

Density-switch campaign:
```bash
nohup python3 /u/97/habibip1/unix/ros2_ws/src/regnonrep/scripts/campaign_web.py \
      --campaign nrlio_den_campaign.py \
      > /u/97/habibip1/unix/campaign_web_den.out 2>&1 &
```

Convergence campaign instead: use `--campaign nrlio_campaign.py`.
Port busy? add `--port 8079`.

## 2. Open it in a browser
```
http://l23-0499:8078
```
Remote: first `ssh -L 8078:localhost:8078 Habibip1@l23-0499`, then open
`http://localhost:8078`.

## 3. Run it
- The page header shows which campaign it drives (e.g. `campaign: nrlio_den_campaign.py`).
- **Rate = 0.5**, **budget = 14** are already set → click **▶ Start**.
- **■ Stop** kills it. It is **resumable** — Start again continues where it left off.
- Live progress bar + scrolling log stream on the page.

---

## Handy checks
```bash
ss -ltn | grep 8078                 # is the panel up?
pkill -f campaign_web.py            # stop the panel (NOT the campaign)
# live progress / final report:
tail -f  .../benchmark_results/_campaign_den/progress.log
cat      .../benchmark_results/_campaign_den/report.md
```
(`_campaign_den/` for the density campaign, `_campaign2/` for convergence.)

## Notes
- **Do not use the bench UI (port 8077) while a campaign runs** — same engine/topics collide.
- Start it from **your own terminal** (`nohup`), not from an assistant session — background jobs there get reaped between turns; `nohup` makes it survive.
- Results land in `benchmark_results/<campaign>/`: `master.csv` (every run),
  `progress.log` (live), `report.md` (analysis, regenerable via
  `python3 <campaign>.py --report`).
