#!/usr/bin/env python3.10
"""
lio_gen_lio_tier4.py  —  gen_liotier4: Tier 4 (intensity) on top of Tier 2.

Ablation ladder: gen_lio ⊂ tier1 ⊂ tier2 ⊂ tier4 (intensity).
Tier 4 adds an INTENSITY / photometric constraint, the standard remedy for the
geometrically-unobservable direction in narrow-FoV indoor scenes (COIN-LIO,
intensity-augmented solid-state SLAM):  along a flat wall geometry gives no
constraint, but the wall's *intensity* usually varies (markings, signs, material
edges), so matching intensity constrains the along-wall motion.

Mechanism (self-contained, numpy only, fail-safe):
  * a voxel INTENSITY MAP accumulates mean reflectivity per world voxel;
  * on weakly-localizable (near-degenerate) scans it estimates the local intensity
    gradient ∇I from the map (finite differences) and solves a small translation
    correction Δp so the scan's intensities line up with the map
    (Lucas-Kanade / photometric:  g·Δp ≈ I_scan − I_map);
  * Δp is clipped and fed as a χ²-gated absolute-pose measurement, so a bad one
    cannot corrupt the state.  Only translation is corrected (the along-wall dof).

This is v1 — a reproduction of published intensity-augmentation adapted to this
pipeline, to be tuned/validated (t4_voxel, t4_apply_loc, t4_max_shift, t4_noise).

    ros2 run regnonrep lio_gen_lio_tier4.py --ros-args -p t4_apply_loc:=0.05
"""
import numpy as np

from lio_base import run_node
from lio_gen_lio_tier2 import SuperLioGenT2

_PK_OFF = 1 << 19
_PK_A = 1 << 20


def _pack(keys):                       # keys (N,3) int -> (N,) int64
    return ((keys[:, 0] + _PK_OFF) * _PK_A + (keys[:, 1] + _PK_OFF)) * _PK_A \
        + (keys[:, 2] + _PK_OFF)


class SuperLioGenT4(SuperLioGenT2):
    NODE_NAME = "super_lio_gen_t4"
    VARIANT_DESC = "+ T4: intensity/photometric constraint (COIN-LIO-style)"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        self.t4_enable = bool(gp("t4_enable", True).value)
        self.t4_voxel = float(gp("t4_voxel", 0.30).value)      # intensity-map voxel [m]
        self.t4_max_pts = int(gp("t4_max_pts", 1500).value)    # scan subsample cap
        self.t4_max_shift = float(gp("t4_max_shift", 0.15).value)  # clip |Δp| [m]
        # apply the intensity correction when localizability < this.  Set high
        # (e.g. 1.0) to apply on every scan (COIN-LIO-style always-on) for A/B;
        # low (0.02) to only rescue near-degenerate scans.
        self.t4_apply_loc = float(gp("t4_apply_loc", 0.20).value)
        self.t4_noise = float(gp("t4_noise", 0.1).value)       # intensity-update meas noise
        self.t4_min_voxels = int(gp("t4_min_voxels", 500).value)
        self._imap = {}                # packed voxel key -> [sumI, count]
        self._t4_scan = None           # raw deskewed scan (xyz) snapshot, aligned with intensity
        self._t4_inten = None
        self._n_t4_applied = 0
        self.get_logger().info(
            f"  T4: intensity voxel={self.t4_voxel} apply_loc<{self.t4_apply_loc} "
            f"max_shift={self.t4_max_shift} enable={self.t4_enable}")

    # snapshot the intensity-aligned raw scan BEFORE Tier-1 accumulation reorders it
    def _gen_preprocess(self):
        s, i = self._scan_undistort, self._scan_intensity
        if (self.t4_enable and s is not None and i is not None
                and s.shape[0] == i.shape[0] and s.shape[0] >= 50):
            n = s.shape[0]
            step = max(1, n // self.t4_max_pts)
            self._t4_scan = s[::step].copy()
            self._t4_inten = i[::step].astype(np.float64).copy()
        else:
            self._t4_scan = self._t4_inten = None
        super()._gen_preprocess()

    def _register(self):
        super()._register()            # Tier-2 geometric registration
        if self.t4_enable:
            try:
                self._t4_refine()
            except Exception:
                pass

    def _output(self):
        super()._output()
        if self.t4_enable:
            try:
                self._t4_update_imap()
            except Exception:
                pass

    # ---- intensity photometric refinement (gated) ------------------------
    def _t4_refine(self):
        if self._t4_scan is None or len(self._imap) < self.t4_min_voxels:
            return
        if getattr(self, "last_loc", 1.0) >= self.t4_apply_loc:  # only weak-geometry scans
            return
        R, p = self.kf.R, self.kf.p
        world = self._t4_scan @ R.T + p
        v = self.t4_voxel
        k0 = np.floor(world / v).astype(np.int64)
        dp = []
        get = self._imap.get
        for j in range(world.shape[0]):
            i, jj, kk = int(k0[j, 0]), int(k0[j, 1]), int(k0[j, 2])
            c = get(_pack(np.array([[i, jj, kk]]))[0])
            if c is None:
                continue
            I0 = c[0] / c[1]
            g = np.zeros(3)
            ok = True
            for ax, (a, b) in enumerate((
                    ((i + 1, jj, kk), (i - 1, jj, kk)),
                    ((i, jj + 1, kk), (i, jj - 1, kk)),
                    ((i, jj, kk + 1), (i, jj, kk - 1)))):
                ca = get(_pack(np.array([a]))[0]); cb = get(_pack(np.array([b]))[0])
                if ca is None or cb is None:
                    ok = False
                    break
                g[ax] = (ca[0] / ca[1] - cb[0] / cb[1]) / (2.0 * v)
            gn = g @ g
            if not ok or gn < 1e-9:
                continue
            r = float(self._t4_inten[j]) - I0            # I_scan - I_map
            dp.append(r * g / gn)                          # Lucas-Kanade step
        if len(dp) < 20:
            return
        step = np.median(np.asarray(dp), axis=0)
        nrm = float(np.linalg.norm(step))
        if not np.isfinite(nrm) or nrm < 1e-4:
            return
        if nrm > self.t4_max_shift:
            step *= self.t4_max_shift / nrm
        p_meas = p + step
        R_n = (self.t4_noise ** 2) * np.eye(6)
        acc, _ = self.kf.update_pose(R.copy(), p_meas, R_n, self.chi2_threshold)
        if acc:
            self._n_t4_applied += 1
            self.last_method = f"{self.last_method}+I"

    # ---- intensity map update -------------------------------------------
    def _t4_update_imap(self):
        if self._t4_scan is None:
            return
        R, p = self.kf.R, self.kf.p
        world = self._t4_scan @ R.T + p
        keys = np.floor(world / self.t4_voxel).astype(np.int64)
        packed = _pack(keys)
        uniq, inv = np.unique(packed, return_inverse=True)
        sumI = np.zeros(uniq.shape[0])
        np.add.at(sumI, inv, self._t4_inten)
        cnt = np.bincount(inv, minlength=uniq.shape[0])
        imap = self._imap
        for u, si, c in zip(uniq.tolist(), sumI.tolist(), cnt.tolist()):
            e = imap.get(u)
            if e is None:
                imap[u] = [si, c]
            else:
                e[0] += si
                e[1] += c


def main():
    run_node(SuperLioGenT4)


if __name__ == "__main__":
    main()
