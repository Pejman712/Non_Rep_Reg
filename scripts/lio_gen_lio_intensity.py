#!/usr/bin/env python3.10
"""
lio_gen_lio_intensity.py  —  gen_lio_intensity: corrected intensity/photometric
variant on top of Tier 2 (supersedes the gen_liotier4 prototype).

Adds a photometric (LiDAR-intensity) constraint to help the geometrically-
unobservable direction in narrow-FoV indoor scenes, with the tier4 defects fixed:

  #1 TRANSLATION-ONLY update — rotation gets ~infinite measurement noise, so it
     carries no information (tier4 fed R_meas=R with finite noise, i.e. a bogus
     zero-residual rotation observation that made rotation over-confident).
  #3 proper GAUSS-NEWTON least-squares solve  Δp = (GᵀG + λI)⁻¹ Gᵀr  over all
     correspondences (tier4 used an ad-hoc component-wise median of per-point
     steps, which can point the wrong way).
  #5 ADAPTIVE noise from the solve covariance  σ_I²·(GᵀG)⁻¹ (+ floor): weak /
     ill-conditioned directions are automatically down-weighted (tier4 used a
     fixed, over-confident 0.1 m).
  #2/#6 intensity NORMALIZATION (scale to ~[0,1]; optional range compensation)
     and a MIN-COUNT guard on map voxels, so ∇I reflects reflectance rather than
     range/angle/1-sample noise; plus robust MAD residual outlier rejection for
     dynamic/occlusion returns.

Still gated to weakly-localizable scans and χ²-gated so a bad correction cannot
corrupt the state.  Inherits Tier-1 accumulation + Tier-2 observability switch.

    ros2 run regnonrep lio_gen_lio_intensity.py --ros-args -p i_apply_loc:=1.0
"""
import numpy as np

from lio_base import run_node
from lio_gen_lio_tier2 import SuperLioGenT2

_PK_OFF = 1 << 19
_PK_A = 1 << 20
# base voxel + its 6 face neighbours, for the finite-difference gradient
_OFFS = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0],
                  [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)


def _pack(keys):                       # keys (N,3) int -> (N,) int64
    return ((keys[:, 0] + _PK_OFF) * _PK_A + (keys[:, 1] + _PK_OFF)) * _PK_A \
        + (keys[:, 2] + _PK_OFF)


class SuperLioGenIntensity(SuperLioGenT2):
    NODE_NAME = "super_lio_gen_intensity"
    VARIANT_DESC = "+ intensity photometric (translation-only LSQ, normalized)"

    def __init__(self):
        super().__init__()
        gp = self.declare_parameter
        self.i_enable = bool(gp("i_enable", True).value)
        self.i_voxel = float(gp("i_voxel", 0.30).value)         # intensity-map voxel [m]
        self.i_max_pts = int(gp("i_max_pts", 1500).value)       # scan subsample cap
        self.i_max_shift = float(gp("i_max_shift", 0.15).value)  # clip |Δp| [m]
        self.i_apply_loc = float(gp("i_apply_loc", 0.20).value)  # apply when localizability < (1.0 = always)
        self.i_min_voxels = int(gp("i_min_voxels", 500).value)
        self.i_min_count = int(gp("i_min_count", 3).value)      # voxel must hold ≥ this many pts
        self.i_min_pts = int(gp("i_min_pts", 30).value)         # min correspondences to solve
        self.i_ref = float(gp("i_ref", 255.0).value)            # scale intensity to ~[0,1]
        self.i_range_comp = bool(gp("i_range_comp", False).value)  # multiply by (r/r0)^2 (Livox is ~normalised)
        self.i_range_ref = float(gp("i_range_ref", 5.0).value)
        self.i_noise = float(gp("i_noise", 0.05).value)         # intensity meas noise (norm units)
        self.i_rot_noise = float(gp("i_rot_noise", 1e3).value)  # ~∞: no rotation info
        self.i_floor = float(gp("i_floor", 0.03).value)         # translation-noise floor [m]
        self._imap = {}                # packed voxel -> [sumI, count]
        self._i_scan = None; self._i_val = None
        self._n_i_applied = 0
        self.get_logger().info(
            f"  Intensity: voxel={self.i_voxel} apply_loc<{self.i_apply_loc} "
            f"max_shift={self.i_max_shift} min_count={self.i_min_count} "
            f"range_comp={self.i_range_comp} enable={self.i_enable}")

    def _norm_i(self, scan_body, inten):
        v = inten.astype(np.float64) / max(self.i_ref, 1e-6)    # ~[0,1]
        if self.i_range_comp:                                    # undo 1/r² falloff
            r = np.clip(np.linalg.norm(scan_body, axis=1), 0.5, 30.0)
            v = v * (r / self.i_range_ref) ** 2
        return v

    # snapshot the intensity-aligned raw scan BEFORE Tier-1 accumulation reorders it
    def _gen_preprocess(self):
        s, i = self._scan_undistort, self._scan_intensity
        if (self.i_enable and s is not None and i is not None
                and s.shape[0] == i.shape[0] and s.shape[0] >= 50):
            n = s.shape[0]
            step = max(1, n // self.i_max_pts)
            sb = s[::step]
            self._i_scan = sb.copy()
            self._i_val = self._norm_i(sb, i[::step])
        else:
            self._i_scan = self._i_val = None
        super()._gen_preprocess()

    def _register(self):
        super()._register()            # Tier-2 geometric registration
        if self.i_enable:
            try:
                self._i_refine()
            except Exception:
                pass

    def _output(self):
        super()._output()
        if self.i_enable:
            try:
                self._i_update_map()
            except Exception:
                pass

    # ---- photometric refinement (gated, translation-only, LSQ) -----------
    def _i_refine(self):
        if self._i_scan is None or len(self._imap) < self.i_min_voxels:
            return
        if getattr(self, "last_loc", 1.0) >= self.i_apply_loc:  # only weak geometry
            return
        R, p = self.kf.R, self.kf.p
        world = self._i_scan @ R.T + p
        v = self.i_voxel
        k0 = np.floor(world / v).astype(np.int64)              # (N,3)
        # packed keys for each point's voxel + its 6 neighbours, vectorised
        allk = (k0[:, None, :] + _OFFS[None, :, :]).reshape(-1, 3)
        pk = _pack(allk).reshape(-1, 7)                        # (N,7)
        get = self._imap.get
        mc = self.i_min_count

        def mean(key):
            e = get(int(key))
            return (e[0] / e[1]) if (e is not None and e[1] >= mc) else None

        G, Rr = [], []
        for j in range(world.shape[0]):
            I0 = mean(pk[j, 0])
            if I0 is None:
                continue
            g = np.empty(3)
            ok = True
            for ax in range(3):
                Ia = mean(pk[j, 1 + 2 * ax]); Ib = mean(pk[j, 2 + 2 * ax])
                if Ia is None or Ib is None:
                    ok = False
                    break
                g[ax] = (Ia - Ib) / (2.0 * v)
            if not ok or g @ g < 1e-9:
                continue
            G.append(g); Rr.append(float(self._i_val[j]) - I0)
        if len(Rr) < self.i_min_pts:
            return
        G = np.asarray(G); Rr = np.asarray(Rr)
        # robust residual outlier rejection (MAD) — drop dynamic/occlusion returns
        med = np.median(Rr); mad = np.median(np.abs(Rr - med)) + 1e-9
        keep = np.abs(Rr - med) < 3.0 * 1.4826 * mad
        G, Rr = G[keep], Rr[keep]
        if G.shape[0] < self.i_min_pts:
            return
        # Gauss-Newton photometric solve  Δp = (GᵀG + λI)⁻¹ Gᵀr
        H = G.T @ G
        b = G.T @ Rr
        lam = (np.trace(H) / 3.0) * 1e-2 + 1e-9
        Hr = H + lam * np.eye(3)
        try:
            dp = np.linalg.solve(Hr, b)
            covt = (self.i_noise ** 2) * np.linalg.inv(Hr)      # adaptive covariance
        except np.linalg.LinAlgError:
            return
        nrm = float(np.linalg.norm(dp))
        if not np.isfinite(nrm) or nrm < 1e-4:
            return
        if nrm > self.i_max_shift:
            dp *= self.i_max_shift / nrm
        covt = covt + (self.i_floor ** 2) * np.eye(3)           # noise floor (no over-confidence)
        R_n = np.zeros((6, 6))
        R_n[0:3, 0:3] = (self.i_rot_noise ** 2) * np.eye(3)     # rotation: ~∞ noise ⇒ no info
        R_n[3:6, 3:6] = covt
        acc, _ = self.kf.update_pose(R.copy(), p + dp, R_n, self.chi2_threshold)
        if acc:
            self._n_i_applied += 1
            self.last_method = f"{self.last_method}+I"

    # ---- intensity map update -------------------------------------------
    def _i_update_map(self):
        if self._i_scan is None:
            return
        R, p = self.kf.R, self.kf.p
        world = self._i_scan @ R.T + p
        keys = np.floor(world / self.i_voxel).astype(np.int64)
        packed = _pack(keys)
        uniq, inv = np.unique(packed, return_inverse=True)
        sumI = np.zeros(uniq.shape[0])
        np.add.at(sumI, inv, self._i_val)
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
    run_node(SuperLioGenIntensity)


if __name__ == "__main__":
    main()
