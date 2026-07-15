#!/usr/bin/env python3.10
"""
lio_tsvd.py  —  Truncated-SVD degeneracy mitigation.

This implements the approach Tuna et al. (2024, "Informed, Constrained, Aligned")
found to be the most consistent, tuning-light degeneracy mitigation for point
cloud registration:

  1. DETECT on the optimization Hessian (X-ICP, Tuna 2023): the GICP information
     matrix H is split into rotational (H_rr) and translational (H_tt) 3x3
     blocks; each is eigen-decomposed and an eigen-direction is flagged
     degenerate when its eigenvalue is small relative to the block's largest
     (relative condition number).

  2. MITIGATE with TSVD: the GICP pose increment (relative to the IMU/non-rep
     prior) is projected onto the well-observed eigen-directions and the
     degenerate directions are TRUNCATED to zero — i.e. along an unobservable
     direction the measurement equals the prior, so the iESKF applies no LiDAR
     correction there and keeps the IMU estimate, while every well-constrained
     direction still gets the full GICP correction.

Unlike the degen-skip / reuse variants, degeneracy is handled CONTINUOUSLY per
scan (no scan-PCA, no binary skip, no map pause) and the well-constrained LiDAR
directions are always used — which is the paper's key argument for active,
during/after-optimization mitigation over "Prior-Only" coasting.

    ros2 run regnonrep lio_tsvd.py --ros-args -p debug_csv:=/tmp/tsvd.csv \
        -p tsvd_eig_ratio:=0.04
"""

import numpy as np

from lio_base import (SuperLioBase, run_node, so3_exp, so3_log,
                      estimate_registration_confidence)


class SuperLioTSVD(SuperLioBase):
    NODE_NAME = "super_lio_tsvd"
    USE_NONREP = True
    USE_GICP = True
    VARIANT_DESC = "GICP + TSVD degeneracy mitigation (Hessian rot/trans blocks)"

    def __init__(self):
        super().__init__()
        # relative threshold: an eigen-direction of a Hessian block is degenerate
        # (truncated) if its eigenvalue < tsvd_eig_ratio * largest eigenvalue.
        self.tsvd_eig_ratio = float(self.declare_parameter("tsvd_eig_ratio", 0.04).value)
        self._n_trunc_scans = 0
        self.get_logger().info(f"  tsvd_eig_ratio={self.tsvd_eig_ratio}")

    def _tsvd_remap(self, H, d_rot, d_trans):
        """Project the (decoupled) GICP increment onto the well-observed
        eigen-directions of the rotational / translational Hessian blocks and
        truncate the degenerate directions to zero.
        Returns (d_rot', d_trans', n_degen)."""
        n_degen = 0
        out = []
        for blk, d in ((H[0:3, 0:3], d_rot), (H[3:6, 3:6], d_trans)):
            B = 0.5 * (blk + blk.T)
            w, V = np.linalg.eigh(B)                 # ascending eigenvalues
            w = np.clip(w, 0.0, None)
            wmax = float(w[-1])
            if wmax <= 1e-12:
                out.append(np.zeros(3))
                n_degen += 3
                continue
            good = w >= self.tsvd_eig_ratio * wmax   # keep well-observed dirs
            n_degen += int(np.sum(~good))
            coeff = V.T @ d                          # increment in eigen-basis
            coeff[~good] = 0.0                        # truncate degenerate dirs
            out.append(V @ coeff)
        return out[0], out[1], n_degen

    def _register(self):
        scan_pts, scan_o3d = self._make_scan_cloud()
        if scan_o3d is None:
            ncorr, rms = self._observe_point_to_plane()
            self.last_method = "p2p"
            self.last_conf = self._p2p_conf(ncorr, rms)
            self.last_chi2 = 0.0
            self.last_accepted = 1 if ncorr > 0 else 0
            return

        feat, pred_pose, _, T_init = self._nonrep_init_guess(scan_o3d)
        R_prior, p_prior = T_init[:3, :3].copy(), T_init[:3, 3].copy()

        accepted, chi2, reg_conf, n_degen = False, 0.0, 0.0, 0
        submap = self._get_submap()
        if submap is not None and len(submap.points) >= 30:
            T_raw, H = self._gicp(scan_o3d, submap, T_init)
            R_gicp = T_raw[:3, :3]
            U, _, Vt = np.linalg.svd(R_gicp)
            R_gicp = U @ Vt
            p_gicp = T_raw[:3, 3].copy()
            reg_conf = estimate_registration_confidence(scan_o3d, submap, T_raw)

            # decoupled GICP increment relative to the prior (rot, trans)
            d_rot = so3_log(R_prior.T @ R_gicp)
            d_trans = p_gicp - p_prior
            # TSVD: truncate the degenerate directions of the increment
            if (H is not None and getattr(H, "shape", None) == (6, 6)
                    and np.all(np.isfinite(H))):
                d_rot, d_trans, n_degen = self._tsvd_remap(H, d_rot, d_trans)
                if n_degen > 0:
                    self._n_trunc_scans += 1

            # remapped measurement: prior along degenerate dirs, GICP elsewhere
            R_meas = R_prior @ so3_exp(d_rot)
            p_meas = p_prior + d_trans
            R_n = self._gicp_noise(H, reg_conf)
            accepted, chi2 = self.kf.update_pose(R_meas, p_meas, R_n, self.chi2_threshold)

        self.last_method = (f"tsvd~{n_degen}d" if (accepted and n_degen > 0)
                            else "gicp" if accepted else "p2p")
        if not accepted:
            self._n_p2p_fallback += 1
            ncorr, rms = self._observe_point_to_plane()
            reg_conf = self._p2p_conf(ncorr, rms)

        self.last_conf = float(reg_conf)
        self.last_chi2 = float(chi2)
        self.last_accepted = int(accepted)
        self._feed_processor(feat, pred_pose, reg_conf)
        if self._scan_counter % 100 == 0:
            self.get_logger().info(
                f"[tsvd] truncation fired on {self._n_trunc_scans}/"
                f"{self._scan_counter} scans so far")

    def shutdown(self):
        n = max(self._scan_counter, 1)
        self.get_logger().info(
            f"[tsvd] FINAL: truncation fired on {self._n_trunc_scans}/{n} scans "
            f"({100.0 * self._n_trunc_scans / n:.1f}%)")
        super().shutdown()


def main():
    run_node(SuperLioTSVD)


if __name__ == "__main__":
    main()
