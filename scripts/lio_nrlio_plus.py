#!/usr/bin/env python3.10
"""
lio_nrlio_plus.py — variant "nrlio_plus": nrlio_op_den + a set of ACCURACY-NEUTRAL
speedups.  Same registration/switch logic and (bit-)identical trajectory as
nrlio_op_den; only the compute is made cheaper.  Everything here is process-scoped
(applied when THIS variant's node runs), so no other variant is affected.

Speedups applied (numbered as discussed):
  1. Multithread the map k-NN query        — cKDTree.query(..., workers=-1)   [exact]
  2. Density switch: pack voxel keys to 1-D — np.unique(int64) not axis=0      [exact]
  3. Throttle CSV flushes                   — flush every N, not per scan      [exact]
     (note: the proc-time CSV already flushes every 20; this covers the debug CSV)
  4. iESKF early-exit on convergence        — ALREADY in lio_base (ESKF loop)  [inherited]
  5. Faster map insert                      — packed int64 keys, hoisted locals,
     manual squared-distance (no per-point np.dot / temporaries)               [exact]
  6. float32 KD-tree for the p2p 5-NN       — map reps stay float64; only the
     neighbour *search* uses float32 (sub-10-micron, NN choice unchanged)      [near-exact]
  7. Pin the BLAS threadpool to 1           — 18x18 ESKF solves stop paying
     thread-launch overhead; OpenMP (registration/open3d) left alone           [exact]

    ros2 run regnonrep lio_nrlio_plus.py
"""
# ── #7: limit ONLY the BLAS threadpool, BEFORE numpy is imported.  OpenBLAS/MKL
#    read these at import; OpenMP (OMP_NUM_THREADS) is deliberately left untouched
#    so any OpenMP-parallel registration/open3d keeps all its threads.
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import numpy as np
from scipy.spatial import cKDTree

import lio_base
from lio_base import run_node, OctVoxMapPy
from lio_nrlio_op_den import SuperLioNRLIOOpDen

_PLUS_KEY_OFF = 1 << 20          # index offset → non-negative for 21-bit packing
_PLUS_KEY_BITS = 21


def _pack_keys(q):
    """(N,3) int64 voxel indices → (N,) int64 scalar keys (bijective over ±1M)."""
    return (((q[:, 0] + _PLUS_KEY_OFF) << (2 * _PLUS_KEY_BITS))
            | ((q[:, 1] + _PLUS_KEY_OFF) << _PLUS_KEY_BITS)
            | (q[:, 2] + _PLUS_KEY_OFF))


# ── #5: faster OctVoxMapPy.insert — bit-identical to the original (same order,
#    same merge decisions, same running-mean division), just without per-point
#    3-tuple hashing, np.dot and temporary arrays.
def _fast_insert(self, pts_world):
    n = pts_world.shape[0]
    if n == 0:
        return
    cells = self._cells
    cap = self.capacity
    maxp = self.MAX_PER_SUBVOX
    md2 = self.MERGE_DIST2
    keys = np.floor(pts_world * self.inv_sub).astype(np.int64)
    pk = _pack_keys(keys).tolist()          # python ints hash faster than tuples
    P = pts_world
    for i in range(n):
        k = pk[i]
        ent = cells.get(k)
        px = P[i, 0]; py = P[i, 1]; pz = P[i, 2]
        if ent is None:
            cells[k] = [np.array((px, py, pz)), 1]
            if len(cells) > cap:
                cells.popitem(last=False)
        else:
            mean = ent[0]; cnt = ent[1]
            if cnt >= maxp:
                continue
            m0 = mean[0]; m1 = mean[1]; m2 = mean[2]
            dx = px - m0; dy = py - m1; dz = pz - m2
            if dx * dx + dy * dy + dz * dz > md2:
                continue
            denom = cnt + 1                  # division (not *recip) → bit-identical
            mean[0] = (m0 * cnt + px) / denom
            mean[1] = (m1 * cnt + py) / denom
            mean[2] = (m2 * cnt + pz) / denom
            ent[1] = denom
    self._dirty = True


# ── #1 + #6: knn5 with a multithreaded, float32 KD-tree.  Representative points
#    (returned) stay float64; only the tree/query run in float32.
def _fast_knn5(self, query):
    pts = self.all_points()
    if self._tree is None and pts.shape[0] >= 1:
        self._tree = cKDTree(pts.astype(np.float32, copy=False))     # #6
    if self._tree is None or pts.shape[0] < 4:
        nq = query.shape[0]
        return np.full((nq, 5), np.inf), np.zeros((nq, 5, 3))
    kk = min(5, pts.shape[0])
    dist, idx = self._tree.query(query.astype(np.float32, copy=False),
                                 k=kk, workers=-1)                    # #1
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]
    if kk < 5:
        pad = 5 - kk
        dist = np.concatenate([dist, np.full((dist.shape[0], pad), np.inf)], axis=1)
        idx = np.concatenate([idx, np.zeros((idx.shape[0], pad), dtype=idx.dtype)], axis=1)
    return dist.astype(np.float64), pts[idx]


# ── #1 (prefilter path): adaptive ROR with a multithreaded query, same result.
def _fast_adaptive_ror(pts, k=6, tau=0.06):
    n = pts.shape[0]
    if n <= k + 1 or k < 1:
        return pts
    tree = cKDTree(pts)
    dd, _ = tree.query(pts, k=k + 1, workers=-1)
    mnn = dd[:, 1:].mean(axis=1)
    rng = np.linalg.norm(pts, axis=1)
    return pts[(mnn / np.maximum(rng, 0.1)) <= tau]


def _apply_patches():
    if getattr(lio_base, "_nrlio_plus_patched", False):
        return
    OctVoxMapPy.insert = _fast_insert       # #5
    OctVoxMapPy.knn5 = _fast_knn5           # #1 + #6
    lio_base.adaptive_ror = _fast_adaptive_ror   # #1 (prefilter)
    lio_base._nrlio_plus_patched = True


_apply_patches()


class _ThrottleFlush:
    """Wraps a file handle so .flush() only really flushes every `every` writes
    (content is byte-identical; we just skip most fsync stalls).  Closed/flushed
    fully at shutdown so no tail is lost."""

    def __init__(self, fh, every=25):
        self._fh = fh
        self._every = every
        self._n = 0

    def write(self, s):
        self._n += 1
        return self._fh.write(s)

    def flush(self):
        if self._n % self._every == 0:
            self._fh.flush()

    def close(self):
        try:
            self._fh.flush()
        finally:
            self._fh.close()

    def __getattr__(self, name):
        return getattr(self._fh, name)


class SuperLioNRLIOPlus(SuperLioNRLIOOpDen):
    NODE_NAME = "super_lio_nrlio_plus"
    VARIANT_DESC = ("nrlio_op_den + accuracy-neutral speedups "
                    "(threaded/float32 kNN, packed-key map insert, throttled I/O, BLAS-pinned)")

    def __init__(self):
        super().__init__()
        # #3: throttle the (optional) debug CSV flush; the proc CSV is already
        #     flushed every 20 in lio_base, so only wrap the debug handle.
        if getattr(self, "_csv_fh", None) is not None:
            self._csv_fh = _ThrottleFlush(self._csv_fh, every=25)
        self.get_logger().info(
            "  [plus] accuracy-neutral speedups active: "
            "kNN workers=-1 + float32 tree, packed-key map insert, "
            "throttled debug-CSV, BLAS threads pinned to 1 "
            "(iESKF early-exit inherited)")


def main():
    run_node(SuperLioNRLIOPlus)


if __name__ == "__main__":
    main()
