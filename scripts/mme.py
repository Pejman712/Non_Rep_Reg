#!/usr/bin/env python3
"""
mme.py — Mean Map Entropy of a point-cloud map (no ground truth needed).

MME measures local map self-consistency / sharpness: for each map point, take the
neighbours within `radius`, form their 3x3 covariance Σ, and

    h(p) = 1/2 * ln( det( 2*pi*e * Σ ) )        MME = mean_p h(p)

Lower = crisper, more consistent map (thin surfaces); higher = blurry/smeared
(drift, double walls).  This is the GT-free metric from MapEval / Razlaw et al.

    python3 mme.py map.pcd [--radius 0.3] [--min-nb 8] [--max-centers 60000]

Prints:  MME=<value>  points=<N>  used=<M>
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree

_LN_2PIE = np.log(2.0 * np.pi * np.e)     # constant term per axis


# ── minimal PCD reader (ascii + uncompressed binary; reads x,y,z by field) ────
_PCD_NP = {("F", 4): np.float32, ("F", 8): np.float64,
           ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32,
           ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32}


def _read_pcd(path, want):
    """Read requested fields (present ones only) from an ascii/binary PCD.
    Returns dict name→float64 (N,) array."""
    with open(path, "rb") as f:
        fields, size, typ, count = [], [], [], []
        npts = 0
        data_kind = "ascii"
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("ascii", "replace").strip()
            if s.startswith("#") or not s:
                continue
            key, *vals = s.split()
            key = key.upper()
            if key == "FIELDS":
                fields = vals
            elif key == "SIZE":
                size = [int(v) for v in vals]
            elif key == "TYPE":
                typ = vals
            elif key == "COUNT":
                count = [int(v) for v in vals]
            elif key == "POINTS":
                npts = int(vals[0])
            elif key == "WIDTH" and npts == 0:
                npts = int(vals[0])
            elif key == "DATA":
                data_kind = vals[0].lower()
                break
        if not count:
            count = [1] * len(fields)
        offs, off = {}, 0
        for nm, sz, c in zip(fields, size, count):
            offs[nm] = (off, sz)
            off += sz * c
        stride = off
        present = [nm for nm in want if nm in fields]
        out = {}
        if data_kind == "binary":
            raw = f.read(stride * npts)
            buf = np.frombuffer(raw, np.uint8)[: stride * npts].reshape(npts, stride)
            for nm in present:
                o, sz = offs[nm]
                dt = _PCD_NP[(typ[fields.index(nm)], sz)]
                out[nm] = buf[:, o:o + sz].copy().view(dt).ravel().astype(np.float64)
        elif data_kind == "ascii":
            arr = np.loadtxt(f, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            for nm in present:
                out[nm] = arr[:, fields.index(nm)]
        else:
            raise ValueError(f"unsupported PCD DATA '{data_kind}' (compressed?)")
        return out


def read_pcd_xyz(path):
    c = _read_pcd(path, ("x", "y", "z"))
    return np.column_stack([c["x"], c["y"], c["z"]])


def read_pcd_xyzi(path):
    """Returns (xyz (N,3), intensity (N,) or None)."""
    c = _read_pcd(path, ("x", "y", "z", "intensity"))
    xyz = np.column_stack([c["x"], c["y"], c["z"]])
    return xyz, c.get("intensity")


def compute_mme(pts, radius=0.3, min_nb=8, max_centers=60000, seed=0):
    """MME over `pts` (N,3).  Returns (mme, n_points, n_used)."""
    n = pts.shape[0]
    if n < min_nb + 1:
        return float("nan"), n, 0
    tree = cKDTree(pts)
    if n > max_centers:                       # subsample query centres (tree stays full)
        centers = pts[np.random.default_rng(seed).choice(n, max_centers, replace=False)]
    else:
        centers = pts
    nbrs = tree.query_ball_point(centers, radius, workers=-1)
    hs = []
    c3 = 3.0 * _LN_2PIE
    for i, idx in enumerate(nbrs):
        if len(idx) < min_nb:
            continue
        cov = np.cov(pts[idx].T)              # 3x3
        det = np.linalg.det(cov)
        if det <= 1e-18 or not np.isfinite(det):
            continue
        hs.append(0.5 * (c3 + np.log(det)))
    if not hs:
        return float("nan"), n, 0
    return float(np.mean(hs)), n, len(hs)


def mme_of_pcd(path, radius=0.3, min_nb=8, max_centers=60000):
    return compute_mme(read_pcd_xyz(path), radius, min_nb, max_centers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcd")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--min-nb", type=int, default=8)
    ap.add_argument("--max-centers", type=int, default=60000)
    a = ap.parse_args()
    mme, n, used = mme_of_pcd(a.pcd, a.radius, a.min_nb, a.max_centers)
    print(f"MME={mme:.4f}  points={n}  used={used}  radius={a.radius}")


if __name__ == "__main__":
    main()
