#!/usr/bin/env python3.10
"""
pose_graph_se2.py  —  minimal SE(2) pose-graph optimizer (Gauss-Newton).

Self-contained (numpy only) since gtsam / g2o aren't available.  Standard
formulation (Grisetti "A tutorial on graph-based SLAM"): nodes are 2-D poses
(x, y, theta); each edge is a relative-pose measurement z (3-vector) with a 3x3
information matrix.  Node 0 is held fixed.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def v2t(p):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([[c, -s, p[0]], [s, c, p[1]], [0.0, 0.0, 1.0]])


def t2v(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def optimize(nodes, edges, iterations=30, tol=1e-5):
    """nodes: (N,3) [x,y,theta]; edges: list of (i, j, z(3,), info(3x3)).
    Node 0 is fixed.  Returns optimized (N,3)."""
    x = np.array(nodes, dtype=float).copy()
    N = len(x)
    for _ in range(iterations):
        H = np.zeros((3 * N, 3 * N))
        b = np.zeros(3 * N)
        for (i, j, z, info) in edges:
            xi, xj = x[i], x[j]
            Ti, Tj, Tz = v2t(xi), v2t(xj), v2t(z)
            e = t2v(np.linalg.inv(Tz) @ (np.linalg.inv(Ti) @ Tj))
            e[2] = wrap(e[2])

            ci, si = np.cos(xi[2]), np.sin(xi[2])
            RiT = np.array([[ci, si], [-si, ci]])
            dRiT = np.array([[-si, ci], [-ci, -si]])      # d(Ri^T)/dtheta
            cz, sz = np.cos(z[2]), np.sin(z[2])
            RzT = np.array([[cz, sz], [-sz, cz]])
            tij = xj[0:2] - xi[0:2]

            A = np.zeros((3, 3))
            B = np.zeros((3, 3))
            A[0:2, 0:2] = -RzT @ RiT
            A[0:2, 2] = RzT @ dRiT @ tij
            A[2, 2] = -1.0
            B[0:2, 0:2] = RzT @ RiT
            B[2, 2] = 1.0

            ii = slice(3 * i, 3 * i + 3)
            jj = slice(3 * j, 3 * j + 3)
            AtO = A.T @ info
            BtO = B.T @ info
            H[ii, ii] += AtO @ A
            H[ii, jj] += AtO @ B
            H[jj, ii] += BtO @ A
            H[jj, jj] += BtO @ B
            b[ii] += AtO @ e
            b[jj] += BtO @ e

        H[0:3, 0:3] += np.eye(3) * 1e9   # fix the first node (gauge)
        try:
            dx = np.linalg.solve(H, -b)
        except np.linalg.LinAlgError:
            break
        x += dx.reshape(N, 3)
        x[:, 2] = wrap(x[:, 2])
        if np.max(np.abs(dx)) < tol:
            break
    return x


def total_error(nodes, edges):
    x = np.array(nodes, dtype=float)
    tot = 0.0
    for (i, j, z, info) in edges:
        e = t2v(np.linalg.inv(v2t(z)) @ (np.linalg.inv(v2t(x[i])) @ v2t(x[j])))
        e[2] = wrap(e[2])
        tot += float(e @ info @ e)
    return tot
