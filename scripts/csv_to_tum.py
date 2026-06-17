#!/usr/bin/env python3.10
"""
Convert an OptiTrack/ROS CSV ground-truth file to TUM format.

Input CSV columns (no header):
  timestamp_ns, x, y, z, roll, pitch, yaw, qx, qy, qz, qw

Output TUM format (space-separated):
  timestamp_sec  tx  ty  tz  qx  qy  qz  qw

Usage:
  python3 csv_to_tum.py <input.csv> <output.tum>
"""

import sys


def convert(csv_path: str, tum_path: str) -> int:
    written = 0
    skipped = 0
    with open(csv_path, "r") as fin, open(tum_path, "w") as fout:
        for lineno, raw in enumerate(fin, 1):
            line = raw.strip().rstrip(",").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                skipped += 1
                continue
            try:
                ts_sec = int(parts[0]) * 1e-9
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = (float(parts[7]), float(parts[8]),
                                   float(parts[9]), float(parts[10]))
            except (ValueError, IndexError):
                skipped += 1
                continue
            fout.write(f"{ts_sec:.9f} {x:.9f} {y:.9f} {z:.9f} "
                       f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            written += 1
    print(f"Converted {written} poses → {tum_path}  (skipped {skipped} bad rows)")
    return written


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: {} <input.csv> <output.tum>".format(sys.argv[0]))
        sys.exit(1)
    n = convert(sys.argv[1], sys.argv[2])
    sys.exit(0 if n > 0 else 1)
