# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# Replicate the kill-test bench exactly for track/1e6 (R=20, lag4,
# store_history=True) in a fresh process to chase the 13.3 vs 23.9
# ms/step discrepancy.
import statistics
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "killtest"))
from mlx_side import bench, make_track


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    r = bench(make_track, 1_000_000, 20, 4)
    ts = r["times_s"]
    print("median", statistics.median(ts), "min", min(ts), "max", max(ts))
    print("peak_mb", statistics.median(r["peak_mb"]))
    print("all times:", [round(t, 3) for t in ts])


if __name__ == "__main__":
    main()
