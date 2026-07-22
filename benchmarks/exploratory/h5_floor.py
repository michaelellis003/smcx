# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# H5: fixed overhead floor — whole bootstrap filter at N=1e3 where
# GPU compute ~ 0; per-step time = Python/dispatch/cadence floor.
# Also lag0 vs lag4 vs async at N=1e3.
import json
import pathlib
import statistics
import sys
import time

import mlx.core as mx

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[1] / "killtest")
)
from mlx_side import bench, make_lgssm, make_sv, make_track

SCRATCH = pathlib.Path(__file__).parent


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    out = {}
    for name, make in [
        ("lgssm", make_lgssm),
        ("sv", make_sv),
        ("track", make_track),
    ]:
        row = {}
        for arm, lag in [("lag0", 0), ("lag4", 4), ("async", 10**9)]:
            r = bench(make, 1000, 10, lag)
            row[arm] = statistics.median(r["times_s"])
        _, _, _, y = make()
        row["T"] = y.shape[0]
        row["per_step_lag4_us"] = row["lag4"] / row["T"] * 1e6
        out[name] = row
        print(name, row, flush=True)
    (SCRATCH / "h5_floor.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
