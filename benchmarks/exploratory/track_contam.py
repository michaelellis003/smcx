# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# Does running earlier kill-test cells in the same process slow
# track/1e6 down (allocator/cache state), reproducing 4.79s?
import statistics
import sys

import mlx.core as mx

sys.path.insert(0, "/Users/michaelellis/Projects/smcx/benchmarks/killtest")
from mlx_side import bench, make_lgssm, make_sv, make_track


def med(r):
    return statistics.median(r["times_s"])


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    print(
        "track/1e6 fresh:", med(bench(make_track, 1_000_000, 5, 4)), flush=True
    )
    # simulate prior cells: sv at 1e6 (12 GB peak) + cpu arm etc.
    print("running sv/1e6 ...", flush=True)
    bench(make_sv, 1_000_000, 5, 4)
    mx.set_default_device(mx.Device(mx.cpu))
    try:
        bench(make_sv, 100_000, 2, 4)  # cpu arm (small, just to mimic)
    finally:
        mx.set_default_device(mx.Device(mx.gpu))
    print(
        "track/1e6 after sv:",
        med(bench(make_track, 1_000_000, 5, 4)),
        flush=True,
    )
    print("cache MB:", mx.get_cache_memory() / 1e6, flush=True)
    mx.clear_cache()
    print(
        "track/1e6 after clear_cache:",
        med(bench(make_track, 1_000_000, 5, 4)),
        flush=True,
    )


if __name__ == "__main__":
    main()
