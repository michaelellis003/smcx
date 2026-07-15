# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Kill-test report generator: gates + pre-registered verdict.

Reads benchmarks/data/{meta,jax_results,mlx_results}.json and prints
the results tables and the PROTOCOL.md criterion application as
markdown. Purely mechanical — the criterion constants (k=3, 3x,
both-N rule, 1.2x/1.5x fail thresholds) are the pre-registered ones.
"""

import json
import math
import pathlib
import statistics as st

DATA = pathlib.Path(__file__).parent.parent / "data"
GRID = (10_000, 100_000, 1_000_000)
K = 3.0


def _med(xs):
    return st.median(xs)


def _iqr(xs):
    q = st.quantiles(xs, n=4)
    return q[2] - q[0]


def kalman_gate(logzs, oracle):
    r = len(logzs)
    sd = st.stdev(logzs)
    err = st.mean(logzs) - oracle
    upper = K * sd / math.sqrt(r)
    lower = -(upper + 0.5 * sd**2)
    return lower <= err <= upper, err, sd


def cross_gate(logzs_a, logzs_b):
    r = len(logzs_a)
    diff = st.mean(logzs_a) - st.mean(logzs_b)
    bound = K * math.sqrt(
        st.stdev(logzs_a) ** 2 / r + st.stdev(logzs_b) ** 2 / r
    )
    return abs(diff) <= bound, diff, bound


def main():
    meta = json.loads((DATA / "meta.json").read_text())
    jax_r = json.loads((DATA / "jax_results.json").read_text())
    mlx_r = json.loads((DATA / "mlx_results.json").read_text())

    print("## Timing (median [min, IQR] seconds; speedup = JAX/MLX-GPU)\n")
    speedups = {}
    for w in ("lgssm", "sv", "track", "track_full"):
        print(f"### {w}\n")
        print(
            "| N | JAX-CPU | MLX-GPU (lag4) | MLX-CPU | speedup | GPU peak MB |"
        )
        print("|---|---|---|---|---|---|")
        for n in GRID:
            j = jax_r["cells"][f"{w}/{n}"]["times_s"]
            cell = mlx_r["cells"][f"{w}/{n}"]
            g = cell["gpu_lag4"]["times_s"]
            c = cell["cpu_lag4"]["times_s"]
            sp = _med(j) / _med(g)
            speedups[w, n] = (sp, _med(g), _med(c))
            print(
                f"| {n:,} | {_med(j):.3f} [{min(j):.3f}, {_iqr(j):.3f}] "
                f"| {_med(g):.3f} [{min(g):.3f}, {_iqr(g):.3f}] "
                f"| {_med(c):.3f} | **{sp:.1f}x** "
                f"| {max(cell['gpu_lag4']['peak_mb']):.0f} |"
            )
        print()

    print("## Cadence sweep (MLX-GPU median s; best arm bolded)\n")
    arms = ("gpu_lag0", "gpu_lag2", "gpu_lag4", "gpu_lag8", "gpu_async")
    print("| cell | " + " | ".join(a[4:] for a in arms) + " |")
    print("|---|" + "---|" * len(arms))
    for w in ("lgssm", "sv", "track"):
        for n in GRID:
            cell = mlx_r["cells"][f"{w}/{n}"]
            meds = {a: _med(cell[a]["times_s"]) for a in arms}
            best = min(meds, key=lambda a: meds[a])
            row = " | ".join(
                f"**{meds[a]:.3f}**" if a == best else f"{meds[a]:.3f}"
                for a in arms
            )
            print(f"| {w}/{n:,} | {row} |")
    print()

    print("## Correctness gates (k=3, R=20, one-sided Jensen)\n")
    gates = {}
    print("| cell | side | gate | err (nats) | SD |")
    print("|---|---|---|---|---|")
    for w, oracle_key in (("lgssm", "lgssm"), ("track", "track")):
        for n in GRID:
            ok_j, err_j, sd_j = kalman_gate(
                jax_r["cells"][f"{w}/{n}"]["logz"],
                meta["oracles"][oracle_key],
            )
            ok_m, err_m, sd_m = kalman_gate(
                mlx_r["cells"][f"{w}/{n}"]["gpu_lag4"]["logz"],
                meta["oracles"][oracle_key],
            )
            gates[w, n] = ok_j and ok_m
            print(
                f"| {w}/{n:,} | jax | {'PASS' if ok_j else 'FAIL'} "
                f"| {err_j:+.3f} | {sd_j:.3f} |"
            )
            print(
                f"| {w}/{n:,} | mlx | {'PASS' if ok_m else 'FAIL'} "
                f"| {err_m:+.3f} | {sd_m:.3f} |"
            )
    for n in GRID:
        ok, diff, bound = cross_gate(
            mlx_r["cells"][f"sv/{n}"]["gpu_lag4"]["logz"],
            jax_r["cells"][f"sv/{n}"]["logz"],
        )
        gates["sv", n] = ok
        print(
            f"| sv/{n:,} | cross | {'PASS' if ok else 'FAIL'} "
            f"| {diff:+.3f} | bound {bound:.3f} |"
        )
    print()

    print("## Pre-registered criterion\n")
    counts = {}
    for w in ("lgssm", "sv", "track"):
        ok = all(
            gates[w, n] and speedups[w, n][0] >= 3.0
            for n in (100_000, 1_000_000)
        )
        counts[w] = ok
        detail = ", ".join(
            f"N={n:,}: {speedups[w, n][0]:.1f}x"
            f"{'' if gates[w, n] else ' GATE-FAIL'}"
            for n in (100_000, 1_000_000)
        )
        print(f"- {w}: {'COUNTS' if ok else 'does not count'} ({detail})")
    n_count = sum(counts.values())
    gpu_vs_cpu_dead = all(
        speedups[w, n][2] / speedups[w, n][1] < 1.2
        for w in ("lgssm", "sv", "track")
        for n in GRID
    )
    jax_close = all(
        speedups[w, n][0] < 1.5 for w in ("lgssm", "sv", "track") for n in GRID
    )
    if jax_close or gpu_vs_cpu_dead:
        verdict = "FAILS"
    elif n_count >= 2:
        verdict = "HOLDS"
    else:
        verdict = "HOLDS WEAKLY"
    print(f"\n**Verdict: the thesis {verdict}** ({n_count}/3 count)")


if __name__ == "__main__":
    main()
