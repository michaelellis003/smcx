# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Area 3: streams — CPU-stream diagnostics concurrent with GPU filtering; sync costs; f64 diag stalls."""

import time

import mlx.core as mx

mx.random.seed(0)
N = 1_000_000
T = 50

CPU = mx.new_stream(mx.cpu)


def gpu_work(p, key):
    # heavy-ish GPU chain
    y = p
    for _ in range(30):
        y = mx.abs(y * 1.0001 + 0.001)
    y = y + mx.random.normal((N,), key=key) * 0.01
    return y


keys = mx.random.split(mx.random.key(0), T)
p0 = mx.random.normal((N,))
mx.eval(keys, p0)
f = mx.compile(gpu_work)
mx.eval(f(p0, keys[0]))
mx.synchronize()


def run(mode):
    p = p0
    mx.synchronize()
    t0 = time.perf_counter()
    diags = []
    for t in range(T):
        p = f(p, keys[t])
        if mode == "gpu_diag":
            d = mx.logsumexp(p) - mx.max(p)  # small reduction on GPU
            diags.append(d)
        elif mode == "cpu_diag_f64":
            with mx.stream(CPU):
                d = mx.logsumexp(p.astype(mx.float64)) - mx.max(
                    p.astype(mx.float64)
                )
            diags.append(d)
        elif mode == "cpu_diag_f64_item":
            with mx.stream(CPU):
                d = mx.logsumexp(p.astype(mx.float64))
            _ = d.item()
        elif mode == "gpu_diag_item":
            d = mx.logsumexp(p)
            _ = d.item()
        mx.async_eval(p)
    mx.eval(p, diags)
    mx.synchronize()
    return time.perf_counter() - t0


for mode in (
    "none",
    "gpu_diag",
    "cpu_diag_f64",
    "gpu_diag_item",
    "cpu_diag_f64_item",
):
    # warm
    run(mode)
    ts = [run(mode) for _ in range(3)]
    print(
        f"mode={mode:>18}: min {min(ts) * 1e3:8.1f} ms  ({min(ts) / T * 1e6:6.0f} us/step)"
    )

print()
print("=== cross-stream dependency serialization test ===")


# diagnostic result fed BACK into next GPU step (dependency) vs independent
def run_feedback(feedback):
    p = p0
    mx.synchronize()
    t0 = time.perf_counter()
    for t in range(T):
        p = f(p, keys[t])
        with mx.stream(CPU):
            d = mx.sum(p.astype(mx.float64)) * 0.0  # f64 diag
        if feedback:
            p = p + d.astype(mx.float32)  # inject dependency GPU<-CPU
        mx.async_eval(p)
    mx.eval(p)
    mx.synchronize()
    return time.perf_counter() - t0


for fb in (False, True):
    run_feedback(fb)
    ts = [run_feedback(fb) for _ in range(3)]
    print(f"cpu f64 diag feedback={fb}: min {min(ts) * 1e3:8.1f} ms")

print()
print(
    "=== does the CPU diag actually overlap? measure GPU-only vs GPU+big CPU job ==="
)


def run_cpu_load(big):
    p = p0
    mx.synchronize()
    t0 = time.perf_counter()
    ds = []
    for t in range(T):
        p = f(p, keys[t])
        if big:
            with mx.stream(CPU):
                d = mx.sum(mx.sort(p))  # expensive CPU op on 1e6 (sort)
            ds.append(d)
        mx.async_eval(p)
    mx.eval(p, ds)
    mx.synchronize()
    return time.perf_counter() - t0


for big in (False, True):
    run_cpu_load(big)
    ts = [run_cpu_load(big) for _ in range(3)]
    print(f"big CPU sort per step={big}: min {min(ts) * 1e3:8.1f} ms")
