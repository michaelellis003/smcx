# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# Decompose TRACK mutate cost: vmapped matvec vs batched matmul,
# vmapped keyed normal, and the L @ noise matvec.
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "killtest"))
from gen_data import track_matrices


def timeit(f, args, batch=20, reps=12, warm=3):
    for _ in range(warm):
        out = f(*args)
        mx.eval(out)
    mx.synchronize()
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(batch):
            out = f(*args)
            mx.async_eval(out)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / batch)
    return statistics.median(ts)


def main():
    mx.eval(mx.compile(lambda a: a + 1)(mx.zeros(8)))
    n = 1_000_000
    f_mat, q_mat, _h, _rd, _rf = track_matrices()
    fj = mx.array(f_mat.astype(np.float32))
    lq = mx.array(np.linalg.cholesky(q_mat).astype(np.float32))
    p = mx.random.normal((n, 4), key=mx.random.key(1))
    keys_n = mx.random.split(mx.random.key(0), n)
    mx.eval(p, keys_n)

    tests = {
        "vmap_matvec_F": mx.compile(lambda pp: mx.vmap(lambda s: fj @ s)(pp)),
        "batched_matmul_F": mx.compile(lambda pp: pp @ fj.T),
        "vmap_keyed_normal4": mx.compile(
            lambda ks: mx.vmap(lambda k: mx.random.normal((4,), key=k))(ks)
        ),
        "vmap_L_at_normal4": mx.compile(
            lambda ks: mx.vmap(lambda k: lq @ mx.random.normal((4,), key=k))(ks)
        ),
        "batched_normal_at_LT": mx.compile(
            lambda k: mx.random.normal((n, 4), key=k) @ lq.T
        ),
    }
    for name, f in tests.items():
        arg = (
            (p,)
            if "F" in name and "normal" not in name
            else ((keys_n,) if "vmap" in name else (mx.random.key(2),))
        )
        t = timeit(f, arg)
        print(f"{name}: {t * 1e6:.1f} us", flush=True)


if __name__ == "__main__":
    main()
