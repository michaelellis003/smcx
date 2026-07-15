# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# H2 (MLX side): per-particle-key vmapped draws vs one batched draw.
import json
import pathlib
import statistics
import time

import mlx.core as mx

SCRATCH = pathlib.Path(__file__).parent


def timeit(f, args, batch, reps=12, warm=3):
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
    key = mx.random.key(0)
    out = {}
    for n in (100_000, 1_000_000):
        batch = 20 if n >= 10**6 else 50
        keys_n = mx.random.split(key, n)
        mx.eval(keys_n)
        row = {}
        # split cost itself
        row["split"] = timeit(
            mx.compile(lambda k: mx.random.split(k, n)), (key,), batch
        )
        for d in (1, 4):
            # (i) vmapped per-key draw
            f_v = mx.compile(
                lambda ks: mx.vmap(lambda k: mx.random.normal((d,), key=k))(ks)
            )
            row[f"vmap_d{d}"] = timeit(f_v, (keys_n,), batch)
            # (ii) one batched draw
            f_b = mx.compile(lambda k: mx.random.normal((n, d), key=k))
            row[f"batched_d{d}"] = timeit(f_b, (key,), batch)
        # scalar draw per particle (shape () per row)
        f_v0 = mx.compile(
            lambda ks: mx.vmap(lambda k: mx.random.normal((), key=k))(ks)
        )
        row["vmap_scalar"] = timeit(f_v0, (keys_n,), batch)
        f_b0 = mx.compile(lambda k: mx.random.normal((n,), key=k))
        row["batched_dN"] = timeit(f_b0, (key,), batch)
        # split + vmap together (what the filter actually pays)
        for d in (1, 4):
            f_sv = mx.compile(
                lambda k: mx.vmap(lambda kk: mx.random.normal((d,), key=kk))(
                    mx.random.split(k, n)
                )
            )
            row[f"split_plus_vmap_d{d}"] = timeit(f_sv, (key,), batch)
        out[str(n)] = row
        print(n, {k: f"{v * 1e6:.1f}us" for k, v in row.items()}, flush=True)
    (SCRATCH / "h2_mlx.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
