# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

# H2 (JAX-CPU side): same comparison in the smcjax venv.
import json
import pathlib
import statistics
import time

import jax
import jax.numpy as jnp
import jax.random as jr

SCRATCH = pathlib.Path(__file__).parent


def timeit(f, args, batch, reps=12, warm=3):
    for _ in range(warm):
        f(*args).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(batch):
            out = f(*args)
        out.block_until_ready()
        ts.append((time.perf_counter() - t0) / batch)
    return statistics.median(ts)


def main():
    key = jr.PRNGKey(0)
    out = {}
    for n in (100_000, 1_000_000):
        batch = 10 if n >= 10**6 else 30
        keys_n = jr.split(key, n)
        keys_n.block_until_ready()
        row = {}
        row["split"] = timeit(jax.jit(lambda k: jr.split(k, n)), (key,), batch)
        for d in (1, 4):
            f_v = jax.jit(lambda ks: jax.vmap(lambda k: jr.normal(k, (d,)))(ks))
            row[f"vmap_d{d}"] = timeit(f_v, (keys_n,), batch)
            f_b = jax.jit(lambda k: jr.normal(k, (n, d)))
            row[f"batched_d{d}"] = timeit(f_b, (key,), batch)
        f_v0 = jax.jit(lambda ks: jax.vmap(lambda k: jr.normal(k, ()))(ks))
        row["vmap_scalar"] = timeit(f_v0, (keys_n,), batch)
        f_b0 = jax.jit(lambda k: jr.normal(k, (n,)))
        row["batched_dN"] = timeit(f_b0, (key,), batch)
        for d in (1, 4):
            f_sv = jax.jit(
                lambda k: jax.vmap(lambda kk: jr.normal(kk, (d,)))(
                    jr.split(k, n)
                )
            )
            row[f"split_plus_vmap_d{d}"] = timeit(f_sv, (key,), batch)
        out[str(n)] = row
        print(n, {k: f"{v * 1e6:.1f}us" for k, v in row.items()}, flush=True)
    (SCRATCH / "h2_jax.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
