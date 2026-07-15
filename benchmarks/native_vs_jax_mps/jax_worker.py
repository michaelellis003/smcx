# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process JAX CPU or jax-mps benchmark worker."""

import argparse
import json
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from common import LGSSM, SCHEMA_VERSION, lgssm_data, summarize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm",
        choices=("jax_cpu", "jax_mps_async", "jax_mps_sync"),
        required=True,
    )
    parser.add_argument("--block", required=True, type=int)
    parser.add_argument("--repeats", required=True, type=int)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--warmups", required=True, type=int)
    parser.add_argument(
        "--workload",
        choices=(
            "eltwise_reduce",
            "gather_scatter",
            "lgssm_pf",
            "matmul",
            "random",
            "scan",
            "systematic",
        ),
        required=True,
    )
    return parser.parse_args()


def _package_version(name: str) -> str | None:
    """Return an installed package version without requiring jax-mps on CPU."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _peak_memory(device) -> int | None:
    """Read the best backend-reported peak-memory counter available."""
    stats = device.memory_stats()
    if not stats:
        return None
    for name in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
        if name in stats:
            return int(stats[name])
    return None


def _eltwise_reduce(size, jax, jnp):
    """Build the elementwise negative control and f64 oracle."""
    x_np = np.linspace(-3.0, 3.0, size, dtype=np.float32)

    def operation(value):
        return jnp.sum(jnp.tanh(value) * jax.nn.sigmoid(value) + 0.1 * value**2)

    x64 = x_np.astype(np.float64)
    expected = np.asarray(
        np.sum(
            np.tanh(x64) / (1.0 + np.exp(-x64)) + 0.1 * x64**2,
            dtype=np.float64,
        )
    )
    return (x_np,), jax.jit(operation), expected


def _gather_scatter(size, jax, jnp):
    """Build deterministic indexed gather/scatter inputs and oracle."""
    rng = np.random.default_rng(20260715)
    base_np = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    indices_np = rng.integers(0, size, size=size, dtype=np.int32)
    updates_np = base_np[indices_np] * np.float32(0.01)

    def operation(base, indices, updates):
        return base.at[indices].add(updates)

    expected = base_np.astype(np.float64)
    np.add.at(expected, indices_np, updates_np.astype(np.float64))
    return (base_np, indices_np, updates_np), jax.jit(operation), expected


def _matmul(size, jax, jnp):
    """Build dense matrix inputs and an f64 checksum oracle."""
    rng = np.random.default_rng(20260715)
    left_np = rng.normal(0.0, 0.1, size=(size, size)).astype(np.float32)
    right_np = rng.normal(0.0, 0.1, size=(size, size)).astype(np.float32)

    def operation(left, right):
        return jnp.sum(left @ right)

    expected = np.asarray(
        np.sum(
            left_np.astype(np.float64) @ right_np.astype(np.float64),
            dtype=np.float64,
        )
    )
    return (left_np, right_np), jax.jit(operation), expected


def _lgssm_pf(size, jax, jnp):
    """Build an adversarial whole-filter JAX bootstrap filter."""
    observations_np, oracle = lgssm_data()
    normalizer = float(np.log(2.0 * np.pi * LGSSM["r"]))
    initial_scale = float(np.sqrt(LGSSM["p0"]))
    transition_scale = float(np.sqrt(LGSSM["q"]))
    log_size = float(np.log(size))

    def logsumexp(values):
        maximum = jnp.max(values)
        return maximum + jnp.log(jnp.sum(jnp.exp(values - maximum)))

    def operation(key, observations):
        key, initial_key = jax.random.split(key)
        particles = LGSSM["m0"] + initial_scale * jax.random.normal(
            initial_key, shape=(size, 1), dtype=jnp.float32
        )
        residual = observations[0] - particles[:, 0]
        unnormalized = -0.5 * (normalizer + residual**2 / LGSSM["r"])
        log_sum = logsumexp(unnormalized)
        log_weights = unnormalized - log_sum
        marginal = log_sum - log_size
        ess = 1.0 / jnp.sum(jnp.exp(2.0 * log_weights))
        identity = jnp.arange(size, dtype=jnp.int32)
        step_keys = jax.random.split(key, observations.shape[0] - 1)

        def step(carry, inputs):
            current_particles, current_log_weights, current_marginal = carry
            step_key, observation = inputs
            resampling_key, transition_key = jax.random.split(step_key)
            previous_ess = 1.0 / jnp.sum(jnp.exp(2.0 * current_log_weights))
            should_resample = previous_ess < 0.5 * size

            def resample(_):
                cdf = jnp.cumsum(jnp.exp(current_log_weights))
                cdf = cdf / cdf[-1]
                offset = jax.random.uniform(resampling_key)
                queries = (offset + jnp.arange(size)) / size
                indices = jnp.searchsorted(cdf, queries, side="right")
                indices = jnp.clip(indices, 0, size - 1)
                return jnp.take(current_particles, indices, axis=0), indices

            def skip(_):
                return current_particles, identity

            parents, ancestors = jax.lax.cond(
                should_resample, resample, skip, operand=None
            )
            noise = jax.random.normal(
                transition_key, shape=parents.shape, dtype=jnp.float32
            )
            next_particles = LGSSM["a"] * parents + transition_scale * noise
            residual = observation - next_particles[:, 0]
            log_observation = -0.5 * (normalizer + residual**2 / LGSSM["r"])
            unnormalized = jnp.where(
                should_resample,
                log_observation,
                current_log_weights + log_observation,
            )
            next_log_sum = logsumexp(unnormalized)
            next_log_weights = unnormalized - next_log_sum
            increment = jnp.where(
                should_resample, next_log_sum - log_size, next_log_sum
            )
            next_ess = 1.0 / jnp.sum(jnp.exp(2.0 * next_log_weights))
            next_carry = (
                next_particles,
                next_log_weights,
                current_marginal + increment,
            )
            outputs = (
                next_particles,
                next_log_weights,
                ancestors,
                next_ess,
                increment,
            )
            return next_carry, outputs

        carry, history = jax.lax.scan(
            step,
            (particles, log_weights, marginal),
            (step_keys, observations[1:]),
        )
        _, _, marginal = carry
        particle_history = jnp.concatenate(
            (particles[None, ...], history[0]), axis=0
        )
        weight_history = jnp.concatenate(
            (log_weights[None, ...], history[1]), axis=0
        )
        ancestor_history = jnp.concatenate(
            (identity[None, ...], history[2]), axis=0
        )
        ess_history = jnp.concatenate((ess[None], history[3]), axis=0)
        increments = jnp.concatenate(
            ((log_sum - log_size)[None], history[4]), axis=0
        )
        return (
            marginal,
            particle_history,
            weight_history,
            ancestor_history,
            ess_history,
            increments,
        )

    inputs = (jax.random.key(20260715), observations_np)
    return inputs, jax.jit(operation), np.asarray(oracle)


def _random(size, jax, jnp):
    """Build a fixed-key normal draw with moment output."""

    def operation(key):
        samples = jax.random.normal(key, shape=(size,), dtype=jnp.float32)
        return jnp.stack((jnp.mean(samples), jnp.var(samples)))

    return (jax.random.key(20260715),), jax.jit(operation), None


def _scan(size, jax, jnp):
    """Build a whole-loop compiled JAX scan and f64 oracle."""
    initial_np = np.linspace(-1.0, 1.0, size, dtype=np.float32)

    def operation(initial):
        def step(state, _):
            return jnp.tanh(0.99 * state + 0.01), None

        state, _ = jax.lax.scan(step, initial, xs=None, length=100)
        return jnp.concatenate((state, jnp.sum(state)[None]))

    expected_state = initial_np.astype(np.float64)
    for _ in range(100):
        expected_state = np.tanh(0.99 * expected_state + 0.01)
    expected = np.concatenate((expected_state, [np.sum(expected_state)]))
    return (initial_np,), jax.jit(operation), expected


def _systematic(size, jax, jnp):
    """Build supplied systematic queries and JAX searchsorted."""
    weights_np = np.full(size, 1.0 / size, dtype=np.float32)
    queries_np = (np.arange(size, dtype=np.float64) + 0.37) / size
    particles_np = np.linspace(-2.0, 2.0, size, dtype=np.float32)

    def operation(weights, queries, particles):
        cdf = jnp.cumsum(weights)
        cdf = cdf / jnp.maximum(cdf[-1], jnp.finfo(jnp.float32).tiny)
        ancestors = jnp.searchsorted(cdf, queries, side="right")
        ancestors = jnp.clip(ancestors, 0, size - 1)
        return jnp.take(particles, ancestors)

    cdf = np.cumsum(weights_np.astype(np.float64))
    cdf /= cdf[-1]
    ancestors = np.searchsorted(cdf, queries_np, side="right")
    ancestors = np.clip(ancestors, 0, size - 1)
    expected = particles_np.astype(np.float64)[ancestors]
    inputs = (weights_np, queries_np.astype(np.float32), particles_np)
    return inputs, jax.jit(operation), expected


def _build_workload(name, size, jax, jnp):
    """Build one explicitly registered workload."""
    builders = {
        "eltwise_reduce": _eltwise_reduce,
        "gather_scatter": _gather_scatter,
        "lgssm_pf": _lgssm_pf,
        "matmul": _matmul,
        "random": _random,
        "scan": _scan,
        "systematic": _systematic,
    }
    return builders[name](size, jax, jnp)


def main() -> None:
    """Run one benchmark block and emit one JSON result."""
    args = _parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)
    expected_backend = "cpu" if args.arm == "jax_cpu" else "mps"
    actual_backend = jax.default_backend()
    if actual_backend != expected_backend:
        raise RuntimeError(
            f"requested {expected_backend} backend, got {actual_backend}"
        )

    host_inputs, compiled, expected = _build_workload(
        args.workload, args.size, jax, jnp
    )
    inputs = tuple(jax.device_put(value) for value in host_inputs)
    for value in inputs:
        value.block_until_ready()

    started = time.perf_counter()
    output = compiled(*inputs)
    jax.block_until_ready(output)
    cold_s = time.perf_counter() - started

    if args.workload == "lgssm_pf":
        actual = float(output[0])
        passed = bool(np.isfinite(actual))
        expected_json = float(expected)
        atol = None
        rtol = None
    elif args.workload == "random":
        actual_array = np.asarray(output)
        mean, variance = (float(value) for value in actual_array)
        mean_limit = 5.0 / np.sqrt(args.size)
        variance_limit = 5.0 * np.sqrt(2.0 / (args.size - 1))
        passed = bool(
            abs(mean) <= mean_limit and abs(variance - 1.0) <= variance_limit
        )
        actual = {"mean": mean, "variance": variance}
        expected_json = {"mean": 0.0, "variance": 1.0}
        atol = {"mean": mean_limit, "variance": variance_limit}
        rtol = 0.0
    else:
        actual_array = np.asarray(output)
        passed = bool(np.allclose(actual_array, expected, rtol=5e-5, atol=5e-6))
        actual = float(np.sum(actual_array, dtype=np.float64))
        expected_json = float(np.sum(expected, dtype=np.float64))
        atol = 5e-6
        rtol = 5e-5

    for _ in range(args.warmups):
        jax.block_until_ready(compiled(*inputs))

    times = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        jax.block_until_ready(compiled(*inputs))
        times.append(time.perf_counter() - started)

    dispatch_mode = "cpu"
    if args.arm.startswith("jax_mps_"):
        dispatch_mode = (
            "async"
            if os.environ.get("JAX_MPS_ASYNC_DISPATCH") == "1"
            else "safe"
        )
    result = {
        "arm": args.arm,
        "backend": actual_backend,
        "block": args.block,
        "cold_s": cold_s,
        "correctness": {
            "actual": actual,
            "atol": atol,
            "expected": expected_json,
            "passed": passed,
            "rtol": rtol,
        },
        "dispatch_mode": dispatch_mode,
        "failure": None,
        "parameters": {"size": args.size},
        "peak_memory_bytes": _peak_memory(jax.devices()[0]),
        "schema_version": SCHEMA_VERSION,
        "summary": summarize(times),
        "times_s": times,
        "versions": {
            "jax": jax.__version__,
            "jax-mps": _package_version("jax-mps"),
            "jaxlib": _package_version("jaxlib"),
            "python": platform.python_version(),
        },
        "workload": args.workload,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
