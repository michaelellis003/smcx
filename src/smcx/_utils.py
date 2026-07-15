# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Entry validation and callback-arity helpers (ADR-0008 item 3).

Structural validation happens eagerly at public-function entry (MLX
shape inference is eager even though compute is lazy), producing
errors that name user-facing parameters. NaN in emissions is
PERMITTED — missing-observation semantics belong to the user's
``log_observation_fn`` (design §4); validation must never reject it.
"""

import inspect

import mlx.core as mx


def canonicalize_emissions(emissions: mx.array) -> mx.array:
    """Accept ``(T,)`` univariate series and canonicalize to ``(T, 1)``."""
    if emissions.ndim == 1:
        return emissions[:, None]
    if emissions.ndim != 2:
        raise ValueError(
            f"emissions must have shape (T,) or (T, emission_dim); "
            f"got ndim={emissions.ndim}"
        )
    return emissions


def canonicalize_inputs(inputs: mx.array, num_timesteps: int) -> mx.array:
    """Validate/canonicalize the optional per-step inputs array.

    ``inputs[t]`` feeds both the transition INTO step t and the
    observation AT step t; ``inputs[0]`` reaches only the initial
    weighting (ADR-0008 item 1 alignment convention).
    """
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    if inputs.shape[0] != num_timesteps:
        raise ValueError(
            f"inputs must have leading dimension T={num_timesteps} "
            f"aligned with emissions; got {inputs.shape[0]}"
        )
    return inputs


def num_positional_params(fn) -> int | None:
    """Count positional parameters, or None if undecidable.

    Undecidable signatures (``*args``, some builtins/partials) return
    None; callers apply the ADR-0008 fallback rule for their site.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    count = 0
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            count += 1
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            return None
    return count


def check_callback_arity(
    fn, name: str, base_arity: int, inputs_supplied: bool
) -> None:
    """Raise a named, actionable error on arity/inputs mismatch.

    Undecidable signatures are trusted (the ADR-0008 rule): a wrong
    guess would reject valid ``functools.partial``/bound-method
    closures, and a genuine mismatch still fails loudly inside vmap.
    """
    n = num_positional_params(fn)
    if n is None:
        return
    expected = base_arity + (1 if inputs_supplied else 0)
    if n != expected:
        if inputs_supplied and n == base_arity:
            raise TypeError(
                f"`inputs` was supplied but `{name}` takes {n} "
                f"arguments; add a trailing `input_t` parameter "
                f"(ADR-0008)."
            )
        raise TypeError(
            f"`{name}` takes {n} positional arguments; expected "
            f"{expected} ({'with' if inputs_supplied else 'without'} "
            f"a trailing `input_t`)."
        )
