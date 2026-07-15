# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

r"""External-authority SMC² baseline: Chopin's `particles` (ADR-0014).

`particles` pins numpy<2 and so conflicts with smcx's runtime; run it
in an isolated environment that ignores this project:

    uv run --no-project --with 'particles>=0.4' \\
        python benchmarks/smc2/particles_side.py 512 512 100

Same LGSSM (unknown AR coefficient a) and the same data generator as
benchmarks/smc2/bench_smc2.py, so the log-evidence is directly
comparable. particles' SMC² default is waste-free with a length-10
move chain (heavier than smcx's 3 PMMH steps); reported as-is — this
is an external reference implementation, not a controlled
algorithm match. Prints one JSON line with the time and logZ.
"""

import json
import math
import sys
import time

import numpy as np
import particles
from particles import distributions as dists
from particles import smc_samplers as ssp
from particles import state_space_models as ssm

A_TRUE, Q, R, P0 = 0.9, 0.5, 0.3, 1.0


def _data(t_len, seed=0):
    rng = np.random.default_rng(seed)
    x = np.empty(t_len)
    x[0] = rng.normal(0.0, math.sqrt(P0))
    for t in range(1, t_len):
        x[t] = A_TRUE * x[t - 1] + rng.normal(0, math.sqrt(Q))
    return x + rng.normal(0, math.sqrt(R), t_len)


class LGSSM(ssm.StateSpaceModel):
    default_params = {"a": 0.9}

    def PX0(self):
        return dists.Normal(loc=0.0, scale=math.sqrt(P0))

    def PX(self, t, xp):
        return dists.Normal(loc=self.a * xp, scale=math.sqrt(Q))

    def PY(self, t, xp, x):
        return dists.Normal(loc=x, scale=math.sqrt(R))


def run_cell(n_theta, n_x, t_len):
    y = _data(t_len)
    prior = dists.StructDist({"a": dists.Uniform(0.5, 1.3)})
    fk = ssp.SMC2(ssm_cls=LGSSM, data=y, prior=prior, init_Nx=n_x)
    alg = particles.SMC(fk=fk, N=n_theta)
    t0 = time.time()
    alg.run()
    elapsed = time.time() - t0
    return {
        "impl": "particles",
        "n_theta": n_theta,
        "n_x": n_x,
        "t_len": t_len,
        "median_s": elapsed,  # single run (particles is not cheap to repeat)
        "logz": float(alg.summaries.logLts[-1]),
    }


if __name__ == "__main__":
    nth, nx, t = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    print(json.dumps(run_cell(nth, nx, t)))
