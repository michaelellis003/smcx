# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen independent outputs for a nonlinear extended Kalman filter.

The canonical outputs were generated on CPU in float64 with Stone Soup
1.9.1 and SciPy 1.18.0:

* CPython 3.13.9
* NumPy 2.5.1
* Stone Soup 1.9.1, commit
  a4336b920a799cfe0a77ecb05867c5deeb371c7a, MIT:
  https://github.com/dstl/Stone-Soup/releases/tag/v1.9.1
  https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/LICENSE
* SciPy 1.18.0, commit
  54ef5423f2e4376230ec3bfda6912a07a50958e3, BSD-3-Clause:
  https://github.com/scipy/scipy/releases/tag/v1.18.0

Stone Soup's ``ExtendedKalmanPredictor`` and ``ExtendedKalmanUpdater`` were
used with ``use_joseph_cov=True`` and forced covariance symmetry. SciPy's
multivariate normal log density supplied the innovation likelihoods.

The values were cross-checked with Dynamax 1.0.2, commit
a216d7feec0d025560a0a194ed5abab538648375, MIT:
https://github.com/probml/dynamax/releases/tag/1.0.2
https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/LICENSE

Dynamax differed by at most 3.35e-10 in filtered means, 1.94e-9 in
covariances, and 9.40e-9 in total evidence because its PSD solve adds a
1e-9 diagonal boost. The maximum Stone Soup innovation-covariance condition
number was 2.618. Only numerical inputs and outputs are retained here; no
implementation code was copied or translated.
"""

import numpy as np

INITIAL_MEAN = np.array([0.25, -0.35], dtype=np.float64)
INITIAL_COVARIANCE = np.array(
    [[0.55, 0.08], [0.08, 0.40]],
    dtype=np.float64,
)
TRANSITION_COVARIANCE = np.array(
    [[0.04, 0.006], [0.006, 0.03]],
    dtype=np.float64,
)
OBSERVATION_COVARIANCE = np.array(
    [[0.12, -0.01], [-0.01, 0.09]],
    dtype=np.float64,
)
EMISSIONS = np.array(
    [
        [0.18, -0.12],
        [0.31, -0.05],
        [-0.08, 0.22],
        [0.42, 0.10],
        [0.05, -0.18],
    ],
    dtype=np.float64,
)

PREDICTED_MEANS = np.array(
    [
        [0.25, -0.35],
        [0.12276054052997835, -0.26816179137230595],
        [0.15781328956761415, -0.20774708858882648],
        [0.07692041533194274, -0.05444923489320876],
        [0.19969568478845784, -0.01206429746143826],
    ],
    dtype=np.float64,
)
PREDICTED_COVARIANCES = np.array(
    [
        [[0.55, 0.08], [0.08, 0.40]],
        [
            [0.1136179591169599, 0.01304807278683837],
            [0.01304807278683837, 0.14682831751764933],
        ],
        [
            [0.08513843674294633, 0.01299081562846191],
            [0.01299081562846191, 0.1025543848181983],
        ],
        [
            [0.07907103018643538, 0.01301339506632127],
            [0.01301339506632127, 0.08624624169179518],
        ],
        [
            [0.07715158760169193, 0.01230080358747652],
            [0.01230080358747652, 0.08009055545441858],
        ],
    ],
    dtype=np.float64,
)
FILTERED_MEANS = np.array(
    [
        [0.19690392695928127, -0.26934656035479865],
        [0.22270658389104078, -0.19916454258384314],
        [0.09818911638876097, -0.04720128315017796],
        [0.22621340837524626, 0.01659021596843913],
        [0.11948403290075882, -0.10124821257992712],
    ],
    dtype=np.float64,
)
FILTERED_COVARIANCES = np.array(
    [
        [
            [0.09415691366043905, -0.00627064056069843],
            [-0.00627064056069843, 0.13798069052260536],
        ],
        [
            [0.05655926060926272, -0.00111554495818018],
            [-0.00111554495818018, 0.08639084184126553],
        ],
        [
            [0.04817794040567638, 0.00131543654720279],
            [0.00131543654720279, 0.06831643954123154],
        ],
        [
            [0.04598811348098338, 0.00157756157036097],
            [0.00157756157036097, 0.06023688212586616],
        ],
        [
            [0.04534578071889979, 0.00154226995981523],
            [0.00154226995981523, 0.05708215892992072],
        ],
    ],
    dtype=np.float64,
)
LOG_EVIDENCE_INCREMENTS = np.array(
    [
        -0.9903633957623901,
        -0.28277294049814994,
        -0.6221645680415542,
        -0.3528577214619353,
        -0.19278446542251593,
    ],
    dtype=np.float64,
)
MARGINAL_LOG_LIKELIHOOD = np.float64(-2.4409430911865453)
