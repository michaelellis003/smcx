# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""Frozen multivariate Kalman filter and RTS smoother reference data.

The canonical outputs were generated on CPU in float64 with statsmodels
0.14.6 at commit ``40e6a84d26ac74623c6b94b718f0987ef0351c53``. The
low-level ``KalmanSmoother`` used its conventional filter, Cholesky solve,
forced covariance symmetry, known initial distribution, and default
predicted-state timing.

Repository (BSD-3-Clause):
https://github.com/statsmodels/statsmodels

Immutable source:
https://github.com/statsmodels/statsmodels/blob/40e6a84d26ac74623c6b94b718f0987ef0351c53/statsmodels/tsa/statespace/kalman_smoother.py

The same model was independently evaluated with Dynamax 1.0.2 at commit
``a216d7feec0d025560a0a194ed5abab538648375``. Its maximum absolute
differences from the canonical arrays are recorded below. The roughly
1e-9 covariance differences are expected because Dynamax's PSD solver
adds ``1e-9 I``.

Repository (MIT):
https://github.com/probml/dynamax

Immutable source:
https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/linear_gaussian_ssm/inference.py

Generation environment: Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0,
JAX/JAXlib 0.10.2, and tfp-nightly 0.26.0.dev20260717, with
``JAX_PLATFORMS=cpu`` and JAX float64 enabled.

Transition arrays use smcx's incoming convention. Entry ``k`` maps state
``k`` to state ``k + 1`` and consumes ``INPUTS[k + 1]``; ``INPUTS[0]``
only affects observation zero. Reference libraries use outgoing
transitions, so their outgoing slot ``k`` was given that destination
input. The conspicuous first input catches an accidental off-by-one use.

Only numerical inputs and outputs are retained. No statsmodels or
Dynamax code was copied or translated, and neither package is needed to
run the smcx tests.
"""

import numpy as np

REFERENCE_METADATA = {
    "canonical_package": "statsmodels",
    "canonical_version": "0.14.6",
    "canonical_commit": "40e6a84d26ac74623c6b94b718f0987ef0351c53",
    "crosscheck_package": "dynamax",
    "crosscheck_version": "1.0.2",
    "crosscheck_commit": "a216d7feec0d025560a0a194ed5abab538648375",
}

DYNAMAX_MAX_ABS_DIFFS = {
    "predicted_means": 3.26893928592753e-10,
    "predicted_covariances": 1.0923089377445905e-09,
    "filtered_means": 3.644584911643989e-10,
    "filtered_covariances": 1.2754513001311807e-09,
    "smoothed_means": 1.0572736297564944e-09,
    "smoothed_covariances": 1.6980152284240546e-09,
    "marginal_loglik": 2.6724968904545676e-09,
}

INITIAL_MEAN = np.array([0.35, -0.25], dtype=np.float64)

INITIAL_COVARIANCE = np.array(
    [
        [0.8, 0.12],
        [0.12, 0.55],
    ],
    dtype=np.float64,
)

TRANSITION_MATRIX = np.array(
    [
        [[0.92, 0.18], [-0.08, 0.84]],
        [[1.03, -0.12], [0.07, 0.91]],
        [[0.81, 0.22], [-0.15, 0.96]],
        [[0.97, 0.05], [0.11, 0.78]],
    ],
    dtype=np.float64,
)

TRANSITION_COVARIANCE = np.array(
    [
        [[0.07, 0.012], [0.012, 0.05]],
        [[0.045, -0.008], [-0.008, 0.065]],
        [[0.09, 0.015], [0.015, 0.04]],
        [[0.055, -0.006], [-0.006, 0.075]],
    ],
    dtype=np.float64,
)

OBSERVATION_MATRIX = np.array(
    [
        [[1.0, 0.25], [-0.15, 0.9]],
        [[0.82, -0.2], [0.12, 1.08]],
        [[1.15, 0.08], [-0.25, 0.72]],
        [[0.93, 0.31], [0.05, 0.88]],
        [[1.05, -0.1], [-0.18, 0.97]],
    ],
    dtype=np.float64,
)

OBSERVATION_COVARIANCE = np.array(
    [
        [[0.21, 0.025], [0.025, 0.16]],
        [[0.18, -0.012], [-0.012, 0.23]],
        [[0.14, 0.018], [0.018, 0.2]],
        [[0.24, 0.032], [0.032, 0.17]],
        [[0.19, -0.02], [-0.02, 0.15]],
    ],
    dtype=np.float64,
)

TRANSITION_BIAS = np.array(
    [
        [0.04, -0.03],
        [-0.02, 0.05],
        [0.03, 0.015],
        [-0.01, -0.04],
    ],
    dtype=np.float64,
)

OBSERVATION_BIAS = np.array(
    [
        [0.025, -0.035],
        [-0.015, 0.02],
        [0.04, 0.0],
        [-0.03, 0.025],
        [0.01, -0.02],
    ],
    dtype=np.float64,
)

TRANSITION_INPUT_MATRIX = np.array(
    [
        [0.16, -0.07],
        [0.05, 0.13],
    ],
    dtype=np.float64,
)

OBSERVATION_INPUT_MATRIX = np.array(
    [
        [0.008, 0.002],
        [-0.003, 0.006],
    ],
    dtype=np.float64,
)

INPUTS = np.array(
    [
        [40.0, -25.0],
        [0.5, -0.8],
        [-1.2, 0.4],
        [0.75, 1.1],
        [-0.3, -0.6],
    ],
    dtype=np.float64,
)

EMISSIONS = np.array(
    [
        [0.42, -0.31],
        [0.15, 0.27],
        [-0.38, 0.44],
        [0.63, -0.18],
        [-0.05, 0.36],
    ],
    dtype=np.float64,
)

PREDICTED_MEANS = np.array(
    [
        [0.35, -0.25],
        [0.3225693238769978, -0.17432519014459172],
        [0.07798724500947518, 0.0546230455544927],
        [-0.07522131234048349, 0.3963073809231631],
        [0.1280252204521634, 0.05473118856384585],
    ],
    dtype=np.float64,
)

PREDICTED_COVARIANCES = np.array(
    [
        [[0.8, 0.12], [0.12, 0.55]],
        [
            [0.20981107576506447, 0.0321396613915184],
            [0.0321396613915184, 0.15081222972236993],
        ],
        [
            [0.16563334362910132, 0.00131504012234527],
            [0.00131504012234527, 0.1366037991444227],
        ],
        [
            [0.13720186989793043, 0.03274110426222385],
            [0.03274110426222385, 0.13325275245796403],
        ],
        [
            [0.1392893880140021, 0.01232531179093509],
            [0.01232531179093509, 0.1259830868146055],
        ],
    ],
    dtype=np.float64,
)

FILTERED_MEANS = np.array(
    [
        [0.17133735554205368, -0.06145024012050881],
        [0.3075850252988913, -0.00978890793014253],
        [-0.23003958012578335, 0.17323067073364123],
        [0.13706978102049686, 0.22135065724562974],
        [0.05732116114721997, 0.2043942019129096],
    ],
    dtype=np.float64,
)

FILTERED_COVARIANCES = np.array(
    [
        [
            [0.1546381743600722, 0.01286901115368026],
            [0.0128690111536803, 0.14392310162346955],
        ],
        [
            [0.11512511339920607, 0.01097852399205528],
            [0.01097852399205528, 0.08409735796664501],
        ],
        [
            [0.06166875452885402, 0.00515841970729779],
            [0.00515841970729779, 0.1012921336336443],
        ],
        [
            [0.08859282082380442, 0.00755295891993547],
            [0.00755295891993547, 0.07990635426031646],
        ],
        [
            [0.07768736873150428, 0.00931026677744522],
            [0.00931026677744522, 0.07112344085890368],
        ],
    ],
    dtype=np.float64,
)

SMOOTHED_MEANS = np.array(
    [
        [0.02160784157204187, 0.07811046160788765],
        [0.15639467819827033, 0.01691769552429248],
        [-0.14443735580793884, 0.10294324445130984],
        [0.10324312510176895, 0.2932010302065158],
        [0.05732116114721997, 0.20439420191290958],
    ],
    dtype=np.float64,
)

SMOOTHED_COVARIANCES = np.array(
    [
        [
            [0.08056912241786768, 0.00344758897519761],
            [0.00344758897519759, 0.08625227507323792],
        ],
        [
            [0.05733801425413797, 0.00618094626058993],
            [0.00618094626058994, 0.06139549876386482],
        ],
        [
            [0.05183283894109298, 0.00246533239841239],
            [0.00246533239841239, 0.06568586478642273],
        ],
        [
            [0.06490715710043368, 0.00346713633582135],
            [0.00346713633582135, 0.06612143009191268],
        ],
        [
            [0.07768736873150428, 0.00931026677744522],
            [0.00931026677744522, 0.07112344085890368],
        ],
    ],
    dtype=np.float64,
)

LOG_EVIDENCE_INCREMENTS = np.array(
    [
        -1.6970962876102613,
        -1.0542306030604955,
        -1.3096676642449818,
        -2.145053302896293,
        -0.8880889283063231,
    ],
    dtype=np.float64,
)

MARGINAL_LOG_LIKELIHOOD = -7.094136786118355
