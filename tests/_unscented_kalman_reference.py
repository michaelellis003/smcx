# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0

"""Frozen independent outputs for an unscented Kalman filter and smoother.

The canonical outputs were generated on CPU in float64 with CPython 3.13.9,
NumPy 2.5.1, Stone Soup 1.9.1, and SciPy 1.18.0. The scaled-rule parameters
were passed explicitly as ``alpha=1``, ``beta=2``, and ``kappa=0``.

Stone Soup 1.9.1, commit
``a4336b920a799cfe0a77ecb05867c5deeb371c7a``, MIT:

* https://github.com/dstl/Stone-Soup/releases/tag/v1.9.1
* https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/stonesoup/functions/__init__.py
* https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/stonesoup/predictor/kalman.py
* https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/stonesoup/updater/kalman.py
* https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/stonesoup/smoother/kalman.py#L271-L325
* https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/LICENSE

``UnscentedKalmanPredictor`` and ``UnscentedKalmanUpdater`` produced the
filtering track, and ``UnscentedKalmanSmoother`` consumed it. All three used
the explicitly stated scaled-rule parameters. The updater used its default
subtractive covariance form, and every stored covariance was forced symmetric.
The retained smoother covariances were also symmetrized, changing raw Stone
Soup entries by less than 1.74e-18. Measurement sigma points were regenerated
after the complete process-noise-inclusive prediction. Recomputing the update
in smcx's residual-sigma form changed array fields by at most 2.78e-16 and
total evidence by 4.44e-16.

SciPy 1.18.0, commit
``54ef5423f2e4376230ec3bfda6912a07a50958e3``, BSD-3-Clause, supplied the
innovation log densities:

* https://github.com/scipy/scipy/releases/tag/v1.18.0
* https://github.com/scipy/scipy/blob/54ef5423f2e4376230ec3bfda6912a07a50958e3/scipy/stats/_multivariate.py
* https://github.com/scipy/scipy/blob/54ef5423f2e4376230ec3bfda6912a07a50958e3/LICENSE.txt

The values were cross-checked with Dynamax 1.0.2, commit
``a216d7feec0d025560a0a194ed5abab538648375``, MIT:

* https://github.com/probml/dynamax/releases/tag/1.0.2
* https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/nonlinear_gaussian_ssm/inference_ukf.py#L79-L122
* https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/nonlinear_gaussian_ssm/inference_ukf.py#L242-L318
* https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/utils/utils.py#L213-L218
* https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/LICENSE

That cross-check used JAX/JAXlib 0.10.2,
tfp-nightly 0.26.0.dev20260717, ``JAX_PLATFORMS=cpu``, and JAX float64.
Dynamax differed by at most 3.46e-10 in predicted means, 1.56e-9 in predicted
covariances, 3.76e-10 in filtered means, 1.94e-9 in filtered covariances, and
9.29e-9 in total evidence because its PSD solve adds a 1e-9 diagonal boost.
Its smoother differed by less than 1.08e-9 in means and 2.50e-9 in
covariances. The maximum Stone Soup innovation-covariance condition number
was 2.613033461095126.

Only numerical inputs and outputs are retained. No implementation code was
copied or translated, and no comparison package is a test dependency.
"""

import numpy as np

ALPHA = 1.0
BETA = 2.0
KAPPA = 0.0

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
        [0.09683613501217056, -0.25747155592017557],
        [0.14215163968835182, -0.1969062656288512],
        [0.06571298376242754, -0.04533050964874276],
        [0.19148792789177288, -0.0038685365445344233],
    ],
    dtype=np.float64,
)
PREDICTED_COVARIANCES = np.array(
    [
        [[0.55, 0.08], [0.08, 0.40]],
        [
            [0.11652645621579616, 0.01382223490738563],
            [0.01382223490738563, 0.14618900885400268],
        ],
        [
            [0.08579626860083231, 0.013010027384036216],
            [0.013010027384036216, 0.10215539120978431],
        ],
        [
            [0.07923045295499463, 0.012966228771887665],
            [0.012966228771887665, 0.08602430288080579],
        ],
        [
            [0.07719082823075876, 0.0122828500457472],
            [0.0122828500457472, 0.07995118112362001],
        ],
    ],
    dtype=np.float64,
)
FILTERED_MEANS = np.array(
    [
        [0.16597280017566024, -0.2618069603844005],
        [0.2031024360429863, -0.189949094422927],
        [0.08374598131466135, -0.03911427746632229],
        [0.21548509269742033, 0.024131557576074006],
        [0.11121904272233052, -0.09453411027467092],
    ],
    dtype=np.float64,
)
FILTERED_COVARIANCES = np.array(
    [
        [
            [0.09767462109020819, -0.004761054507920942],
            [-0.004761054507920942, 0.13795695008979458],
        ],
        [
            [0.057455634912345115, -0.0009155842750076916],
            [-0.0009155842750076916, 0.08610242105279653],
        ],
        [
            [0.04847621843615829, 0.0013221092674976101],
            [0.0013221092674976101, 0.06813174335258701],
        ],
        [
            [0.046110136505925735, 0.0015818280960216164],
            [0.0015818280960216164, 0.06012674750321503],
        ],
        [
            [0.04541482912631204, 0.0015594461287094241],
            [0.0015594461287094241, 0.05700662098798591],
        ],
    ],
    dtype=np.float64,
)
SMOOTHED_MEANS = np.array(
    [
        [0.19207014197706154, -0.08918541784614414],
        [0.1810768806356682, -0.06256674996308037],
        [0.14908531891971977, -0.02967933437804084],
        [0.18401701327216213, -0.040984287677055684],
        [0.11121904272233052, -0.09453411027467092],
    ],
    dtype=np.float64,
)
SMOOTHED_COVARIANCES = np.array(
    [
        [
            [0.05756351733138948, -0.004582080543671107],
            [-0.004582080543671107, 0.06814112710437469],
        ],
        [
            [0.04119810299179997, -0.002789666471017912],
            [-0.002789666471017912, 0.053734840494212226],
        ],
        [
            [0.03708336653291222, -0.001706053868571588],
            [-0.001706053868571588, 0.0484751842077438],
        ],
        [
            [0.03787155747645921, -0.0010753231476699114],
            [-0.0010753231476699114, 0.048913011062869174],
        ],
        [
            [0.04541482912631204, 0.0015594461287094241],
            [0.0015594461287094241, 0.05700662098798591],
        ],
    ],
    dtype=np.float64,
)
LOG_EVIDENCE_INCREMENTS = np.array(
    [
        -1.0010520526701658,
        -0.29493991285642895,
        -0.6051732165244941,
        -0.35373079851937933,
        -0.19830841019809403,
    ],
    dtype=np.float64,
)
MARGINAL_LOG_LIKELIHOOD = np.float64(-2.4532043907685623)
