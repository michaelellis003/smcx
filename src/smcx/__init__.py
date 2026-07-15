# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""smcx.

Sequential Monte Carlo for Apple silicon, built on MLX
"""

from smcx.weights import ess, log_ess, log_normalize, normalize

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ess",
    "log_ess",
    "log_normalize",
    "normalize",
]
