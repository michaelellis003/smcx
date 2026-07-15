# Copyright Contributors to the smcx project.
# SPDX-License-Identifier: Apache-2.0

"""__all__ lock test (ADR-0008 item 5, v0.1 phase-in rule).

During v0.1: smcx.__all__ must be a subset of (smcjax's 32 names
union the ADR-cited additions), and every name v0.1 implements must
be present. The full direction — all 32 smcjax names present —
activates at the release completing the port (v0.2). Never assert a
count.
"""

import smcx

SMCJAX_ALL = {
    "LiuWestPosterior",
    "ParticleFilterPosterior",
    "ParticleFilterResult",
    "ParticleState",
    "__version__",
    "auxiliary_filter",
    "bootstrap_filter",
    "crps",
    "cumulative_log_score",
    "diagnose",
    "ess",
    "liu_west_filter",
    "log_bayes_factor",
    "log_ess",
    "log_ml_increments",
    "log_normalize",
    "multinomial",
    "normalize",
    "param_weighted_mean",
    "param_weighted_quantile",
    "pareto_k_diagnostic",
    "particle_diversity",
    "posterior_predictive_sample",
    "replicated_log_ml",
    "residual",
    "simulate",
    "stratified",
    "systematic",
    "tail_ess",
    "weighted_mean",
    "weighted_quantile",
    "weighted_variance",
}

# Every addition beyond smcjax's export list cites its ratifying ADR.
ADDITIONS = {
    "DegenerateWeightsError": "ADR-0003",
}

V01_IMPLEMENTED = {
    "DegenerateWeightsError",
    "ParticleFilterPosterior",
    "ParticleFilterResult",
    "ParticleState",
    "__version__",
    "bootstrap_filter",
    "ess",
    "log_ess",
    "log_normalize",
    "multinomial",
    "normalize",
    "residual",
    "simulate",
    "stratified",
    "systematic",
}


def test_all_is_subset_of_smcjax_plus_cited_additions():
    extras = set(smcx.__all__) - SMCJAX_ALL
    uncited = extras - set(ADDITIONS)
    assert not uncited, f"exports without a ratifying ADR: {uncited}"


def test_v01_names_are_present_and_sorted():
    assert set(smcx.__all__) >= V01_IMPLEMENTED
    assert list(smcx.__all__) == sorted(smcx.__all__)


def test_exports_resolve():
    for name in smcx.__all__:
        assert getattr(smcx, name) is not None
