# 0030. Native square-root conditionally linear-Gaussian RBPF

Date: 2026-07-20 | Status: proposed | Supersedes: — | Superseded-by: —

## Context

Rao--Blackwellization can sample a nonlinear or discrete state while
integrating a conditionally linear-Gaussian state exactly. The first smcx
slice must preserve the callable model boundary from ADR-0019, the input
alignment from ADR-0022, and structured sampled states from ADR-0024.
Limited-precision Kalman algebra is the main numerical risk: square-root
filtering is the established remedy [Kaminski, Bryson, and Schmidt
(1971)](https://doi.org/10.1109/TAC.1971.1099816) and Bierman (1977),
while a [float32 study](https://arxiv.org/abs/2208.06452) found a
conventional update diverged where a QR square-root form survived.
ADR-0012 also records a silent f32 Cholesky failure in the prior Metal
stack.

## Options considered

- Delegate conditional filtering to Dynamax per particle — reduces local
  algebra, but makes an optional model library own smcx inference and its
  undocumented SLDS implementation is not a valid oracle.
- Propagate full covariances with a conventional or Joseph update — is
  simpler, but makes the less stable limited-precision path the default.
- Implement a native square-root bootstrap RBPF — keeps inference and its
  numerical contract in smcx while leaving models as plain callbacks.
- Generalize immediately to guided proposals or arbitrary analytic
  subfilters — broadens the interface before a demonstrated requirement.

## Decision

We will implement a native bootstrap RBPF for a sampled state ``r_t`` and a
dense Gaussian state ``x_t``. The first model family is

```text
r_t ~ p(r_t | r_(t-1), u_t)
x_t = F_t x_(t-1) + b_t + epsilon_t,  epsilon_t ~ N(0, Q_t)
y_t = H_t x_t + d_t + eta_t,          eta_t ~ N(0, R_t).
```

``F_t``, ``b_t``, and ``Q_t`` may depend on ``r_(t-1)``, ``r_t``, and
``u_t``; emission terms may depend on ``r_t`` and ``u_t``. Paired
input-free and input-aware callback Protocols in ``types.py`` will expose
plain signatures for: sampled-state initialization and prior transition;
``initial_gaussian(r_0[, u_0]) -> (m_0, P_0)``;
``linear_transition(r_prev, r_t[, u_t]) -> (F_t, b_t, Q_t)``; and
``linear_emission(r_t[, u_t]) -> (H_t, d_t, R_t)``. These are parameter
callbacks, not model or distribution objects. The sampled transition prior
is the proposal, so the first slice has no guided-proposal interface or
proposal correction.

smcx will own propagation, normalized log weighting, ESS, conditional
resampling, genealogy, predictive Gaussian log likelihood, evidence, and
all Kalman prediction/update algebra. Explicit keys reach only sampling and
resampling. A future optional Dynamax adapter may translate public model
parameters into these callbacks; it must never call a Dynamax filter,
smoother, fitting routine, or RBPF.

The conditional carry and posterior will store a mean and lower Cholesky
factor ``L_t`` with ``P_t = L_t L_t.T``. Prediction and measurement update
will use QR-based square-root operations and triangular solves, with no
explicit inverse. QR signs are normalized so stored factors have positive
diagonals. Callback covariances and any reconstructed fallback covariance
are symmetrized as ``(C + C.T) / 2``; asymmetry beyond a dtype-scaled
tolerance is invalid. Factorization tries diagonal jitter
``lambda * max(trace(C) / dim, finfo(dtype).tiny)`` for
``lambda = 0, 10*eps, 100*eps, 1000*eps``. If a square-root stage still
fails its finite/positive-diagonal checks, smcx will reconstruct the
covariance: prediction uses the symmetrized covariance equation and the
measurement update uses Joseph form. The result is symmetrized and factored
with the same bounded schedule. Failure of both paths is surfaced at the
loop-shell evaluation boundary rather than returning a corrupt factor.

The hot loop preserves its input dtype: physical Metal runs in float32 and
does not promote, while CPU float64 is an oracle path. Square-root behavior,
the near-singular gate, and jitter/fallback use must be tested on both.
Changing the default away from square-root propagation requires a
superseding ADR with recorded CPU-f64 and physical-Metal evidence.

``RBPFPosterior`` will contain ``marginal_loglik``, sampled-state
``filtered_particles``, ``filtered_conditional_means``,
``filtered_conditional_scale_trils``, normalized
``filtered_log_weights``, ``ancestors``, ``ess``, and
``log_evidence_increments``. The factors represent filtered covariances
after assimilating ``y_t``. With ``store_history=False``, the sampled-state
PyTree, means, factors, weights, and ancestors retain only the final step
with a leading time axis of length one; ESS and evidence-increment traces
remain length ``T``, matching ADR-0011.

At time zero, ``u_0`` reaches sampled-state initialization, the initial
Gaussian, and the emission terms before ``y_0`` is assimilated. For
``t > 0``, ``u_t`` reaches the sampled transition into ``t``, linear
dynamics into ``t``, and emission at ``t``. Sampled states may be any
nonempty fixed-structure PyTree accepted by ADR-0024. Conditional means and
factors remain dense arrays with fixed dimension, shape, and dtype.

Dynamax 1.0.2's undocumented SLDS ``rbpfilter`` is disqualified as an
implementation source or correctness oracle. In the
[pinned source](https://github.com/probml/dynamax/blob/a216d7feec0d025560a0a194ed5abab538648375/dynamax/slds/inference.py),
the step samples new modes but rebuilds its carry from previous modes, and
its weights omit the transition/proposal correction. Validation will use
independent mathematical oracles and ADR-0023's isolated-validator policy.
No external inference code will be copied or translated.

## Consequences

smcx gains one stable, model-library-independent RBPF contract that covers
switching and nonlinear sampled states. Cholesky histories cost the same
quadratic storage order as covariances and make reconstruction explicit,
while QR updates cost more than a naive covariance update. Model authors
must supply well-shaped dense Gaussian terms, and guided proposals,
non-Gaussian analytic subfilters, variable-dimensional marginalized states,
and Dynamax recipes remain later work.
