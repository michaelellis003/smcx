# Why the square-root pass uses QR, and why there is no SVD form

*Decision record, 2026-08-06. Companion to the square-root entry
points (`sqrt_kalman_filter`, `sqrt_rts_smoother`, `as_covariance`).
Published because the trade-off may be useful to other researchers
choosing a factorization for state-space software.*

## The setting

The covariance-form Kalman recursions can lose positive
semidefiniteness in float32 when transitions are ill-conditioned.
Measured on smcx's regression fixtures (state dimension 4, partial
observation, tiny process noise): the covariance form survives
transition condition number $10^4$ over 200 steps, and beyond that
($10^5$ with $Q = 10^{-10}$) fails loudly — a named eager rejection
at the smoother boundary, not silent wrongness. A square-root form
carries a triangular factor $L$ with $P = L L^\top$ instead of $P$
itself, so semidefiniteness holds by construction.

Two orthogonal factorizations can propagate the factor. The QR array
form (Särkkä and Svensson 2023, chapter 6) stacks scaled factors and
takes one QR per predict and one blocked QR per update. The SVD form
(Zhang and Li 1996) carries each covariance as its singular value
decomposition; R's `dlm` package (Petris) implements its entire
filter and smoother this way — there is no covariance path in `dlm`
at all — and independently arrived at the same interface shape smcx
uses: the filter returns factors, and a utility (`dlmSvd2var` there,
`as_covariance` here) reconstructs covariances on request.

## The decision

smcx ships the QR form, opt-in beside the unchanged covariance-form
default, and does not ship an SVD form. Three measured or structural
reasons:

1. **Cost.** On jitted CPU at filter-relevant sizes (batched 8-by-8),
   SVD costs about 4.2 times a QR — on top of the QR form's own
   1.1-1.2 times premium over the covariance form. Both
   factorizations run on the Metal backend, so the choice is cost,
   not availability.
2. **The QR form already meets every goal the square-root form
   exists for.** It runs both float32 regimes the covariance form
   rejects, tracks the float64 reference at roughly $3 \times
   10^{-7}$ relative on the fixtures, and is positive semidefinite
   by construction. What SVD adds beyond this is rank-revealing
   behavior when a predicted covariance is exactly singular — a
   regime where the smoother's gain is ill-posed in any
   factorization, and which no measured fixture has produced.
3. **The architecture keeps the SVD door open at zero present
   cost.** Square-root results are their own record type
   (`SqrtGaussianFilterPosterior`), deliberately not substitutable
   for the covariance record. An SVD form later would be one more
   sibling with its own factor record, touching nothing shipped.
   Adding it now would be a third parallel surface on speculation.

If a workflow produces a genuine rank-collapse regime the QR form
cannot handle, an SVD form becomes an evidence-gated issue, with
Zhang and Li (1996) and the `dlm` source as the implementation
references.

## References

- Särkkä, S., and Svensson, L. (2023). Bayesian Filtering and
  Smoothing, second edition, chapter 6.
  https://doi.org/10.1017/9781108917407
- Zhang, Y., and Li, X. R. (1996). Fixed-interval smoothing
  algorithm based on singular value decomposition.
- Petris, G. dlm: Bayesian and Likelihood Analysis of Dynamic Linear
  Models. https://cran.r-project.org/package=dlm
