# Export to ArviZ

Install `smcx[arviz]`, then call
`smcx.to_arviz(posterior, key=jr.key(7), num_draws=1_000)`.

One posterior is one chain; independent runs make multiple chains. Draws are
seeded equal-weight resamples. Log weights and diagnostics use `sample_stats`;
evidence is an attribute. Supply `unconstrained=` for aligned u-space values.

Filter draws are per-time filtering marginals, not joint trajectories. The
bridge uses ArviZ's public [`from_dict` APIs](https://python.arviz.org/) under
its [Apache-2.0 license](https://github.com/arviz-devs/arviz/blob/main/LICENSE).
