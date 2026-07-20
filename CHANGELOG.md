# CHANGELOG

<!-- version list -->

## v1.5.0 (2026-07-20)

### Features

- **bootstrap**: Add resumable init and step
  ([#37](https://github.com/michaelellis003/smcx/pull/37),
  [`4788478`](https://github.com/michaelellis003/smcx/commit/4788478cf0695370805458459137049577d26814))

### Testing

- **bootstrap**: Freeze checkpoint compatibility
  ([#36](https://github.com/michaelellis003/smcx/pull/36),
  [`37ca409`](https://github.com/michaelellis003/smcx/commit/37ca409a171970bae44c174b1e4d5d37cc748de5))


## v1.4.0 (2026-07-20)

### Chores

- **deps**: Add ArviZ reporting extra ([#34](https://github.com/michaelellis003/smcx/pull/34),
  [`0dcfc12`](https://github.com/michaelellis003/smcx/commit/0dcfc125d67e2b5006798a87bde49e33997cc284))

### Documentation

- Repair planning state ([#20](https://github.com/michaelellis003/smcx/pull/20),
  [`d11022d`](https://github.com/michaelellis003/smcx/commit/d11022d6a219ebaa025a4c4e159797cfdb97b64c))

- **adr**: Define ArviZ bridge contract ([#28](https://github.com/michaelellis003/smcx/pull/28),
  [`291b7c6`](https://github.com/michaelellis003/smcx/commit/291b7c6052e9e049c24cd707597138058e4cdd7b))

- **adr**: Define native RBPF contract ([#33](https://github.com/michaelellis003/smcx/pull/33),
  [`90b19ce`](https://github.com/michaelellis003/smcx/commit/90b19cea875bc3f6b2473faab42c50300f07584b))

- **adr**: Define streaming filter checkpoints
  ([#23](https://github.com/michaelellis003/smcx/pull/23),
  [`4e906d0`](https://github.com/michaelellis003/smcx/commit/4e906d0c1263497d7b37c9bb22d41195afc9fbf6))

- **adr**: Specify static posterior updating
  ([#32](https://github.com/michaelellis003/smcx/pull/32),
  [`fb72808`](https://github.com/michaelellis003/smcx/commit/fb72808495c6fc4a57366a370ce7f3b9f406d5b6))

- **guides**: Add custom model authoring guide
  ([#24](https://github.com/michaelellis003/smcx/pull/24),
  [`c7e19dc`](https://github.com/michaelellis003/smcx/commit/c7e19dceb405b268f662a7f4d03ca4c32c4ee9b9))

### Features

- **to-arviz**: Add ArviZ reporting bridge ([#35](https://github.com/michaelellis003/smcx/pull/35),
  [`f6755b7`](https://github.com/michaelellis003/smcx/commit/f6755b728815cf66dd907c3d96bd6452ee50f8d7))


## v1.3.1 (2026-07-20)

### Performance Improvements

- Reduce filter memory after all-algorithm profiling
  ([#19](https://github.com/michaelellis003/smcx/pull/19),
  [`e8875b7`](https://github.com/michaelellis003/smcx/commit/e8875b7b03c4e77fac1c8c4305ae234067e683f1))


## v1.3.0 (2026-07-19)

### Features

- **filters**: Support structured latent state PyTrees
  ([`350f511`](https://github.com/michaelellis003/smcx/commit/350f5114850f6ee00fbc24ec421cbb83f22800d5))


## v1.2.1 (2026-07-19)

### Bug Fixes

- Validate SMC algorithms against independent implementations
  ([#17](https://github.com/michaelellis003/smcx/pull/17),
  [`da20e1a`](https://github.com/michaelellis003/smcx/commit/da20e1a222e0619d5b561819f8d033ae0668f068))


## v1.2.0 (2026-07-19)

### Features

- **inputs**: Add exogenous inputs to model callbacks
  ([#16](https://github.com/michaelellis003/smcx/pull/16),
  [`52f3b62`](https://github.com/michaelellis003/smcx/commit/52f3b62d1ab31d8e1958de27188e9cdb0f05a826))


## v1.1.0 (2026-07-18)

### Bug Fixes

- Restore the Zhang-Stephens candidate grid in the Pareto-k fit
  ([#14](https://github.com/michaelellis003/smcx/pull/14),
  [`84dbda7`](https://github.com/michaelellis003/smcx/commit/84dbda7e03b6165387990b5a1ffe2e83a33e99cf))

### Documentation

- Draw the diagnostics boundary (ADR-0020) ([#13](https://github.com/michaelellis003/smcx/pull/13),
  [`4bbb03f`](https://github.com/michaelellis003/smcx/commit/4bbb03f9d57eb5f55b3f7ec75bca139642739df9))

### Features

- Genealogy diagnostics — trajectories and log-ML variance
  ([#15](https://github.com/michaelellis003/smcx/pull/15),
  [`85bc967`](https://github.com/michaelellis003/smcx/commit/85bc967ca3c79621cae28de5ac6963f77191a612))


## v1.0.2 (2026-07-18)

### Bug Fixes

- State the Pareto-k threshold as a reliability boundary
  ([#12](https://github.com/michaelellis003/smcx/pull/12),
  [`246acce`](https://github.com/michaelellis003/smcx/commit/246acce5ded1e718ff12a220272f92a61a452968))

### Continuous Integration

- Exclude example notebooks from ty
  ([`812bada`](https://github.com/michaelellis003/smcx/commit/812bada2d8313443af5549ac4acf412479a71b17))

### Documentation

- Add the thesis regime-switching HMM example notebook
  ([`de795d0`](https://github.com/michaelellis003/smcx/commit/de795d05bf7ed114efdaa08211fc95bcd80da8f7))

- Bust the cached PyPI badge
  ([`553e1f8`](https://github.com/michaelellis003/smcx/commit/553e1f8c0843820d22345f92376d9bc84c0848d1))

- Document the trunk-based branching workflow
  ([#10](https://github.com/michaelellis003/smcx/pull/10),
  [`7ba6165`](https://github.com/michaelellis003/smcx/commit/7ba6165e6b9aabde5b9301292be8fb2de7bf7639))

- Rewrite the roadmap for the released library
  ([#11](https://github.com/michaelellis003/smcx/pull/11),
  [`ebfb884`](https://github.com/michaelellis003/smcx/commit/ebfb884e165da825ad457a9d0a2f12ade0b8097f))


## v1.0.1 (2026-07-17)

### Bug Fixes

- Stop semantic-release rewriting the version fallback
  ([`9995c1d`](https://github.com/michaelellis003/smcx/commit/9995c1d45ae30ac4e5d0c04995b5599eae7bd6dc))

- **build**: Correct the PyPI metadata for the JAX library
  ([`610761d`](https://github.com/michaelellis003/smcx/commit/610761d93b35fcfbe42af377232658e3b9fc3dd1))

- **build**: Make docs targets call mkdocs
  ([`85b48e0`](https://github.com/michaelellis003/smcx/commit/85b48e0703ca05536387b7d414a650f2eada66b3))

### Continuous Integration

- Restore the conventional-title PR check
  ([`51c5fda`](https://github.com/michaelellis003/smcx/commit/51c5fdaa3a1387f44ce041323fbcb1e6df8c5f6a))

### Documentation

- Codify the model-free engine boundary (ADR-0019)
  ([`095c786`](https://github.com/michaelellis003/smcx/commit/095c786a1f80e9ae3f36dcc3875d1d787a885fb1))

- Correct CITATION.cff references for the JAX library
  ([`42a478b`](https://github.com/michaelellis003/smcx/commit/42a478b7df4269557c40d7bee70e381cbbf3dff2))

- Drop smcjax mentions from user-facing pages
  ([`23ed45f`](https://github.com/michaelellis003/smcx/commit/23ed45f337a64ae2bd2c139dd6d763667eaa3735))

- Fix the README example and restore contributor docs
  ([`b9674eb`](https://github.com/michaelellis003/smcx/commit/b9674eb005b639796b174a7b8be6ec840182e929))

- Keep only the docs-site sources in the repo
  ([`2eefe11`](https://github.com/michaelellis003/smcx/commit/2eefe11f695c5160f927ff33cdf7ed0301ab41f4))

- Trim CITATION.cff to the citation metadata
  ([`aef9df2`](https://github.com/michaelellis003/smcx/commit/aef9df2f3f46b27025a3ed656ec0be6b82d0c632))

### Testing

- Make increment-sum tolerances float32-aware
  ([`8f1f7d2`](https://github.com/michaelellis003/smcx/commit/8f1f7d2531fa41f19d0f4bf40d66533b4db37149))

- Make increment-sum tolerances float32-aware
  ([`d9e28ea`](https://github.com/michaelellis003/smcx/commit/d9e28ea6a3c782e31323618fd03aa0390f4f5376))


## v1.0.0 (2026-07-17)

- Initial Release
