# CHANGELOG

<!-- version list -->

## v1.13.20 (2026-07-25)

### Bug Fixes

- **resampling**: Validate custom ancestor outputs
  ([#205](https://github.com/michaelellis003/smcx/pull/205),
  [`76b6101`](https://github.com/michaelellis003/smcx/commit/76b610113cd357bac81d24a28a1d284455c36aac))


## v1.13.19 (2026-07-25)

### Bug Fixes

- **mps**: Select filter containment at lowering
  ([#204](https://github.com/michaelellis003/smcx/pull/204),
  [`8e1e7e1`](https://github.com/michaelellis003/smcx/commit/8e1e7e152641b01988ea0f44d1ee19fa3a61e0a0))

### Testing

- **mps**: Keep hosted Metal smoke best-effort
  ([#206](https://github.com/michaelellis003/smcx/pull/206),
  [`a09c86d`](https://github.com/michaelellis003/smcx/commit/a09c86d1434713207bba869f8c39464032110ecb))


## v1.13.18 (2026-07-25)

### Bug Fixes

- **bootstrap**: Enforce checkpoint invariants
  ([#203](https://github.com/michaelellis003/smcx/pull/203),
  [`3d5cbd5`](https://github.com/michaelellis003/smcx/commit/3d5cbd51816e5724c6a0639fd115192fb9b21fcf))


## v1.13.17 (2026-07-25)

### Bug Fixes

- **smc2**: Preserve shifted likelihood components
  ([#202](https://github.com/michaelellis003/smcx/pull/202),
  [`6235bd6`](https://github.com/michaelellis003/smcx/commit/6235bd6b635fd89584c0317d346fcd35fed79001))


## v1.13.16 (2026-07-25)

### Bug Fixes

- **weights**: Preserve large-offset invariance
  ([#201](https://github.com/michaelellis003/smcx/pull/201),
  [`a04197c`](https://github.com/michaelellis003/smcx/commit/a04197c949373447107e6ca9e28e9091b100ca74))


## v1.13.15 (2026-07-25)

### Bug Fixes

- **weights**: Reject non-normalizable stages
  ([#200](https://github.com/michaelellis003/smcx/pull/200),
  [`eaf391f`](https://github.com/michaelellis003/smcx/commit/eaf391f86580d497ff25d93daa1bc08d178aa980))


## v1.13.14 (2026-07-25)

### Bug Fixes

- **numerics**: Require float32 weight precision
  ([#199](https://github.com/michaelellis003/smcx/pull/199),
  [`60f0ce0`](https://github.com/michaelellis003/smcx/commit/60f0ce0ef20fec18d5a62d9ef94da80a95baef8e))


## v1.13.13 (2026-07-25)

### Bug Fixes

- **resampling**: Restore monotone float32 CDFs
  ([#198](https://github.com/michaelellis003/smcx/pull/198),
  [`b3d1ac8`](https://github.com/michaelellis003/smcx/commit/b3d1ac82699c950b70ef4374677da80a371cafe2))


## v1.13.12 (2026-07-25)

### Bug Fixes

- **resampling**: Reject non-normalizable weights
  ([#197](https://github.com/michaelellis003/smcx/pull/197),
  [`1700cb7`](https://github.com/michaelellis003/smcx/commit/1700cb721e2b30d7647f0078028e6319ba96296a))


## v1.13.11 (2026-07-25)

### Bug Fixes

- **weights**: Validate public log-weight inputs
  ([#196](https://github.com/michaelellis003/smcx/pull/196),
  [`4f117f9`](https://github.com/michaelellis003/smcx/commit/4f117f9513a3e8418cb33baa566ede108611532e))


## v1.13.10 (2026-07-25)

### Bug Fixes

- **diagnostics**: Compensate cumulative predictive log scores
  ([#195](https://github.com/michaelellis003/smcx/pull/195),
  [`4d7c232`](https://github.com/michaelellis003/smcx/commit/4d7c232c9ad7e3ee10401008167073568f4e0474))


## v1.13.9 (2026-07-25)

### Bug Fixes

- **filters**: Compensate evidence accumulation
  ([#194](https://github.com/michaelellis003/smcx/pull/194),
  [`da5b82d`](https://github.com/michaelellis003/smcx/commit/da5b82d76b5911c51ee76d9a61b55c433eba7063))

### Documentation

- **diagnostics**: Define genealogy variance scope
  ([#192](https://github.com/michaelellis003/smcx/pull/192),
  [`7890794`](https://github.com/michaelellis003/smcx/commit/7890794d13dda5d7522993116349326cbae57c0c))


## v1.13.8 (2026-07-24)

### Bug Fixes

- **bootstrap**: Validate incremental particle count
  ([#193](https://github.com/michaelellis003/smcx/pull/193),
  [`dd5f5ad`](https://github.com/michaelellis003/smcx/commit/dd5f5ade753678d1acbc65dd136826dc81f48edf))


## v1.13.7 (2026-07-24)

### Bug Fixes

- **runner**: Contain MPS scan history corruption
  ([#191](https://github.com/michaelellis003/smcx/pull/191),
  [`fe8273c`](https://github.com/michaelellis003/smcx/commit/fe8273cac4295a78e3044c2a7e2bdf7fa6efabf0))

### Build System

- **packaging**: Include license in release artifacts
  ([#188](https://github.com/michaelellis003/smcx/pull/188),
  [`ddd4d6e`](https://github.com/michaelellis003/smcx/commit/ddd4d6ef8b19568e6deaef2f74fa6438c33d0d16))

### Chores

- **licensing**: Restrict header updates to owned files
  ([#189](https://github.com/michaelellis003/smcx/pull/189),
  [`a752928`](https://github.com/michaelellis003/smcx/commit/a752928a4be876db7fabf65741a57c194d52d439))

- **profiling**: Harden campaign lock ([#190](https://github.com/michaelellis003/smcx/pull/190),
  [`e8911c8`](https://github.com/michaelellis003/smcx/commit/e8911c8ab8e11ebffe33139fdf65cd5787277302))

### Continuous Integration

- **dependabot**: Harden identity gate ([#186](https://github.com/michaelellis003/smcx/pull/186),
  [`85d0b98`](https://github.com/michaelellis003/smcx/commit/85d0b989d7e5658ffe5399e26fafbc3c1d0777f6))

- **docs**: Deploy released locked revision
  ([#187](https://github.com/michaelellis003/smcx/pull/187),
  [`365ee94`](https://github.com/michaelellis003/smcx/commit/365ee9410910ec0d1c8af7d343fa31bcb996c858))

- **release**: Enforce Metal attestation reviewer
  ([#184](https://github.com/michaelellis003/smcx/pull/184),
  [`4240c6a`](https://github.com/michaelellis003/smcx/commit/4240c6a6238bfe49f95a6aa831f5d910669b9c7d))

### Testing

- **platform**: Enforce backend selector contract
  ([#185](https://github.com/michaelellis003/smcx/pull/185),
  [`fb9b428`](https://github.com/michaelellis003/smcx/commit/fb9b428f964838904f8b30f4bc34b68e29790192))


## v1.13.6 (2026-07-24)

### Bug Fixes

- **diagnostics**: Track cumulative particle ancestry
  ([#183](https://github.com/michaelellis003/smcx/pull/183),
  [`87cb43d`](https://github.com/michaelellis003/smcx/commit/87cb43da72462c6572066ca7db1d888e7fb04b64))

### Chores

- Remove dead profiling and reference artifacts
  ([#181](https://github.com/michaelellis003/smcx/pull/181),
  [`f72177d`](https://github.com/michaelellis003/smcx/commit/f72177db5dd7e9a233749add38f60d1c16eb0e22))

### Documentation

- Align posterior and diagnostic contracts
  ([#173](https://github.com/michaelellis003/smcx/pull/173),
  [`8d08f8e`](https://github.com/michaelellis003/smcx/commit/8d08f8ed34c48967f6cec18616d8607648b765fe))

- Correct API rendering and public claims ([#172](https://github.com/michaelellis003/smcx/pull/172),
  [`6390740`](https://github.com/michaelellis003/smcx/commit/63907406e3e69da0961fd01ce125f2ed125d1a8e))

- Correct platform and public contracts ([#182](https://github.com/michaelellis003/smcx/pull/182),
  [`9937ab6`](https://github.com/michaelellis003/smcx/commit/9937ab61e340c60e8e1fbc359b957f648f65d68a))

### Refactoring

- **liu-west**: Extract pure scan step ([#148](https://github.com/michaelellis003/smcx/pull/148),
  [`37b15ee`](https://github.com/michaelellis003/smcx/commit/37b15ee2dba944f741c568c7db0d46d1ae293ff6))


## v1.13.5 (2026-07-24)

### Bug Fixes

- **filters**: Validate callback outputs throughout
  ([#143](https://github.com/michaelellis003/smcx/pull/143),
  [`5bdcafb`](https://github.com/michaelellis003/smcx/commit/5bdcafb5aa47fda169544eb5e93354dca0cec7ab))

- **simulation**: Validate emission callback outputs
  ([#136](https://github.com/michaelellis003/smcx/pull/136),
  [`68acd42`](https://github.com/michaelellis003/smcx/commit/68acd428a7d006eda3fbfe235116db328eac3b9b))

- **smc2**: Validate callback output contracts
  ([#141](https://github.com/michaelellis003/smcx/pull/141),
  [`ffce30d`](https://github.com/michaelellis003/smcx/commit/ffce30d925b969e7e0bd6d7b6caa6efc6b8fde8a))

- **tempering**: Validate callback outputs at every stage
  ([#142](https://github.com/michaelellis003/smcx/pull/142),
  [`8d10479`](https://github.com/michaelellis003/smcx/commit/8d104791697ed33bc47d6a58392ba7548427abb2))

### Refactoring

- **auxiliary**: Extract pure scan step ([#145](https://github.com/michaelellis003/smcx/pull/145),
  [`5c112e3`](https://github.com/michaelellis003/smcx/commit/5c112e310e6d255e36436ec56af8838c5d2e8015))

- **guided**: Extract pure scan step ([#146](https://github.com/michaelellis003/smcx/pull/146),
  [`3be2275`](https://github.com/michaelellis003/smcx/commit/3be2275396f819c7e54692be0a6c469725b88c07))

- **numerics**: Share compensated summation
  ([#147](https://github.com/michaelellis003/smcx/pull/147),
  [`b65f117`](https://github.com/michaelellis003/smcx/commit/b65f117f5c91b62e961257dd2192e564c53f7f25))

- **types**: Name replicated filter callback contract
  ([#144](https://github.com/michaelellis003/smcx/pull/144),
  [`7e70af5`](https://github.com/michaelellis003/smcx/commit/7e70af5e3674f6efc90895f9652e390ac6d9ca0f))


## v1.13.4 (2026-07-24)

### Bug Fixes

- **diagnostics**: Align parameter and lag contracts
  ([#140](https://github.com/michaelellis003/smcx/pull/140),
  [`420487b`](https://github.com/michaelellis003/smcx/commit/420487b067f9df187b27217b69a025bc47c17e4a))

- **resampling**: Validate public structural inputs
  ([#139](https://github.com/michaelellis003/smcx/pull/139),
  [`bca028c`](https://github.com/michaelellis003/smcx/commit/bca028c62e129052777e9a1103dddd4798e1279f))


## v1.13.3 (2026-07-24)

### Bug Fixes

- **diagnostics**: Handle singleton Pareto estimates
  ([#135](https://github.com/michaelellis003/smcx/pull/135),
  [`abc986f`](https://github.com/michaelellis003/smcx/commit/abc986f7862d5a66d481da928ae67c7e5481c031))

- **reporting**: Preserve adaptive tempering stages
  ([#137](https://github.com/michaelellis003/smcx/pull/137),
  [`2bbe5e4`](https://github.com/michaelellis003/smcx/commit/2bbe5e42e6282e9e3322c6a598f00594d46645ae))

- **resampling**: Preserve dtype-specific tail mass
  ([#134](https://github.com/michaelellis003/smcx/pull/134),
  [`b8a847e`](https://github.com/michaelellis003/smcx/commit/b8a847ede4d929f0c3c205d8996235c9e22947b7))

- **tempering**: Use a representable acceptance floor
  ([#138](https://github.com/michaelellis003/smcx/pull/138),
  [`ba26a74`](https://github.com/michaelellis003/smcx/commit/ba26a747a53be36782a889f5d3bc51f050130b3e))

### Chores

- **benchmarks**: Archive native MLX comparison harness
  ([#127](https://github.com/michaelellis003/smcx/pull/127),
  [`45cf05b`](https://github.com/michaelellis003/smcx/commit/45cf05b2977a50baa1fccf2bb07e1a5619fea9f6))

- **benchmarks**: Retire MLX harnesses ([#125](https://github.com/michaelellis003/smcx/pull/125),
  [`54c50c6`](https://github.com/michaelellis003/smcx/commit/54c50c651d881d6a831023cf9612a65857eea788))

- **deps**: Bump actions/checkout from 7.0.0 to 7.0.1
  ([#123](https://github.com/michaelellis003/smcx/pull/123),
  [`19127b0`](https://github.com/michaelellis003/smcx/commit/19127b00284656f8f4a76263557ff269bd779569))

- **deps**: Bump pypa/gh-action-pypi-publish from 1.14.0 to 1.14.1
  ([#124](https://github.com/michaelellis003/smcx/pull/124),
  [`7910a5f`](https://github.com/michaelellis003/smcx/commit/7910a5ffd8f22f54b9ae265bf3d4f7e3e84e053f))

- **docs**: Remove retired notebook environment
  ([#126](https://github.com/michaelellis003/smcx/pull/126),
  [`da70ddf`](https://github.com/michaelellis003/smcx/commit/da70ddff453ec6aebddb6820fa5a661817e383c9))


## v1.13.2 (2026-07-24)

### Bug Fixes

- **validation**: Reject malformed inference inputs
  ([#114](https://github.com/michaelellis003/smcx/pull/114),
  [`e570af8`](https://github.com/michaelellis003/smcx/commit/e570af8999f31bd15938696cd100c1b63ee80a6b))

### Chores

- **benchmarks**: Retire exploratory harnesses
  ([#119](https://github.com/michaelellis003/smcx/pull/119),
  [`74f1b94`](https://github.com/michaelellis003/smcx/commit/74f1b9441afba441f00a06864095868e09447a8b))

- **docs**: Retire thesis notebook ([#120](https://github.com/michaelellis003/smcx/pull/120),
  [`15d94ff`](https://github.com/michaelellis003/smcx/commit/15d94ff32d61059ecb2108cec0275b7ca9d514e1))

### Refactoring

- **bootstrap**: Share particle update core
  ([#122](https://github.com/michaelellis003/smcx/pull/122),
  [`7ba9055`](https://github.com/michaelellis003/smcx/commit/7ba905584750f833f5597dbcb1434739dc048e62))

### Testing

- **validation**: Cover malformed callback contracts
  ([#121](https://github.com/michaelellis003/smcx/pull/121),
  [`1304a61`](https://github.com/michaelellis003/smcx/commit/1304a6193b0159be499019576d694821fc011ef5))


## v1.13.1 (2026-07-24)

### Bug Fixes

- **diagnostics**: Reject incomplete time histories
  ([#113](https://github.com/michaelellis003/smcx/pull/113),
  [`5d97c8c`](https://github.com/michaelellis003/smcx/commit/5d97c8c2478cd26d889d37809568a8bd2c1092f5))

### Chores

- Remove stale tooling and metadata ([#116](https://github.com/michaelellis003/smcx/pull/116),
  [`3fb1f70`](https://github.com/michaelellis003/smcx/commit/3fb1f70f1d279c0d328899d19a1b91ab05a518e1))

- **repo**: Make cleanup and deployments revision-safe
  ([#115](https://github.com/michaelellis003/smcx/pull/115),
  [`4e3d8a2`](https://github.com/michaelellis003/smcx/commit/4e3d8a2fdea668a0be0723ed6bc5e4921659b2a0))

### Continuous Integration

- **coverage**: Route uploads to smcx with OIDC
  ([#118](https://github.com/michaelellis003/smcx/pull/118),
  [`958e003`](https://github.com/michaelellis003/smcx/commit/958e00327cf7c9e3c98ce12935ea24618bebbc49))

### Documentation

- **particles**: Compose auxiliary-guided runner kernels
  ([#110](https://github.com/michaelellis003/smcx/pull/110),
  [`ac9572d`](https://github.com/michaelellis003/smcx/commit/ac9572da46dad4c3829ccde99d1bc7fc05ead0dd))

### Testing

- **bootstrap**: Assert observable device behavior
  ([#117](https://github.com/michaelellis003/smcx/pull/117),
  [`7c783d6`](https://github.com/michaelellis003/smcx/commit/7c783d6dbec013079d1fa0d21105a23f3d68af6e))


## v1.13.0 (2026-07-24)

### Features

- **tempering**: Accept caller-owned mutation kernels
  ([#109](https://github.com/michaelellis003/smcx/pull/109),
  [`f449935`](https://github.com/michaelellis003/smcx/commit/f4499357c3b1053554f82d0160d9e7b7c4000af7))


## v1.12.0 (2026-07-24)

### Bug Fixes

- **reporting**: Preserve particle diagnostic dimensions
  ([#107](https://github.com/michaelellis003/smcx/pull/107),
  [`e6b0e47`](https://github.com/michaelellis003/smcx/commit/e6b0e47eb5350309d36a397ada8486109d925598))

### Features

- **resampling**: Accept caller-owned criteria
  ([#108](https://github.com/michaelellis003/smcx/pull/108),
  [`6d7c340`](https://github.com/michaelellis003/smcx/commit/6d7c340e471a9a2a9edd14bd427668d4c44fba97))

### Testing

- **kalman**: Validate unscented filter independently
  ([#106](https://github.com/michaelellis003/smcx/pull/106),
  [`0225dfb`](https://github.com/michaelellis003/smcx/commit/0225dfb2d4713f4952e55a31e63ef6a4fc827775))


## v1.11.0 (2026-07-24)

### Features

- **kalman**: Expose scaled unscented filter
  ([#105](https://github.com/michaelellis003/smcx/pull/105),
  [`b3a5aec`](https://github.com/michaelellis003/smcx/commit/b3a5aecb361e8f92fbe20ac90174cf3ff1054a00))

### Refactoring

- **kalman**: Share nonlinear filter inputs
  ([#104](https://github.com/michaelellis003/smcx/pull/104),
  [`6a785c5`](https://github.com/michaelellis003/smcx/commit/6a785c5ce941733e8ff553fff55f72c03f264a11))


## v1.10.0 (2026-07-24)

### Features

- **kalman**: Add scaled unscented numerical core
  ([#103](https://github.com/michaelellis003/smcx/pull/103),
  [`cd24b10`](https://github.com/michaelellis003/smcx/commit/cd24b10ce838fb5659ca19fd6a6497a71da12408))


## v1.9.0 (2026-07-23)

### Features

- **kalman**: Add explicit-Jacobian extended filter
  ([#102](https://github.com/michaelellis003/smcx/pull/102),
  [`92a7335`](https://github.com/michaelellis003/smcx/commit/92a7335f7e9aec22c60fc95431057f3cf157ba45))


## v1.8.0 (2026-07-23)

### Features

- **runner**: Add caller-owned particle execution
  ([#101](https://github.com/michaelellis003/smcx/pull/101),
  [`9915ae6`](https://github.com/michaelellis003/smcx/commit/9915ae616577d604ff5c3316e3f713f73044bb47))


## v1.7.0 (2026-07-23)

### Chores

- Remove local paths and internal review language
  ([#50](https://github.com/michaelellis003/smcx/pull/50),
  [`e8bdfd9`](https://github.com/michaelellis003/smcx/commit/e8bdfd9efe45b2e63d996fded4cb60b81d8d7e4a))

- **deps**: Bump jupyterlab from 4.6.1 to 4.6.2
  ([#91](https://github.com/michaelellis003/smcx/pull/91),
  [`f442568`](https://github.com/michaelellis003/smcx/commit/f442568f5a746f5d86c1a512e1f7a799c8d989e9))

- **tempering**: Retire one-off accuracy campaign
  ([#85](https://github.com/michaelellis003/smcx/pull/85),
  [`87c6b2c`](https://github.com/michaelellis003/smcx/commit/87c6b2cb2a77528bcbb1c28b9e182065096934ee))

### Documentation

- Complete scoring-rule attribution ([#51](https://github.com/michaelellis003/smcx/pull/51),
  [`72d552e`](https://github.com/michaelellis003/smcx/commit/72d552ef5839400d8a86c3188ff53388489c11e1))

- Execute filtering tutorial ([#54](https://github.com/michaelellis003/smcx/pull/54),
  [`ff0ac89`](https://github.com/michaelellis003/smcx/commit/ff0ac89e37efb77b97a55b51f6959fc696fca63b))

- Remove internal decision index ([#58](https://github.com/michaelellis003/smcx/pull/58),
  [`90b3926`](https://github.com/michaelellis003/smcx/commit/90b3926bf36280b0c72c391692908ca4e30150c0))

- Remove internal integration records ([#59](https://github.com/michaelellis003/smcx/pull/59),
  [`6fa59d7`](https://github.com/michaelellis003/smcx/commit/6fa59d76cac196658260249ec2799f9c1e156801))

- Remove internal planning records ([#57](https://github.com/michaelellis003/smcx/pull/57),
  [`18dbbb5`](https://github.com/michaelellis003/smcx/commit/18dbbb5d55a4d842c1474e35cf794d7ad64efbdf))

- Remove licensing inventory ([#56](https://github.com/michaelellis003/smcx/pull/56),
  [`627fc98`](https://github.com/michaelellis003/smcx/commit/627fc98a11d774985d266f0896ce5bdd22f0da2f))

- Remove remaining internal decision records
  ([#60](https://github.com/michaelellis003/smcx/pull/60),
  [`caaecbf`](https://github.com/michaelellis003/smcx/commit/caaecbfe24eb359a8f08e3463884d9c03cfa3e8b))

- Remove stale public artifacts ([#47](https://github.com/michaelellis003/smcx/pull/47),
  [`921ef42`](https://github.com/michaelellis003/smcx/commit/921ef42abdef711ef0d05f90d99a79574da659e3))

- Simplify contributor and documentation entry points
  ([#49](https://github.com/michaelellis003/smcx/pull/49),
  [`56129e4`](https://github.com/michaelellis003/smcx/commit/56129e41cd416b4467eca3f72f33c2e6eb935e7e))

- Simplify public documentation and attribution
  ([#48](https://github.com/michaelellis003/smcx/pull/48),
  [`e0a75d0`](https://github.com/michaelellis003/smcx/commit/e0a75d0650a10d32f182b5c96786927f340286ed))

- Tighten the public documentation ([#52](https://github.com/michaelellis003/smcx/pull/52),
  [`6368872`](https://github.com/michaelellis003/smcx/commit/63688729597ebb3dc69ded0f5dfa2ea0f6b8d54a))

- **benchmarks**: Report tempering accuracy ([#84](https://github.com/michaelellis003/smcx/pull/84),
  [`2af605a`](https://github.com/michaelellis003/smcx/commit/2af605ad4ea454fc4b006517eed7ac68becbf01e))

- **contributing**: Simplify public templates
  ([#46](https://github.com/michaelellis003/smcx/pull/46),
  [`465a6ac`](https://github.com/michaelellis003/smcx/commit/465a6ac2a601b9398fc12b2feea59c9da54eb037))

### Features

- Add exact Kalman filtering and RTS smoothing
  ([#100](https://github.com/michaelellis003/smcx/pull/100),
  [`05e1af9`](https://github.com/michaelellis003/smcx/commit/05e1af940bd5575e4b39765d4aedd149a7450202))

### Testing

- Focus regressions on package behavior ([#88](https://github.com/michaelellis003/smcx/pull/88),
  [`63c877c`](https://github.com/michaelellis003/smcx/commit/63c877c7f1b417cc97e6dc5a4d2067b7f42c233e))

- Remove non-product campaign checks ([#86](https://github.com/michaelellis003/smcx/pull/86),
  [`0709227`](https://github.com/michaelellis003/smcx/commit/0709227c5815a6f54635ea931d1dea1e03359b40))

- Remove profiling implementation locks ([#89](https://github.com/michaelellis003/smcx/pull/89),
  [`2b4304e`](https://github.com/michaelellis003/smcx/commit/2b4304e0022e7420726527a6e894f7c8ed34eb14))

- Scan only tracked public documentation ([#61](https://github.com/michaelellis003/smcx/pull/61),
  [`2352190`](https://github.com/michaelellis003/smcx/commit/2352190087f2ae17ffa1c31b30b4da6293fd7859))

- Streamline diagnostic coverage ([#90](https://github.com/michaelellis003/smcx/pull/90),
  [`41569e6`](https://github.com/michaelellis003/smcx/commit/41569e6a59f70f1bacd7985e96715529c4861526))

- **tempering**: Add current-RWM smoke worker
  ([#62](https://github.com/michaelellis003/smcx/pull/62),
  [`ecdeeaa`](https://github.com/michaelellis003/smcx/commit/ecdeeaa98ce4c1a399afaeeb970222206cb629b6))

- **tempering**: Add replicated accuracy worker
  ([#66](https://github.com/michaelellis003/smcx/pull/66),
  [`00ee716`](https://github.com/michaelellis003/smcx/commit/00ee716fdeef753ef358591183a6bb8683f1888c))

- **tempering**: Add standard timing worker ([#64](https://github.com/michaelellis003/smcx/pull/64),
  [`183ffee`](https://github.com/michaelellis003/smcx/commit/183ffee2d47b521f31e103cae20e0fc249f629c0))

- **tempering**: Aggregate accuracy evidence
  ([#77](https://github.com/michaelellis003/smcx/pull/77),
  [`acc8ca2`](https://github.com/michaelellis003/smcx/commit/acc8ca2a982a81f8be4fd9c2bd425dae5772d81f))

- **tempering**: Classify timing report evidence
  ([#75](https://github.com/michaelellis003/smcx/pull/75),
  [`c486df0`](https://github.com/michaelellis003/smcx/commit/c486df0571d0b3343c448662520d51ba5b19ad72))

- **tempering**: Enforce supervisor evidence boundaries
  ([#73](https://github.com/michaelellis003/smcx/pull/73),
  [`1489d97`](https://github.com/michaelellis003/smcx/commit/1489d97b35f6d42da878c9b001b79e55f58f5660))

- **tempering**: Freeze accuracy campaign plan
  ([#42](https://github.com/michaelellis003/smcx/pull/42),
  [`07ffd99`](https://github.com/michaelellis003/smcx/commit/07ffd99f056db43ce91352c1ccdf92d76b7606ad))

- **tempering**: Freeze accuracy efficiency losses
  ([#44](https://github.com/michaelellis003/smcx/pull/44),
  [`cedda39`](https://github.com/michaelellis003/smcx/commit/cedda392d1959504a48391f67c064d6630de5eda))

- **tempering**: Freeze campaign artifacts ([#68](https://github.com/michaelellis003/smcx/pull/68),
  [`c67297c`](https://github.com/michaelellis003/smcx/commit/c67297caedb05c4ffe9f6c17222ef1c2bcec015c))

- **tempering**: Freeze replicated accuracy gates
  ([#43](https://github.com/michaelellis003/smcx/pull/43),
  [`76602ca`](https://github.com/michaelellis003/smcx/commit/76602ca01d14684472a032ecb3dd4ed813109c92))

- **tempering**: Harden campaign artifacts ([#69](https://github.com/michaelellis003/smcx/pull/69),
  [`b2da942`](https://github.com/michaelellis003/smcx/commit/b2da9423d0b1af45b2004732097a6baec4e98260))

- **tempering**: Integrate campaign evidence
  ([#82](https://github.com/michaelellis003/smcx/pull/82),
  [`d79fa94`](https://github.com/michaelellis003/smcx/commit/d79fa94bca41e8a068ca91598028312ed6e4c65a))

- **tempering**: Isolate campaign workers ([#70](https://github.com/michaelellis003/smcx/pull/70),
  [`b261a98`](https://github.com/michaelellis003/smcx/commit/b261a98af455b9b8124f7b39aa0a730dab3d7b31))

- **tempering**: Load campaign report evidence
  ([#74](https://github.com/michaelellis003/smcx/pull/74),
  [`e4d09e6`](https://github.com/michaelellis003/smcx/commit/e4d09e6fe9b80597e3fbc9ec85d0b0a7e09c998b))

- **tempering**: Publish campaign report ([#83](https://github.com/michaelellis003/smcx/pull/83),
  [`4d55dd5`](https://github.com/michaelellis003/smcx/commit/4d55dd5f9d1fd388585a57e3236f0e4c83b550f8))

- **tempering**: Register accuracy target contracts
  ([#41](https://github.com/michaelellis003/smcx/pull/41),
  [`a7cfeaf`](https://github.com/michaelellis003/smcx/commit/a7cfeafd7734a67fcbc5ac46937a0114dac47422))

- **tempering**: Render campaign evidence ([#79](https://github.com/michaelellis003/smcx/pull/79),
  [`b01d28b`](https://github.com/michaelellis003/smcx/commit/b01d28b7422e3a14578003abd81f45a65615b16b))

- **tempering**: Render campaign figures ([#81](https://github.com/michaelellis003/smcx/pull/81),
  [`fe6f333`](https://github.com/michaelellis003/smcx/commit/fe6f3337d5b330bfdeddbeffa8ae6db34ffeef88))

- **tempering**: Render campaign Markdown ([#80](https://github.com/michaelellis003/smcx/pull/80),
  [`f50c184`](https://github.com/michaelellis003/smcx/commit/f50c18407f6111ceb9209e010cd23f140804e10a))

- **tempering**: Retain callback device inputs
  ([#63](https://github.com/michaelellis003/smcx/pull/63),
  [`e4fd246`](https://github.com/michaelellis003/smcx/commit/e4fd246d40f36f078bfa27221bc3e010aa867d95))

- **tempering**: Retain campaign launch attempts
  ([#76](https://github.com/michaelellis003/smcx/pull/76),
  [`8ad0e8a`](https://github.com/michaelellis003/smcx/commit/8ad0e8abe6bae89ec911cb36b9af949bfa18420c))

- **tempering**: Retain partial timing evidence
  ([#65](https://github.com/michaelellis003/smcx/pull/65),
  [`f8bb66c`](https://github.com/michaelellis003/smcx/commit/f8bb66ce98dff63e5549d6889272ac4350217c88))

- **tempering**: Retain public measurement evidence
  ([#78](https://github.com/michaelellis003/smcx/pull/78),
  [`6e7d657`](https://github.com/michaelellis003/smcx/commit/6e7d6570504609562e5e3c699c79e6e3f0109d24))

- **tempering**: Retain supervisor evidence ([#72](https://github.com/michaelellis003/smcx/pull/72),
  [`6fc888d`](https://github.com/michaelellis003/smcx/commit/6fc888de1473a8956410a32bd20e47698e70000d))

- **tempering**: Retain timing failure boundaries
  ([#67](https://github.com/michaelellis003/smcx/pull/67),
  [`740e44c`](https://github.com/michaelellis003/smcx/commit/740e44c59cfe20aa49b803c9939aec1a0b1e2ba8))

- **tempering**: Supervise frozen campaign ([#71](https://github.com/michaelellis003/smcx/pull/71),
  [`da38fb0`](https://github.com/michaelellis003/smcx/commit/da38fb0ca47bbbca2f2b16eb18f6a7a90ab31d56))


## v1.6.0 (2026-07-21)

### Documentation

- **adr**: Define Metal scan-history containment
  ([#39](https://github.com/michaelellis003/smcx/pull/39),
  [`335f9ce`](https://github.com/michaelellis003/smcx/commit/335f9cecde145c5e5b3d89590e3a3c02f44a30dd))

### Features

- **bootstrap**: Add chunked checkpoint updates
  ([#40](https://github.com/michaelellis003/smcx/pull/40),
  [`f66852c`](https://github.com/michaelellis003/smcx/commit/f66852ccb89b8a3599db81cfd7f3dff6c895f1e2))


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
