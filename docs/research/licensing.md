# Licensing, attribution, and upstream strategy

*Snapshot: 2026-07-19. Licenses verified against actual LICENSE files
/ CRAN fields, not memory. Re-check when a new reference project is
added or before porting any code. Binding rules extracted into
`AGENTS.md`; attribution artifacts: `NOTICE`, `CITATION.cff`.*

## License table

| Project | License | What smcx takes | Obligation |
|---|---|---|---|
| mlx | MIT (© 2023 Apple Inc.) | runtime dependency | none; courtesy credit |
| jax-mps | Apache-2.0 (no NOTICE upstream) | optional Metal runtime dependency; public PJRT backend only | dependency/API use: none; no source copied or translated |
| numpy | BSD-3 | dependency; `np.searchsorted` test oracle | none |
| smcjax | Apache-2.0 (same author) | code translation (diagnostics), API contract, fixtures | NOTICE line + ported-file headers (hygiene) |
| blackjax | Apache-2.0 (no NOTICE upstream) | resampling contract (idea); possible port source | idea: none; if ported: Apache §4 headers + NOTICE line |
| particles (Chopin) | MIT | design ideas (FK core) | none; cite book + repo |
| dynamax | MIT | design ideas (containers); optional notebook dependency; public LGSSM APIs called by the profiling adapter, with no code copied or translated | dependency/API use: none; cite the exact release, commit, public methods, and upstream license |
| tensorflow-probability | Apache-2.0 | design ideas (criterion/trace fns) | none |
| pymc | Apache-2.0 | design ideas (UX, ArviZ interop) | none |
| filterpy | MIT | design idea (free-function resamplers) | none |
| pfilter | MIT (no holder named in file) | design idea | none; credit John Williamson by name if ever ported |
| pyfilter | MIT | design idea (proposals) | none |
| Stone-Soup | MIT (8 co-holders) | cautionary tale | none |
| samplex | MIT (© 2023 Thomas Edwards) | MLX prior art, RNG-pattern proof | none; courtesy credit |
| arviz | Apache-2.0 (LICENSE has unfilled placeholder — attribute "ArviZ developers") | test strategy, Pareto-k conventions | conventions: none; if `_gpdfit` ported: §4 — prefer reimplementation from papers |
| **pomp** (R) | **GPL-3** | design idea (plug-and-play) ONLY | **red line: never port code**; cite King, Nguyen & Ionides (2016, JSS) |
| nimbleSMC (R) | BSD-3 \| GPL-2 \| GPL-3 (user's choice) | design idea | none (ideas only; BSD option exists if ever needed) |

## Obligations by borrowing mode

1. **Design ideas / API shapes**: not copyrightable (17 U.S.C.
   §102(b); *Google v. Oracle* 2021 held API reimplementation fair use
   even assuming protection). Obligation: none. Norm: credit in docs,
   `CITATION.cff` `references`, docstring References sections.
2. **Porting/translating code = derivative work** (§101 includes
   "translation"; GNU FAQ; Rosen). Line-by-line JAX→MLX translation
   keeping structure/names/constants carries the source license.
   Clean reimplementation from the paper/spec is a new work.
   - MIT → Apache-2.0: compatible (ASF Category A); preserve the
     copyright + permission notice in or beside the ported file.
   - Apache-2.0 → Apache-2.0: §4 — ship the license, prominent change
     notices on modified files, retain source-form notices, carry
     upstream NOTICE contents (BlackJAX/ArviZ have no NOTICE).
   - ASF policy reserves NOTICE for legally required attributions;
     this repo additionally sanctions voluntary ported-code
     provenance lines (per AGENTS.md — the smcjax line is one).
     Design credits go in docs, never NOTICE.
3. **Test-strategy borrowing** (cases, properties, tolerances,
   thresholds): §102(b) ideas — no obligation; comment crediting the
   source of the test design. Copying test *code* is ordinary copying.

## Red lines — never port/translate code from

1. **pomp** — GPL-3.
2. **avehtari/PSIS** — GPL-3 (commonly assumed BSD; it is not). Safe
   Pareto-k lineage: Zhang & Stephens (2009) + Vehtari et al. (2024)
   papers → smcjax's own Apache-2.0 implementation.
3. **Numerical Recipes** — proprietary; its `gammln` is a different
   (g=5) coefficient set, so no accidental overlap with ours.
4. Any GPL R code generally.

## Provenance findings

- **Lanczos g=7/n=9 coefficients** (used in our `lgamma`): the
  Godfrey (2001)/GSL set (`lanczos_7_c[9]` in GSL `specfunc/gamma.c`;
  reproduced on Wikipedia citing Godfrey 2001 and Pugh 2004; used in
  GSL/Boost/CPython/musl). Mathematical facts, unencumbered, and NOT
  the Numerical Recipes set. Cite Lanczos (1964) + Godfrey (2001) in
  the docstring.
- **Zhang–Stephens GPD fit**: implement via smcjax's lineage (own
  code matching the papers, NumPyro-convention-compatible), citing
  Zhang & Stephens (2009) and Vehtari et al. (2024). Do not port from
  avehtari/PSIS (GPL-3).
- **Dynamax profiling adapter**: original smcx glue calls the public
  distribution methods in Dynamax 1.0.2 (commit
  `a216d7feec0d025560a0a194ed5abab538648375`). No Dynamax source was copied,
  translated, or vendored. The adapter docstring carries immutable links to
  the MIT-licensed API source and license; the campaign records the installed
  distribution version and `uv.lock` digest.

## Upstream assessment (verified July 2026)

MLX: no CLA; conservative maintainers; **standing July 2026
moratorium** on PRs adding unary/binary primitives pending the SIMD
rework (draft PR #3019).

| Item | Verified state | Action |
|---|---|---|
| lgamma/digamma | PR #3181 closed by maintainer ("maintenance costs… few requests"); issue #2050 open; #3330 floats a future `mx.special` submodule | **Vendor + track.** No new PR (moratorium). Add a supportive use-case comment on #3330 with our 1.3e-6 f32 accuracy data. Revisit after #3019 lands and `mx.special` materializes |
| searchsorted | PR #2817 closed as an **abandoned draft — not rejected on merits**; issue #1255 open with maintainer on record: "I'm open to adding a little binary search implementation… it's open" | **Vendor now, PR later.** After the kill test proves the workload, offer a clean benchmarked PR referencing the #1255 invitation; ask first whether it can be a composed op given the moratorium |
| `categorical(num_samples=M)` O(N·M) memory | **Unreported upstream** (no existing issue found) | **File a performance issue now** with our measured data (400 MB at 1e4², attempted 4 TB at 1e6²; 499 MB/9.4 ms inverse-CDF workaround). Creates the public record ADR-0004 can cite |
| gamma/beta/dirichlet samplers | never requested upstream | nothing for v0; optionally mention under #3330 |
| scan/while_loop | issue #1441 unanswered; `mx.while_loop` PR declined on principle | wait/do nothing — Python-loop-over-compiled-step is the sanctioned pattern |
| numpy | no gaps | nothing |
| smcjax | own project | **Open coordinated-change issues now**, priority: (1) `simulate` initial-sampler fix, (2) pluggable `resampling_criterion`, (3) guided filter — the three smcx already implements/fixes — then Model-bundle, smoothing hooks, ArviZ bridge, store_history, dtype parameterization |

## Policy (adopted in AGENTS.md)

- **Contribute-first** only when a maintainer has signaled openness on
  the record, the change is self-contained with tests/benchmarks, and
  smcx doesn't block on the merge. Never block the kill test on
  upstream.
- **Vendor-with-tracking-issue** is the default for critical-path
  gaps: in-library implementation + one smcx tracking issue per
  vendored capability linking the upstream issue; re-check on every
  mlx floor bump; delete vendored code the release after upstream
  ships it.
- **File issues, not PRs, for behavior/performance defects.**
- **Wait** where maintainers rejected the direction on principle.
