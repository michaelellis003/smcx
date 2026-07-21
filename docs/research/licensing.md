# Licensing and attribution

*Snapshot: 2026-07-19. Licenses verified against actual LICENSE files
/ CRAN fields, not memory. Re-check when a new reference project is
added or before porting any code.*

## License table

| Project | License | What smcx takes | Obligation |
|---|---|---|---|
| mlx | MIT (© 2023 Apple Inc.) | transitive Metal runtime through jax-mps | dependency/API use: none; no source copied or translated |
| jax-mps | Apache-2.0 (no NOTICE upstream) | optional Metal runtime dependency; public PJRT backend only | dependency/API use: none; no source copied or translated |
| numpy | BSD-3 | dependency; `np.searchsorted` test oracle | none |
| smcjax | Apache-2.0 (same author) | code translation (diagnostics), API contract, fixtures | ported-file provenance headers |
| blackjax | Apache-2.0 (no NOTICE upstream) | resampling contract (idea); possible port source | idea: none; Apache §4 applies if code is ported |
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
     Design credits go in documentation rather than license files.
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
