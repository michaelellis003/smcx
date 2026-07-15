# AGENTS.md — process constitution

Rules for anyone (human or AI agent) changing this repository. Tool-
specific extras live in `CLAUDE.md` (untracked); this file is the
committed source of truth for process.

## The development loop (TDD-first)

0. **DECIDE** — if the work adds/pins a dependency, diverges from
   smcjax's public API, or picks among numerical algorithms with real
   alternatives: write an ADR first (`docs/adr/`, copy
   `0000-template.md`, next number, status `proposed`). Accepted ADRs
   are immutable; reverse via a superseding ADR.
1. Define work — GitHub issue with Given/When/Then acceptance
   criteria. **The issue is the spec** for single-session work.
2. Branch — `<type>/<issue-id>-<short-description>`.
3. RED — write one failing test.
4. GREEN — minimal code to pass.
5. REFACTOR — clean up, tests still pass.
6. COMMIT — conventional commit per cycle
   (`feat|fix|test|refactor|docs|chore(scope): description`).
7. Repeat 3–6 until acceptance criteria are met.
8. DOCS — apply the documentation triggers below.
9. PUSH — `uv run pre-commit run --all-files` + full tests first.
   Before merging to main: run the full suite on a local M-series
   GPU. CI's gpu-smoke job runs the suite on Metal when the runner
   exposes a GPU (GitHub's macOS runners currently expose a
   paravirtual device — observed 2026-07-15) but that is
   best-effort, not guaranteed; the local run remains the
   authoritative Metal gate, attested at release by the
   environment approval.
10. PR — self-review; target < 400 changed lines.
11. MERGE — squash to main; semantic-release automates versioning.

For multi-session features (a new module, a port): write
`specs/<branch-name>.md` first — files and interfaces involved, what
is out of scope, an end-to-end verification command. `specs/` is
gitignored (specs are prompts, not documentation) but must not be
lost or unreviewable: **paste the spec into the GitHub issue or PR
description** when opening it — that is the durable, reviewable copy.
Anything durable in a spec graduates to an ADR (decisions) or docs
(behavior) before merge. For one-sentence diffs: no spec, no plan,
just TDD.

## Documentation system

| Document | Type | Update trigger |
|---|---|---|
| `docs/adr/` | Immutable | New significant decision (step 0); never edit accepted ADRs |
| `ROADMAP.md` | Living | The moment priorities change; strike completed items at each minor release; refresh the date stamp |
| `docs/` site (quickstart, how-tos, reference) | Living | Same PR as any user-facing change; delete dead docs, don't TODO them |
| `docs/design/*.md` | Snapshot | Never updated — superseded by a new dated snapshot if ever needed |
| `docs/research/*.md` | Snapshot | Re-run and re-date when assumptions decay (MLX audit: on every mlx floor bump) |
| `CHANGELOG.md` | Immutable | Automated by semantic-release — never hand-edit |
| `README.md` | Living | Thesis/status/scope changes; kill-test verdicts |

Documentation triggers (step 8, mirrored in the PR template):

- Public API changed → docstrings + quickstart still accurate.
- Architecturally significant decision made → ADR exists.
- smcjax divergence → ADR + noted in reference docs.
- Roadmap item completed → struck through in `ROADMAP.md`.
- No dead docs left behind.

## Scope

Check `ROADMAP.md` Non-goals before proposing features; do not
implement anything listed there. Prefer the smallest change that
satisfies the acceptance criteria. The smcjax-derived API is the
inherited surface and smcjax is a frozen reference (ADR-0010):
public API changes still require an ADR, but no cross-repo
coordination exists anymore.

## Engineering standards

Evidence and sources: `docs/research/engineering-practices.md`.

- **Typing** (ADR-0007): jaxtyping shape annotations on all array
  params; beartype import hook in conftest enforces them in tests;
  never `from __future__ import annotations`; user closures typed as
  callback Protocols in `types.py`, not bare `Callable`; vendored
  `typings/mlx/core.pyi` regenerated on every mlx floor bump.
- **Validation**: structural checks (shapes, dtypes, sizes) at
  public-function entry in plain Python (MLX shape inference is
  eager). Data-dependent checks (degenerate weights, NaN) live in the
  loop shell at eval boundaries (detection latency up to k−1 steps
  under eval-every-k; one step lagged under async_eval; immediate at
  k=1) — they cannot raise inside a compiled step. One custom exception
  (`DegenerateWeightsError`), one warning category
  (`SMCNumericsWarning`), always with `stacklevel`.
- **Stochastic tests, three tiers**: (1) exact seeded tests only for
  determinism contracts and frozen parity fixtures; (2) moment tests
  with tolerance = 5× the derived estimator SE, derivation in a
  comment (MC-error-honest, extending f32-honest); (3) distributional
  tests at a committed seed with documented α — a failure at the
  committed seed is information, never re-rolled. Hypothesis via
  numpy strategies converted at the boundary; `deadline=None`,
  `derandomize=True` in CI.
- **Functional core**: step is pure scan-shaped
  `(carry, inputs_t, key_t) -> (carry, outputs_t)`; value-dependent
  control flow in the shell; NamedTuples everywhere; per-step keys
  pre-split in the shell. Unit-test the step uncompiled + one
  compiled-equivalence test per algorithm.
- **Benchmarks**: GPU numbers are local-only on M-series hardware; CI
  never asserts timings. Results are dated markdown in
  `benchmarks/results/` with a metadata header (hardware, macOS, mlx
  version, N, T, warm-up, median-of-≥5, peak memory). Every public
  performance claim cites a committed results file.
- Pre-1.0 changes land at minor bumps without deprecation shims,
  **except** `FutureWarning` for anything that silently changes
  numerical output at a fixed key. Post-1.0: NEP 23.

## Licensing and attribution

Evidence and license table: `docs/research/licensing.md`.

- **Red lines — never port/translate code from**: pomp (GPL-3),
  avehtari/PSIS (GPL-3), Numerical Recipes (proprietary), any GPL
  code. Design *ideas* from them are fine; cite them in docs.
- **Porting code is creating a derivative work** (translation
  included). MIT sources: preserve the copyright + permission notice
  beside the ported code. Apache-2.0 sources (smcjax, BlackJAX,
  ArviZ): Apache §4 — provenance header on the ported file ("Ported
  to MLX from X (URL), Apache-2.0. Modified: …") always; a NOTICE
  line is *legally required* only when the upstream ships a NOTICE
  file whose contents must be carried (none of the current sources
  do). We additionally sanction voluntary NOTICE lines for
  ported-code provenance (the existing smcjax line is one). Design
  credits — no code ported — go in README/`CITATION.cff`, never
  NOTICE.
- **Algorithm docstrings cite** the paper(s) and, where relevant, the
  reference implementation lineage (e.g. lgamma: Lanczos 1964 +
  Godfrey 2001 coefficient set as used in GSL; Pareto-k: Zhang &
  Stephens 2009 + Vehtari et al. 2024).
- **Upstream policy** (ADR-0006): vendor critical-path gaps with a
  tracking issue linking upstream; re-check tracking issues +
  regenerate MLX stubs + re-run the mlx audit on every mlx floor
  bump; delete vendored code the release after upstream ships it.
  File evidence-backed issues for upstream defects; PR only where a
  maintainer invited it and we don't block on the merge.

## Project constraints (pointers, not prose)

- Numerics and MLX rules: ADR-0003/0004/0005 and
  `docs/research/mlx-audit.md` (hazard list). f32 log-domain in the
  hot loop; explicit RNG keys; no keyless randomness anywhere.
- Style: 80-char lines, 4-space indent, single quotes, Google
  docstrings, type hints everywhere; license headers via
  `make license`.
- Tests before implementation, always; f32-honest tolerances with the
  justification in a comment.
