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
10. PR — self-review; target < 400 changed lines.
11. MERGE — squash to main; semantic-release automates versioning.

For multi-session features (a new module, a port): write
`specs/<branch-name>.md` first — files and interfaces involved, what
is out of scope, an end-to-end verification command. `specs/` is
gitignored: specs are prompts, not documentation. Anything durable in
a spec graduates to an ADR (decisions) or docs (behavior) before
merge. For one-sentence diffs: no spec, no plan, just TDD.

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
satisfies the acceptance criteria. API parity with smcjax is the
default; divergence is an ADR-level event.

## Project constraints (pointers, not prose)

- Numerics and MLX rules: ADR-0003/0004/0005 and
  `docs/research/mlx-audit.md` (hazard list). f32 log-domain in the
  hot loop; explicit RNG keys; no keyless randomness anywhere.
- Style: 80-char lines, 4-space indent, single quotes, Google
  docstrings, type hints everywhere; license headers via
  `make license`.
- Tests before implementation, always; f32-honest tolerances with the
  justification in a comment.
