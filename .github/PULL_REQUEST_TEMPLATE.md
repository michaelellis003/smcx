## Summary

<!-- What does this PR do? Keep it to 1-3 bullet points. -->

-

## Related Issue

<!-- Link to the issue this PR addresses. Use "Closes #123" to auto-close. -->

## Changes

<!-- List the key changes made. -->

-

## Test Plan

<!-- How was this tested? -->

- [ ] Tests pass locally (`uv run pytest -v --cov`)
- [ ] Full suite passed on a local M-series GPU (CI is CPU-only
      unless the gpu-smoke job reports Metal available)
- [ ] Coverage meets minimum threshold (`fail_under` in `pyproject.toml`)
- [ ] Linting passes (`uv run pre-commit run --all-files`)

## Documentation triggers

<!-- Check each that applies; delete lines that don't. -->

- [ ] Public API changed → docstrings + quickstart still accurate
- [ ] Architecturally significant decision → ADR added (`docs/adr/`)
- [ ] smcjax divergence → ADR + noted in reference docs
- [ ] Roadmap item completed → struck through in `ROADMAP.md`
- [ ] No dead docs left behind
