# 0023. External implementations are one-time isolated validators

Date: 2026-07-18 | Status: accepted | Supersedes: — | Superseded-by: —

## Context

smcx needs independent evidence for its particle filters, SMC samplers,
and resamplers without making outside implementations part of the
runtime or test dependency graph. Reimplementing those packages locally
would weaken independence and create avoidable derivative-work risk.
Stochastic comparisons also cannot use particle-by-particle equality
across unrelated random-number generators.

## Options considered

- Add validators to the dev/test environment — reproducible, but couples
  ordinary tests to large, conflicting, and cross-language dependencies.
- Keep a permanent validation framework — reproducible, but leaves more
  research infrastructure than this project wants to maintain.
- Run pinned validators once in isolated environments, then promote only
  reviewed numerical references and provenance into dependency-free tests.

## Decision

We will execute authoritative outside implementations as pinned black
boxes in temporary isolated environments. We will not copy or translate
their algorithm code. Comparisons will use preregistered exact-oracle and
Monte-Carlo-SE gates. After confirmation, temporary adapters and
environments will be removed; permanent tests will contain only frozen
data or summary values with immutable source, version, license, seed, and
derivation citations.

## Consequences

Ordinary installation and tests remain independent of comparator
packages, and production fixes can be derived from mathematical
requirements rather than copied code. Reproducing a campaign later will
require reconstructing the cited temporary adapters from its recorded
commands and source pins. Frozen stochastic references must retain their
replicate counts and uncertainty; a single random trajectory is never an
oracle.
