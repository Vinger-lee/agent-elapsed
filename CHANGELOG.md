# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] — 2026-10-23

The formula layer was re-derived and re-solved; the distribution family, the serial-summation
assumption, the Monte Carlo quantiles and the seed semantics are unchanged. Full derivations,
measured verification and deliberate non-goals: [docs/math-upgrade.md](docs/math-upgrade.md).

### Changed

- **φ is now a mean-preserving dispersion parameter.** `ConstTable.log_params` lowers the effective
  log-mean by `(s² − σ²)/2` whenever it widens σ, so `E[X] = exp(μ + σ²/2)` no longer depends on φ.
  Previously φ inflated σ alone, which pushed the predicted mean and every upper quantile upward
  while claiming to only widen the interval. Sampling and closed-form moments share this one entry
  point.
- **φ is solved, not guessed.** The model's mean relative variance is a monotone exponential sum in
  φ (the between-class term is φ-free and kept as a constant), so the fixed point is found by
  bisection on `[1, 25]`; at the interior solution the analytic CV² equals the holdout residual mean
  square exactly. The previous ratio heuristic could not attain its own target.
- **`estimate_phi` no longer samples.** It needs no Monte Carlo; `m` and `seed_base` remain accepted
  for call compatibility.
- Second moments are evaluated once per operation as `E[X²] = E[X]²·exp(s²)` with explicit
  non-finite guards, and `task_moments` no longer recomputes step moments per class.
- Weighted ridge Poisson regression: weighted intercept start, diagonal penalty, Cholesky solve with
  scaled jitter and `lstsq` fallback, gradient-norm stopping rule, relative objective tolerance.
  The unpenalised-intercept score equation `Σλ̂ = Σŵy` still holds exactly.
- Log-step NIG posterior accumulates its scatter with exact summation on prior-centred deviations.
- `gate.evaluate` reports `paired_diff`, `paired_diff_se` and `paired_z`, and accepts `min_effect`
  (default `0`, i.e. the previous decision rule byte for byte).
- `offline.calibrate.interval_loss` is now the single implementation of the (P50, P90) pinball mean,
  shared by the gate and the drift detector.

### Added

- Distribution-free quantile intervals: `predict()` returns `p50_ci`, `p90_ci`, `p50_se`, `p90_se`,
  and `EstResult` carries `p50_se` / `p90_se`. The interval inverts the exact binomial rank test, so
  it needs no density estimate and collapses correctly at the zero atom.
- `tests/test_math_upgrade.py` (23 tests): mean preservation, exact dispersion solve, quantile
  interval coverage, NIG agreement with a 60-digit decimal reference, IRLS convergence and
  invariants, loss contract.
- Documentation: [docs/math-upgrade.md](docs/math-upgrade.md) (what changed and what deliberately did
  not), [docs/validation.md](docs/validation.md) (reproducible acceptance record), this changelog,
  `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, a CI workflow and a docs-parity test.
- README rewritten and translated into eight languages (`README.md`, `README_CN.md`,
  `docs/i18n/README.{ja,ko,fr,es,ru,ar}.md`), with `tests/test_docs.py` enforcing switcher, document
  table and link parity.

### Fixed

- A class-mixture dispersion bug surfaced by the new solve: the between-class variance is φ-free and
  must stay in the constant term, otherwise a correct mixture table is pushed to the φ ceiling.
- The stdio bridge test no longer fails on CPython 3.13+, where the JSON scanner stopped recursing.

## [0.1.0] — 2026-10-22

First working version.

### Added

- Compound Poisson × lognormal model of task duration over four operation types, with closed-form
  moments and a task-class mixture that draws one latent class per path.
- Monte Carlo quantiles (5000 paths) with no normal approximation; explicit treatment of the zero
  atom.
- Randomized predictive rank (PIT), quantile coverage bracket, KS diagnostic and paired pinball
  acceptance gate with atomic table swap and rollback.
- Offline fitting: ridge Poisson count regression, log-step NIG posterior, synthetic generator with
  hidden ground truth, EWMA drift detector, inline-data HTML report, explicit CLI with documented
  exit codes.
- Host embedding: task hooks and display policy, `WidgetState` protocol, TUI text renderer and a
  resident line-oriented JSON bridge.
- 157 tests (standard library `unittest` only), later extended to 205 by the calibration corrections
  and to 229 by this release.
