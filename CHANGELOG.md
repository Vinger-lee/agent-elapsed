# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [1.2.0] — 2026-10-23

这版补的是"能不能真的用起来"：闭环、安装、以及几个会让宿主替我们崩掉的边界。

### Added

- **`TelemetryRecorder`**（`agent_elapsed/telemetry.py`）：把 `ElapsedHooks` 的生命周期翻译成
  遥测行 —— 自动写任务记录、`step()` 上下文管理器写入步事件、取消记删失、失败记完整观测、
  重复终结幂等、坏行不抛异常。此前组件只有"显示"和"落盘"两半，中间没有任何东西把它们连起来，
  "越用越准"在代码里无法成立。
- **`examples/record_telemetry.py`**：无宿主跑通整条链路（录制 → 落盘 → 自动 refit）。
- **`offline.cli init` / `synth`**：写出随包先验、生成带真值的合成遥测。新用户不再需要自己造数据。
- **包内资源**：`agent_elapsed/data/constants.json`（冷启动先验）与 `offline/templates/report.html`
  随包分发，用 `importlib.resources` 读取。
- **`[project.scripts]`**：`agent-elapsed-bridge`、`agent-elapsed-offline`。
- **`agent_elapsed.__all__`**：宿主可以从包根一次性导入公开入口。
- **`offline.partition`**：把"记录在库里但模型读不了"的任务隔离并计数，写进 report 与 CLI 输出。
- **落盘校验**：`total_wall` 非有限/负数、坏概率形状、缺 `task_id/model_id/host` 一律进
  rejects，不再抛异常打断宿主写入。
- **运行时保护**：`predict` 拒绝 `m < 2`；单次采样超过 `MAX_EVENTS`（20 000 000 步）时给出可读
  错误而不是去分配上百 GB；空样本的分位数区间改为报错而不是返回 NaN。
- **拟合诊断**：`refit` 输出 `step_diagnostics`（每个操作的观测数、中位数与先验中位数之差、
  位置先验占 β 的比例）。

### Changed

- **位置先验权重与尺度先验解耦**：位置（冷启动中位数）只保留 1 个伪观测，尺度仍按
  `PRIOR_EQUIV_N=20` 收缩。此前 285 个真实观测之后中位数仍被拉高 33%，而这个偏差会让第一次
  重拟合的覆盖率冲到 1.0、KS 拒绝 —— 恰恰是新宿主最需要它的时候。完整 NIG 后验仍可通过
  `posterior()` 取得。
- **预测用的 σ 取 scatter 口径**（不含 location gap）：位置先验猜错时不再假装成步长离散度。
- **用户可见文案**：`band` 由"大约 3–9 分钟"改为"通常 3 分钟 · 九成在 9 分钟内"。
  P50–P90 对连续校准分布只含约 40% 概率，旧措辞把区间说大了一倍多。`format_band` 保留但不再
  用于展示。
- 报告模板移入 `offline/templates/`，`tests/test_viz.py` 改为通过包读取。

### Fixed

- `pip install agent-elapsed` 之后 `report` 必崩（模板路径指向仓库外）。
- `TelemetryStore.append_task` 在坏概率形状上抛 `ValueError`，把宿主写入路径打断。
- `offline/__init__.py` 的过时说明（"可依赖 scipy"，项目从未依赖）。

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
