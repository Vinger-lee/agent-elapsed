# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-268%20passing-22bb33" alt="268 tests">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="Runtime: numpy only">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="Model">
  <br>
  <b>⏱️ An honest ETA for long-running agent tasks — computed locally, in milliseconds, with no LLM in the loop.</b>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README_CN.md">中文</a> ·
  <a href="docs/i18n/README.ja.md">日本語</a> ·
  <a href="docs/i18n/README.ko.md">한국어</a> ·
  <a href="docs/i18n/README.fr.md">Français</a> ·
  <a href="docs/i18n/README.es.md">Español</a> ·
  <a href="docs/i18n/README.ru.md">Русский</a> ·
  <a href="docs/i18n/README.ar.md">العربية</a>
</p>

`agent-elapsed` is a drop-in elapsed-time widget for agents that run long tasks. When a host agent
starts a task, the component produces a *"6–14 minutes"* style interval **in parallel** with the task
itself; when the task ends, prediction and reality are drawn on the same axis and the model slowly
gets better at its job.

Inside it is deliberately small: task hooks, a state protocol, a TUI text renderer and a
line-oriented JSON bridge. No web page, no HTTP server, no separate window. At runtime it needs
**nothing but numpy** — no LLM call, no fitting, no telemetry reads, no network access.

> **Important.** Everything here is validated against **synthetic telemetry and independent numerical
> experiments**. There is no claim of measured accuracy on real agent logs, because no real log ships
> with the repository. `constants.json` is a cold-start prior, not a trained model. The component
> hides its interval until a fitted table has passed the calibration gate — a prior is never dressed
> up as a validated prediction.

---

## Contents

- [Why this exists](#why-this-exists)
- [Quickstart](#quickstart)
- [The model, in full](#the-model-in-full)
- [Quantiles and their uncertainty](#quantiles-and-their-uncertainty)
- [Calibration and the acceptance gate](#calibration-and-the-acceptance-gate)
- [Embedding it in a host](#embedding-it-in-a-host)
- [Recording telemetry](#recording-telemetry-the-loop-that-makes-it-self-correcting)
- [Offline commands](#offline-commands)
- [Telemetry contract](#telemetry-contract)
- [Repository layout](#repository-layout)
- [Verification](#verification)
- [What it does not claim](#what-it-does-not-claim)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

---

## Why this exists

An agent that says "this will take a while" is not giving an estimate. A useful estimate has to
satisfy four constraints at once, and they pull in different directions:

1. **Parallel.** It must exist while the task is still running, not after it finishes.
2. **Cheap.** It cannot burn the host's token budget or its own wall clock — a prediction that takes
   longer than the task is worthless.
3. **Honest.** Too narrow an interval is worse than no interval: it teaches users to distrust every
   future number.
4. **Self-correcting.** It must improve from the host's own task history, without a data pipeline.

The design follows from those constraints. One-shot classification (which an LLM can do) plus a
closed-form‑and‑sampling statistical model (which an LLM must not do) splits the problem: the host
supplies task features and class probabilities, this component owns all the arithmetic. Because the
model is a small parametric family, the moments are closed-form, the quantiles come from a fixed
number of Monte Carlo paths, and a whole prediction is deterministic given a seed — reproducible,
auditable and about five milliseconds on a laptop.

---

## Quickstart

Requirements: Python ≥3.11 and numpy ≥2.0. The test suite uses only the standard library.

```bash
python3 -c "import numpy; print(numpy.__version__)"   # needs >= 2.0
python3 -m unittest discover -s tests -t . -v          # 268 tests
```

Or install it and use the scripts:

```bash
pip install agent-elapsed
agent-elapsed-offline init --constants constants.json      # bundled cold-start prior
agent-elapsed-offline synth --out data --tasks 400        # synthetic telemetry with ground truth
agent-elapsed-offline refit --store data --constants constants.json --env 'm-a|h-a'
agent-elapsed-bridge --constants constants.json --env default   # stdio bridge for other languages
```

Watch the full host lifecycle in a terminal:

```bash
python3 examples/tui_host.py            # add --plain for frames without terminal control codes
```

Run the resident line-oriented bridge that a non-Python host can drive:

```bash
python3 -m agent_elapsed.bridge --constants constants.json --env default --long-task-seconds 60
```

Predict one task directly:

```python
import json
from agent_elapsed.constants import load_tables, pick_env
from agent_elapsed.estimate import estimate_task

table = pick_env(load_tables("constants.json"), "my-model|my-host")
g = {
    "task_class": "patch",
    "touched_files": "2-3",
    "verification_loops": "2",
    "needs_network": False,
    "plan_steps": "4-8",
}
result = estimate_task(g, [("patch", .65), ("debug", .35)], table=table, seed=7)

result.p50, result.p90, result.expected      # quantiles and the analytic mean
result.p50_se, result.p90_se                 # Monte Carlo standard error of each quantile
result.calibrating                           # True => do not show the interval yet
saved = json.dumps(result.to_dict(), ensure_ascii=False)
```

Persist `fit_id`, `seed` and `g_schema_version` together with the estimate you actually showed the
user. Afterwards the exact same prediction can be replayed against the real duration — that is what
makes offline calibration honest rather than retrospective.

---

## The model, in full

A task is a **serial** chain of steps grouped into four operation types:
`llm`, `tool`, `build`, `net`. Given a feature vector `g` and a latent task class `C = c`:

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

Assumptions, stated plainly: conditionally on the class, operation types are independent; step
durations are i.i.d. within an operation type and independent of the counts; every step is serial and
non-overlapping; parameters and class weights are fixed for one prediction.

**φ is a pure dispersion parameter.** Enlarging σ and lowering the log-mean by exactly half the
increment keeps the lognormal mean fixed:

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independent of φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (grows with φ)
```

That matters because what is being repaired is *spread*, not *level*. An earlier version inflated σ
without compensating μ, so φ quietly pushed the predicted mean and every upper quantile upward while
claiming to only widen the interval.

Because counts are Poisson and independent of step durations:

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

Note the second moment, not the variance: a compound Poisson's variance is `λ·E[X²]`.

Class uncertainty is handled by the law of total variance, with **one latent class drawn per task
path** and shared by all operation types:

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (within-class)
       + Σ_c p_c (M_c − E[T])²  (between-class)
```

Collapsing the mixture into a single Poisson with the averaged rate loses the between-class term. For
the hand-checkable fixture in `tests/test_math_audit.py` (jumps of exactly 1 s, class rates 1 vs 9
across four operations, weights ½/½) the true mean is 20 and the true variance is **276**, while the
averaged-rate model gives 20. Dropping the shared-class correlation makes the distribution far too
narrow — nearly half of the paths belong below 10 s, and the averaged model puts about 1 % there.

The mixture also means operation types are **not** independent unconditionally: their class-induced
covariance is carried by squaring the complete class mean `M_c`.

**How φ gets its value.** Only the second moment depends on φ, through `exp(φ·σ_j²)`, so the model's
mean relative variance is a monotone exponential sum

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ between-class relative variance ]     ← φ-free
                             + Σ_j B_j · exp(φ · σ_j²)
```

with `B_j = mean_t[ λ̄_tj · E[X_j]² / E[T_t]² ]`. The target `r = mean_t[ ((y_t − E[T_t])/E[T_t])² ]`
comes from holdout residuals, and the fixed point is found by bisection on `[1, 25]`:

* `f(1) ≥ 0` → φ = 1 (the table is already wide enough; φ is not allowed to *narrow* below step noise);
* `f(25) ≤ 0` → φ = 25 (clipped; the table is not "fixed", it is reported as bounded);
* otherwise the interior solution, where the analytic CV² equals the residual mean square **exactly**.

For equal σ the closed form `φ = 1 + ln(r)/σ²` is available and used as a cross-check.

---

## Quantiles and their uncertainty

Quantiles are Monte Carlo — 5000 paths by default — and no normal approximation is attempted anywhere.
The atom at zero is handled explicitly:

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

so the predictive distribution is not continuous. `predict()` therefore reports, next to `p50` and
`p90`, a distribution-free interval obtained by inverting the exact binomial rank test: the number of
samples below the population quantile is `Binomial(m, τ)`, hence

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # order statistics, clipped to 0..m−1
```

No density estimate is needed, which matters exactly where a kernel/SE formula breaks: at the atom the
interval collapses to a single point instead of inventing a narrow band.

Measured behaviour (`tests/test_math_upgrade.py`, 300 independent repetitions at m = 2000):

| τ | nominal | measured coverage |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

At m = 5000 on the reference table the P50 interval is about ±22 s (SE ≈ 11 s) and P90 about ±89 s
(SE ≈ 46 s). That is the resolution of 5000 paths; a tighter P90 costs paths, not cleverness. Heavy
tails are the reason: at φ = 4 the sample *variance* reaches only 83 % of its analytic value at 120k
paths and 96 % at 400k, while quantiles converge much faster.

---

## Calibration and the acceptance gate

A fitted table is never trusted blindly. A candidate must beat the online table on a **temporally
split** holdout before it is swapped in atomically.

### What is measured

* **Randomized predictive rank (PIT).** For observation `y` against `m` predictive samples,
  `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)` with independent `u ~ U(0,1)`. Under
  exchangeability `R` is uniform **even with ties and even for a fully degenerate distribution** —
  which is what makes the zero atom safe to evaluate.
* **Quantile coverage bracket.** For a discrete quantile the correct condition is
  `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)`, not an equality. The tolerance is two standard errors,
  `2√(τ(1−τ)/n)`, because a fixed ±0.05 is barely one σ at n = 20 and would reject fair models.
* **KS on the PIT sample**, as a diagnostic with its asymptotic p-value: an approximation, not a proof
  of the data-generating model.
* **Paired pinball loss.** The same task with the same seed is predicted by both models, so the
  comparison is paired. `gate.evaluate` also reports `paired_diff`, `paired_diff_se` and `paired_z`, so
  a rejection carries its uncertainty instead of hiding a coin flip.

### Thresholds and state transitions

| Rule | Value |
|---|---|
| Complete tasks in the store | ≥ 80 |
| Tasks on the fitting side | ≥ 60 |
| Holdout | `max(20, ceil(20 %))`, always the most recent slice |
| New step observations per operation | ≥ 50 |
| Dispersion φ | clipped to `[1, 25]`, solved by bisection |
| Acceptance | calibration passes **and** paired pinball is not worse (`require_strict` demands strictly better) |

A candidate is written with a temporary file plus `os.replace`, keeping the previous table as
`constants.prev.json`; `rollback` restores it. A rejected candidate leaves both the live table and the
rollback copy untouched. When the store holds telemetry for an environment but not enough of it, the
CLI writes the prior and explicitly does **not** promote a candidate — "insufficient data" is exit
code 2, never a silent pass.

Drift is detected on the same pinball series that scores the model: `log1p` of each score, EWMA over
the most recent 100 points, a `2σ` band from a fixed stable baseline, and two consecutive crossings
before alerting. The baseline is deliberately *not* updated with the data being tested, otherwise a
permanent shift would quietly migrate into the baseline and hide itself.

---

## Embedding it in a host

Three integration surfaces, one lifecycle core:

| Host kind | Surface | Entry point |
|---|---|---|
| Python agent | hooks + text renderer | `ElapsedHooks`, `render_tui(state)` |
| Native UI (Qt, SwiftUI, web view, …) | state subscription | `WidgetState` → your own widgets |
| Any other language | resident process, one JSON object per line | `python -m agent_elapsed.bridge` |

Display gating is intentionally conservative, because a wrong interval is worse than a hidden one:

* the default long-task threshold is **60 s** and is configurable;
* `long_task=True` from the host shows the widget immediately;
* otherwise the widget appears once a **calibrated** P50 crosses the threshold, or real elapsed time
  does;
* while `calibrating` is true only real elapsed time and the word "calibrating" are shown — the prior
  interval and the over-estimate warning stay hidden;
* ordinary chat never starts a timer or an estimate.

Full protocol, thread boundaries and error handling: [docs/integration.md](docs/integration.md).

---

## Recording telemetry (the loop that makes it self-correcting)

Nothing above ever learns unless real durations land in the store. `ElapsedHooks` only draws the
widget; `TelemetryStore` only appends validated rows. `TelemetryRecorder` is the adapter that turns
one into the other, and it is the difference between "a nice estimate" and "an estimate that gets
better":

```python
from agent_elapsed import ElapsedHooks, TelemetryRecorder, TelemetryStore, load_tables, pick_env

table = pick_env(load_tables("constants.json"), "my-model|my-host")
store = TelemetryStore("data/events.jsonl", "data/tasks.jsonl", "data/rejects.jsonl")

with ElapsedHooks(table) as hooks:
    recorder = TelemetryRecorder(hooks, store, model_id="my-model", host="my-host")

    recorder.start("t-1", "修复接口并运行测试", g=g, probs=[("patch", .7), ("debug", .3)],
                   long_task=True)
    with recorder.step("t-1", "llm"):
        ...                       # 一次 LLM 往返
    with recorder.step("t-1", "tool", tool="pytest") as step:
        ...
        step["ok"] = False        # 失败步骤照样记录时长
    recorder.finish("t-1")        # 取消用 cancel()，失败用 fail()
```

Decisions baked into the adapter, each for a reason:

* **Steps are written immediately**, not at task end. If the host crashes or the user cancels, the
  steps that already happened are complete observations — exactly the data censoring handling needs.
* **`cancel()` marks the task censored**: a cancelled task's wall time is only a lower bound, so it
  must not enter counts, coverage or acceptance. Its steps still inform step lengths.
* **`fail()` is a complete observation**: the task really ended and really spent that time.
* **A task with no features is still recorded** (`g_pred` = `{}`). Losing a real duration because the
  classifier had not run yet would bias the sample; instead `offline.partition.partition_tasks`
  quarantines the row and counts it, so the skip is visible rather than silent.

Try the whole loop with no host at all:

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```


---

## Offline commands

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

`report` also accepts `--store data --estimate out/estimate.json --actual 500` to draw the real landing
point on the same axis, the P90 upper coverage of the last 100 complete tasks, the error table grouped
by `g_pred.task_class`, and a drift badge. The HTML inlines its data and loads nothing from the
network.

| Exit code | Meaning |
|---|---|
| `0` | success |
| `1` | rejected by the acceptance gate |
| `2` | insufficient data (the JSON lists deficits and descriptive checks) |
| `3` | no telemetry for that environment |
| `4` | no rollback copy |

---

## Telemetry contract

Two tables, both JSONL, both append-only. Derived quantities (`X`, `n_j`) are deliberately **not**
stored: anything derived can drift away from its source.

* Step event: `task_id`, `op`, `t_start`, `t_end`, `ok`, `tool`, `v`
* Task record: `task_id`, `g_pred`, `g_pred_probs`, `g_final`, `total_wall`, `censored`, `model_id`,
  `host`, `estimator_wall`, `g_schema_version`, `v`

Rules the code enforces rather than documents:

* `t_end > t_start`, finite timestamps, `op` in the known set, no duplicate `(task_id, op, t_start)`;
* violating rows go to `rejects.jsonl` — never silently dropped, because the rows that get dropped are
  usually the very long or abnormal ones, and dropping them biases the mean downward in the same
  direction as censoring;
* fitting reads `g_pred` only; touching `g_final` would be label leakage, and a test fits twice with
  `g_final` tampered and requires identical coefficients;
* cancelled tasks contribute their completed steps to step-length fitting (a step that happened is a
  complete observation) but are excluded from counts, coverage and acceptance (their total is only a
  lower bound);
* every task id is unique; steps and tasks are joined by it.

---

## Repository layout

```
agent-elapsed/
├── constants.json               # cold-start prior table (per environment)
├── agent_elapsed/               # runtime: no fitting, no network (telemetry writes only)
│   ├── model.py                 #   compound Poisson × lognormal, moments, sampler, quantiles
│   ├── features.py              #   21-column one-hot encoding of g, top-2 class mixing
│   ├── constants.py             #   table shape, env selection, the φ entry point, bundled prior
│   ├── estimate.py              #   the single runtime entry point (EstResult)
│   ├── hooks.py                 #   task lifecycle, display policy, background estimate
│   ├── telemetry.py             #   hooks → store: task records and step events
│   ├── presentation.py          #   the shared "通常 6 分钟 · 九成在 14 分钟内" text
│   ├── tui.py                   #   TUI text block renderer
│   ├── bridge.py                #   resident stdio JSON protocol
│   ├── data/constants.json      #   the bundled cold-start prior (so pip install needs no checkout)
│   └── schema.py, store.py      #   validated telemetry contract, append-only store
├── offline/                     # fitting, calibration, gate, drift, synthetic data, report
│   ├── fit_counts.py            #   weighted ridge Poisson regression (IRLS/Newton)
│   ├── fit_step.py              #   log-step Normal-Inverse-Gamma posterior
│   ├── calibrate.py             #   randomized PIT, coverage, KS, pinball
│   ├── gate.py                  #   dispersion solve, paired acceptance, atomic swap, rollback
│   ├── drift.py                 #   EWMA drift detector, discount weights
│   ├── partition.py             #   quarantine for rows the model cannot read (counted, not dropped)
│   ├── synth.py                 #   hidden-truth generator (the only place with ground truth)
│   ├── templates/report.html    #   inline-data report template (shipped as package data)
│   └── viz.py, cli.py           #   offline HTML report, init/synth/refit/report/rollback
├── examples/tui_host.py         # a complete host lifecycle in a terminal
├── examples/record_telemetry.py # the whole loop: hooks → store → refit → report
├── docs/                        # see the documentation table below
└── tests/                       # 268 tests, standard-library unittest only
```

---

## Verification

```bash
python3 -B -m unittest discover -s tests -t .     # 268 tests
```

End-to-end on synthetic telemetry (400 tasks, seed 61, environment `m-a|h-a`, 15 047 step events, 351
complete tasks, 32 holdout tasks):

| Quantity | Result |
|---|---|
| Acceptance gate | accepted |
| P50 / P80 / P90 upper coverage | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| Candidate paired pinball | 242.8 |
| Prior paired pinball | 481.9 |
| Paired difference / z | −239.0 / −2.77 |
| Fitted φ | 1.0 (lower clip: the table's own dispersion already covers the holdout) |

Step-length recovery against the hidden truth in `offline/synth.py`:

| op | μ fitted / truth | σ fitted / truth | step observations |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

A single 5000-path prediction takes about **5 ms** on a laptop.

These numbers show that the implementation is internally consistent and recovers a known generator.
They are not evidence about real agent logs. Full record: [docs/validation.md](docs/validation.md).

---

## What it does not claim

Being explicit about the boundary is part of the product:

* **No ground truth about real tasks.** No real log ships here; `constants.json` is a prior.
* **Serial summation is not wall time.** Overlapping tool calls, nested timers, parallel branches and
  unrecorded queueing all break the model. Parallel work needs an execution graph, which this version
  deliberately does not invent.
* **Conditional independence is an assumption**, not a finding. Retry correlations are not modelled;
  φ is a dispersion knob, not a correlation model.
* **Point estimates of parameters.** The NIG posterior shrinks step parameters, but predictions use
  point estimates; count-regression and classifier uncertainty are not propagated.
* **Cancellation may bias the fit** if it correlates with duration; there is no survival model, only
  the lower-bound-aware split described above.
* **Monte Carlo quantiles carry sampling error** — now reported rather than hidden, but still error.
* **The KS p-value is asymptotic.** It is a diagnostic near the p > 0.05 threshold, not a proof.
* **The classifier belongs to the host.** Class probabilities are taken as given; if they are
  miscalibrated, so is the mixture.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/design.md](docs/design.md) | Data boundaries, statistical definitions, acceptance criteria |
| [docs/embedded-widget.md](docs/embedded-widget.md) | Embedding the widget in a host TUI / native UI |
| [docs/integration.md](docs/integration.md) | Integration code, event protocol, thread boundaries |
| [docs/accuracy-audit.md](docs/accuracy-audit.md) | Derivations, calibration fixes, where accuracy claims stop |
| [docs/math-upgrade.md](docs/math-upgrade.md) | The v1.1 formula upgrade: derivations, verification, non-goals |
| [docs/implementation-notes.md](docs/implementation-notes.md) | Implementation details and corrections to earlier drafts |
| [docs/validation.md](docs/validation.md) | Reproducible acceptance record |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development workflow, test and translation rules |
| [SECURITY.md](SECURITY.md) | Threat model and how to report a vulnerability |

### Languages

`README.md` (English) and [`README_CN.md`](README_CN.md) (中文) are the authoritative pair; the
translations under [`docs/i18n/`](docs/i18n/) track them section by section. Code identifiers, feature
levels, telemetry field names and JSON keys are deliberately **not** translated — translating them
would break host integrations. Every language file is checked by `tests/test_docs.py`, which asserts
that the language switcher, the document table and every referenced path resolve.

---

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

[MIT](LICENSE) © agent-elapsed contributors
