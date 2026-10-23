# Contributing

Thanks for taking an interest. This repository is small on purpose: a statistics core, a fitting
layer and a thin host-embedding surface. Contributions that respect those boundaries are welcome.

## Ground rules

1. **The runtime stays client-side.** `agent_elapsed/` must not read telemetry, fit parameters, call
   a model or touch the network. `tests/test_boundaries.py` enforces this; a change that violates it
   will be rejected regardless of what it buys.
2. **Honesty over polish.** Never present a cold-start prior as a validated prediction, never report a
   coverage number as accuracy, and never let a Monte Carlo artefact look like a model improvement.
   If a limit exists, write it down (see `docs/accuracy-audit.md`, "What it does not claim").
3. **No approximations that are not labelled.** Closed-form moments are exact *under the stated
   assumptions*; quantiles are Monte Carlo. A normal approximation, a kernel density or a CLT shortcut
   needs a very good reason and must be labelled in the docstring and in the docs.
4. **Reproducibility.** Anything that samples takes an integer seed and must be deterministic for it.
   Do not read or write NumPy's global legacy random state.

## Development setup

```bash
python3 -c "import numpy; print(numpy.__version__)"   # needs >= 2.0, Python >= 3.11
python3 -m unittest discover -s tests -t . -v
```

There is no packaging step for development: the tests import the packages straight from the repository
root. `python3 -m compileall -q agent_elapsed offline examples tests` should also stay clean.

## What a change needs

| Change | Required alongside it |
|---|---|
| New or changed formula | A test that recomputes the quantity from its definition, plus a line in `docs/accuracy-audit.md` or `docs/math-upgrade.md` |
| New threshold or exit code | Update the tables in `README.md`, `README_CN.md`, all `docs/i18n/*` files and the CLI help |
| New test | Update the test-count badge in every README file (`tests/test_docs.py` fails otherwise) |
| New document under `docs/` | Add it to the documentation table in every README file |
| New telemetry field | Bump `g_schema_version`, update `schema.py`, the contract section of every README and `docs/design.md` |
| Behaviour change | An entry under `## [Unreleased]` in `CHANGELOG.md` |

## Test discipline

* `unittest` only — the runtime dependency is numpy, and the test layer deliberately adds nothing.
* Prefer tests that would fail on a plausible wrong implementation. "It returns a number" is not a
  test; "the analytic CV² equals the residual mean square to 1e-9" is.
* When you fix a bug, first write the test that reproduces it, then fix it. Several tests in this
  repository exist because a previous version passed its own suite while being wrong.
* Statistical tests need a fixed seed and a stated tolerance with a reason, not a tolerance chosen to
  make the run green. If a test is inherently stochastic, assert a band derived from the sampling
  distribution and say so in a comment.

## Translations

`README.md` (English) and `README_CN.md` (Chinese) are authoritative; the files under `docs/i18n/`
follow them section by section. When you change a number, a command, a threshold or a table row in the
main README, change it in every translation in the same commit.

Untranslated by design: code identifiers, operation names (`llm`, `tool`, `build`, `net`), feature
levels (`task_class=patch`, …), telemetry field names, JSON keys and shell commands. Translating them
breaks host integrations.

`tests/test_docs.py` verifies that every language file exists, carries the full eight-language
switcher, lists every document, resolves every relative link and quotes the same test count.

## Commit and pull request style

* One logical change per commit; imperative subject line ("fix the between-class dispersion term",
  not "fixed stuff").
* The body explains *why*, and states what was verified and what was not. Numbers belong in the body
  when they are the evidence.
* Do not mix formatting-only churn with behaviour changes.
* If a change alters predictions, say so explicitly in the pull request: which numbers move, and in
  which direction.

## Reporting a bug

Include the inputs (`g`, class probabilities, the constant table's `fit_id`), the seed, the observed
and expected outputs, and the Python/numpy versions. If the issue is statistical rather than
programmatic, say which quantity you think is wrong and how you checked it — a hand-computable case is
the most useful possible report.
