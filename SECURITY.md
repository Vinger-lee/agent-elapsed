# Security policy

## What this project does with data

`agent-elapsed` is designed so that the honest answer to "what could leak?" is "very little":

* The **runtime** (`agent_elapsed/`) makes no network requests, reads no telemetry files, calls no
  model and writes nothing except the append-only store paths the host passes in. A prediction is a
  pure function of `(g, class probabilities, constant table, seed, path count)`.
* The **offline layer** (`offline/`) reads the JSONL store the host points it at and writes the
  constant table, a rollback copy, a reject log and an HTML report. The report inlines its data and
  loads no external resources, so opening it does not phone home.
* **No credentials** are read, stored or transmitted; there is nothing to configure and no API key.
* Telemetry contains **task identifiers, timestamps, feature levels and durations** — no prompts, no
  model output, no user text. Whether task identifiers are sensitive is a property of the host, so
  treat the store with the same care as the host's own logs.

## Threat model

The interesting risks are data-handling, not remote exploitation:

| Risk | Mitigation |
|---|---|
| A host logs prompt text into `unrecognised` fields | The schema only stores the documented fields; unknown keys are ignored, and `rejects.jsonl` records the raw row — review it before sharing |
| A report is published with real durations | The report is generated locally, inline, and `out/` is git-ignored; check before attaching it to an issue |
| Malformed telemetry crashes a long-running process | Rows are validated, rejected rows are logged rather than raised, and the stdio bridge turns a bad line into one error event instead of exiting |
| Fitting silently uses leaked labels | Only `g_pred` is read; `g_final` is covered by a leakage test |
| A resource-exhausting table (huge μ or φ) | Parameters are validated finite, `eta` is clipped, and step moments raise instead of propagating `inf` |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Send a private report through GitHub's
[security advisories](https://github.com/Vinger-lee/agent-elapsed/security/advisories/new) for this
repository, including:

* what the issue is and which file or function it lives in,
* a minimal reproduction (inputs, seed, versions),
* the impact you believe it has,
* any mitigation you have already tested.

You can expect an acknowledgement within a few days. Once a fix is ready it ships with a note in
`CHANGELOG.md`; credit is given unless you ask otherwise.

## Supported versions

The `main` branch is the supported version. There are no long-term branches; fixes land on `main`.
