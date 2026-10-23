# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-267%20passing-22bb33" alt="267 tests">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="numpy only">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="Model">
  <br>
  <b>⏱️ Честная оценка времени для долгих задач агента — локально, за миллисекунды, без LLM.</b>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="../../README_CN.md">中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ar.md">العربية</a>
</p>

`agent-elapsed` — виджет прошедшего времени: интервал «6–14 минут» идёт **параллельно** задаче; рантайм — numpy.

> **Важно.** Только **синтетическая телеметрия и независимые численные опыты**; реальных логов нет.
> `constants.json` — априор, не модель; без ворот интервал скрыт: априор не выдаётся за прогноз.

## Полная модель

Шаги **последовательны**: `llm`, `tool`, `build`, `net`; класс `C = c`:

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

**φ — чистый разброс**: среднее не меняется:

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independent of φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (grows with φ)
```

Счётчики пуассоновские, независимы от длительностей:

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

Класс — закон полной дисперсии:

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (within-class)
       + Σ_c p_c (M_c − E[T])²  (between-class)
```

**φ** — корень монотонной суммы, бисекция `[1, 25]`:

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ between-class relative variance ]     ← φ-free
                             + Σ_j B_j · exp(φ · σ_j²)
```

`r` — остатки отложки.

## Квантили и их неопределённость

5000 путей Монте-Карло, атом в нуле:

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

`predict()` даёт интервал биномиального ранга:

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # order statistics, clipped to 0..m−1
```

| τ | номинал | покрытие (m = 2000, 300 повторов) |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

m = 5000: P50 ≈ ±22 с (SE ≈ 11 с), P90 ≈ ±89 с (SE ≈ 46 с).

## Калибровка и ворота приёмки

* **PIT**: `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)`, `u ~ U(0,1)`; равномерен и при совпадениях.
* **Покрытие**: `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)`; допуск `2√(τ(1−τ)/n)`.
* **KS на PIT**: p асимптотический.
* **Парная pinball**: `paired_diff`, `paired_diff_se`, `paired_z`.

| Правило | Значение |
|---|---|
| Полных задач | ≥ 80 |
| Задач на подгонке | ≥ 60 |
| Отложка | `max(20, ceil(20 %))` |
| Шагов/операцию | ≥ 50 |
| Разброс φ | `[1, 25]`, бисекция |
| Приёмка | калибровка **и** pinball не хуже |

`os.replace`, `constants.prev.json`, `rollback`; мало данных — код 2, не молча.

Дрейф: `log1p`, EWMA-100, `2σ`, два подряд выхода.

## Встраивание в хост

| Хост | Вход |
|---|---|
| Python-агент | `ElapsedHooks`, `render_tui(state)` |
| Нативный UI (Qt, SwiftUI) | `WidgetState` → свои виджеты |
| Другой язык | `python -m agent_elapsed.bridge` |

* **60 с**; `long_task=True` — сразу; иначе **калиброванный** P50 или реальное время;
* `calibrating`: лишь время;
* обычный чат не считает.

## Запись телеметрии (цикл, который делает самокоррекцию реальной)

Ничто из описанного выше не учится, пока реальные длительности не попадают в хранилище. `ElapsedHooks` только рисует виджет; `TelemetryStore` только дописывает провалидированные строки. `TelemetryRecorder` — адаптер, превращающий одно в другое, и именно он отличает «красивую оценку» от «оценки, которая становится лучше»:

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

Решения, зашитые в адаптер, каждое по своей причине:

* **Шаги пишутся сразу**, а не в конце задачи. Если хост упадёт или пользователь отменит задачу, уже состоявшиеся шаги — полные наблюдения; ровно те данные, которые нужны для учёта цензурирования.
* **`cancel()` помечает задачу цензурированной**: wall time отменённой задачи — только нижняя граница, поэтому он не должен попадать в счётчики, покрытие или приёмку. Её шаги по-прежнему уточняют длины шагов.
* **`fail()` — полное наблюдение**: задача действительно завершилась и действительно потратила это время.
* **Задача без признаков всё равно записывается** (`g_pred` = `{}`). Потерять реальную длительность из-за того, что классификатор ещё не отработал, означало бы сместить выборку; вместо этого `offline.partition.partition_tasks` отправляет строку в карантин и учитывает её, так что пропуск виден, а не молчалив.

Прогнать весь цикл вообще без хоста:

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```

## Оффлайн-команды

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

| Код | Что |
|---|---|
| `0` | успех |
| `1` | отклонено |
| `2` | мало данных |
| `3` | нет телеметрии |
| `4` | нет отката |

## Контракт телеметрии

JSONL, только добавление строк.

* Шаг: `task_id`, `op`, `t_start`, `t_end`, `ok`, `tool`, `v`
* Задача: `task_id`, `g_pred`, `g_pred_probs`, `g_final`, `total_wall`, `censored`, `model_id`, `host`,
  `estimator_wall`, `g_schema_version`, `v`

* `rejects.jsonl` — не молча;
* только `g_pred`; `g_final` — утечка меток;
* отмена — шаги в подгонку, не в счётчики;
* task id уникален.

## Проверка

```bash
python3 -B -m unittest discover -s tests -t .     # 267 tests
```

Синтетика: 400 задач, seed 61, `m-a|h-a`, 15 047 шагов, 351 полная, 32 отложки.

| Что | Итог |
|---|---|
| Ворота | accepted |
| Покрытие P50 / P80 / P90 | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| Pinball кандидата / априора | 242.8 / 481.9 |
| Разность / z | −239.0 / −2.77 |
| φ | 1.0 (нижняя обрезка) |

| op | μ / истина | σ / истина | шагов |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

**5 мс**; не данные о реальных логах.

## Чего он не обещает

* **Нет истины**: логов нет.
* **Сумма шагов ≠ wall time**: перекрытия, очередь.
* **Условная независимость — допущение.**
* **Параметры — точечные оценки.**
* **Отмена смещает подгонку.**
* **Квантили Монте-Карло: ошибка выборки.**
* **p-значение KS асимптотично.**
* **Классификатор — у хоста.**

## Документация

| Документ | Раздел |
|---|---|
| [design.md](../../docs/design.md) | Данные |
| [embedded-widget.md](../../docs/embedded-widget.md) | UI |
| [integration.md](../../docs/integration.md) | Код |
| [accuracy-audit.md](../../docs/accuracy-audit.md) | Точность |
| [math-upgrade.md](../../docs/math-upgrade.md) | v1.1 |
| [implementation-notes.md](../../docs/implementation-notes.md) | Детали |
| [validation.md](../../docs/validation.md) | Приёмка |
| [CHANGELOG.md](../../CHANGELOG.md) | Релизы |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Разработка |
| [SECURITY.md](../../SECURITY.md) | Угрозы |

## Лицензия

[MIT](../../LICENSE) © agent-elapsed contributors
