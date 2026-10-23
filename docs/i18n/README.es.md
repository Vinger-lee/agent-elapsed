# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-268%20passing-22bb33" alt="268 tests">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="Runtime: numpy only">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="Model">
  <br>
  <b>⏱️ ETA honesta para tareas largas de agentes: local, milisegundos, sin LLM en el bucle.</b>
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

`agent-elapsed` da un intervalo tipo *"6–14 minutos"* **en paralelo** con la tarea y, al acabar, dibuja
predicción y realidad en el mismo eje. Dentro: hooks, estado, TUI, puente JSON; en ejecución solo
**numpy** — sin LLM, sin ajuste, sin telemetría, sin red.

> **Importante.** Validado con **telemetría sintética y experimentos numéricos independientes**; no
> hay registros reales, así que no se declara exactitud medida. `constants.json` es prior de arranque
> en frío, no modelo entrenado. El intervalo se oculta hasta pasar la puerta de calibración: un prior
> no es una predicción validada.

---

## El modelo, en detalle

Una tarea es una cadena **serial** de pasos en cuatro tipos: `llm`, `tool`, `build`, `net`. Con `g` y
clase latente `C = c`:

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

Supuestos: tipos independientes dada la clase; duraciones i.i.d. por tipo y libres del conteo; pasos
seriales; parámetros fijos por predicción.

**φ es dispersión pura.** Subir σ y bajar μ medio incremento deja la media fija:

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independiente de φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (crece con φ)
```

Se repara la *dispersión*, no el *nivel*.

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

Segundo momento, no varianza: el Poisson compuesto da `λ·E[X²]`. La clase entra por varianza total, con
**una clase latente por ruta** compartida:

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (intraclase)
       + Σ_c p_c (M_c − E[T])²  (entre clases)
```

La tasa promediada pierde el término entre clases: en `tests/test_math_audit.py` (saltos de 1 s, tasas
1 vs 9, pesos ½/½) la varianza real es **276**, no 20.

**Valor de φ.** Solo el segundo momento depende de φ (`exp(φ·σ_j²)`); la varianza relativa media es una
suma exponencial monótona:

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ varianza relativa entre clases ]     ← independiente de φ
                             + Σ_j B_j · exp(φ · σ_j²)
```

con `B_j = mean_t[ λ̄_tj · E[X_j]² / E[T_t]² ]` y objetivo
`r = mean_t[ ((y_t − E[T_t])/E[T_t])² ]` de holdout. Bisección en `[1, 25]`:

* `f(1) ≥ 0` → φ = 1 (tabla ancha; no *estrecha* bajo el ruido de paso);
* `f(25) ≤ 0` → φ = 25 (recortado, reportado acotado);
* si no, solución interior: CV² = residual **exacto**.

Con σ iguales, `φ = 1 + ln(r)/σ²` sirve de verificación cruzada.

---

## Cuantiles y su incertidumbre

Monte Carlo, 5000 rutas, sin aproximación normal. Átomo en cero:

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

`predict()` añade a `p50`/`p90` un intervalo libre de distribución, invirtiendo el test exacto de rangos
binomial (`Binomial(m, τ)` muestras bajo el cuantil):

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # estadísticos de orden, recortados a 0..m−1
```

En el átomo el intervalo se reduce a un punto, sin inventar banda. `tests/test_math_upgrade.py`, 300 repeticiones,
m = 2000:

| τ | nominal | cobertura medida |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

Con m = 5000: P50 ≈ ±22 s (SE ≈ 11 s), P90 ≈ ±89 s (SE ≈ 46 s) — la resolución de 5000 rutas.

---

## Calibración y la puerta de aceptación

Un candidato debe ganar a la tabla en línea en un holdout **partido en el tiempo** antes del reemplazo
atómico.

### Qué se mide

* **PIT aleatorizado**: `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)`; uniforme **incluso con
  empates o degeneración**.
* **Banda de cobertura**: `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)`, no igualdad; tolerancia `2√(τ(1−τ)/n)`.
* **KS sobre PIT**, p-valor asintótico: diagnóstico, no prueba.
* **Pinball emparejado** con la misma semilla; `gate.evaluate` da `paired_diff`, `paired_diff_se`,
  `paired_z`: el rechazo lleva su incertidumbre.

### Umbrales y transiciones

| Regla | Valor |
|---|---|
| Tareas completas | ≥ 80 |
| Tareas de ajuste | ≥ 60 |
| Holdout | `max(20, ceil(20 %))`, tramo reciente |
| Pasos nuevos por operación | ≥ 50 |
| φ | `[1, 25]`, bisección |
| Aceptación | calibración OK **y** pinball no peor (`require_strict`: mejor) |

Escritura con temporal y `os.replace`; la previa queda en `constants.prev.json`, `rollback` la
restaura. Un rechazo no toca nada. Datos insuficientes: se escribe el prior, **no** se promueve
candidato — código 2, nunca pase silencioso.

Deriva: `log1p` del pinball, EWMA de 100 puntos, banda `2σ` de base fija, dos cruces seguidos; la base
no se contamina con los datos bajo prueba.

---

## Integrarlo en un host

| Host | Superficie | Entrada |
|---|---|---|
| Agente Python | hooks + texto | `ElapsedHooks`, `render_tui(state)` |
| UI nativa (Qt, SwiftUI, web view) | estado | `WidgetState` → tus widgets |
| Otro lenguaje | residente, JSON por línea | `python -m agent_elapsed.bridge` |

Gating conservador: un intervalo erróneo es peor que uno oculto.

* umbral **60 s**, configurable;
* `long_task=True` lo muestra al instante;
* si no, cuando un P50 **calibrado** o el tiempo real cruzan el umbral;
* con `calibrating`: solo tiempo real y "calibrando", sin prior ni aviso;
* el chat común no inicia temporizador.

Protocolo, hilos y errores: [docs/integration.md](../../docs/integration.md).

---

## Registrar la telemetría (el bucle que hace real la autocorrección)

Nada de lo anterior aprende si las duraciones reales no llegan al almacén. `ElapsedHooks` solo dibuja
el widget; `TelemetryStore` solo añade filas validadas. `TelemetryRecorder` es el adaptador que une
uno y otro, y es la diferencia entre "una estimación bonita" y "una estimación que mejora":

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

Decisiones del adaptador, cada una con su motivo:

* **Los pasos se escriben al instante**, no al cerrar la tarea. Si el host se cae o se cancela, los
  pasos ya ocurridos son observaciones completas — justo lo que necesita el manejo de censura.
* **`cancel()` marca la tarea como censurada**: el tiempo de pared de una tarea cancelada es solo una
  cota inferior, así que no puede entrar en conteos, cobertura ni aceptación. Sus pasos sí informan
  las duraciones de paso.
* **`fail()` es una observación completa**: la tarea terminó de verdad y gastó ese tiempo de verdad.
* **Una tarea sin características se registra igual** (`g_pred` = `{}`). Perder una duración real
  porque el clasificador aún no había corrido sesgaría la muestra; en su lugar
  `offline.partition.partition_tasks` pone la fila en cuarentena y la cuenta, así el salto se ve en
  lugar de ser silencioso.

Prueba el bucle completo sin ningún host:

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```

---

## Comandos offline

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

`report` acepta `--store data --estimate out/estimate.json --actual 500`: aterrizaje real en el mismo
eje, cobertura P90 de las últimas 100 tareas, errores por `g_pred.task_class` y deriva. HTML inline,
sin red.

| Código | Significado |
|---|---|
| `0` | éxito |
| `1` | rechazado por la puerta |
| `2` | datos insuficientes (déficits en el JSON) |
| `3` | sin telemetría del entorno |
| `4` | sin copia de rollback |

---

## Contrato de telemetría

Dos tablas JSONL de solo añadir; derivadas (`X`, `n_j`) **no** se guardan.

* Paso: `task_id`, `op`, `t_start`, `t_end`, `ok`, `tool`, `v`
* Tarea: `task_id`, `g_pred`, `g_pred_probs`, `g_final`, `total_wall`, `censored`, `model_id`, `host`,
  `estimator_wall`, `g_schema_version`, `v`

Reglas que el código impone:

* `t_end > t_start`, tiempos finitos, `op` conocido, sin `(task_id, op, t_start)` repetido;
* filas malas → `rejects.jsonl`, nunca descarte silencioso: son las muy largas y sesgan a la baja;
* ajuste solo con `g_pred`; `g_final` sería fuga (prueba con `g_final` manipulado exige coeficientes iguales);
* canceladas: pasos al ajuste de duraciones, fuera de conteos, cobertura y aceptación;
* `task_id` único enlaza pasos y tareas.

---

## Verificación

```bash
python3 -B -m unittest discover -s tests -t .     # 268 pruebas
```

Extremo a extremo (400 tareas, semilla 61, `m-a|h-a`, 15 047 eventos, 351 completas, 32 holdout):

| Cantidad | Resultado |
|---|---|
| Puerta | accepted |
| Cobertura superior P50 / P80 / P90 | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| Pinball candidato | 242.8 |
| Pinball prior | 481.9 |
| Diferencia / z | −239.0 / −2.77 |
| φ ajustado | 1.0 (recorte inferior: la tabla ya cubre el holdout) |

Recuperación frente a la verdad oculta (`offline/synth.py`):

| op | μ ajustado / verdad | σ ajustado / verdad | pasos |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

Una predicción de 5000 rutas: unos **5 ms**. Coherencia interna y recuperación de un generador
conocido, no evidencia sobre registros reales: [docs/validation.md](../../docs/validation.md).

---

## Lo que no afirma

* **Sin verdad real**: no hay logs; `constants.json` es un prior.
* **Suma serial ≠ wall time**: solapes, anidamientos, ramas y colas la rompen.
* **Independencia condicional es supuesto**, no hallazgo; φ no modela correlación.
* **Parámetros puntuales**: el NIG contrae, pero no propaga incertidumbre de conteo ni clasificador.
* **Cancelación sesga** si correlaciona con la duración; no hay modelo de supervivencia.
* **Cuantiles MC con error de muestreo**: se reporta, sigue siendo error.
* **KS asintótico**: diagnóstico cerca de p > 0.05, no prueba.
* **El clasificador es del host**: mal calibrado, mezcla mal calibrada.

---

## Documentación

| Documento | Contenido |
|---|---|
| [docs/design.md](../../docs/design.md) | Límites de datos, criterios |
| [docs/embedded-widget.md](../../docs/embedded-widget.md) | Widget en TUI / UI nativa |
| [docs/integration.md](../../docs/integration.md) | Integración, eventos, hilos |
| [docs/accuracy-audit.md](../../docs/accuracy-audit.md) | Derivaciones, límites |
| [docs/math-upgrade.md](../../docs/math-upgrade.md) | Upgrade v1.1, verificación |
| [docs/implementation-notes.md](../../docs/implementation-notes.md) | Detalles, correcciones |
| [docs/validation.md](../../docs/validation.md) | Aceptación reproducible |
| [CHANGELOG.md](../../CHANGELOG.md) | Historial de versiones |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Flujo, pruebas, traducción |
| [SECURITY.md](../../SECURITY.md) | Amenazas, reporte de fallos |

### Idiomas

`README.md` y [`README_CN.md`](../../README_CN.md) son el par autoritativo; `docs/i18n/` los sigue por
secciones. Identificadores, campos y claves JSON **no** se traducen. `tests/test_docs.py` verifica
selector, tabla y rutas.

## Licencia

[MIT](../../LICENSE) © agent-elapsed contributors
