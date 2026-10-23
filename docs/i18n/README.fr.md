# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="Étoiles">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="Licence : MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-267%20passing-22bb33" alt="267 tests">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="Exécution : numpy uniquement">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="Modèle">
  <br>
  <b>⏱️ Une ETA honnête pour les tâches d'agent longues — calculée localement, en millisecondes, sans LLM dans la boucle.</b>
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

`agent-elapsed` produit un intervalle *« 6–14 minutes »* **en parallèle** de la tâche, puis trace
prédiction et réalité sur le même axe ; **numpy** suffit, sans LLM ni réseau.

> **Important.** Validé seulement sur **télémétrie synthétique et expériences numériques
> indépendantes** ; aucun journal réel n'est livré, donc aucune précision mesurée n'est revendiquée.
> `constants.json` est un prior de démarrage à froid : l'intervalle reste masqué avant la calibration,
> et un prior n'est jamais présenté comme validé.

---

## Le modèle, en entier

Chaîne **sérielle** d'étapes `llm`, `tool`, `build`, `net` ; pour `g` et `C = c` :

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

Hypothèses : indépendance conditionnelle, durées i.i.d., étapes sérielles. **φ = dispersion seule** :
σ monte, la log-moyenne baisse d'autant, la moyenne reste fixe.

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independent of φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (grows with φ)
```

Comptages indépendants des durées :

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

**Une classe par trajectoire**, variance totale :

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (within-class)
       + Σ_c p_c (M_c − E[T])²  (between-class)
```

Un taux moyen unique donne **20** au lieu de **276**, et ~1 % sous 10 s contre ~50 %
(`tests/test_math_audit.py`). **φ** ne touche que le second moment, `exp(φ·σ_j²)` :

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ between-class relative variance ]     ← φ-free
                             + Σ_j B_j · exp(φ · σ_j²)
```

`B_j = mean_t[ λ̄_tj · E[X_j]² / E[T_t]² ]` ; cible `r = mean_t[ ((y_t − E[T_t])/E[T_t])² ]`,
dichotomie `[1, 25]` :

* `f(1) ≥ 0` → φ = 1, jamais sous le bruit ;
* `f(25) ≤ 0` → φ = 25, borné et signalé ;
* sinon solution intérieure : CV² = carré moyen résiduel **exactement**.

---

## Quantiles et leur incertitude

Monte Carlo, 5000 trajectoires, sans approximation normale ; atome en zéro :

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

Intervalle sans hypothèse de loi, à côté de `p50`/`p90`, par rang binomial exact :

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # order statistics, clipped to 0..m−1
```

`tests/test_math_upgrade.py`, 300 répétitions, m = 2000 :

| τ | nominal | couverture |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

m = 5000 : P50 ±22 s (SE ≈ 11 s), P90 ±89 s (SE ≈ 46 s).

---

## Calibration et porte d'acceptation

Un candidat doit battre la table en ligne sur un holdout **séparé temporellement**.

### Mesures

* **PIT randomisé** `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)` : uniforme, atome compris.
* **Couverture** `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)`, tolérance `2√(τ(1−τ)/n)`.
* **KS** sur le PIT, p-value asymptotique : diagnostic.
* **Pinball appariée** : `paired_diff`, `paired_diff_se`, `paired_z`.

### Seuils

| Règle | Valeur |
|---|---|
| Tâches complètes | ≥ 80 |
| Côté ajustement | ≥ 60 |
| Holdout | `max(20, ceil(20 %))`, tranche récente |
| Étapes par opération | ≥ 50 |
| φ | `[1, 25]`, dichotomie |
| Acceptation | calibration OK **et** pinball non pire (`require_strict`) |

Écriture atomique (`os.replace`), `constants.prev.json` pour `rollback` ; un rejet ne bouge rien.
Données insuffisantes : prior, **aucune** promotion, code 2. Dérive : `log1p`, EWMA 100 points, `2σ`
sur base fixe, deux franchissements ; la base n'intègre jamais les données testées.

---

## Intégration dans un hôte

| Type d'hôte | Surface | Point d'entrée |
|---|---|---|
| Agent Python | hooks + rendu texte | `ElapsedHooks`, `render_tui(state)` |
| UI native (Qt, SwiftUI, web view) | abonnement à l'état | `WidgetState` → vos widgets |
| Autre langage | processus résident, JSON par ligne | `python -m agent_elapsed.bridge` |

* seuil **60 s**, configurable ;
* `long_task=True` affiche immédiatement ;
* sinon P50 **calibré** au-dessus du seuil, ou temps réel ;
* `calibrating` : seuls le temps réel et le mot s'affichent ;
* conversation ordinaire : ni chronomètre ni estimation.

---

## Enregistrer la télémétrie (la boucle qui rend l'auto-correction réelle)

Rien de ce qui précède n'apprend jamais rien tant que des durées réelles n'arrivent pas dans le
store. `ElapsedHooks` ne fait que dessiner le widget ; `TelemetryStore` ne fait qu'ajouter des lignes
validées. `TelemetryRecorder` est l'adaptateur qui transforme l'un en l'autre, et c'est la différence
entre « une belle estimation » et « une estimation qui s'améliore » :

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

Choix inscrits dans l'adaptateur, chacun pour une raison :

* **Les étapes sont écrites immédiatement**, pas à la fin de la tâche. Si l'hôte plante ou si
  l'utilisateur annule, les étapes déjà passées sont des observations complètes — exactement les
  données dont la gestion de la censure a besoin.
* **`cancel()` marque la tâche comme censurée** : le temps mural d'une tâche annulée n'est qu'une
  borne inférieure, il ne doit donc entrer ni dans les comptages, ni dans la couverture, ni dans
  l'acceptation. Ses étapes informent quand même les longueurs d'étape.
* **`fail()` est une observation complète** : la tâche s'est réellement terminée et a réellement
  consommé ce temps.
* **Une tâche sans features est tout de même enregistrée** (`g_pred` = `{}`). Perdre une durée
  réelle parce que le classifieur n'avait pas encore tourné biaiserait l'échantillon ; à la place,
  `offline.partition.partition_tasks` met la ligne en quarantaine et la compte, de sorte que le saut
  soit visible plutôt que silencieux.

Essayez toute la boucle sans aucun hôte :

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```

---

## Commandes hors ligne

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

| Code | Signification |
|---|---|
| `0` | succès |
| `1` | rejeté par la porte |
| `2` | données insuffisantes |
| `3` | aucune télémétrie |
| `4` | aucune copie de rollback |

---

## Contrat de télémétrie

Deux tables JSONL en ajout seul ; dérivés non stockés (`X`, `n_j`).

* Étape : `task_id`, `op`, `t_start`, `t_end`, `ok`, `tool`, `v`
* Tâche : `task_id`, `g_pred`, `g_pred_probs`, `g_final`, `total_wall`, `censored`, `model_id`, `host`,
  `estimator_wall`, `g_schema_version`, `v`

* `t_end > t_start`, horodatages finis, `op` connu, pas de doublon ; lignes fautives dans
  `rejects.jsonl`, jamais perdues — les jeter biaise comme la censure ;
* ajustement sur `g_pred` seul, jamais `g_final` (fuite ; test : mêmes coefficients) ;
* annulation : étapes terminées comptées pour les durées, exclues des comptages ;
* `task_id` unique, clé de jointure.

---

## Vérification

```bash
python3 -B -m unittest discover -s tests -t .     # 267 tests
```

Bout en bout, synthétique : 400 tâches, graine 61, `m-a|h-a`, 15 047 événements, 351 complètes, 32 holdout.

| Grandeur | Résultat |
|---|---|
| Porte d'acceptation | accepted |
| Couverture P50 / P80 / P90 | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| Pinball candidat | 242.8 |
| Pinball prior | 481.9 |
| Différence / z | −239.0 / −2.77 |
| φ ajusté | 1.0 (table déjà couvrante) |

| op | μ ajusté / vérité | σ ajusté / vérité | observations |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

**5 ms** pour 5000 trajectoires ; cela valide l'implémentation, pas les journaux réels.
[docs/validation.md](../../docs/validation.md).

---

## Ce qu'il ne prétend pas

* **Aucune vérité terrain** : `constants.json` est un prior.
* **Sériel ≠ wall time** : chevauchements et branches parallèles cassent le modèle.
* **Indépendance conditionnelle** : hypothèse ; φ n'est pas une corrélation.
* **Estimations ponctuelles** : incertitude non propagée.
* **Annulation biaisante** si corrélée à la durée.
* **Erreur MC** sur les quantiles, rapportée mais réelle.
* **KS asymptotique** : diagnostic près de p > 0.05.
* **Classifieur = hôte** : probabilités prises telles quelles.

---

## Documentation

| Document | Contenu |
|---|---|
| [docs/design.md](../../docs/design.md) | Frontières, critères |
| [docs/embedded-widget.md](../../docs/embedded-widget.md) | TUI / UI native |
| [docs/integration.md](../../docs/integration.md) | Protocole, threads |
| [docs/accuracy-audit.md](../../docs/accuracy-audit.md) | Dérivations, limites |
| [docs/math-upgrade.md](../../docs/math-upgrade.md) | Formules v1.1 |
| [docs/implementation-notes.md](../../docs/implementation-notes.md) | Détails |
| [docs/validation.md](../../docs/validation.md) | Dossier d'acceptation |
| [CHANGELOG.md](../../CHANGELOG.md) | Versions |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | Workflow, tests |
| [SECURITY.md](../../SECURITY.md) | Menaces |

## Licence

[MIT](../../LICENSE) © agent-elapsed contributors
