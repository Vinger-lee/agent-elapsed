# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="النجوم">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-268%20passing-22bb33" alt="268">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="numpy">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="النموذج">
  <br>
  <b>⏱️ تقدير زمني صادق للمهام الطويلة — محليًا، بالميلي ثانية، دون LLM في الحلقة.</b>
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

`agent-elapsed` ودجة زمن للمهام الطويلة: فترة «6–14 دقيقة» تُحسب **بالتوازي** مع المهمة، ثم يُرسم
التوقّع والواقع على محور واحد — بـ**numpy** فقط.

> **Important.** التحقق على **قياسات اصطناعية وتجارب عددية** فقط، لا سجلات حقيقية. `constants.json`
> أوليّ للبدء البارد لا نموذج مدرَّب، والمجال مخفي حتى تجتاز جدولًا ملاكم بوابة المعايرة، فلا يُزيَّن
> الأوليّ أبدًا كتوقّع متحقَّق.

---

## النموذج كاملًا

سلسلة **تسلسلية** في أربعة أنواع (`llm`, `tool`, `build`, `net`) بمعطى `g` وصنف كامن `C = c`:

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

الأنواع مستقلة بشرط الصنف، والمدد i.i.d.، والخطوات غير متداخلة.

**φ تشتّت محض**: رفع σ وخفض μ بنصف الزيادة يحفظ المتوسط، فالمُصلَح *الانتشار* لا *المستوى*:

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independent of φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (grows with φ)
```

تباين بواسون المركّب يستخدم العزم الثاني:

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

قانون التباين الكلي، بصنف كامن لكل مسار:

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (within-class)
       + Σ_c p_c (M_c − E[T])²  (between-class)
```

الدمج في معدل واحد يُفقد حد ما بين الأصناف (20 مقابل **276**).

**φ**: العزم الثاني وحده يعتمد عليها، والهدف `r` من بواقي holdout، والتنصيف على `[1, 25]`:

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ between-class relative variance ]     ← φ-free
                             + Σ_j B_j · exp(φ · σ_j²)
```

φ = 1 إن `f(1) ≥ 0`، و25 مقصوصًا إن `f(25) ≤ 0`، وإلا الجذر الداخلي.

---

## الشرائح المئوية وعدم يقينها

مونت كارلو (5000 مسار) بلا تقريب طبيعي؛ والذرّة عند الصفر:

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

فالتوزيع غير متصل؛ والفترة من عكس اختبار الرتبة الثنائي (`Binomial(m, τ)`):

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # order statistics, clipped to 0..m−1
```

بلا تقدير كثافة، فتنطبق الفترة عند الذرّة إلى نقطة. (300 تكرار، m = 2000):

| τ | اسمي | التغطية |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

عند m = 5000: P50 ±22 ث (SE ≈ 11)، P90 ±89 ث (SE ≈ 46).

---

## المعايرة وبوابة القبول

المرشّح يتفوّق على الحي في holdout **مقسّم زمنيًا** قبل الاستبدال الذري.

* **PIT** بـ`u ~ U(0,1)`: منتظم حتى مع التعادلات والانحلال.
* **التغطية** `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)` لا مساواة؛ والسماحية `2√(τ(1−τ)/n)`.
* **KS** على PIT: تشخيص لا برهان.
* **pinball مزدوج** بنفس المهمة والبذرة: `paired_diff`, `paired_diff_se`, `paired_z`.

| القاعدة | القيمة |
|---|---|
| مهام مكتملة | ≥ 80 |
| جانب الملاءمة | ≥ 60 |
| Holdout | `max(20, ceil(20 %))`، أحدث شريحة |
| مشاهدات جديدة لكل عملية | ≥ 50 |
| φ | `[1, 25]` بالتنصيف |
| القبول | المعايرة **و** pinball ليس أسوأ (`require_strict`) |

`os.replace` مع `constants.prev.json` و`rollback`؛ ونقص البيانات رمز 2.
والانحراف: `log1p`، EWMA على 100 نقطة، `2σ` من أساس مستقر، وعبوران متتاليان.

---

## التضمين في مضيف

| نوع المضيف | السطح | نقطة الدخول |
|---|---|---|
| وكيل Python | خطافات + عارض | `ElapsedHooks`, `render_tui(state)` |
| واجهة أصلية (Qt، SwiftUI، web) | اشتراك حالة | `WidgetState` ← ودجاتك |
| لغة أخرى | عملية مقيمة، JSON لكل سطر | `python -m agent_elapsed.bridge` |

العرض محافظ: **60 ث**؛ `long_task=True` فورًا؛ وإلا عند تجاوز P50 **المعايَر**؛ و`calibrating`
يُظهر الزمن الحقيقي فقط؛ والمحادثة لا تُوقَّت. [docs/integration.md](../../docs/integration.md)

---

## تسجيل القياسات (الحلقة التي تجعل التصحيح الذاتي ممكنًا)

لا شيء مما سبق يتعلّم فعلًا ما لم تصل المدد الحقيقية إلى المخزن. `ElapsedHooks` يرسم الودجة فقط؛
و`TelemetryStore` يضيف صفوفًا مُتحقَّقًا منها فقط. أما `TelemetryRecorder` فهو المُهايئ الذي يحوّل
أحدهما إلى الآخر، وهو الفرق بين «تقدير لطيف» و«تقدير يتحسّن»:

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

قرارات مثبَّتة في المُهايئ، ولكلٍّ منها سبب:

* **تُكتب الخطوات فورًا**، لا عند نهاية المهمة. إذا انهار المضيف أو ألغى المستخدم، فالخطوات التي
  حدثت بالفعل مشاهدات كاملة — وهي بالضبط البيانات التي تحتاجها معالجة الرقابة.
* **`cancel()` يعلّم المهمة كمرقَّبة**: الزمن الفعلي لمهمة ملغاة حدٌّ أدنى فقط، لذا يجب ألا يدخل
  العدّ ولا التغطية ولا القبول. وخطواتها تظل تفيد أطوال الخطوات.
* **`fail()` مشاهدة كاملة**: المهمة انتهت فعلًا وأنفقت ذلك الوقت فعلًا.
* **المهمة التي لا ميزات لها تُسجَّل أيضًا** (`g_pred` = `{}`). إن فقدان مدة حقيقية لأن المصنّف لم
  يكن قد عمل بعد سيُحيّز العيّنة؛ وبدلًا من ذلك يعزل `offline.partition.partition_tasks` الصف
  ويعدّه، فيكون التخطّي مرئيًا لا صامتًا.

جرّب الحلقة كاملة دون أي مضيف:

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```

---

## الأوامر دون اتصال

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

`report` يقبل `--actual 500`؛ وHTML بلا شبكة.

| رمز | المعنى |
|---|---|
| `0` | نجاح |
| `1` | مرفوض بالبوابة |
| `2` | بيانات غير كافية |
| `3` | لا قياسات للبيئة |
| `4` | لا نسخة استرجاع |

---

## عقد القياسات

JSONL بالإضافة فقط؛ والمشتقات (`X`, `n_j`) لا تُخزَّن.

* خطوة: `task_id`, `op`, `t_start`, `t_end`, `ok`, `tool`, `v`
* مهمة: `task_id`, `g_pred`, `g_pred_probs`, `g_final`, `total_wall`, `censored`, `model_id`,
  `host`, `estimator_wall`, `g_schema_version`, `v`

* `t_end > t_start`، `op` معروف، ولا تكرار `(task_id, op, t_start)`؛ والمخالف إلى `rejects.jsonl`
  لا يُسقط صامتًا؛
* الملاءمة تقرأ `g_pred` فقط، و`g_final` تسريب؛
* المهام الملغاة تُسهم في ملاءمة الأطوال لا في الأعداد؛ والمعرّف فريد.

---

## التحقق

```bash
python3 -B -m unittest discover -s tests -t .     # 268 tests
```

من طرف إلى طرف (400 مهمة، بذرة 61، `m-a|h-a`، 15 047 حدثًا، 351 مكتملة، 32 holdout):

| الكمية | النتيجة |
|---|---|
| البوابة | مقبول |
| تغطية P50 / P80 / P90 | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| pinball مرشّح / أوليّ | 242.8 / 481.9 |
| الفرق / z | −239.0 / −2.77 |
| φ | 1.0 (قصّ أدنى) |

| op | μ ملاكم / حقيقة | σ ملاكم / حقيقة | مشاهدات |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

توقّع واحد نحو **5 ms**؛ والأرقام اتساق داخلي لا دليل على سجلات حقيقية:
[docs/validation.md](../../docs/validation.md).

---

## ما لا يدّعيه

* لا حقيقة أرضية؛ `constants.json` أوليّ.
* التسلسل ليس wall time؛ التوازي يكسر النموذج.
* الاستقلال الشرطي افتراض؛ φ مقبض تشتّت.
* تقديرات نقطية بلا نشر لعدم اليقين.
* الإلغاء قد يحيّز الملاءمة؛ لا نموذج بقاء.
* شرائح مونت كارلو بخطأ معاينة.
* KS مقاربة لا برهان.
* المصنِّف ملك المضيف.

---

## الوثائق

| المستند | المحتوى |
|---|---|
| [docs/design.md](../../docs/design.md) | الحدود والتعريفات |
| [docs/embedded-widget.md](../../docs/embedded-widget.md) | تضمين الودجة |
| [docs/integration.md](../../docs/integration.md) | كود التكامل والأحداث |
| [docs/accuracy-audit.md](../../docs/accuracy-audit.md) | الاشتقاقات وحدود الدقة |
| [docs/math-upgrade.md](../../docs/math-upgrade.md) | ترقية صيغ v1.1 |
| [docs/implementation-notes.md](../../docs/implementation-notes.md) | تفاصيل التنفيذ |
| [docs/validation.md](../../docs/validation.md) | سجل قبول قابل للإعادة |
| [CHANGELOG.md](../../CHANGELOG.md) | تاريخ الإصدارات |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | سير التطوير والترجمة |
| [SECURITY.md](../../SECURITY.md) | نموذج التهديد والإبلاغ |

المعرّفات والحقول لا تُترجم.

---

## الترخيص

[MIT](../../LICENSE) © agent-elapsed contributors
