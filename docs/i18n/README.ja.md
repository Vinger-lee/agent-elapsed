# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="スター数">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="ライセンス: MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-267%20passing-22bb33" alt="267 件のテスト">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="実行時は numpy のみ">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="モデル">
  <br>
  <b>⏱️ 長時間走るエージェントタスクに、誠実な残り時間を — ローカル計算、ミリ秒、LLM 呼び出しなし。</b>
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

`agent-elapsed` は長時間走るタスク向けの、そのまま差し込める経過時間ウィジェットです。タスクと**並行**に「6〜14 分」の区間を出し、終了後は予測と実績を同じ軸に描きます。内部はフック、状態プロトコル、TUI レンダラ、行指向 JSON ブリッジのみで、実行時に要るのは **numpy だけ**です。

> **重要。** 検証はすべて**合成テレメトリと独立な数値実験**に基づきます。実ログは同梱されず、実ログ上の実測精度は主張しません。`constants.json` は学習済みモデルではないコールドスタートの事前分布で、フィット表がキャリブレーションゲートを通るまで区間は隠されます。事前分布を検証済み予測として装うことはしません。

---

## 完全なモデル

タスクは `llm`、`tool`、`build`、`net` の 4 操作に分類されたステップの**直列**チェーン。`g` と潜在クラス `C = c` に対し:

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

**φ は純粋なばらつきパラメータ。** σ を増やし対数平均を増分の半分だけ下げれば平均は不変:

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independent of φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (grows with φ)
```

直すのは**水準**ではなく**ばらつき**です。

件数がポアソンでステップ長と独立なので:

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

複合ポアソンの分散は `λ·E[X²]`。

クラス不確かさは全分散の法則。**潜在クラスは経路ごとに 1 回だけ抽選**し、全操作種別で共有:

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (within-class)
       + Σ_c p_c (M_c − E[T])²  (between-class)
```

平均レート＋単一ポアソンに潰すとクラス間項が落ちます。

**φ の決め方。** 2 次モーメントに `exp(φ·σ_j²)` を通してのみ効くので、平均相対分散は単調な指数和:

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ between-class relative variance ]     ← φ-free
                             + Σ_j B_j · exp(φ · σ_j²)
```

`B_j = mean_t[ λ̄_tj · E[X_j]² / E[T_t]² ]`、`r = mean_t[ ((y_t − E[T_t])/E[T_t])² ]` はホールドアウト残差、`[1, 25]` で二分法:

* `f(1) ≥ 0` → φ = 1(十分広い。狭小化は不可);
* `f(25) ≤ 0` → φ = 25(クリップ、有界として報告);
* それ以外は内点解(解析 CV² = 残差平均平方、厳密)。

σ が等しければ `φ = 1 + ln(r)/σ²` で相互検証。

---

## 分位数とその不確かさ

分位数はモンテカルロ(既定 5000 経路)で、正規近似は使いません。ゼロの原子は明示的に:

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

分布は連続でないため、`predict()` は `p50`、`p90` と並べて分布無仮定の区間も返します。二項の正確な順位検定の反転で、母分位点未満の標本数は `Binomial(m, τ)`:

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # order statistics, clipped to 0..m−1
```

実測(`tests/test_math_upgrade.py`、m = 2000、300 回の独立反復):

| τ | 名目 | 実測カバレッジ |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

m = 5000 では P50 約 ±22 秒(SE ≈ 11 秒)、P90 約 ±89 秒(SE ≈ 46 秒)。より狭い P90 は経路数を要します。

---

## キャリブレーションと受け入れゲート

候補は**時間分割**ホールドアウトで現行表に勝たなければ、アトミックに差し替えられません。

### 何を測るか

* **ランダム化 PIT。** `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)`、`u ~ U(0,1)`。交換可能性の下で一様 — **同順位でも完全退化分布でも**成り立ち、ゼロ原子も安全。
* **カバレッジの挟み込み。** 条件は `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)` で等式ではない。許容幅は `2√(τ(1−τ)/n)`(固定 ±0.05 は n = 20 で 1σ 強)。
* **PIT 上の KS** と漸近 p 値は診断であり、証明ではありません。
* **ペア pinball。** 同一タスク・同一シードのペア比較で、`paired_diff`、`paired_diff_se`、`paired_z` も返す。

### 閾値と状態遷移

| 規則 | 値 |
|---|---|
| ストア内の完了タスク | ≥ 80 |
| フィット側のタスク | ≥ 60 |
| ホールドアウト | `max(20, ceil(20 %))`、常に直近の区間 |
| 操作種別ごとの新規ステップ観測 | ≥ 50 |
| ばらつき φ | `[1, 25]` にクリップ、二分法で求解 |
| 受け入れ | キャリブレーション合格 **かつ** ペア pinball が悪化しない(`require_strict` は厳密な改善を要求) |

候補は一時ファイル + `os.replace`、旧表は `constants.prev.json`、`rollback` で復元。棄却候補は現行表もコピーも触りません。テレメトリ不足時は事前分布を書き候補を**昇格させず**、終了コード 2(無言通過なし)。

ドリフトは採点と同じ pinball 系列:`log1p`、直近 100 点の EWMA、固定安定ベースラインの `2σ` 帯、2 回連続逸脱で警報。ベースラインは検査対象データで更新しない(永久シフトが隠れるため)。

---

## ホストへの組み込み

3 つの統合面、1 つのライフサイクルコア:

| ホスト種別 | 面 | エントリポイント |
|---|---|---|
| Python エージェント | フック + テキストレンダラ | `ElapsedHooks`、`render_tui(state)` |
| ネイティブ UI(Qt、SwiftUI、WebView…) | 状態購読 | `WidgetState` → 独自ウィジェット |
| その他の言語 | 常駐プロセス、1 行 1 JSON | `python -m agent_elapsed.bridge` |

表示ゲーティングは保守的(誤った区間は隠すより悪い):

* 既定の長タスク閾値は **60 秒**、設定可能;
* `long_task=True` で即座に表示;
* **キャリブレーション済み**の P50 か実経過時間が閾値を越えたとき;
* `calibrating` 中は実経過時間と「calibrating」だけを表示(区間と過大推定警告は隠す);
* 通常のチャットは計時も推定も開始しません。

プロトコル、スレッド境界、エラー処理:[../../docs/integration.md](../../docs/integration.md)。

---

## テレメトリの記録(自己修正が成立する仕組み)

上の仕組みは、実測の所要時間がストアに残らなければ何も学習しません。`ElapsedHooks` はウィジェットを描くだけで、`TelemetryStore` は検証済みの行を追記するだけです。`TelemetryRecorder` はその 2 つをつなぐアダプタであり、「良い見積もり」と「良くなっていく見積もり」の違いそのものです:

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

アダプタに組み込んだ判断と、それぞれの理由:

* **ステップは即座に書き込みます**——タスク終了時ではありません。ホストのクラッシュやユーザーのキャンセルがあっても、既に起きたステップは完全な観測であり、まさに打ち切り(censoring)処理が必要とするデータです。
* **`cancel()` はタスクを打ち切りとして記録します**: キャンセルされたタスクの実時間は下界にすぎないため、件数・カバレッジ・受け入れに入れてはいけません。ステップはステップ長の推定には寄与します。
* **`fail()` は完全な観測です**: タスクは実際に終わり、実際にその時間を使っています。
* **特徴量のないタスクも記録します**(`g_pred` = `{}`)。分類器がまだ走っていないという理由で実測時間を落とすと標本が偏ります。代わりに `offline.partition.partition_tasks` がその行を隔離(quarantine)し件数に入れるので、スキップは黙って消えずに見えます。

ホストなしでループ全体を試せます:

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```

---

## オフラインコマンド

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

`report` は `--store data --estimate out/estimate.json --actual 500` も受け付け、着地点、直近 100 完了タスクの P90 上側カバレッジ、`g_pred.task_class` 別誤差表、ドリフトバッジを出します。HTML はデータを内蔵し外部読み込みなし。

| 終了コード | 意味 |
|---|---|
| `0` | 成功 |
| `1` | 受け入れゲートによる棄却 |
| `2` | データ不足(JSON に不足分と記述的チェックを列挙) |
| `3` | その環境のテレメトリがない |
| `4` | ロールバックコピーがない |

---

## テレメトリ契約

2 つの JSONL、追記専用。派生量(`X`、`n_j`)は**保存しません**。

* ステップイベント:`task_id`、`op`、`t_start`、`t_end`、`ok`、`tool`、`v`
* タスク記録:`task_id`、`g_pred`、`g_pred_probs`、`g_final`、`total_wall`、`censored`、`model_id`、`host`、`estimator_wall`、`g_schema_version`、`v`

コードが強制する規則:

* `t_end > t_start`、有限なタイムスタンプ、`op` は既知集合、`(task_id, op, t_start)` の重複なし;
* 違反行は `rejects.jsonl` へ。黙って捨てない(捨てるのは超長・異常行で、平均が打ち切りと同方向に偏る);
* フィットは `g_pred` のみ読む。`g_final` はラベル漏洩(改竄して 2 回フィットし係数一致を要求するテストあり);
* キャンセルタスクの完了ステップはステップ長フィットに入るが、件数・カバレッジ・受け入れからは除外(総時間は下界);
* タスク ID は一意で、ステップとタスクはこれで結合する。

---

## 検証

```bash
python3 -B -m unittest discover -s tests -t .     # 267 tests
```

合成テレメトリのエンドツーエンド(400 タスク、seed 61、環境 `m-a|h-a`、15047 ステップイベント、351 完了タスク、32 ホールドアウト):

| 項目 | 結果 |
|---|---|
| 受け入れゲート | accepted |
| P50 / P80 / P90 上側カバレッジ | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| 候補のペア pinball | 242.8 |
| 事前分布のペア pinball | 481.9 |
| ペア差 / z | −239.0 / −2.77 |
| フィット後の φ | 1.0(下側クリップ:表自身のばらつきがホールドアウトを覆う) |

`offline/synth.py` の隠れた真値に対するステップ長の回収:

| op | μ フィット / 真値 | σ フィット / 真値 | ステップ観測数 |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

5000 経路の 1 回の予測はノート PC で約 **5 ミリ秒**。数字は内部整合と真値回収の証拠で、実ログの証拠では**ありません**。詳細:[../../docs/validation.md](../../docs/validation.md)。

---

## 約束しないこと

* **実タスクの真値はない。** 実ログは同梱されず、`constants.json` は事前分布。
* **直列の総和は wall time ではない。** 重なる呼び出し、入れ子タイマー、並列分岐、未記録の待ち行列はモデルを壊す。
* **条件付き独立は仮定であり発見ではない。** 再試行相関は未モデル化で、φ はばらつきのつまみ。
* **パラメータは点推定。** 件数回帰と分類器の不確かさは伝播しない。
* **キャンセルはフィットを偏らせ得る**(所要時間と相関する場合)。生存モデルはない。
* **モンテカルロ分位数は標本誤差を伴う** — 報告はするが誤差である。
* **KS の p 値は漸近的。** 診断であって証明ではない。
* **分類器はホストのもの。** クラス確率が較正されていなければ混合もずれる。

---

## ドキュメント

| 文書 | 内容 |
|---|---|
| [../../docs/design.md](../../docs/design.md) | データ境界、統計的定義、受け入れ基準 |
| [../../docs/embedded-widget.md](../../docs/embedded-widget.md) | ホスト TUI / ネイティブ UI への組み込み |
| [../../docs/integration.md](../../docs/integration.md) | 統合コード、イベントプロトコル、スレッド境界 |
| [../../docs/accuracy-audit.md](../../docs/accuracy-audit.md) | 導出、キャリブレーション修正、精度主張の限界 |
| [../../docs/math-upgrade.md](../../docs/math-upgrade.md) | v1.1 の式の更新:導出、検証、非目標 |
| [../../docs/implementation-notes.md](../../docs/implementation-notes.md) | 実装詳細と初期草稿への修正 |
| [../../docs/validation.md](../../docs/validation.md) | 再現可能な受け入れ記録 |
| [../../CHANGELOG.md](../../CHANGELOG.md) | リリース履歴 |
| [../../CONTRIBUTING.md](../../CONTRIBUTING.md) | 開発フロー、テストと翻訳の規則 |
| [../../SECURITY.md](../../SECURITY.md) | 脅威モデルと脆弱性の報告方法 |

## ライセンス

[MIT](../../LICENSE) © agent-elapsed contributors
