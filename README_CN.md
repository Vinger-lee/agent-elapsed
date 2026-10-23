# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-267%20passing-22bb33" alt="267 项测试">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="运行时仅 numpy">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="模型">
  <br>
  <b>⏱️ 给 Agent 的长任务一个诚实的预计完成时间 —— 本地计算、毫秒级、不调用任何大模型。</b>
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

`agent-elapsed` 是给"会跑很久的 Agent 任务"用的用时预估组件。宿主安排任务时，它在任务**并行**进行
的同时给出一个"大约 6–14 分钟"的区间；任务结束后，预估与实际被画在同一根轴上，模型慢慢把预估变准。

内部刻意做小：任务 hook、状态协议、TUI 文本渲染、逐行 JSON 桥。没有网页、没有 HTTP 服务、没有独立窗口。
运行时**只需要 numpy** —— 不调用 LLM、不做拟合、不读遥测、不访问网络。

> **重要。** 本仓库的一切验证都来自**合成遥测与独立数值实验**。仓库不含真实任务日志，因此不声明任何
> 真实 Agent 上的实测准确率。`constants.json` 是冷启动先验，不是已训练模型。在拟合表通过校准门之前，
> 组件隐藏预测区间 —— 先验永远不会被包装成"已验证的预测"。

---

## 目录

- [为什么需要它](#为什么需要它)
- [快速开始](#快速开始)
- [完整模型](#完整模型)
- [分位数与它的不确定度](#分位数与它的不确定度)
- [校准与验收门](#校准与验收门)
- [嵌入宿主](#嵌入宿主)
- [记录遥测](#记录遥测让越用越准真正成立的那一段)
- [离线命令](#离线命令)
- [遥测契约](#遥测契约)
- [仓库结构](#仓库结构)
- [验证](#验证)
- [它不承诺什么](#它不承诺什么)
- [文档](#文档)
- [参与开发](#参与开发)
- [许可](#许可)

---

## 为什么需要它

一个只会说"这要跑一会儿"的 Agent 并没有给出预估。有用的预估必须同时满足四个互相拉扯的约束：

1. **并行**。任务还在跑时就要有数字，而不是跑完再补。
2. **廉价**。不能吃掉宿主的 token 预算或自己的墙钟时间 —— 预估比任务本身还慢就毫无意义。
3. **诚实**。过窄的区间比没有区间更糟：它会让用户不再相信之后所有数字。
4. **自校正**。必须能从宿主自己的任务历史里变准，而且不需要额外的数据管道。

这套设计就是这四个约束的直接后果。一次性的语义分类（大模型可以做）加上闭式矩 + 抽样的统计模型
（大模型不该做）把问题切开：宿主提供任务特征与类别概率，这个组件独占所有算术。因为模型是一个小的参数族，
矩是闭式的、分位数来自固定条数的蒙特卡洛路径、给定种子整次预测完全确定 —— 可复现、可审计，
在笔记本上约五毫秒。

---

## 快速开始

需要 Python ≥3.11 与 numpy ≥2.0。测试只用标准库。

```bash
python3 -c "import numpy; print(numpy.__version__)"   # 需要 ≥2.0
python3 -m unittest discover -s tests -t . -v          # 267 项
```

或者装成包直接用脚本：

```bash
pip install agent-elapsed
agent-elapsed-offline init --constants constants.json      # 写出随包分发的冷启动先验
agent-elapsed-offline synth --out data --tasks 400         # 生成带真值的合成遥测
agent-elapsed-offline refit --store data --constants constants.json --env 'm-a|h-a'
agent-elapsed-bridge --constants constants.json --env default   # 给其他语言的常驻桥
```

在终端里看完整宿主生命周期：

```bash
python3 examples/tui_host.py            # 加 --plain 可打印不带终端控制码的帧
```

启动可被非 Python 宿主驱动的常驻桥：

```bash
python3 -m agent_elapsed.bridge --constants constants.json --env default --long-task-seconds 60
```

直接预估一个任务：

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

result.p50, result.p90, result.expected      # 分位数与解析均值
result.p50_se, result.p90_se                 # 各分位数的蒙特卡洛标准误
result.calibrating                           # True => 现在还不该展示区间
saved = json.dumps(result.to_dict(), ensure_ascii=False)
```

请把 `fit_id`、`seed` 与 `g_schema_version` 和"当时展示给用户的那个预估"一起落盘。之后就能用同一条预估
对真实耗时复盘 —— 这才是离线校准可信、而不是事后挑选的原因。

---

## 完整模型

一个任务是若干步骤的**串行**链，步骤归入四类操作：`llm`、`tool`、`build`、`net`。给定特征向量 `g`
与隐类别 `C = c`：

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

假设写清楚：给定类别后各操作类型相互独立；同一操作类型的步长独立同分布且与步数独立；所有步骤串行、互不重叠；
参数与类别权重在一次预测内固定。

**φ 是纯离散度参数。** 放大 σ 的同时把对数均值下移恰好一半的增量，对数正态的均值就不动：

```
E[X_j]   =  exp(μ_j + σ_j²/2)          与 φ 无关
E[X_j²]  =  E[X_j]² · exp(s_j²)        随 φ 增长
```

这很重要，因为要修的是**离散度**而不是**水平**。早期版本放大 σ 却不补偿 μ，于是 φ 声称只加宽区间，
实际却把预测均值和所有上分位数一起推高了。

因为步数是泊松且与步长独立：

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

注意是二阶矩而不是方差：复合泊松的方差是 `λ·E[X²]`。

类别不确定性用全方差公式处理，**每条任务路径只抽一次隐类别**，所有操作类型共用它：

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            类别内
       + Σ_c p_c (M_c − E[T])²  类别间
```

把混合压成"平均率 + 单个泊松"会丢掉类别间那一项。`tests/test_math_audit.py` 里可手算的夹具
（每步恰好 1 秒，四类操作的类别率 1 与 9，权重 ½/½）真值均值 20、真值方差 **276**，
而平均率模型只给出 20 —— 丢掉共享类别的相关性会让分布窄得离谱：近一半路径落在 10 秒以内，
平均率模型只有约 1%。

混合还意味着操作类型在**无条件**下并不独立：类别诱导的协方差通过"对整个类别均值 `M_c` 求平方"被保留下来。

**φ 的值怎么来。** 只有二阶矩随 φ 变化，且只经 `exp(φ·σ_j²)` 进入，所以模型的平均相对方差是单调指数和：

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ 类别间相对方差 ]          ← 与 φ 无关
                             + Σ_j B_j · exp(φ · σ_j²)
```

其中 `B_j = mean_t[ λ̄_tj · E[X_j]² / E[T_t]² ]`。目标是留出残差
`r = mean_t[ ((y_t − E[T_t])/E[T_t])² ]`，在 `[1, 25]` 上二分求根：

* `f(1) ≥ 0` → φ = 1（表已经够散；φ 不允许把区间收窄到步长噪声以下）；
* `f(25) ≤ 0` → φ = 25（夹住，并且不假装已经修好）；
* 否则取内点解 —— 此处解析 CV² 与残差均方**精确相等**。

等 σ 时有闭式 `φ = 1 + ln(r)/σ²`，用来交叉验证。

---

## 分位数与它的不确定度

分位数是蒙特卡洛：默认 5000 条路径，任何地方都不做正态近似。零步骤处的原子被显式处理：

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

所以预测分布不是连续的。因此 `predict()` 在 `p50`、`p90` 之外还会给出一个分布无关的区间，
它由二项精确秩检验反解而来：落在总体分位数以下的样本数服从 `Binomial(m, τ)`，于是

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # 次序统计量，截断到 0..m−1
```

不需要密度估计，而这恰恰是核密度/标准误公式失效的地方：在原子处区间自然退化成一点，
而不是编造一条很窄的带。

实测行为（`tests/test_math_upgrade.py`，m = 2000，300 次独立重复）：

| τ | 名义 | 实测覆盖 |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

真值表 m = 5000 时，P50 区间约 ±22 秒（标准误 ≈ 11 秒），P90 约 ±89 秒（标准误 ≈ 46 秒）。
这就是 5000 条路径的分辨率；想要更窄的 P90 只能加路径，不能靠聪明。重尾是原因所在：
φ = 4 时样本**方差**在 12 万条路径下只到解析值的 83 %，40 万条才到 96 %，而分位数收敛快得多。

---

## 校准与验收门

拟合表从不盲信。候选表必须在**按时间切分**的留出集上打赢线上表，才会被原子替换。

### 量什么

* **随机化预测秩（PIT）。** 观测 `y` 对 `m` 条预测样本，
  `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)`，`u ~ U(0,1)` 独立。在可交换性下 `R` 均匀，
  **即便有并列值、即便分布完全退化** —— 这正是零原子也能安全评估的原因。
* **分位数覆盖夹判据。** 离散分位数的正确条件是 `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)`，不是等式。
  容差取两个标准误 `2√(τ(1−τ)/n)`：定死的 ±0.05 在 n = 20 时只有一个 σ 多一点，会误杀合格模型。
* **PIT 上的 KS**，用它渐近 p 值作诊断：这是近似，不是"数据生成过程为真"的证明。
* **配对 pinball 损失。** 同一任务、同一种子，两个模型各跑一次，比较是配对的。`gate.evaluate`
  同时输出 `paired_diff`、`paired_diff_se`、`paired_z`，让拒绝/接受的理由带上不确定度，
  而不是把一次蒙特卡洛抖动当成胜负。

### 阈值与状态迁移

| 规则 | 取值 |
|---|---|
| 存储中完整任务数 | ≥ 80 |
| 拟合侧任务数 | ≥ 60 |
| 留出集 | `max(20, ceil(20 %))`，永远取最近的一段 |
| 每类操作的新增步观测 | ≥ 50 |
| 离散度 φ | 夹在 `[1, 25]`，二分求根 |
| 验收 | 校准通过 **且** 配对 pinball 不更差（`require_strict` 要求严格更好） |

候选表用临时文件加 `os.replace` 写入，旧表存为 `constants.prev.json`；`rollback` 可恢复。
被拒绝的候选不会改动线上表，也不会动回滚副本。当某个环境有遥测但样本不足时，CLI 写出先验并**明确不**
上线候选 —— "数据不足"是退出码 2，绝不静默通过。

漂移检测跑在给模型打分的同一条 pinball 序列上：对分数取 `log1p`，最近 100 点做 EWMA，
用固定稳定基线的 `2σ` 带，连续两次越界才报警。基线刻意不把正在被检测的数据吸进去，
否则永久漂移会悄悄迁移进基线、把自己藏起来。

---

## 嵌入宿主

三个接入面，同一个生命周期内核：

| 宿主类型 | 接入面 | 入口 |
|---|---|---|
| Python Agent | hook + 文本渲染 | `ElapsedHooks`、`render_tui(state)` |
| 原生 UI（Qt、SwiftUI、WebView…） | 订阅状态 | `WidgetState` → 宿主自己的控件 |
| 其他语言 | 常驻进程，一行一个 JSON | `python -m agent_elapsed.bridge` |

展示门控刻意保守，因为"错的区间"比"不显示"更糟：

* 默认长任务门槛 **60 秒**，可配置；
* 宿主显式 `long_task=True` 时立刻显示；
* 否则要等**已校准**的 P50 达到门槛，或真实已用时达到门槛；
* `calibrating` 为真时只显示真实已用时和"校准中"，隐藏先验区间与超出预估提示；
* 普通聊天永不启动计时或预估。

协议、线程边界、错误处理见 [docs/integration.md](docs/integration.md)。

---

## 记录遥测（让"越用越准"真正成立的那一段）

上面所有内容都不会自己变准，除非真实耗时真的落进库里。`ElapsedHooks` 只负责画组件，
`TelemetryStore` 只负责追加校验过的行，`TelemetryRecorder` 是把前者翻译成后者的适配器——
它决定了这套东西到底是"一个不错的预估"还是"一个会变准的预估"：

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
        step["ok"] = False        # 失败步骤照样记时长
    recorder.finish("t-1")        # 取消用 cancel()，失败用 fail()
```

适配器里每个取舍都有原因：

* **步事件立刻落盘**，不等任务结束。宿主崩了或被用户取消时，已经发生的步骤就是完整观测，
  正是删失处理需要的那批数据。
* **`cancel()` 记删失**：取消任务的总时长只是下界，不能进计数、覆盖率或验收；它的步长仍然有效。
* **`fail()` 是完整观测**：任务确实结束了，也确实花掉了那些时间。
* **没有分类也要落盘**（`g_pred` 为空字典）。因为分类还没跑就丢掉真实耗时会让样本有偏；
  改成让 `offline.partition.partition_tasks` 隔离并计数，跳过这件事就看得见而不是静默发生。

不用宿主也能把整条链路跑一遍：

```bash
python3 examples/record_telemetry.py --tasks 120 --out demo-data   # 录制 + 自动尝试一次 refit
```


---

## 离线命令

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

`report` 还可加 `--store data --estimate out/estimate.json --actual 500`：把真实落点画到同一根轴上、
显示最近 100 个完整任务的 P90 上限覆盖率、按 `g_pred.task_class` 分组的误差表、以及漂移徽章。
HTML 数据内联，不从网络加载任何东西。

| 退出码 | 含义 |
|---|---|
| `0` | 成功 |
| `1` | 验收门拒绝 |
| `2` | 数据不足（JSON 里列出缺口与描述性检查） |
| `3` | 该环境没有遥测 |
| `4` | 没有回滚副本 |

---

## 遥测契约

两张表，都是 JSONL，都是追加写。派生量（`X`、`n_j`）刻意**不落盘**：任何派生值都会和源不一致。

* 步事件：`task_id`、`op`、`t_start`、`t_end`、`ok`、`tool`、`v`
* 任务记录：`task_id`、`g_pred`、`g_pred_probs`、`g_final`、`total_wall`、`censored`、`model_id`、
  `host`、`estimator_wall`、`g_schema_version`、`v`

代码强制而不是只写在文档里的规则：

* `t_end > t_start`、时间戳有限、`op` 在已知集合内、`(task_id, op, t_start)` 不重复；
* 违规行写入 `rejects.jsonl`，绝不静默丢弃 —— 被丢掉的往往正是超长或异常那批，
  静默丢会让均值朝偏低方向系统性偏移，与删失同向；
* 拟合只读 `g_pred`；读 `g_final` 属于标签泄漏，有一条测试用被篡改的 `g_final` 拟合两次并要求系数完全一致；
* 取消任务的已完成步骤进入步长拟合（发生过的步骤就是完整观测），但不进入计数、覆盖率与验收
  （它们只是下界）；
* 任务 ID 唯一，步骤与任务靠它关联。

---

## 仓库结构

```
agent-elapsed/
├── constants.json               # 冷启动先验表（按环境分组）
├── agent_elapsed/               # 运行时：不拟合、不联网（只写遥测）
│   ├── model.py                 #   复合泊松 × 对数正态、矩、采样、分位数
│   ├── features.py              #   g 的 21 列 one-hot 编码，top-2 类别混合
│   ├── constants.py             #   表结构、环境选择、φ 的唯一入口、随包先验
│   ├── estimate.py              #   运行时唯一入口（EstResult）
│   ├── hooks.py                 #   任务生命周期、展示策略、后台预估
│   ├── telemetry.py             #   hooks → store：任务记录与步事件
│   ├── presentation.py          #   "通常 6 分钟 · 九成在 14 分钟内"的唯一实现
│   ├── tui.py                   #   TUI 文本块渲染
│   ├── bridge.py                #   常驻 stdio JSON 协议
│   ├── data/constants.json      #   随包分发的冷启动先验（装完就不需要检出目录）
│   └── schema.py, store.py      #   校验过的遥测契约、追加写存储
├── offline/                     # 拟合、校准、验收门、漂移、合成数据、报告
│   ├── fit_counts.py            #   加权 ridge 泊松回归（IRLS / Newton）
│   ├── fit_step.py              #   对数步长的 Normal-Inverse-Gamma 后验
│   ├── calibrate.py             #   随机化 PIT、覆盖率、KS、pinball
│   ├── gate.py                  #   离散度求解、配对验收、原子替换、回滚
│   ├── drift.py                 #   EWMA 漂移检测、折扣权重
│   ├── partition.py             #   模型读不了的记录：隔离并计数，不静默丢
│   ├── synth.py                 #   隐藏真值生成器（唯一有标准答案的地方）
│   ├── templates/report.html    #   内联数据报告模板（作为包数据分发）
│   └── viz.py, cli.py           #   离线 HTML 报告、init/synth/refit/report/rollback
├── examples/tui_host.py         # 终端里的完整宿主生命周期
├── examples/record_telemetry.py # 整条闭环：hooks → store → refit → report
├── docs/                        # 见下方文档表
└── tests/                       # 267 项测试，仅标准库 unittest
```

---

## 验证

```bash
python3 -B -m unittest discover -s tests -t .     # 267 项
```

合成遥测端到端（400 任务、seed 61、环境 `m-a|h-a`、15047 条步事件、351 个完整任务、32 个留出任务）：

| 项目 | 结果 |
|---|---|
| 验收门 | accepted |
| P50 / P80 / P90 上限覆盖率 | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| 候选配对 pinball | 242.8 |
| 先验配对 pinball | 481.9 |
| 配对差 / z | −239.0 / −2.77 |
| 拟合 φ | 1.0（下界夹取：表自身离散度已覆盖留出残差） |

对照 `offline/synth.py` 隐藏真值的步长回收：

| op | μ 拟合 / 真值 | σ 拟合 / 真值 | 步观测数 |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

单次 5000 路径预估在笔记本上约 **5 毫秒**。

这些数字说明实现内部自洽、能从已知生成器回收真值，**不**构成真实 Agent 日志上的证据。
完整记录见 [docs/validation.md](docs/validation.md)。

---

## 它不承诺什么

把边界写清楚本身是产品的一部分：

* **没有真实任务的正确答案。** 仓库不含真实日志，`constants.json` 是先验。
* **串行求和不是 wall time。** 重叠的工具调用、嵌套计时、并行分支、未记录的排队都会打破模型。
  并行需要执行图，本版本刻意不凭空造一个。
* **条件独立是假设，不是结论。** 重试相关性没有建模；φ 是离散度旋钮，不是相关性模型。
* **参数是点估计。** NIG 后验收缩步长参数，但预测仍用点估计；计数回归与分类器的不确定性没有传播。
* **取消可能给拟合带来偏差**（若取消与耗时相关）；没有生存模型，只有上文那个"区分下界"的切分。
* **蒙特卡洛分位数带抽样误差** —— 现在已经报告出来，但它仍然是误差。
* **KS 的 p 值是渐近的。** 它在 p > 0.05 阈值附近是诊断，不是证明。
* **分类器属于宿主。** 类别概率被当作既定输入；它偏了，混合分布也就偏了。

---

## 文档

| 文件 | 内容 |
|---|---|
| [docs/design.md](docs/design.md) | 数据边界、统计定义、验收判据的设计依据 |
| [docs/embedded-widget.md](docs/embedded-widget.md) | 嵌入宿主 TUI / 原生 UI 的增量设计 |
| [docs/integration.md](docs/integration.md) | 接入代码、事件协议、线程边界 |
| [docs/accuracy-audit.md](docs/accuracy-audit.md) | 数学推导、校准修正、准确性边界 |
| [docs/math-upgrade.md](docs/math-upgrade.md) | v1.1 公式升级的推导、验证与明确未做的事 |
| [docs/implementation-notes.md](docs/implementation-notes.md) | 实现细节与早期草稿的修正记录 |
| [docs/validation.md](docs/validation.md) | 可复现的验收记录 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开发流程、测试与翻译规则 |
| [SECURITY.md](SECURITY.md) | 威胁模型与漏洞上报方式 |

### 语言

`README.md`（English）与 `README_CN.md`（中文）是权威版本；[`docs/i18n/`](docs/i18n/) 下的译文
逐节跟随。代码标识符、特征档位、遥测字段名与 JSON 键刻意**不翻译** —— 翻了会破坏宿主集成。
每种语言文件都由 `tests/test_docs.py` 检查：语言切换、文档表、以及所有被引用的路径必须能落地。

---

## 参与开发

欢迎 issue 与 PR，请先读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

[MIT](LICENSE) © agent-elapsed contributors
