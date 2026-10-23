# agent-elapsed

嵌入 Agent 自身 TUI / 原生 UI 的长任务用时组件。组件提供任务 hook、状态数据和 TUI 文本渲染，不包含网页、HTTP 服务或独立窗口。用户安排任务时开始计时；普通聊天与短任务默认隐藏，完成、取消或失败后自动收起。

运行时仅依赖 numpy，使用复合泊松 × 对数正态采样（默认 5000 条路径），不调用 LLM、不拟合、不读遥测、不访问网络。语义分类调用与真实日志适配由宿主 Agent 提供。

## 环境与测试

需要 Python ≥3.11 与 numpy ≥2.0；测试只用标准库 `unittest`，不必安装项目本身。

```bash
python3 -c "import numpy; print(numpy.__version__)"   # 需要 ≥2.0；没有就 pip install numpy
python3 -m unittest discover -s tests -t . -v
```

下面所有命令都在仓库根目录执行；`offline.*` / `examples/` 的 `sys.path` 已指向根目录，无需额外配置。

## 直接嵌入宿主

查看 TUI 宿主中的完整生命周期：

```bash
python3 examples/tui_host.py            # --plain 打印不带终端控制码的帧
```

- Python 宿主：`ElapsedHooks` 提供任务开始、完成、取消、失败 hook，后台计算一次预估；`render_tui(state)` 返回宿主布局中的文本块。
- 原生 UI 宿主：订阅 `WidgetState`，把任务列表绑定到宿主自己的控件；空列表时折叠该区域。
- 其他语言：启动常驻 `python -m agent_elapsed.bridge`，通过逐行 UTF-8 JSON 发送任务事件、接收组件状态。无需增加 Python 以外的运行时依赖。

默认长任务门槛为 **60 秒，可配置**。宿主明确标记 `long_task=True` 时立即展示；否则仅在已校准预估 P50 达标或实际耗时达标时显示。冷启动先验不能区分任务长短，不用它自动触发展示；未校准时只显示真实已用时及“校准中”，隐藏预测区间和超出预估提示。普通聊天事件不启动计时或估算。

接入代码、协议及线程边界见 [docs/integration.md](docs/integration.md)。这些是通用组件和宿主示例；具体 Agent 需要把自己的任务生命周期接到 hook，并在自己的布局中挂载组件。

## 底层预估接口

```python
import json
from agent_elapsed.constants import load_tables, pick_env
from agent_elapsed.estimate import estimate_task

tables = load_tables('constants.json')
table = pick_env(tables, 'my-model|my-host')
g = {
    'task_class': 'patch',
    'touched_files': '2-3',
    'verification_loops': '2',
    'needs_network': False,
    'plan_steps': '4-8',
}
result = estimate_task(g, [('patch', .65), ('debug', .35)], table=table, seed=7)
saved_estimate = json.dumps(result.to_dict(), ensure_ascii=False)
```

`constants.json` 是冷启动先验，不能当作已训练模型。`fit_id`、`seed` 与 `g_schema_version` 应随开始时的预估保存，结束后使用同一份预估复盘。底层接口保留原始先验结果用于诊断；自定义宿主也必须先检查 `result.calibrating`，只有已校准时才用 `agent_elapsed.presentation.format_band(result.p50, result.p90)` 展示预测区间。

数学推导、已修复问题及准确性证据见 [docs/accuracy-audit.md](docs/accuracy-audit.md)；
公式层这轮升级（φ 保均值与解析求根、分位数不确定度、IRLS 数值细节）见 [docs/math-upgrade.md](docs/math-upgrade.md)。闭式矩是在模型假设下正确的计算，分位数是蒙特卡洛近似；没有能够无条件精确预知未来任务耗时的公式。目前只有合成数据验证，没有真实 Agent 的预测准确率。实际已用时来自单调时钟，与预测分开处理。

## 遥测与离线命令

`TelemetryStore(events_path, tasks_path, rejects_path)` 提供追加写接口。步事件记录 `task_id/op/t_start/t_end/ok/tool/v`；任务记录包含 `g_pred/g_pred_probs/g_final/total_wall/censored/model_id/host/estimator_wall/g_schema_version/v`，详细契约见设计文档。失败步骤保留；取消任务的完整步骤参与步长拟合，任务计数、覆盖率和验收比较排除取消任务。每个任务 ID 必须唯一，步骤和任务通过它关联。

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

复盘报告可加 `--store data --estimate out/estimate.json --actual 500`：显示实际落点、最近 100 个完整任务的 P90 上限覆盖率、按 `g_pred.task_class` 分组的误差表及漂移徽章。HTML 内联数据，不依赖服务器或外部资源。`--estimate` 是任务开始时保存的 `EstResult.to_dict()` JSON。

重拟合先按 `(model_id, host)` 隔离环境，再按时间划分；每类操作在拟合侧新增有效观测至少 50，完整任务总数至少 80、拟合任务至少 60。验证集是最后 `max(20, ceil(20%))` 个完整任务。候选必须通过 PIT/KS、覆盖率容差及同任务同种子的配对 pinball 比较，才原子替换当前表，并保留 `constants.prev.json`。拒绝时现有常数和回滚副本保持原样。

退出码：`0` 成功；`1` 验收拒绝；`2` 数据不足（JSON 中列出缺口和描述性检查）；`3` 未找到环境遥测；`4` 没有回滚副本。冷启动在已有环境遥测但样本不足时写出先验，不把候选自动上线。

## 统计与集成说明

- 21 列 one-hot 保留全部水平，截距不惩罚、系数 ridge；计数只使用 `g_pred`。
- 类别混合每条路径只抽一次共享任务类别；总方差包含类别内及类别间两部分，不能用平均泊松率代替混合分布。
- `phi >= 1` 是**保均值的纯离散度**：σ 放大、对数均值同步下移 `(s²−σ²)/2`，`E[X]` 与 φ 无关；
  φ 由留出残差解析求出（模型相对方差对 φ 是单调指数和，二分求根），内点解处解析 CV² 与残差均方精确相等。
  它只修离散度，不修偏差，也不是重试相关性的解。
- 校准使用随机化预测秩处理零步骤原子；离散分位数检查严格下方覆盖率与上限覆盖率夹住名义水平，连续分布下恢复双侧判据。
- 分位数带蒙特卡洛不确定度：`predict()` / `EstResult` 输出 `p50_ci/p90_ci` 与 `p50_se/p90_se`（秩区间，分布无关，
  零步骤原子处自然退化）。5000 条路径下真值表的 P90 区间约 ±89 s——这就是 m 的分辨率，想更窄就加路径。
- 漂移检测对右偏 pinball 采用 `log1p` 后的最近 100 项 EWMA、基线 2σ 和连续两次越界规则；报警后缩短拟合折扣半衰期。
- “P90 覆盖率”指 `actual <= P90`，不是实际落在 P50–P90 显示带中的比例；连续校准分布的该显示带约包含 40% 概率。
- 宿主提供任务意图、特征分类及遥测适配器；新增 hook 已实现后台数学估算和用时记录。若分类依赖模型，模型调用及独立资源配额由宿主安排。

## 文档

| 文件 | 内容 |
|---|---|
| [docs/design.md](docs/design.md) | 数据边界、统计定义、验收判据的设计依据 |
| [docs/embedded-widget.md](docs/embedded-widget.md) | 嵌入宿主 TUI / 原生 UI 的增量设计 |
| [docs/integration.md](docs/integration.md) | 接入代码、事件协议、线程边界 |
| [docs/accuracy-audit.md](docs/accuracy-audit.md) | 数学推导、校准修正、准确性边界（哪里能承诺、哪里不能） |
| [docs/math-upgrade.md](docs/math-upgrade.md) | v1.1 公式升级的推导、验证与明确未做的事 |
| [docs/implementation-notes.md](docs/implementation-notes.md) | 实现细节与早期草稿的修正记录 |
| [docs/validation.md](docs/validation.md) | 可复现的验收记录 |

## 许可

MIT，见 [LICENSE](LICENSE)。
