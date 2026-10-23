# 长任务嵌入组件设计

本增量落实用户要求：兼容多种 Agent；用户安排任务时通过 hook 开始计时；组件直接挂载在宿主 TUI / UI 中；普通聊天及短任务隐藏。旧版 HTML 保留作离线复盘，不再承担运行时展示。

采用共享生命周期核心、TUI 文本渲染和原生 UI 状态绑定。组件没有网页、HTTP 服务或独立窗口；Python 调用和标准输入输出协议共用行为。先前增加的 Web 自定义元素和网页演示已按用户纠正删除。

## 触发及可见性

- 宿主负责区分聊天和任务，不以关键词猜测用户意图。`prompt_received` 只有 `intent=task` 才等价于 `task_started`；普通 `message` 不启动计时或估算。
- `ElapsedHooks.on_task_start(task_id, title, g=None, probs=(), long_task=None)` 在入口记录单调时钟，后台最多计算一次开始时预估，不阻塞主任务。
- `DisplayPolicy.long_task_seconds` 默认 60 秒，必须是有限正数。宿主 `long_task=True` 立即显示；否则实际耗时达到门槛时显示；已校准预估的 P50 达到门槛也可提前显示。`long_task=False` 禁用预估提前展示，实际超时仍会显示。
- 冷启动先验不参与自动长短判定，因为先验系数为零，尚不能区分工作量。`calibrating=True` 时不展示用时区间或超出预估提示，只显示任务状态、真实已用时及校准标识；原始预估仅保留供诊断。缺少特征或预估失败时继续真实计时。
- 完成、取消、失败都立即移除组件，返回实际用时与开始时保存的预估供宿主记录。迟到预估不允许恢复已结束任务。
- 同时执行的任务按唯一任务 ID 隔离；重复开始事件不重置时钟或重算；结束后复用 ID 拒绝。生命周期实例对应一个宿主会话。

## 组件与协议

`snapshot()` 只包含可见任务；`subscribe()` 推送变化；宿主定时调用 `refresh()` 更新已用时。订阅回调可能来自后台线程，宿主需要派发回自己的 UI 线程。核心不读遥测、不写文件、不访问网络、不调用 LLM，数学入口保持纯函数。

TUI 的 `render_tui(state, width=80)` 返回任务标题和用时信息的文本块；空状态返回空串，不输出终端控制码，不争夺宿主终端。原生 UI 订阅 `WidgetState` 或读取桥的状态，把同一份数据绑定到宿主自己的任务控件；空数组折叠任务区域。没有虚构百分比、倒计时或剩余时间。

非 Python 宿主通过常驻 `python -m agent_elapsed.bridge` 子进程传递逐行 JSON：`task_started`、`task_finished`、`task_cancelled`、`task_failed`、`prompt_received`、`message`、`tick`、`shutdown`。输出 `elapsed.updated` 状态和 `elapsed.finished` 结果，错误为 `elapsed.error`。桥接进程每秒刷新，EOF 或 shutdown 清理组件。每次启动独立进程的 hook 需要宿主把事件转发到同一个常驻桥，不能期待进程间自动共享内存。

## 交付与验收

提供可运行的 TUI 宿主示例，展示聊天、短任务、长任务、完成和取消；原生 UI 使用状态协议接入，不提供网页演示。

验证普通聊天不计算、短任务隐藏、校准和冷启动分支、实际超时兜底、后台不阻塞、重复事件、并行隔离、取消时晚到结果、计算异常、组件隐藏清理、JSON 协议及 TUI 宿主示例。使用标准库 unittest，不新增依赖或虚拟环境。数学准确性结论见 `docs/accuracy-audit.md`。
