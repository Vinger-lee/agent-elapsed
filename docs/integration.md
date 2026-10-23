# 在 Agent 的 TUI / UI 内接入

组件由宿主任务事件驱动，不单独打开计时窗口，不接管终端，也不在普通聊天中显示。库只依赖 numpy，不包含网页、Web 组件或 HTTP 服务；原生 UI 的控件与布局由宿主提供。

## Python 生命周期 hook

```python
from agent_elapsed.constants import load_tables, pick_env
from agent_elapsed.hooks import DisplayPolicy, ElapsedHooks

table = pick_env(load_tables('constants.json'), 'my-model|my-host')
hooks = ElapsedHooks(table, policy=DisplayPolicy(long_task_seconds=60))

g = {
    'task_class': 'patch', 'touched_files': '2-3',
    'verification_loops': '2', 'needs_network': False, 'plan_steps': '4-8',
}

# 在宿主接受任务时调用；不要等待返回的 Future，主 Agent 同时开始干活。
hooks.on_task_start('session-7/task-1', '修复接口并运行测试',
                    g=g, probs=[('patch', .7), ('debug', .3)], long_task=True)

# 以下三种终结事件选一种，放在主任务实际结束的位置。
outcome = hooks.on_task_finish('session-7/task-1')
# 取消时调用 hooks.on_task_cancel(task_id)，失败时调用 hooks.on_task_fail(task_id)。

hooks.close()  # 宿主会话退出时释放后台资源并清理组件。
```

每个宿主会话共用一个 `ElapsedHooks`，任务 ID 在该会话内唯一。重复开始事件不重置计时或重新估算；重复结束返回 `None`；已结束 ID 不允许用于新任务。并行任务各自计时、各自清理。

`outcome` 包含 `task_id/status/total_wall/estimate/estimator_wall/estimation_error`。`total_wall` 是从任务开始到终结事件的真实秒数，采用单调时钟；`estimate` 保留开始时得到的估算，不在任务结束时重新猜一次。预估尚未完成或失败时它为 `None`，结束 hook 不等待后台计算；迟到结果被丢弃。

这层不自动写遥测。宿主把 `outcome`、开始时的 `g/probs`、自身 `model_id/host` 等信息映射到现有 `TelemetryStore`；中断而未完成的任务标记为删失，不能当作完整任务拟合计数或验证覆盖率。

## 聊天、短任务和长任务

意图判断由已有宿主规划器或任务调度器提供，组件不按关键词猜测，也不为每条聊天额外调用模型：

```python
from agent_elapsed.bridge import handle_event

handle_event(hooks, {'event': 'prompt_received', 'intent': 'chat', 'content': '你好'})
# 不注册任务、不估算、不显示。

handle_event(hooks, {
    'event': 'prompt_received', 'intent': 'task',
    'task_id': 'session-7/task-2', 'title': '整理模块并验证', 'long_task': True,
})
# 没有 g 时也会正常计时，只显示任务状态和已用时。
```

`long_task` 的三种取值：

| 值 | 展示时机 |
|---|---|
| `True` | 宿主已确认是长任务，立即显示，预估在后台补入 |
| `None`（默认） | 已校准 P50 ≥ 门槛时提前显示，或实际用时 ≥ 门槛时显示 |
| `False` | 不凭预估提前显示；原本的短任务实际超过门槛后仍会显示 |

默认门槛为 60 秒。冷启动表不参与自动长短判断：其各类任务的系数尚无差别，不能据此把每个小任务都显示成几分钟。任务一旦结束、取消或失败就收起；“聊天隐藏”不会抹掉同时仍在执行的长任务。

只更新已用时，不推算剩余时间、倒计时或完成百分比。已校准后才显示 P50–P90 预估区间，它不是 90% 概率带；超出 P90 时显示“已超过预估区间”。未校准时仅显示已用时及“校准中”，不提供时间承诺。

## TUI 直接挂入已有布局

```python
from queue import Empty, SimpleQueue
from agent_elapsed.tui import render_tui

updates = SimpleQueue()
unsubscribe = hooks.subscribe(updates.put)

# 在宿主的刷新循环内执行：
state = hooks.refresh()
while True:
    try:
        state = updates.get_nowait()  # 合并初始推送及本帧的更新，绘制最新状态。
    except Empty:
        break
task_region_text = render_tui(state, width=72)
print(task_region_text)  # 实际宿主把文本赋给自身已有的任务区域。

unsubscribe()
```

空状态返回空串；宿主同时折叠该区域的边距。非空时是任务标题和时间信息组成的文本块，支持中文列宽截断，移除标题中的终端控制字符。组件本身不 `print`、不发送 ANSI 光标命令；Textual、Rich 或自制 TUI 可直接消费返回值。

订阅在注册时先推送当前状态，在生命周期变化、预估完成和 `refresh()` 时再次推送。**回调可能来自后台线程**，应像上例先入队，再由 UI 线程绘制；异步框架也可用 `loop.call_soon_threadsafe` 派发。宿主每秒调用一次 `refresh()` 即可。不要在订阅回调里执行耗时工作或等待预估 Future。回调内调用 `close()` 会立即清理组件并取消排队的估算，但不会等待工作线程；让当前回调返回后，正在运行的估算自行退出，避免线程互等。

完整宿主演示：`python examples/tui_host.py`；`--plain` 禁用示例宿主的清屏，`--cancel` 演示取消。

## 原生 UI 状态绑定

原生 UI 与 TUI 使用同一个 `WidgetState`。通过 `hooks.subscribe(...)` 接收更新，在宿主 UI 线程消费 `state.to_dict()['tasks']`：空数组折叠任务区域；非空数组按 `task_id` 更新任务控件。Qt、桌面应用或 Agent 自己的 UI 都使用自身控件绘制，不加载网页。

任务数据包含 `title`、`elapsed_seconds`、`elapsed_label`、`band`、`calibrating` 和 `over_estimate`。其中已用时来自单调时钟，预估来自数学模型。`calibrating=True` 时 `band=None`、`over_estimate=False`；原始 `estimate` 仅供诊断，宿主不能绕过这些门控把先验显示为已验证预测。模型假设、校准方法及真实数据缺口见 [accuracy-audit.md](accuracy-audit.md)。

## 非 Python Agent：常驻 hook 桥

在项目根目录启动并保持进程存活：

```powershell
python3 -m agent_elapsed.bridge --constants constants.json --env default --long-task-seconds 60
```

每行写入一个 UTF-8 JSON 对象，写完立即 flush。例如：

```json
{"v":1,"event":"prompt_received","intent":"chat","content":"你好"}
{"v":1,"event":"task_started","task_id":"s1/t1","title":"修复接口并运行测试","long_task":true,"g":{"task_class":"patch","touched_files":"2-3","verification_loops":"2","needs_network":false,"plan_steps":"4-8"},"probs":[["patch",0.7],["debug",0.3]]}
{"v":1,"event":"task_finished","task_id":"s1/t1"}
{"v":1,"event":"shutdown"}
```

协议输入还支持 `task_cancelled`、`task_failed`、`tick` 和忽略的普通 `message`。`prompt_received` 仅在 `intent` 精确为 `task` 时启动任务。桥每秒自动刷新；坏行（包括 JSON 嵌套过深）输出错误后继续处理下一行。EOF 或 `shutdown` 会清理可见状态并释放资源。

输出有三种事件：

| event | 用途 |
|---|---|
| `elapsed.updated` | `tasks` 数组仅含可见任务，供宿主原生任务控件更新 |
| `elapsed.finished` | `task` 是 `TaskOutcome.to_dict()`，供宿主记录实际用时 |
| `elapsed.error` | `message` 描述输入错误，不中断后续事件 |

Node.js 宿主可以直接使用标准库：

```javascript
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

const bridge = spawn('python', ['-m', 'agent_elapsed.bridge', '--constants', 'constants.json'],
  { cwd: process.cwd(), stdio: ['pipe', 'pipe', 'inherit'] });
createInterface({ input: bridge.stdout }).on('line', line => {
  const event = JSON.parse(line);
  if (event.event === 'elapsed.updated') console.log('宿主组件状态：', event.tasks);
  if (event.event === 'elapsed.finished') console.log('任务实际用时：', event.task.total_wall);
});
bridge.stdin.write(JSON.stringify({ event: 'task_started', task_id: 'node/1',
  title: '运行长任务', long_task: true }) + '\n');
bridge.stdin.write(JSON.stringify({ event: 'task_finished', task_id: 'node/1' }) + '\n');
bridge.stdin.end();
```

示例用 `console.log` 展示传输；实际宿主把事件派发给自己的 TUI/UI。任务工作应发生在 start 与 finish 之间。一个会话共用一个桥进程；若某个 Agent 的 hook 每次启动新进程，需要宿主转发到常驻桥，或在其插件进程内持有 Python 生命周期实例。该组件不会自动修改 Codex、Claude Code 等产品的原生界面。
