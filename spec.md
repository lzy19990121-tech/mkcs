好，这里给你一段**“可以直接贴给 Claude Code 的启动总结指令”**。这段话的目标只有一个：
让它像一个守纪律的工程师，在 Git 驱动下，用 skills 思路搭交易 agent 的地基，而不是乱发挥。

🚀 Claude Code 启动总结指令（直接可用）

项目目标
我正在搭建一个 可扩展的自动交易 agent（美股 + 港股），采用 “skills + agent 编排” 架构。
重要原则：LLM 不参与实时下单决策，只负责盘后/低频的策略与配置生成；交易时段由纯 Python 代码执行。

🧱 总体架构约束（必须遵守）

使用 Git 驱动开发

所有修改只发生在当前分支

每完成一个小模块就 commit（写清晰 commit message）

不要自动 push

采用 skills 模块化设计

每个 skill 都是可替换、可测试的独立模块

skill 之间只通过结构化数据通信（禁止隐式耦合）

统一数据模型（这是系统地基）

Bar / Quote / Signal / OrderIntent / Trade / Position

上层逻辑不得依赖具体券商或数据源

🧩 当前阶段的开发范围（不要越界）

只做系统地基，不涉及真实交易策略优劣：

不接真实券商 API（先用 paper broker）

不做高频 / tick 级交易

不让 LLM 参与实时下单路径

📁 期望的 Repo 结构（可按需微调）
core/
  models.py          # 统一数据结构（dataclass）
skills/
  market_data/       # 数据源适配（先用 mock / csv）
  session/           # 交易时段与日历
  strategy/          # 规则策略（占位即可）
  risk/              # 风控与仓位
broker/
  paper.py           # 模拟交易 broker
agent/
  runner.py          # 编排主循环
storage/
  db.py              # sqlite / 简单持久化
reports/
  daily.py           # 盘后复盘报告

🔧 技术选择约束

语言：Python

数据结构：dataclass

存储：sqlite（或文件）

不引入复杂框架

每个模块要有最小自测方式（哪怕是 main 函数）

🧠 Agent / Skill 职责边界（非常重要）

skills：只做“确定性计算”

agent：负责任务编排、状态管理

LLM（你）：只参与

盘后分析

参数 / 策略配置生成

异常复盘

✅ 当前第一阶段目标（MVP）

跑通一条完整闭环：
Market Data → Strategy → Risk → Paper Broker → Trade Log → Daily Report

新增约束（当前架构）：
- 使用回放引擎推进时间（ReplayEngine + RunContext）
- 订单必须通过 submit_order / on_bar 撮合，禁止同bar成交
- 事件阶段需包含 order_submit / order_fill / order_reject

哪怕策略很弱，只要流程完整即可。

📝 输出要求

所有代码放在 repo 内

写清楚 docstring / README

每一步说明“为什么这样设计”

若需要做假设，必须显式写出来

请从创建项目基础骨架开始。
