# MKCS Web UI - Demo Guide

## 启动服务

### 后端启动

```bash
# 启动 Flask 后端服务器
python -m web.app --data-dir runs --debug

# 或者指定端口
python -m web.app --data-dir runs --port 5000 --debug
```

后端服务将在 `http://localhost:5000` 启动。

### 前端启动

```bash
# 进入前端目录
cd web/frontend

# 安装依赖（首次运行）
npm install

# 启动开发服务器
npm run dev
```

前端服务将在 `http://localhost:5173` 启动。

## 功能演示流程

### 1. 回测研究 (/runs)

访问 `http://localhost:5173/runs` 查看所有回测运行结果：

- **搜索**: 在搜索框中输入回测 ID 或策略名称进行过滤
- **排序**: 点击表头按日期、收益率等字段排序
- **对比**: 勾选两条记录，点击"对比选中"按钮进入对比页面
- **详情**: 点击任意回测 ID 查看详细信息

### 2. 回测详情 (/runs/:id)

回测详情页包含以下标签页：

- **概览**: 查看关键指标（起始资金、最终权益、收益率、夏普比率、最大回撤等）
- **权益曲线**: 交互式权益曲线图表，支持缩放和查看数值
- **交易**: 交易记录表格，支持按股票、方向、盈亏状态筛选
- **风控拒绝**:
  - 表格视图: 查看所有被风控拒绝的交易记录
  - 统计图表: TopN 拒绝原因柱状图 + 时间线视图
- **文件**: 浏览和预览回测目录中的所有文件

### 3. 策略对比 (/compare)

1. 在回测列表页勾选两条记录
2. 点击"对比选中"按钮
3. 查看对比结果：
   - 关键指标对比卡片（收益率、夏普比率、最大回撤等）
   - 权益曲线叠加对比图
   - 交易统计对比

### 4. 风控中心 (/risk)

访问 `http://localhost:5173/risk` 查看风控状态和事件（开发中）。

### 5. 规则管理 (/rules)

访问 `http://localhost:5173/rules` 管理交易规则版本（开发中）。

## 数据准备

确保 `runs/` 目录下有回测结果数据。每个回测目录应包含：

```
runs/
├── exp_0d71ac54/
│   ├── summary.json         # 回测摘要
│   ├── equity_curve.csv     # 权益曲线
│   ├── trades.csv           # 交易记录
│   ├── risk_rejects.csv     # 风控拒绝记录
│   └── risk_report.md       # 风险报告（可选）
└── ...
```

### summary.json 示例

```json
{
  "backtest_date": "2024-01-15",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "strategy": "MAStrategy",
  "initial_capital": 100000,
  "final_equity": 115000,
  "total_return": 15.0,
  "total_trades": 150,
  "win_rate": 55.0,
  "sharpe_ratio": 1.2,
  "max_drawdown": 8.5,
  "winning_trades": 82,
  "losing_trades": 68
}
```

### equity_curve.csv 格式

```csv
timestamp,equity,cash
2023-01-01 00:00:00,100000,100000
2023-01-02 00:00:00,100500,99500
...
```

### trades.csv 格式

```csv
timestamp,symbol,side,price,quantity,commission,pnl
2023-01-05 09:30:00,AAPL,BUY,150.25,100,1.5,
2023-01-10 14:30:00,AAPL,SELL,155.50,100,1.55,500
...
```

### risk_rejects.csv 格式

```csv
timestamp,symbol,action,reason,confidence
2023-02-15 10:00:00,TSLA,BUY,Position size exceeds limit,0.95
...
```

## 技术栈

### 前端
- React 18.2
- Vite
- Ant Design 5.12
- Zustand (状态管理)
- React Router 6.20
- Lightweight Charts (图表)

### 后端
- Flask
- Flask-SocketIO
- SQLAlchemy

## 开发说明

### 添加新页面

1. 在 `web/frontend/src/pages/` 创建页面组件
2. 在 `web/frontend/src/App.jsx` 添加路由
3. 在 `web/frontend/src/components/Sidebar.jsx` 添加菜单项

### 添加新 API

1. 在 `web/api/` 创建蓝图文件
2. 在 `web/api/__init__.py` 注册蓝图
3. 在 `web/frontend/src/services/api.js` 添加 API 调用函数

## 故障排除

### 前端无法连接后端

确保后端已启动，检查 CORS 配置。

### 回测数据不显示

检查 `runs/` 目录路径和文件格式是否正确。

### 图表不显示

检查控制台错误，可能需要安装 `lightweight-charts` 包：

```bash
cd web/frontend
npm install lightweight-charts
```
