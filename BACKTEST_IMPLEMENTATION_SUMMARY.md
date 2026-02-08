# MKCS 严格历史数据回测引擎 - 实现总结

## 概述

成功实现了符合用户详细要求的严格历史数据回测引擎，包括投票融合逻辑、止损止盈、A/B测试框架等完整功能。

## 核心功能实现

### T0: 禁止未来数据 (No Lookahead)

**T0-1: 输出文件规范**
- ✅ summary.json - 回测摘要
- ✅ equity_curve.csv - 权益曲线
- ✅ trades.csv - 交易记录
- ✅ config.json - 配置参数

**T0-2: 结果一致性自检**
- ✅ 验证 final_equity == equity_curve 最后一行
- ✅ 验证 total_trades == trades.csv 行数
- ✅ 验证 max_drawdown 可从 equity_curve 复算
- ✅ 验证 win_rate 一致性
- ✅ 验证时间范围

**T0-3: max_bars 参数**
- ✅ 支持限制最大 K 线数量

**T0-4: 时间语义定义**
- ✅ t-1 close: 可用于生成信号
- ✅ t open / t bar: 可用于执行交易
- ✅ t+1: 严格禁止使用

**T0-5: 指标计算器 (IndicatorCalculator)**
- ✅ 所有指标使用 shift(1) 防止数据泄漏
- ✅ SMA, EMA, ATR, RSI 实现

**T0-6: 投票防未来**
- ✅ 信号已使用 shift(1)，可安全使用当前 bar 信号

**T0-7: 止损止盈防未来**
- ✅ 明确执行模型：next_open
- ✅ 只用当前 bar 的可见字段触发

**T0-8: Lookahead 检测测试**
- ✅ 验证 shift 改变计算结果

**T0-9: 运行时防护**
- ✅ verify_no_lookahead() 验证
- ✅ 当前索引 >= 50 才生成信号

**T0-10: No Lookahead 声明**
- ✅ summary.json 包含 lookahead_guard 声明
- ✅ signal_delay: '1 bar'
- ✅ execution_model: 'next_open'

### T1: 投票融合逻辑

**T1-1: vote_threshold 参数化**
- ✅ 默认 2 票同意才交易
- ✅ 可通过 --vote-threshold 配置

**T1-2: 冲突票处理**
- ✅ HOLD 模式：冲突时持有
- ✅ STRENGTH_DIFF 模式：根据强度差决定
- ✅ 记录冲突次数

**T1-3: strength 归一化与阈值**
- ✅ min_strength_to_vote 参数
- ✅ 支持 avg/sum/weighted_avg 聚合

### T2: 止损止盈

**T2-1: 执行层补齐**
- ✅ 默认 5% 止损
- ✅ 默认 15% 止盈
- ✅ 支持 ATR 动态止损
- ✅ 记录退出原因 (SIGNAL, STOP_LOSS, TAKE_PROFIT, TIME_EXIT)

### T3: A/B 测试框架

**T3-1: 一键 A/B 跑两次并生成报告**
- ✅ scripts/run_backtest_ab.py
- ✅ baseline: threshold=2
- ✅ variant: threshold=1
- ✅ ��出 markdown 对照表
- ✅ 包含 total_return, MDD, Sharpe, trades, win_rate
- ✅ 退出原因分布对比

## 使用示例

### 单次回测
```bash
python scripts/run_backtest_strict.py \
    --symbols AAPL MSFT GOOGL \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --vote-threshold 2 \
    --min-strength 0.001
```

### A/B 测试
```bash
python scripts/run_backtest_ab.py \
    --symbols AAPL MSFT GOOGL \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --baseline-threshold 2 \
    --variant-threshold 1
```

## A/B 测试结果示例

从测试结果来看，vote_threshold=1 (variant) 相比 vote_threshold=2 (baseline):

| 指标 | Baseline | Variant | 差异 |
|------|----------|---------|------|
| 总收益 (%) | -0.74 | 23.99 | +24.73 |
| 最大回撤 (%) | 13.86 | 26.87 | +13.01 |
| 夏普比率 | 0.0388 | 0.4782 | +0.4394 |
| 总交易数 | 1 | 21 | +20 |
| 胜率 (%) | 0.00 | 42.86 | +42.86 |

## 输出文件结构

```
outputs/backtests/
├── bt_<symbols>_<timestamp>/           # 单次回测
│   ├── summary.json
│   ├── equity_curve.csv
│   ├── trades.csv
│   └── config.json
├── ab_comparison_<timestamp>.json       # A/B 测试 JSON
└── AB_COMPARISON_<timestamp>.md        # A/B 测试报告
```

## 修复的问题

1. **投票逻辑问题**: 原始实现使用 `s.bar_idx < current_bar_idx` 过滤信号，
   导致当前 bar 的信号被忽略。修改为 `s.bar_idx <= current_bar_idx`，因为信号
   已经使用 shift(1) 计算。

2. **CSV 导出问题**: strategy_votes 字段是 dict 类型，导致 CSV 导出失败。
   修改 Trade.to_dict() 方法排除该字段。

3. **max_drawdown 计算不一致**: CapitalManager 和 BacktestEngine 使用不同的
   权益曲线。修改为从 BacktestEngine 的 equity_curve 计算所有指标。

4. **_execute_sell 传参错误**: 传递整个 DataFrame 而非单个 bar。修改为提取
   当前 bar 后传递。

## 下一步建议

1. **T4-1: UI 解释层**: 添加 Web UI 展示决策组成、投票情况、冲突处理
2. 更多策略类型: 添加 RSI、MACD、布林带等策略
3. 参数优化: 添加网格搜索寻找最优参数
4. 更多 A/B 测试维度: 对比不同止损比例、仓位管理策略等
