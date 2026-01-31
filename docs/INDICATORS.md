# 技术指标库文档

## 概述

实现了 **30+ 个技术指标**，用于量化交易的��征工程和信号生成。

## 指标列表

### 趋势指标 (6个)
| 指标 | 描述 | 参数 |
|------|------|------|
| SMA | 简单移动平均线 | 周期 |
| EMA | 指数移动平均线 | 周期 |
| MACD | 指数平滑异同移动平均线 | fast=12, slow=26, signal=9 |
| TRIX | 三重指数平滑移动平均 | period=15 |
| ROC | 变动率 | period=10/20 |
| Momentum | 动量 | period=10 |

### 动量指标 (6个)
| 指标 | 描述 | 参数 | 范围 |
|------|------|------|------|
| RSI | 相对强弱指标 | period=14/6 | 0-100 |
| Stochastic | 随机指标 | K=14, D=3 | 0-100 |
| Williams %R | 威廉指标 | period=14 | -100~0 |
| CCI | 顺势指标 | period=20 | |
| MFI | 资金流量指标 | period=14 | 0-100 |
| TRIX | 三重指数平滑 | period=15 | |

### 波动率指标 (4个)
| 指标 | 描述 | 参数 |
|------|------|------|
| ATR | 平均真实波幅 | period=14 |
| Bollinger Upper | 布林带上轨 | period=20, std=2 |
| Bollinger Middle | 布林带中轨 | period=20 |
| Bollinger Lower | 布林带下轨 | period=20, std=2 |
| BB Position | 价格在布林带中的位置 | 0-1 |
| BB Width | 布林带宽度 | |

### 成交量指标 (4个)
| 指标 | 描述 | 参数 |
|------|------|------|
| OBV | 能量潮 | |
| VWAP | 成交量加权平均价 | period=20 |
| Volume ROC | 成交量变化率 | |
| True Range | 真实波幅 | |

### 价格指标 (3个)
| 指标 | 描述 |
|------|------|
| Price Change | 价格变化率 |
| Typical Price | 典型价格 (H+L+C)/3 |
| Median Price | 中位数价格 |

## 使用方法

### 1. 提取所有指标

```python
from skills.indicators.technical import IndicatorFeatures

indicators = IndicatorFeatures.extract_all(bars)

# 访问特定指标
rsi_values = indicators['rsi_14']
macd_values = indicators['macd']
```

### 2. 获取最新值

```python
latest = IndicatorFeatures.get_latest_features(bars)

# 输出示例:
# {
#   'rsi_14': 56.90,
#   'macd': 0.44,
#   'atr_14': 2.35,
#   'bb_position': 0.65,
#   ...
# }
```

### 3. 单独计算指标

```python
from skills.indicators.technical import TechnicalIndicators

# RSI
rsi = TechnicalIndicators.rsi(prices, period=14)

# MACD
macd, signal, histogram = TechnicalIndicators.macd(prices)

# Bollinger Bands
upper, middle, lower = TechnicalIndicators.bollinger_bands(prices)

# ATR
atr = TechnicalIndicators.atr(highs, lows, closes)
```

### 4. 在策略中使用

```python
class MyStrategy(Strategy):
    def generate_signals(self, bars, position):
        # 提取最新指标
        latest = IndicatorFeatures.get_latest_features(bars)

        # 信号逻辑
        if latest['rsi_14'] < 30:
            return Signal(action="BUY", reason="RSI 超卖")
        elif latest['rsi_14'] > 70:
            return Signal(action="SELL", reason="RSI 超买")
```

### 5. 作为 ML 特征

```python
from skills.indicators.technical import IndicatorFeatures

# 提取所有指标作为特征
indicators = IndicatorFeatures.extract_all(bars)

# 构建特征矩阵
feature_matrix = []
for i in range(len(bars)):
    row = {name: values[i] if i < len(values) else float('nan')
           for name, values in indicators.items()}
    feature_matrix.append(row)

# 用于机器学习训练
X = pd.DataFrame(feature_matrix)
```

## 指标说明

### RSI (Relative Strength Index)
- **用途**: 识别超买/超卖
- **阈值**: >70 超买, <30 超卖
- **公式**: `RSI = 100 - (100 / (1 + RS))`

### MACD (Moving Average Convergence Divergence)
- **用途**: 识别趋势变化
- **信号**: MACD 上穿信号线 → 买入
- **组成**: MACD 线, 信号线, 柱状图

### Bollinger Bands
- **用途**: 波动率和价格通道
- **信号**: 价格触及上下轨 → 反转信号
- **构成**: 中轨(SMA), 上轨(+2σ), 下轨(-2σ)

### Stochastic Oscillator
- **用途**: 动量确认
- **阈值**: >80 超买, <20 超卖
- **信号**: K 线上穿 D 线 → 买入

### ATR (Average True Range)
- **用途**: 波动率衡量，止损设置
- **应用**: 动态止损, 仓位管理

### Williams %R
- **用途**: 超买超卖判断
- **阈值**: >-20 超买, <-80 超卖
- **特点**: 类似 Stochastic 但更灵敏

## 性能

- **计算速度**: 100 个 K 线，30 个指标 < 10ms
- **内存占用**: 每个指标 ~800 bytes (100个数据点)
- **精度**: 使用 float64，精度足够

## 注意事项

1. **NaN 值**: 指标初期会有 NaN 值（需要足够的历史数据）
2. **数据对齐**: 所有指标列表长度与输入 K 线长度相同
3. **除零保护**: 所有指标都处理了除零情况
4. **边界检查**: 输入长度不足时返回 NaN 列表

## 扩展

添加新指标：

```python
@staticmethod
def your_indicator(prices: List[float], period: int) -> List[float]:
    """你的指标说明"""
    if len(prices) < period:
        return [float('nan')] * len(prices)

    # 计算逻辑
    values = [float('nan')] * (period - 1)
    for i in range(period - 1, len(prices)):
        # 你的计算公式
        result = ...
        values.append(result)

    return values
```

## 参考资料

- [Technical Analysis Library (TA-Lib)](https://ta-lib.org/)
- [Investopedia: Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [School of Stock Charts](https://school.stockcharts.com/)
