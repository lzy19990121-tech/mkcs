# MKCS 更新日志 (release1.0)

## 2026-02-10 - SPL-7a Bug 修复

### Bug 修复

#### 1. SPL-7a TrendAlert 属性 (`analysis/online/risk_monitor.py`)
- **问题**: `'TrendAlert' object has no attribute 'slope'`
- **修复**: 更新为使用新的 `TrendAlert` 属性结构
  ```python
  # 旧代码
  "slope": alert.slope,
  "r_squared": alert.r_squared,

  # 新代码
  "magnitude": alert.magnitude.value,
  "current_value": alert.current_value,
  "threshold": alert.threshold,
  ```

#### 2. RiskEventStore 方法名
- **问题**: `'RiskEventStore' object has no attribute 'store_event'`
- **修复**: `store_event()` → `write_event()`

#### 3. AlertingManager trends 字典访问 (`analysis/online/alerting.py`)
- **问题**: `trends["volatility"].slope` 访问失败
- **修复**: 使用字典键访问并映射 direction 值
  ```python
  direction_map = {"UP": 1, "DOWN": -1, "NEUTRAL": 0}
  metrics["volatility_trend"] = float(direction_map.get(trend.get("direction"), 0))
  ```

### 相关提交

- `57ab658` - Fix: SPL-7a TrendAlert 属性���和 RiskEventStore 方法名

---

## 历史记录

详见 `CHANGELOG.phase1.md`
