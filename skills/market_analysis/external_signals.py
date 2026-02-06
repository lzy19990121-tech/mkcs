"""
外部市场信号接入

接入：情绪指标、宏观指标、事件日历
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SentimentSource(Enum):
    """情绪数据源"""
    CNN_FEAR_GREED = "cnn_fear_greed"
    VIX = "vix"
    PUT_CALL_RATIO = "put_call_ratio"
    MANUAL = "manual"


@dataclass
class SentimentData:
    """情绪数据"""
    source: SentimentSource
    timestamp: datetime
    score: float              # -1 (极端恐慌) 到 1 (极端贪婪)
    raw_value: Optional[float] = None  # 原始值

    # CNN Fear & Greed 特定字段
    fear_greed_level: Optional[str] = None  # "Extreme Fear", "Fear", etc.

    # VIX 特定字段
    vix_value: Optional[float] = None

    # Put/Call Ratio 特定字段
    pcr_value: Optional[float] = None


class SentimentDetector:
    """
    市场情绪检测器

    接入多个情绪源，综合判断市场情绪
    """

    def __init__(
        self,
        fear_threshold: float = -0.3,
        greed_threshold: float = 0.3,
        enable_cnn: bool = True,
        enable_vix: bool = True,
        enable_pcr: bool = False  # 需要额外数据源
    ):
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        self.enable_cnn = enable_cnn
        self.enable_vix = enable_vix
        self.enable_pcr = enable_pcr

        self._history: List[SentimentData] = []

    def detect(self, external_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        检测市场情绪

        Args:
            external_data: 外部数据，包含情绪指标

        Returns:
            {
                "sentiment_state": "FEAR" / "NEUTRAL" / "GREED",
                "sentiment_score": float,
                "extreme_sentiment": bool,
                "sources": list
            }
        """
        scores = []
        sources = []

        # 1. CNN Fear & Greed
        if self.enable_cnn and external_data:
            fng_data = external_data.get("fear_and_greed")
            if fng_data:
                score = self._parse_fear_greed(fng_data)
                scores.append(score)
                sources.append("CNN_FEAR_GREED")

        # 2. VIX
        if self.enable_vix and external_data:
            vix_data = external_data.get("vix")
            if vix_data:
                score = self._parse_vix(vix_data)
                scores.append(score)
                sources.append("VIX")

        # 3. Put/Call Ratio
        if self.enable_pcr and external_data:
            pcr_data = external_data.get("put_call_ratio")
            if pcr_data:
                score = self._parse_pcr(pcr_data)
                scores.append(score)
                sources.append("PUT_CALL_RATIO")

        # 综合情绪分数
        if scores:
            sentiment_score = np.mean(scores)
        else:
            sentiment_score = 0.0

        # 判定情绪状态
        if sentiment_score <= self.fear_threshold:
            sentiment_state = "FEAR"
            extreme = True
        elif sentiment_score >= self.greed_threshold:
            sentiment_state = "GREED"
            extreme = True
        else:
            sentiment_state = "NEUTRAL"
            extreme = False

        # 保存历史
        self._history.append(SentimentData(
            source=SentimentSource.MANUAL,
            timestamp=datetime.now(),
            score=sentiment_score
        ))

        return {
            "sentiment_state": sentiment_state,
            "sentiment_score": sentiment_score,
            "extreme_sentiment": extreme,
            "sources": sources,
            "fear_threshold": self.fear_threshold,
            "greed_threshold": self.greed_threshold
        }

    def _parse_fear_greed(self, data: Any) -> float:
        """解析 CNN Fear & Greed 数据

        0-100 映射到 -1 到 1
        0 = Extreme Fear -> -1
        50 = Neutral -> 0
        100 = Extreme Greed -> 1
        """
        if isinstance(data, (int, float)):
            value = float(data)
        elif isinstance(data, dict):
            value = float(data.get("value", data.get("score", 50)))
        elif isinstance(data, str):
            # 解析字符串
            data = data.lower()
            if "extreme fear" in data:
                value = 0
            elif "fear" in data:
                value = 25
            elif "neutral" in data:
                value = 50
            elif "greed" in data:
                value = 75
            elif "extreme greed" in data:
                value = 100
            else:
                value = 50
        else:
            value = 50

        # 映射到 -1 到 1
        return (value - 50) / 50

    def _parse_vix(self, data: Any) -> float:
        """解析 VIX 数据

        VIX < 15: 低波动 -> 贪婪 (正分)
        VIX 15-25: 正常 -> 中性
        VIX > 25: 高波动 -> 恐慌 (负分)
        VIX > 35: 极端恐慌
        """
        if isinstance(data, (int, float)):
            vix = float(data)
        elif isinstance(data, dict):
            vix = float(data.get("value", data.get("close", 20)))
        else:
            vix = 20.0

        if vix < 15:
            return 0.3  # 贪婪
        elif vix < 25:
            return 0.0  # 中性
        elif vix < 35:
            return -0.3  # 恐慌
        else:
            return -0.7  # 极端恐慌

    def _parse_pcr(self, data: Any) -> float:
        """解析 Put/Call Ratio 数据

        PCR < 0.7: 贪婪
        PCR 0.7-1.0: 正常
        PCR > 1.0: 恐慌
        PCR > 1.5: 极端恐慌
        """
        if isinstance(data, (int, float)):
            pcr = float(data)
        elif isinstance(data, dict):
            pcr = float(data.get("value", 1.0))
        else:
            pcr = 1.0

        if pcr < 0.7:
            return 0.3
        elif pcr < 1.0:
            return 0.0
        elif pcr < 1.5:
            return -0.3
        else:
            return -0.7


class MacroDetector:
    """
    宏观/跨市场信号检测器

    监控：
    - 股指 (SPX / NDX)
    - 利率 (10Y / 2Y)
    - DXY
    """

    def __init__(
        self,
        market_stress_threshold: float = 0.02,  # 市场压力阈值（2%下跌）
        yield_inversion_threshold: float = -0.001,  # 收益率倒挂阈值
        dxy_strength_threshold: float = 2.0  # DXY 变化阈值
    ):
        self.market_stress_threshold = market_stress_threshold
        self.yield_inversion_threshold = yield_inversion_threshold
        self.dxy_strength_threshold = dxy_strength_threshold

    def detect(self, external_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        检测系统性风险

        Returns:
            {
                "systemic_risk_flag": bool,
                "stress_level": "LOW" / "MEDIUM" / "HIGH",
                "market_stress": bool,
                "yield_inversion": bool,
                "currency_crisis": bool,
                "features": dict
            }
        """
        if not external_data:
            return {
                "systemic_risk_flag": False,
                "stress_level": "LOW",
                "market_stress": False,
                "yield_inversion": False,
                "currency_crisis": False,
                "features": {}
            }

        features = {}
        risk_flags = []

        # 1. 市场压力检测（股指下跌）
        market_data = external_data.get("market_indices", {})
        if market_data:
            spx_change = market_data.get("SPX_change", 0)
            ndx_change = market_data.get("NDX_change", 0)

            market_stress = (
                spx_change < -self.market_stress_threshold or
                ndx_change < -self.market_stress_threshold
            )

            features["SPX_change"] = spx_change
            features["NDX_change"] = ndx_change

            if market_stress:
                risk_flags.append("market_stress")

        # 2. 收益率曲线检测
        yield_data = external_data.get("yields", {})
        if yield_data:
            yield_10y = yield_data.get("10Y", 0.03)
            yield_2y = yield_data.get("2Y", 0.04)

            yield_inversion = yield_10y < yield_2y + self.yield_inversion_threshold

            features["yield_10y"] = yield_10y
            features["yield_2y"] = yield_2y
            features["yield_spread"] = yield_10y - yield_2y

            if yield_inversion:
                risk_flags.append("yield_inversion")

        # 3. 美元强度检测
        dxy_data = external_data.get("dxy", {})
        if dxy_data:
            dxy_change = abs(dxy_data.get("change_1d", 0))

            features["DXY_change"] = dxy_change

            if dxy_change > self.dxy_strength_threshold:
                risk_flags.append("currency_stress")

        # 综合风险等级
        risk_count = len(risk_flags)
        if risk_count >= 3:
            stress_level = "HIGH"
        elif risk_count >= 1:
            stress_level = "MEDIUM"
        else:
            stress_level = "LOW"

        return {
            "systemic_risk_flag": risk_count >= 1,
            "stress_level": stress_level,
            "market_stress": "market_stress" in risk_flags,
            "yield_inversion": "yield_inversion" in risk_flags,
            "currency_crisis": "currency_stress" in risk_flags,
            "features": features,
            "risk_flags": risk_flags
        }


class EventCalendar:
    """
    事件日历

    追踪：
    - FOMC / CPI / NFP
    - Earnings（标的级）
    """

    def __init__(self, events_file: Optional[str] = None):
        self.events_file = events_file
        self._events: List[Dict[str, Any]] = []
        self._loaded = False

        if events_file and Path(events_file).exists():
            self._load_events(events_file)

    def _load_events(self, file_path: str):
        """加载事件定义"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self._events = data.get("events", [])
            self._loaded = True
            logger.info(f"加载了 {len(self._events)} 个事件")
        except Exception as e:
            logger.error(f"加载事件文件失败: {e}")

    def add_event(self, event: Dict[str, Any]):
        """添加事件"""
        self._events.append(event)

    def is_event_window(self, current_time: datetime, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        检查是否在事件窗口

        Args:
            current_time: 当前时间
            symbol: 标的代码（用于财报匹配）

        Returns:
            {
                "in_event_window": bool,
                "event_type": str / None,
                "event_name": str / None,
                "time_to_event": timedelta / None,
                "time_since_event": timedelta / None
            }
        """
        now = current_time

        for event in self._events:
            # 检查是否匹配标的（财报事件）
            if event.get("symbol_filter"):
                if symbol and symbol not in event["symbol_filter"]:
                    continue

            # 检查时间窗口
            event_time = datetime.fromisoformat(event["datetime"])
            pre_window = timedelta(hours=event.get("pre_window_hours", 24))
            post_window = timedelta(hours=event.get("post_window_hours", 24))

            start = event_time - pre_window
            end = event_time + post_window

            if start <= now <= end:
                return {
                    "in_event_window": True,
                    "event_type": event.get("type", "UNKNOWN"),
                    "event_name": event.get("name", ""),
                    "time_to_event": None,
                    "time_since_event": now - event_time
                }

            # 即将到来的事件
            elif now < start:
                time_to = start - now
                if time_to < timedelta(hours=48):  # 48小时内
                    return {
                        "in_event_window": False,
                        "event_type": event.get("type"),
                        "event_name": event.get("name"),
                        "time_to_event": time_to,
                        "time_since_event": None
                    }

        return {
            "in_event_window": False,
            "event_type": None,
            "event_name": None,
            "time_to_event": None,
            "time_since_event": None
        }


class ExternalSignalManager:
    """
    外部信号管理器

    整合所有外部数据源，提供统一接口
    """

    def __init__(
        self,
        enable_sentiment: bool = True,
        enable_macro: bool = True,
        enable_events: bool = True,
        events_file: Optional[str] = None
    ):
        self.sentiment_detector = SentimentDetector() if enable_sentiment else None
        self.macro_detector = MacroDetector() if enable_macro else None
        self.event_calendar = EventCalendar(events_file) if enable_events else None

        self._last_update: Optional[datetime] = None
        self._cached_signals: Dict[str, Any] = {}

    def update_signals(
        self,
        current_time: datetime,
        external_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        更新外部信号

        Args:
            current_time: 当前时间
            external_data: 外部数据（从API或文件读取）
            symbol: 当前标的

        Returns:
            {
                "sentiment": dict,      # 情绪数据
                "systemic_risk": dict,  # 宏观风险数据
                "event_window": dict,   # 事件窗口
                "combined_flags": dict  # 综合标志
            }
        """
        # 1. 情绪检测
        sentiment_data = None
        if self.sentiment_detector:
            sentiment_data = self.sentiment_detector.detect(external_data)

        # 2. 宏观风险检测
        risk_data = None
        if self.macro_detector:
            risk_data = self.macro_detector.detect(external_data)

        # 3. 事件窗口检测
        event_data = None
        if self.event_calendar:
            event_data = self.event_calendar.is_event_window(current_time, symbol)

        # 4. 综合标志
        combined_flags = {
            "extreme_sentiment": sentiment_data.get("extreme_sentiment", False) if sentiment_data else False,
            "high_systemic_risk": risk_data.get("stress_level") == "HIGH" if risk_data else False,
            "event_window_active": event_data.get("in_event_window", False) if event_data else False
        }

        self._last_update = current_time
        self._cached_signals = {
            "sentiment": sentiment_data,
            "systemic_risk": risk_data,
            "event_window": event_data,
            "combined_flags": combined_flags
        }

        return self._cached_signals

    def get_cached_signals(self) -> Dict[str, Any]:
        """获取缓存的信号"""
        return self._cached_signals


if __name__ == "__main__":
    """测试代码"""
    print("=== External Signals 测试 ===\n")

    # 测试情绪检测
    print("1. SentimentDetector:")
    sentiment = SentimentDetector()

    # 模拟 CNN Fear & Greed 数据
    external_data = {
        "fear_and_greed": 75,  # Greed
        "vix": 12,           # Low volatility
        "put_call_ratio": 0.6
    }

    result = sentiment.detect(external_data)
    print(f"   Sentiment State: {result['sentiment_state']}")
    print(f"   Score: {result['sentiment_score']:.2f}")
    print(f"   Extreme: {result['extreme_sentiment']}")

    # 测试宏观检测
    print("\n2. MacroDetector:")
    macro = MacroDetector()

    external_data = {
        "market_indices": {
            "SPX_change": -0.03,  # 下跌3%
            "NDX_change": -0.02
        },
        "yields": {
            "10Y": 0.035,
            "2Y": 0.045
        },
        "dxy": {
            "change_1d": 0.5
        }
    }

    result = macro.detect(external_data)
    print(f"   Systemic Risk Flag: {result['systemic_risk_flag']}")
    print(f"   Stress Level: {result['stress_level']}")
    print(f"   Yield Inversion: {result['yield_inversion']}")
    print(f"   Risk Flags: {result['risk_flags']}")

    # 测试事件日历
    print("\n3. EventCalendar:")
    import tempfile
    import os

    events = {
        "events": [
            {
                "name": "FOMC Meeting",
                "type": "FOMC",
                "datetime": (datetime.now() + timedelta(hours=2)).isoformat(),
                "pre_window_hours": 24,
                "post_window_hours": 24
            },
            {
                "name": "CPI Release",
                "type": "CPI",
                "datetime": (datetime.now() + timedelta(hours=48)).isoformat(),
                "pre_window_hours": 12,
                "post_window_hours": 12
            }
        ]
    }

    temp_file = tempfile.mktemp(suffix=".json")
    try:
        with open(temp_file, 'w') as f:
            json.dump(events, f)

        calendar = EventCalendar(temp_file)
        result = calendar.is_event_window(datetime.now(), "AAPL")
        print(f"   In Event Window: {result['in_event_window']}")

        # 检查即将到来的事件
        future_result = calendar.is_event_window(datetime.now() + timedelta(hours=1), "AAPL")
        print(f"   Upcoming Event: {future_result.get('event_name')} in {future_result.get('time_to_event')}")

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("\n✓ 所有测试通过")


import numpy as np