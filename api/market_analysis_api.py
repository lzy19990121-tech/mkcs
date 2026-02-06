"""
市场分析 API 接口

为 Web UI 提供市场分析数据和可解释性信息
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from decimal import Decimal

from core.models import Bar
from skills.market_analysis.market_state import MarketState, RegimeType
from skills.market_analysis import MarketManager, get_market_manager

logger = logging.getLogger(__name__)


class MarketAnalysisAPI:
    """市场分析 API

    为前端提供：
    1. 当前市场状态
    2. 市场状态历史
    3. 策略可行性评估
    4. 风控建议
    """

    def __init__(self, market_manager: Optional[MarketManager] = None):
        """初始化 API

        Args:
            market_manager: 市场管理器（如果不提供，使用全局单例）
        """
        self.market_manager = market_manager or get_market_manager()

    def get_current_market_state(
        self,
        bars: List[Bar],
        external_data: Optional[Dict[str, Any]] = None,
        symbol: str = "DEFAULT"
    ) -> Dict[str, Any]:
        """获取当前市场状态（API 格式）

        Args:
            bars: K��数据
            external_data: 外部数据
            symbol: 标的代码

        Returns:
            {
                "status": "ok",
                "data": {
                    "regime": str,
                    "regime_display": str,
                    "volatility": str,
                    "liquidity": str,
                    "sentiment": str,
                    "should_reduce_risk": bool,
                    "position_scale_factor": float,
                    "recommendations": list,
                    "explanation": str,
                    "timestamp": str
                }
            }
        """
        try:
            state = self.market_manager.analyze(bars, external_data=external_data, symbol=symbol)

            return {
                "status": "ok",
                "data": {
                    "regime": state.regime.value,
                    "regime_display": self._get_regime_display(state.regime.value),
                    "volatility": state.volatility_state.value,
                    "volatility_display": self._get_volatility_display(state.volatility_state.value),
                    "liquidity": state.liquidity_state.value,
                    "sentiment": state.sentiment_state.value,
                    "should_reduce_risk": state.should_reduce_risk,
                    "position_scale_factor": state.position_scale_factor,
                    "trend_strength": state.trend_strength,
                    "trend_direction": state.trend_direction,
                    "confidence": state.confidence,
                    "recommendations": self._generate_recommendations(state),
                    "explanation": self._generate_explanation(state),
                    "systemic_risks": self._format_systemic_risks(state.systemic_risk),
                    "timestamp": state.timestamp.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"获取市场状态失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_market_state_history(
        self,
        limit: int = 100
    ) -> Dict[str, Any]:
        """获取市场状态历史

        Args:
            limit: 返回记录数

        Returns:
            历史状态列表
        """
        try:
            history = self.market_manager.get_state_history(limit)

            return {
                "status": "ok",
                "data": {
                    "total": len(history),
                    "states": [
                        {
                            "timestamp": s.timestamp.isoformat(),
                            "regime": s.regime.value,
                            "volatility": s.volatility_state.value,
                            "liquidity": s.liquidity_state.value,
                            "confidence": s.confidence,
                            "position_scale": s.position_scale_factor
                        }
                        for s in history
                    ]
                }
            }
        except Exception as e:
            logger.error(f"获取历史状态失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_strategy_feasibility(
        self,
        strategy_type: str,
        bars: Optional[List[Bar]] = None
    ) -> Dict[str, Any]:
        """获取策略可行性评估

        Args:
            strategy_type: 策略类型 (ma, breakout, ml)
            bars: K线数据（用于获取当前市场状态）

        Returns:
            {
                "status": "ok",
                "data": {
                    "strategy_type": str,
                    "allowed": bool,
                    "position_scale": float,
                    "reason": str,
                    "dynamic_params": dict,
                    "recommendation": str
                }
            }
        """
        try:
            # 获取当前市场状态
            market_state = None
            if bars:
                market_state = self.market_manager.analyze(bars)

            # 评估可行性
            feasibility = self.market_manager.evaluate_strategy(strategy_type, market_state)

            # 获取动态参数
            dynamic_params = feasibility.get("dynamic_params", {})

            return {
                "status": "ok",
                "data": {
                    "strategy_type": strategy_type,
                    "allowed": feasibility["strategy_allowed"],
                    "position_scale": feasibility["position_scale"],
                    "reason": feasibility["reason"],
                    "dynamic_params": dynamic_params,
                    "recommendation": self._generate_strategy_recommendation(feasibility)
                }
            }
        except Exception as e:
            logger.error(f"评估可行性失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_risk_recommendations(
        self,
        bars: List[Bar],
        current_positions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取风控建议

        Args:
            bars: K线数据
            current_positions: 当前持仓

        Returns:
            风控建议
        """
        try:
            state = self.market_manager.analyze(bars)

            recommendations = {
                "position_limits": self._get_position_limits(state),
                "risk_level": self._get_risk_level(state),
                "actions": self._generate_risk_actions(state),
                "warnings": self._generate_warnings(state)
            }

            return {
                "status": "ok",
                "data": recommendations
            }
        except Exception as e:
            logger.error(f"获取风控建议失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def get_market_summary(
        self,
        bars: List[Bar],
        external_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """获取市场摘要（综合视图）

        Args:
            bars: K线数据
            external_data: 外部数据

        Returns:
            综合市场摘要
        """
        try:
            state = self.market_manager.analyze(bars, external_data=external_data)

            # 评估各策略
            strategies = ["ma", "breakout", "ml"]
            strategy_status = {}
            for strategy in strategies:
                feasibility = self.market_manager.evaluate_strategy(strategy, state)
                strategy_status[strategy] = {
                    "allowed": feasibility["strategy_allowed"],
                    "scale": feasibility["position_scale"],
                    "reason": feasibility["reason"]
                }

            return {
                "status": "ok",
                "data": {
                    "market_state": {
                        "regime": state.regime.value,
                        "volatility": state.volatility_state.value,
                        "liquidity": state.liquidity_state.value,
                        "sentiment": state.sentiment_state.value,
                    },
                    "risk_assessment": {
                        "should_reduce_risk": state.should_reduce_risk,
                        "risk_level": self._get_risk_level(state),
                        "position_scale": state.position_scale_factor
                    },
                    "strategy_status": strategy_status,
                    "recommendations": self._generate_recommendations(state),
                    "explanation": self._generate_explanation(state),
                    "timestamp": state.timestamp.isoformat()
                }
            }
        except Exception as e:
            logger.error(f"获取市场摘要失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _get_regime_display(self, regime: str) -> str:
        """获取市场状态显示名称"""
        displays = {
            "TREND": "趋势市场",
            "RANGE": "震荡市场",
            "HIGH_VOL": "高波动",
            "CRISIS": "危机模式",
            "UNKNOWN": "未知"
        }
        return displays.get(regime, regime)

    def _get_volatility_display(self, volatility: str) -> str:
        """获取波动率显示名称"""
        displays = {
            "LOW": "低波动",
            "NORMAL": "正常波动",
            "HIGH": "高波动",
            "EXTREME": "极端波动"
        }
        return displays.get(volatility, volatility)

    def _get_risk_level(self, state: MarketState) -> str:
        """获取风险等级"""
        if state.regime == RegimeType.CRISIS:
            return "CRITICAL"
        if state.volatility_state.value == "EXTREME":
            return "VERY_HIGH"
        if state.liquidity_state.value == "FROZEN":
            return "CRITICAL"
        if state.systemic_risk.has_any_risk():
            return "HIGH"
        if state.should_reduce_risk:
            return "ELEVATED"
        return "NORMAL"

    def _generate_recommendations(self, state: MarketState) -> List[str]:
        """生成建议列表"""
        recommendations = []

        if state.regime == RegimeType.TREND:
            recommendations.append("趋势市场：趋势策略可正常使用")
            recommendations.append("建议使用动态止损保护利润")

        elif state.regime == RegimeType.RANGE:
            recommendations.append("震荡市场：建议降低趋势策略仓位")
            recommendations.append("考虑使用均值回归策略")

        elif state.regime == RegimeType.CRISIS:
            recommendations.append("危机模式：强烈建议降低仓位")
            recommendations.append("避免开新仓，优先保护本金")

        if state.volatility_state.value == "EXTREME":
            recommendations.append("极端波动：收紧止损，降低单次仓位")

        if state.liquidity_state.value == "THIN":
            recommendations.append("流动性不足：避免大额交易")

        if state.sentiment_state.value == "FEAR":
            recommendations.append("市场恐慌：注意假突破，等待确认")
        elif state.sentiment_state.value == "GREED":
            recommendations.append("市场贪婪：注意回调风险")

        return recommendations

    def _generate_explanation(self, state: MarketState) -> str:
        """生成市场状态解释"""
        regime_desc = self._get_regime_display(state.regime.value)
        vol_desc = self._get_volatility_display(state.volatility_state.value)

        explanation = f"当前市场处于{regime_desc}，波动率水平为{vol_desc}。"

        if state.trend_direction == "UP":
            explanation += f" 趋势方向向上，强度为{state.trend_strength:.0%}。"
        elif state.trend_direction == "DOWN":
            explanation += f" 趋势方向向下，强度为{state.trend_strength:.0%}。"

        if state.should_reduce_risk:
            explanation += " 建议降低风险暴露，减少仓位。"
        else:
            explanation += " 市场条件良好，可正常交易。"

        return explanation

    def _format_systemic_risks(self, risks) -> List[Dict[str, Any]]:
        """格式化系统性风险"""
        risk_list = []

        if risks.event_window:
            risk_list.append({
                "type": "event_window",
                "description": "处于重大事件窗口期",
                "severity": "HIGH"
            })

        if risks.high_systemic_risk:
            risk_list.append({
                "type": "high_systemic_risk",
                "description": "系统性风险较高",
                "severity": "HIGH"
            })

        if risks.cross_market_stress:
            risk_list.append({
                "type": "cross_market_stress",
                "description": "跨市场压力",
                "severity": "MEDIUM"
            })

        if risks.currency_crisis:
            risk_list.append({
                "type": "currency_crisis",
                "description": "货币危机风险",
                "severity": "HIGH"
            })

        return risk_list

    def _generate_strategy_recommendation(self, feasibility: Dict[str, Any]) -> str:
        """生成策略建议"""
        if not feasibility["strategy_allowed"]:
            return f"策略不建议使用：{feasibility['reason']}"

        scale = feasibility["position_scale"]
        if scale >= 0.8:
            return "策略条件良好，可正常使用"
        elif scale >= 0.5:
            return f"策略可以使用，建议降低仓位至{scale:.0%}"
        else:
            return f"策略需谨慎使用，建议大幅降低仓位至{scale:.0%}"

    def _get_position_limits(self, state: MarketState) -> Dict[str, float]:
        """获取仓位限制建议"""
        scale = state.position_scale_factor

        return {
            "max_single_position": 0.2 * scale,
            "max_total_position": 0.8 * scale,
            "recommended_scale": scale
        }

    def _generate_risk_actions(self, state: MarketState) -> List[str]:
        """生成风控行动建议"""
        actions = []

        if state.should_reduce_risk:
            actions.append("降低仓位")

        if state.volatility_state.value == "EXTREME":
            actions.append("收紧止损")

        if state.liquidity_state.value in ["THIN", "FROZEN"]:
            actions.append("避免开新仓")

        if state.regime == RegimeType.CRISIS:
            actions.append("考虑平仓保护")

        return actions

    def _generate_warnings(self, state: MarketState) -> List[str]:
        """生成警告信息"""
        warnings = []

        if state.regime == RegimeType.CRISIS:
            warnings.append("市场处于危机模式，交易风险极高")

        if state.volatility_state.value == "EXTREME":
            warnings.append("波动率极端，可能产生较大损失")

        if state.liquidity_state.value == "FROZEN":
            warnings.append("市场流动性枯竭，可能无法平仓")

        if state.systemic_risk.has_any_risk():
            warnings.append("存在系统性风险，需谨慎交易")

        return warnings


# 创建全局 API 实例
_global_api: Optional[MarketAnalysisAPI] = None


def get_market_analysis_api() -> MarketAnalysisAPI:
    """获取全局市场分析 API 实例"""
    global _global_api
    if _global_api is None:
        _global_api = MarketAnalysisAPI()
    return _global_api


if __name__ == "__main__":
    """测试代码"""
    print("=== MarketAnalysisAPI 测试 ===\n")

    # 创建示例数据
    from datetime import timedelta

    bars = []
    for i in range(50):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        price = 100 + i * 0.5 + (i % 5 - 2) * 0.3

        bars.append(Bar(
            symbol="AAPL",
            timestamp=date,
            open=Decimal(str(price - 0.2)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price)),
            volume=1000000,
            interval="1d"
        ))

    api = MarketAnalysisAPI()

    # 测试1: 获取当前市场状态
    print("1. 当前市场状态:")
    result = api.get_current_market_state(bars)
    if result["status"] == "ok":
        data = result["data"]
        print(f"   市场状态: {data['regime_display']}")
        print(f"   波动率: {data['volatility_display']}")
        print(f"   应降低风险: {data['should_reduce_risk']}")
        print(f"   仓位缩放: {data['position_scale_factor']:.2f}")
        print(f"   解释: {data['explanation']}")

    # 测试2: 策略可行性
    print("\n2. 策略可行性:")
    result = api.get_strategy_feasibility("ma", bars)
    if result["status"] == "ok":
        data = result["data"]
        print(f"   MA 策略允许: {data['allowed']}")
        print(f"   仓位缩放: {data['position_scale']:.2f}")
        print(f"   原因: {data['reason']}")
        print(f"   建议: {data['recommendation']}")

    # 测试3: 市场摘要
    print("\n3. 市场摘要:")
    result = api.get_market_summary(bars)
    if result["status"] == "ok":
        data = result["data"]
        print(f"   风险等级: {data['risk_assessment']['risk_level']}")
        print(f"   策略状态: {data['strategy_status']}")
        print(f"   建议: {data['recommendations']}")

    # 测试4: 风控建议
    print("\n4. 风控建议:")
    result = api.get_risk_recommendations(bars)
    if result["status"] == "ok":
        data = result["data"]
        print(f"   风险等级: {data['risk_level']}")
        print(f"   仓位限制: {data['position_limits']}")
        print(f"   建议行动: {data['actions']}")
        print(f"   警告: {data['warnings']}")

    print("\n✓ 所有测试通过")
