"""
UI/可解释性模块 - Phase 7

提供决策可视化、回放、解释功能
不替代系统决策
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from core.schema import (
    MarketState,
    AlphaOpinion,
    MetaDecision,
    RiskDecision,
    ExecutionResult,
    AlphaGatingConfig
)


@dataclass
class DecisionTrace:
    """决策追溯"""
    timestamp: datetime
    step: str  # "market_analysis", "alpha_opinion", "meta_decision", "risk_check", "execution"
    data: Dict[str, Any]
    reason: str = ""


@dataclass
class RunReport:
    """运行报告"""
    run_id: str
    start_time: datetime
    end_time: datetime
    traces: List[DecisionTrace]
    final_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "traces": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "step": t.step,
                    "data": t.data,
                    "reason": t.reason
                }
                for t in self.traces
            ],
            "final_state": self.final_state
        }


class DecisionReplay:
    """决策回放器"""

    def __init__(self):
        self._runs: Dict[str, RunReport] = {}

    def create_run(
        self,
        run_id: str,
        start_time: datetime
    ) -> RunReport:
        """创建新的运行记录"""
        run = RunReport(
            run_id=run_id,
            start_time=start_time,
            end_time=start_time,
            traces=[],
            final_state={}
        )
        self._runs[run_id] = run
        return run

    def add_trace(
        self,
        run_id: str,
        step: str,
        data: Dict[str, Any],
        reason: str = "",
        timestamp: Optional[datetime] = None
    ):
        """添加决策追溯"""
        if run_id not in self._runs:
            return

        trace = DecisionTrace(
            timestamp=timestamp or datetime.now(),
            step=step,
            data=data,
            reason=reason
        )
        self._runs[run_id].traces.append(trace)

    def finish_run(
        self,
        run_id: str,
        end_time: datetime,
        final_state: Dict[str, Any]
    ):
        """结束运行"""
        if run_id in self._runs:
            self._runs[run_id].end_time = end_time
            self._runs[run_id].final_state = final_state

    def get_run(self, run_id: str) -> Optional[RunReport]:
        """获取运行报告"""
        return self._runs.get(run_id)

    def list_runs(self) -> List[str]:
        """列出所有运行ID"""
        return list(self._runs.keys())


class ExplainabilityEngine:
    """
    可解释性引擎

    提供决策解释、可视化数据生成
    """

    def __init__(self):
        self.replay = DecisionReplay()

    def explain_decision(
        self,
        market_state: MarketState,
        opinions: List[AlphaOpinion],
        meta_decision: MetaDecision,
        risk_decision: RiskDecision,
        execution_result: ExecutionResult
    ) -> Dict[str, Any]:
        """
        解释完整决策流程

        Returns:
            包含完整决策解释的字典
        """
        explanation = {
            "timestamp": market_state.timestamp.isoformat(),
            "symbol": market_state.symbol,
            "market_context": self._explain_market(market_state),
            "strategy_opinions": self._explain_opinions(opinions),
            "meta_decision": self._explain_meta(meta_decision),
            "risk_control": self._explain_risk(risk_decision, market_state),
            "execution": self._explain_execution(execution_result),
            "decision_chain": self._build_decision_chain(
                market_state, opinions, meta_decision, risk_decision, execution_result
            )
        }

        return explanation

    def _explain_market(self, state: MarketState) -> Dict[str, Any]:
        """解释市场状态"""
        return {
            "regime": {
                "value": state.regime,
                "confidence": state.regime_confidence,
                "description": self._get_regime_description(state.regime)
            },
            "volatility": {
                "state": state.volatility_state,
                "percentile": state.volatility_percentile,
                "trend": state.volatility_trend,
                "description": self._get_volatility_description(state.volatility_state)
            },
            "liquidity": {
                "state": state.liquidity_state,
                "volume_ratio": state.volume_ratio,
                "description": self._get_liquidity_description(state.liquidity_state)
            },
            "sentiment": {
                "state": state.sentiment_state,
                "score": state.sentiment_score,
                "description": self._get_sentiment_description(state.sentiment_state)
            }
        }

    def _explain_opinions(self, opinions: List[AlphaOpinion]) -> List[Dict[str, Any]]:
        """解释策略观点"""
        result = []
        for opinion in opinions:
            result.append({
                "strategy": opinion.strategy_name,
                "direction": self._direction_to_str(opinion.direction),
                "strength": opinion.strength,
                "strength_label": self._get_strength_label(opinion.strength),
                "confidence": opinion.confidence,
                "confidence_label": self._get_confidence_label(opinion.confidence),
                "horizon": opinion.horizon,
                "reason": opinion.reason,
                "is_disabled": opinion.is_disabled,
                "disabled_reason": opinion.disabled_reason if opinion.is_disabled else None,
                "position_signal": opinion.get_position_signal()
            })
        return result

    def _explain_meta(self, decision: MetaDecision) -> Dict[str, Any]:
        """解释 Meta 决策"""
        return {
            "target_position": decision.target_position,
            "target_weight": decision.target_weight,
            "decision_confidence": decision.decision_confidence,
            "consensus_level": decision.consensus_level,
            "consensus_description": self._get_consensus_description(decision.consensus_level),
            "active_strategies": decision.active_strategies,
            "disabled_strategies": decision.disabled_strategies,
            "reasoning": decision.reasoning
        }

    def _explain_risk(
        self,
        decision: RiskDecision,
        market_state: MarketState
    ) -> Dict[str, Any]:
        """解释风控决策"""
        return {
            "action": decision.risk_action,
            "action_description": self._get_risk_action_description(decision.risk_action),
            "scale_factor": decision.scale_factor,
            "original_position": decision.original_position if hasattr(decision, 'original_position') else "N/A",
            "scaled_position": decision.scaled_target_position,
            "reason": decision.risk_reason,
            "max_position": decision.max_position,
            "cooldown_info": self._get_cooldown_info(decision)
        }

    def _explain_execution(self, result: ExecutionResult) -> Dict[str, Any]:
        """解释执行结果"""
        return {
            "status": result.status,
            "status_description": self._get_execution_status_description(result.status),
            "mode": result.execution_mode,
            "target_position": result.target_position,
            "actual_position": result.actual_position,
            "target_price": result.target_price,
            "fill_price": result.fill_price,
            "fill_quantity": result.fill_quantity,
            "slippage": result.slippage,
            "slippage_description": f"{result.slippage:.2%}",
            "current_positions": result.current_positions,
            "cash_balance": result.cash_balance
        }

    def _build_decision_chain(
        self,
        market_state: MarketState,
        opinions: List[AlphaOpinion],
        meta_decision: MetaDecision,
        risk_decision: RiskDecision,
        execution_result: ExecutionResult
    ) -> List[Dict[str, Any]]:
        """构建决策链"""
        chain = []

        # Step 1: Market Analysis
        chain.append({
            "step": 1,
            "name": "Market Analysis",
            "input": "K线数据",
            "output": f"Regime={market_state.regime}, Volatility={market_state.volatility_state}",
            "description": "分析市场状态，不产生预测"
        })

        # Step 2: Alpha Strategies
        enabled_count = sum(1 for o in opinions if not o.is_disabled)
        disabled_count = sum(1 for o in opinions if o.is_disabled)
        chain.append({
            "step": 2,
            "name": "Alpha Strategies",
            "input": "MarketState + K线",
            "output": f"{enabled_count} 个启用策略, {disabled_count} 个禁用策略",
            "description": "策略输出观点，不决定仓位"
        })

        # Step 3: Meta Decision
        chain.append({
            "step": 3,
            "name": "Meta / Brain",
            "input": "AlphaOpinions + MarketState",
            "output": f"目标仓位 {meta_decision.target_position:.0f} ({meta_decision.target_weight:.1%})",
            "description": f"{meta_decision.reasoning}"
        })

        # Step 4: Risk Control
        chain.append({
            "step": 4,
            "name": "Risk Gate",
            "input": "MetaDecision + MarketState",
            "output": f"{risk_decision.risk_action}, 缩放系数 {risk_decision.scale_factor:.2f}",
            "description": risk_decision.risk_reason
        })

        # Step 5: Execution
        chain.append({
            "step": 5,
            "name": "Execution",
            "input": "RiskDecision + 当前价格",
            "output": f"{execution_result.status}, 成交 {execution_result.fill_quantity:.0f} @ {execution_result.fill_price:.2f}",
            "description": f"模式: {execution_result.execution_mode}"
        })

        return chain

    def _get_regime_description(self, regime: str) -> str:
        """获取市场状态描述"""
        descriptions = {
            "TREND": "趋势市场 - 价格有明显方向性",
            "RANGE": "震荡市场 - 价格在一定区间内波动",
            "CRISIS": "危机模式 - 极端市场条件，需要谨慎"
        }
        return descriptions.get(regime, f"未知状态: {regime}")

    def _get_volatility_description(self, volatility: str) -> str:
        """获取波动率描述"""
        descriptions = {
            "LOW": "低波动 - 价格变化较小",
            "NORMAL": "正常波动 - 价格变化在正常范围内",
            "HIGH": "高波动 - 价格变化较大",
            "EXTREME": "极端波动 - 价格变化剧烈，风险极高"
        }
        return descriptions.get(volatility, f"未知: {volatility}")

    def _get_liquidity_description(self, liquidity: str) -> str:
        """获取流动性描述"""
        descriptions = {
            "NORMAL": "正常流动性 - 市场交易活跃",
            "THIN": "流动性不足 - 交易量偏低，可能有滑点",
            "FROZEN": "流动性枯竭 - 几乎无法交易"
        }
        return descriptions.get(liquidity, f"未知: {liquidity}")

    def _get_sentiment_description(self, sentiment: str) -> str:
        """获取情绪描述"""
        descriptions = {
            "GREED": "贪婪 - 市场过度乐观",
            "NEUTRAL": "中性 - 市场情绪平衡",
            "FEAR": "恐惧 - 市场过度悲观",
            "PANIC": "恐慌 - 极度恐惧，可能超跌"
        }
        return descriptions.get(sentiment, f"未知: {sentiment}")

    def _get_strength_label(self, strength: float) -> str:
        """获取强度标签"""
        if strength < 0.3:
            return "弱"
        elif strength < 0.7:
            return "中"
        else:
            return "强"

    def _get_confidence_label(self, confidence: float) -> str:
        """获取置信度标签"""
        if confidence < 0.4:
            return "低"
        elif confidence < 0.7:
            return "中"
        else:
            return "高"

    def _direction_to_str(self, direction: int) -> str:
        """方向转字符串"""
        if direction > 0:
            return "做多"
        elif direction < 0:
            return "做空"
        else:
            return "中性"

    def _get_consensus_description(self, level: str) -> str:
        """获取共识级别描述"""
        descriptions = {
            "STRONG": "强共识 - 所有策略方向一致",
            "WEAK": "弱共识 - 策略方向部分一致或有冲突",
            "NONE": "无共识 - 策略方向冲突严重"
        }
        return descriptions.get(level, f"未知: {level}")

    def _get_risk_action_description(self, action: str) -> str:
        """获取风控动作描述"""
        descriptions = {
            "APPROVE": "通过 - 无风控限制",
            "SCALE_DOWN": "缩放 - 降低仓位规模",
            "PAUSE": "暂停 - 暂停新开仓",
            "DISABLE": "禁止 - 禁止交易"
        }
        return descriptions.get(action, f"未知: {action}")

    def _get_execution_status_description(self, status: str) -> str:
        """获取执行状态描述"""
        descriptions = {
            "FILLED": "成交 - 订单完全成交",
            "PARTIAL": "部分成交 - 订单部分成交",
            "FAILED": "失败 - 订单执行失败",
            "SKIPPED": "跳过 - 未执行订单"
        }
        return descriptions.get(status, f"未知: {status}")

    def _get_cooldown_info(self, decision: RiskDecision) -> Optional[Dict[str, Any]]:
        """获取冷却信息"""
        if hasattr(decision, 'cooldown_until') and decision.cooldown_until:
            return {
                "active": True,
                "until": decision.cooldown_until.isoformat(),
                "reason": decision.cooldown_reason if hasattr(decision, 'cooldown_reason') else ""
            }
        return {"active": False}

    def generate_html_report(
        self,
        explanation: Dict[str, Any],
        run_id: str
    ) -> str:
        """生成 HTML 报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MKCS 决策报告 - {run_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .section {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
        .decision-chain {{ display: flex; flex-direction: column; gap: 10px; }}
        .chain-step {{ background: white; border-left: 4px solid #007acc; padding: 15px; margin-left: 20px; }}
        .chain-step.step-1 {{ border-color: #4CAF50; }}
        .chain-step.step-2 {{ border-color: #2196F3; }}
        .chain-step.step-3 {{ border-color: #FF9800; }}
        .chain-step.step-4 {{ border-color: #f44336; }}
        .chain-step.step-5 {{ border-color: #9C27B0; }}
        .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .badge.enabled {{ background: #4CAF50; color: white; }}
        .badge.disabled {{ background: #f44336; color: white; }}
        .metric {{ display: inline-block; margin: 5px 10px 5px 0; }}
        .metric-label {{ color: #666; font-size: 14px; }}
        .metric-value {{ color: #333; font-weight: bold; font-size: 16px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 MKCS 决策报告</h1>
        <p><strong>运行 ID:</strong> {run_id}</p>
        <p><strong>时间:</strong> {explanation['timestamp']}</p>
        <p><strong>标的:</strong> {explanation['symbol']}</p>

        <h2>📊 市场环境</h2>
        <div class="section">
"""

        # 市场环境
        mc = explanation['market_context']
        html += f"""
            <div class="metric">
                <span class="metric-label">市场状态:</span>
                <span class="metric-value">{mc['regime']['value']}</span>
                <small>{mc['regime']['description']}</small>
            </div>
            <div class="metric">
                <span class="metric-label">波动率:</span>
                <span class="metric-value">{mc['volatility']['state']}</span>
                <small>{mc['volatility']['description']}</small>
            </div>
            <div class="metric">
                <span class="metric-label">流动性:</span>
                <span class="metric-value">{mc['liquidity']['state']}</span>
                <small>{mc['liquidity']['description']}</small>
            </div>
            <div class="metric">
                <span class="metric-label">情绪:</span>
                <span class="metric-value">{mc['sentiment']['state']}</span>
                <small>{mc['sentiment']['description']}</small>
            </div>
        </div>

        <h2>📈 策略观点</h2>
        <div class="section">
            <table>
                <thead>
                    <tr>
                        <th>策略</th>
                        <th>方向</th>
                        <th>强度</th>
                        <th>置信度</th>
                        <th>状态</th>
                        <th>理由</th>
                    </tr>
                </thead>
                <tbody>
"""

        for opinion in explanation['strategy_opinions']:
            badge_class = "enabled" if not opinion['is_disabled'] else "disabled"
            status = "启用" if not opinion['is_disabled'] else f"禁用: {opinion['disabled_reason']}"
            html += f"""
                    <tr>
                        <td>{opinion['strategy']}</td>
                        <td>{opinion['direction']}</td>
                        <td>{opinion['strength']:.2f} ({opinion['strength_label']})</td>
                        <td>{opinion['confidence']:.2f} ({opinion['confidence_label']})</td>
                        <td><span class="badge {badge_class}">{status}</span></td>
                        <td>{opinion['reason']}</td>
                    </tr>
"""

        html += """
                </tbody>
            </table>
        </div>

        <h2>🧩 Meta 决策 (Brain)</h2>
        <div class="section">
"""

        meta = explanation['meta_decision']
        html += f"""
            <div class="metric">
                <span class="metric-label">目标仓位:</span>
                <span class="metric-value">{meta['target_position']:.0f} 股</span>
            </div>
            <div class="metric">
                <span class="metric-label">目标权重:</span>
                <span class="metric-value">{meta['target_weight']:.2%}</span>
            </div>
            <div class="metric">
                <span class="metric-label">共识级别:</span>
                <span class="metric-value">{meta['consensus_level']}</span>
                <small>{meta['consensus_description']}</small>
            </div>
            <p><strong>决策理由:</strong> {meta['reasoning']}</p>
            <p><strong>启用策略:</strong> {', '.join(meta['active_strategies'])}</p>
        </div>

        <h2>🛡️ 风控决策</h2>
        <div class="section">
"""

        risk = explanation['risk_control']
        html += f"""
            <div class="metric">
                <span class="metric-label">风控动作:</span>
                <span class="metric-value">{risk['action']}</span>
                <small>{risk['action_description']}</small>
            </div>
            <div class="metric">
                <span class="metric-label">缩放系数:</span>
                <span class="metric-value">{risk['scale_factor']:.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">调整后仓位:</span>
                <span class="metric-value">{risk['scaled_position']:.0f} 股</span>
            </div>
            <p><strong>风控理由:</strong> {risk['reason']}</p>
        </div>

        <h2>⚡ 执行结果</h2>
        <div class="section">
"""

        exec_result = explanation['execution']
        html += f"""
            <div class="metric">
                <span class="metric-label">状态:</span>
                <span class="metric-value">{exec_result['status']}</span>
                <small>{exec_result['status_description']}</small>
            </div>
            <div class="metric">
                <span class="metric-label">成交价格:</span>
                <span class="metric-value">{exec_result['fill_price']:.2f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">成交数量:</span>
                <span class="metric-value">{exec_result['fill_quantity']:.0f} 股</span>
            </div>
            <div class="metric">
                <span class="metric-label">滑点:</span>
                <span class="metric-value">{exec_result['slippage_description']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">现金余额:</span>
                <span class="metric-value">{exec_result['cash_balance']:.0f}</span>
            </div>
        </div>

        <h2>🔗 决策链</h2>
        <div class="decision-chain">
"""

        for step in explanation['decision_chain']:
            html += f"""
            <div class="chain-step step-{step['step']}">
                <h3>步骤 {step['step']}: {step['name']}</h3>
                <p><strong>输入:</strong> {step['input']}</p>
                <p><strong>输出:</strong> {step['output']}</p>
                <p><strong>说明:</strong> {step['description']}</p>
            </div>
"""

        html += """
        </div>
    </div>
</body>
</html>
"""
        return html

    def save_html_report(
        self,
        explanation: Dict[str, Any],
        run_id: str,
        filepath: str
    ):
        """保存 HTML 报告"""
        html = self.generate_html_report(explanation, run_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)


if __name__ == "__main__":
    """测试代码"""
    print("=== UI/可解释性 测试 ===\n")

    # 创建测试数据
    now = datetime.now()

    market_state = MarketState(
        timestamp=now,
        symbol="AAPL",
        regime="TREND",
        regime_confidence=0.8,
        volatility_state="NORMAL",
        volatility_trend="STABLE",
        volatility_percentile=0.6,
        liquidity_state="NORMAL",
        volume_ratio=1.2,
        sentiment_state="NEUTRAL",
        sentiment_score=0.0
    )

    opinions = [
        AlphaOpinion(
            strategy_name="MA",
            timestamp=now,
            symbol="AAPL",
            direction=1,
            strength=0.8,
            confidence=0.7,
            horizon="daily",
            reason="MA 金叉"
        ),
        AlphaOpinion(
            strategy_name="Breakout",
            timestamp=now,
            symbol="AAPL",
            direction=1,
            strength=0.6,
            confidence=0.6,
            horizon="swing",
            reason="向上突破"
        ),
        AlphaOpinion(
            strategy_name="ML",
            timestamp=now,
            symbol="AAPL",
            direction=0,
            strength=0.0,
            confidence=0.0,
            horizon="intraday",
            reason="无明确信号"
        )
    ]

    meta_decision = MetaDecision(
        timestamp=now,
        symbol="AAPL",
        target_position=100,
        target_weight=0.1,
        active_strategies=["MA", "Breakout", "ML"],
        disabled_strategies={},
        decision_confidence=0.7,
        consensus_level="WEAK",
        reasoning="单边信号 (2个策略)"
    )

    risk_decision = RiskDecision(
        timestamp=now,
        symbol="AAPL",
        scaled_target_position=100,
        scale_factor=1.0,
        risk_action="APPROVE",
        risk_reason="风控通过"
    )

    execution_result = ExecutionResult(
        timestamp=now,
        symbol="AAPL",
        execution_mode="paper",
        target_position=100,
        actual_position=100,
        target_price=150.0,
        fill_price=150.15,
        status="FILLED",
        fill_quantity=100,
        slippage=0.001,
        current_positions={"AAPL": 100},
        cash_balance=95000
    )

    # 测试可解释性引擎
    engine = ExplainabilityEngine()

    print("1. 生成决策解释:")
    explanation = engine.explain_decision(
        market_state, opinions, meta_decision, risk_decision, execution_result
    )

    print(f"   市场状态: {explanation['market_context']['regime']['value']}")
    print(f"   策略数量: {len(explanation['strategy_opinions'])}")
    print(f"   Meta决策: 目标仓位 {explanation['meta_decision']['target_position']}")
    print(f"   风控动作: {explanation['risk_control']['action']}")
    print(f"   执行状态: {explanation['execution']['status']}")

    print("\n2. 生成 HTML 报告:")
    html = engine.generate_html_report(explanation, "test_run_001")
    print(f"   HTML 长度: {len(html)} 字符")

    print("\n3. 测试决策回放:")
    replay = engine.replay
    run = replay.create_run("test_run_001", now)
    replay.add_trace("test_run_001", "market_analysis", {"regime": "TREND"}, "市场分析")
    replay.add_trace("test_run_001", "alpha_opinion", {"MA": 1}, "MA策略做多")
    replay.add_trace("test_run_001", "meta_decision", {"position": 100}, "决定仓位100")
    replay.finish_run("test_run_001", now + timedelta(seconds=1), {"final_position": 100})

    saved_run = replay.get_run("test_run_001")
    print(f"   运行 ID: {saved_run.run_id}")
    print(f"   追踪数量: {len(saved_run.traces)}")
    print(f"   最终状态: {saved_run.final_state}")

    print("\n✓ 所有测试通过")
