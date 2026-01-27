"""
TUI观察列表界面

使用rich库实现终端UI，显示：
1. 观察列表
2. 选中标的详情
3. 最新订单/成交
4. 风控状态
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

from events.event_log import get_event_logger
from broker.paper import PaperBroker


class WatchlistTUI:
    """观察列表TUI

    显示内容：
    - 左侧：观察列表
    - 右上：选中标的详情
    - 右中：最新订单/成交
    - 右下：风控状态
    """

    def __init__(self, config_path: str = "config.yaml"):
        """初始化TUI

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.console = Console()
        self.event_logger = get_event_logger()
        self.selected_index = 0
        self.broker = None  # 将从外部注入

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        config_file = Path(config_path)
        if not config_file.exists():
            # 返回默认配置
            return {
                'watchlist': ['AAPL', 'GOOGL', 'MSFT'],
                'market': {'type': 'US'},
                'account': {'initial_cash': 100000},
                'risk': {
                    'max_position_ratio': 0.2,
                    'max_positions': 5,
                    'max_daily_loss_ratio': 0.05
                }
            }

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    @property
    def watchlist(self) -> List[str]:
        """获取观察列表"""
        return self.config.get('watchlist', [])

    @property
    def selected_symbol(self) -> Optional[str]:
        """获取当前选中的标的"""
        if 0 <= self.selected_index < len(self.watchlist):
            return self.watchlist[self.selected_index]
        return None

    def set_broker(self, broker: PaperBroker):
        """设置经纪商

        Args:
            broker: 模拟经纪商实例
        """
        self.broker = broker

    def render(self) -> Layout:
        """渲染TUI布局

        Returns:
            Layout对象
        """
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        layout["body"].split_row(
            Layout(name="watchlist", ratio=1),
            Layout(name="details", ratio=2)
        )

        layout["details"].split_column(
            Layout(name="symbol_detail", ratio=1),
            Layout(name="orders", ratio=1),
            Layout(name="risk", ratio=1)
        )

        # 填充内容
        layout["header"].update(self._render_header())
        layout["watchlist"].update(self._render_watchlist())
        layout["symbol_detail"].update(self._render_symbol_detail())
        layout["orders"].update(self._render_orders())
        layout["risk"].update(self._render_risk_status())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """渲染头部"""
        market = self.config.get('market', {}).get('type', 'US')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        header_text = (
            f"[bold cyan]自动交易系统 TUI[/bold cyan] | "
            f"市场: {market} | "
            f"时间: {now}"
        )

        return Panel(
            header_text,
            box=box.DOUBLE,
            style="bold white"
        )

    def _render_watchlist(self) -> Panel:
        """渲染观察列表"""
        table = Table(
            title="观察列表",
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE
        )

        table.add_column("#", width=3)
        table.add_column("标的", width=10)
        table.add_column("状态", width=10)
        table.add_column("最新价", width=10)

        for i, symbol in enumerate(self.watchlist):
            # 标记选中
            prefix = "→ " if i == self.selected_index else "  "

            # 获取状态
            status = self._get_symbol_status(symbol)
            price = self._get_symbol_price(symbol)

            # 高亮选中行
            style = "on blue" if i == self.selected_index else ""

            table.add_row(
                f"{prefix}{i+1}",
                Text(symbol, style=style),
                Text(status, style=style),
                Text(price, style=style)
            )

        return Panel(
            table,
            title="[bold]观察列表[/bold]",
            box=box.ROUNDED
        )

    def _render_symbol_detail(self) -> Panel:
        """渲染标的详情"""
        symbol = self.selected_symbol

        if not symbol:
            return Panel(
                "[yellow]请选择一个标的[/yellow]",
                title="[bold]标的详情[/bold]",
                box=box.ROUNDED
            )

        # 查询最新事件
        events = self.event_logger.query_events(symbol=symbol)
        latest_events = events[-5:] if events else []

        # 构建详情内容
        lines = []
        lines.append(f"[bold cyan]标的代码:[/bold cyan] {symbol}\n")

        # 最新信号
        signal_events = [e for e in latest_events if e.get('stage') == 'signal_gen']
        if signal_events:
            latest = signal_events[-1]
            action = latest['payload'].get('action', 'N/A')
            conf = latest['payload'].get('confidence', 0)
            lines.append(f"[bold]最新信号:[/bold] {action} (置信度: {conf:.2%})")
            lines.append(f"[bold]原因:[/bold] {latest['reason']}\n")

        # 持仓情况
        if self.broker:
            position = self.broker.get_position(symbol)
            if position:
                lines.append(f"[bold]持仓:[/bold] {position.quantity}股")
                lines.append(f"[bold]成本:[/bold] ${position.avg_price:.2f}")
                lines.append(f"[bold]市值:[/bold] ${position.market_value:,.2f}")
                lines.append(f"[bold]浮盈:[/bold] ${position.unrealized_pnl:,.2f}\n")
            else:
                lines.append("[bold]持仓:[/bold] 无\n")

        # 最新事件
        lines.append("[bold]最近事件:[/bold]")
        for event in latest_events[-3:]:
            ts = datetime.fromisoformat(event['ts']).strftime('%H:%M:%S')
            stage = event.get('stage', 'N/A')
            reason = event['reason']
            lines.append(f"  {ts} [{stage}] {reason}")

        content = '\n'.join(lines)

        return Panel(
            content,
            title=f"[bold]{symbol} 详情[/bold]",
            box=box.ROUNDED
        )

    def _render_orders(self) -> Panel:
        """渲染最新订单/成交"""
        # 查询最近的订单执行事件
        events = self.event_logger.query_events(
            stage='order_exec'
        )
        events = events[-10:] if events else []

        table = Table(
            title="最新成交",
            show_header=True,
            header_style="bold green",
            box=box.SIMPLE
        )

        table.add_column("时间", width=8)
        table.add_column("标的", width=8)
        table.add_column("方向", width=6)
        table.add_column("价格", width=8)
        table.add_column("数量", width=8)

        for event in events:
            ts = datetime.fromisoformat(event['ts']).strftime('%H:%M:%S')
            symbol = event['symbol']
            payload = event['payload']
            side = payload.get('side', 'N/A')
            price = payload.get('price', 0)
            quantity = payload.get('quantity', 0)

            # 颜色标记
            side_style = "green" if side == "BUY" else "red"

            table.add_row(
                ts,
                symbol,
                Text(side, style=side_style),
                f"${price:.2f}",
                str(quantity)
            )

        return Panel(
            table,
            title="[bold]最新成交[/bold]",
            box=box.ROUNDED
        )

    def _render_risk_status(self) -> Panel:
        """渲染风控状态"""
        risk_config = self.config.get('risk', {})

        lines = []

        # 风控参数
        lines.append("[bold cyan]风控参数[/bold cyan]\n")
        lines.append(f"单只股票仓位限制: {risk_config.get('max_position_ratio', 0.2)*100:.0f}%")
        lines.append(f"持仓数量限制: {risk_config.get('max_positions', 5)}")
        lines.append(f"单日亏损限制: {risk_config.get('max_daily_loss_ratio', 0.05)*100:.0f}%\n")

        # 当前状态
        lines.append("[bold cyan]当前状态[/bold cyan]\n")

        if self.broker:
            # 账户信息
            cash = self.broker.get_cash_balance()
            equity = self.broker.get_total_equity()
            positions = self.broker.get_positions()

            lines.append(f"可用资金: ${cash:,.2f}")
            lines.append(f"总权益: ${equity:,.2f}")
            lines.append(f"持仓数量: {len(positions)}/{risk_config.get('max_positions', 5)}")
            lines.append(f"仓位比例: {equity/cash*100 if cash > 0 else 0:.1f}%")

            # 计算单日盈亏
            pnl = self.broker.get_total_pnl()
            daily_loss_ratio = pnl / self.broker.initial_cash
            limit = risk_config.get('max_daily_loss_ratio', 0.05)

            if daily_loss_ratio < -limit:
                lines.append(f"\n[red]⚠️  单日亏损超限: {daily_loss_ratio*100:.2f}%[/red]")
            else:
                lines.append(f"\n单日盈亏: {daily_loss_ratio*100:+.2f}%")

        content = '\n'.join(lines)

        return Panel(
            content,
            title="[bold]风控状态[/bold]",
            box=box.ROUNDED
        )

    def _render_footer(self) -> Panel:
        """渲染底部"""
        footer_text = (
            "[bold cyan]操作:[/bold cyan] "
            "↑↓: 选择 | q: 退出 | r: 刷新"
        )

        return Panel(
            footer_text,
            box=box.DOUBLE
        )

    def _get_symbol_status(self, symbol: str) -> str:
        """获取标的状态

        Args:
            symbol: 标的代码

        Returns:
            状态文本
        """
        # 检查是否有持仓
        if self.broker:
            position = self.broker.get_position(symbol)
            if position:
                return "持有"

        # 检查最新信号
        events = self.event_logger.query_events(symbol=symbol)
        signal_events = [e for e in events if e.get('stage') == 'signal_gen']

        if signal_events:
            latest = signal_events[-1]
            action = latest['payload'].get('action', 'N/A')
            return action

        return "观察"

    def _get_symbol_price(self, symbol: str) -> str:
        """获取标的最新价格

        Args:
            symbol: 标的代码

        Returns:
            价格字符串
        """
        # TODO：从数据源获取实时价格
        # 如果有持仓，显示成本价
        if self.broker:
            position = self.broker.get_position(symbol)
            if position:
                return f"${position.avg_price:.2f}"

        return "-"

    def move_selection(self, delta: int):
        """移动选择

        Args:
            delta: 移动方向（1: 向下, -1: 向上）
        """
        self.selected_index = (self.selected_index + delta) % len(self.watchlist)

    def run(self):
        """运行TUI（交互模式）"""
        print("启动TUI界面...")
        print("注意：此功能需要集成到主程序中")
        print("请使用 run_simple_display() 查看静态显示")


def run_simple_display():
    """运行简单显示（测试用）"""
    import time

    print("\n=== TUI 简单显示演示 ===\n")

    # 创建TUI
    tui = WatchlistTUI()

    # 创建模拟经纪商
    from broker.paper import PaperBroker
    from decimal import Decimal

    broker = PaperBroker(initial_cash=100000)

    # 执行一笔交易
    from core.models import Signal, OrderIntent
    signal = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.85,
        reason="金叉买入"
    )

    intent = OrderIntent(
        signal=signal,
        timestamp=datetime.now(),
        approved=True,
        risk_reason="OK"
    )

    trade = broker.execute_order(intent)

    # 记录事件
    logger = get_event_logger()
    logger.log_event(
        ts=datetime.now(),
        symbol="AAPL",
        stage="order_exec",
        payload={"side": "BUY", "price": 150.0, "quantity": 100},
        reason="订单执行成功"
    )

    tui.set_broker(broker)

    # 渲染布局
    layout = tui.render()

    # 打印到控制台
    console = Console()
    console.print(layout)

    print("\n✓ TUI显示完成")


if __name__ == "__main__":
    """自测代码"""
    run_simple_display()
