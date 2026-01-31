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
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

from events.event_log import get_event_logger
from broker.paper import PaperBroker


class StateCallback:
    """状态回调接口

    TUI 通过此接口获取状态，而不是直接访问 Broker
    """

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取持仓信息"""
        return None

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有持仓"""
        return {}

    def get_cash_balance(self) -> float:
        """获取现金余额"""
        return 0.0

    def get_total_equity(self) -> float:
        """获取总权益"""
        return 0.0

    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return 0.0


class WatchlistTUI:
    """观察列表TUI

    显示内容：
    - 左侧：观察列表
    - 右上：选中标的详情
    - 右中：最新订单/成交
    - 右下：风控状态

    注意：TUI 只通过 EventLogger 读取状态，不直接访问 Broker
    """

    def __init__(self, config_path: str = "config.yaml", state_callback: Optional[StateCallback] = None):
        """初始化TUI

        Args:
            config_path: 配置文件路径
            state_callback: 状态回调接口（可选）
        """
        self.config = self._load_config(config_path)
        self.console = Console()
        self.event_logger = get_event_logger()
        self.selected_index = 0
        self.state_callback = state_callback  # 替代直接 broker 访问

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

    def set_state_callback(self, callback: StateCallback):
        """设置状态回调

        Args:
            callback: 状态回调接口
        """
        self.state_callback = callback

    def _get_position_from_events(self, symbol: str) -> Optional[Dict[str, Any]]:
        """从事件日志中获取持仓信息"""
        # 查询成交事件来推断持仓
        events = self.event_logger.query_events(symbol=symbol, stage="order_fill")
        if not events:
            return None

        # 计算持仓
        quantity = 0
        avg_price = 0.0
        total_cost = 0.0

        for event in events:
            payload = event.get('payload', {})
            side = payload.get('side', '')
            price = payload.get('price', 0)
            qty = payload.get('quantity', 0)

            if side == 'BUY':
                total_cost += price * qty
                quantity += qty
                if quantity > 0:
                    avg_price = total_cost / quantity
            elif side == 'SELL':
                quantity -= qty
                if quantity <= 0:
                    quantity = 0
                    total_cost = 0
                    avg_price = 0

        if quantity == 0:
            return None

        # 获取最新价格
        latest_price = self._get_latest_price(symbol)
        market_value = latest_price * quantity if latest_price else total_cost
        unrealized_pnl = (latest_price - avg_price) * quantity if latest_price else 0

        return {
            'symbol': symbol,
            'quantity': quantity,
            'avg_price': avg_price,
            'market_value': market_value,
            'unrealized_pnl': unrealized_pnl
        }

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """从事件日志获取最新价格"""
        events = self.event_logger.query_events(symbol=symbol)
        for event in reversed(events):
            payload = event.get('payload', {})
            if 'price' in payload:
                return payload['price']
        return None

    def _get_account_from_events(self) -> Dict[str, float]:
        """从事件日志获取账户信息"""
        # 查询系统事件获取初始资金
        events = self.event_logger.query_events(stage="system")
        initial_cash = 100000.0  # 默认值

        for event in events:
            payload = event.get('payload', {})
            if 'initial_cash' in payload:
                initial_cash = payload['initial_cash']
                break

        # 计算已实现盈亏
        pnl = 0.0
        fill_events = self.event_logger.query_events(stage="order_fill")
        for event in fill_events:
            payload = event.get('payload', {})
            # 这里简化处理，实际需要更复杂的计算
            commission = payload.get('commission', 0)
            pnl -= commission

        return {
            'cash': initial_cash + pnl,  # 简化计算
            'equity': initial_cash + pnl,
            'pnl': pnl
        }

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
        position = self._get_position_from_events(symbol)
        if position:
            lines.append(f"[bold]持仓:[/bold] {position['quantity']}股")
            lines.append(f"[bold]成本:[/bold] ${position['avg_price']:.2f}")
            lines.append(f"[bold]市值:[/bold] ${position['market_value']:,.2f}")
            lines.append(f"[bold]浮盈:[/bold] ${position['unrealized_pnl']:,.2f}\n")
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
        # 查询最近的订单事件
        events = self.event_logger.query_events()
        order_events = [e for e in events if e.get("stage") in ["order_submit", "order_fill", "order_reject"]]
        order_events = order_events[-10:] if order_events else []

        table = Table(
            title="最新订单事件",
            show_header=True,
            header_style="bold green",
            box=box.SIMPLE
        )

        table.add_column("时间", width=8)
        table.add_column("标的", width=8)
        table.add_column("状态", width=10)
        table.add_column("方向", width=6)
        table.add_column("价格", width=8)
        table.add_column("数量", width=8)

        for event in order_events:
            ts = datetime.fromisoformat(event['ts']).strftime('%H:%M:%S')
            symbol = event['symbol']
            payload = event['payload']
            stage = event.get("stage", "N/A")
            status = {
                "order_submit": "SUBMIT",
                "order_fill": "FILL",
                "order_reject": "REJECT"
            }.get(stage, "N/A")
            side = payload.get('side', 'N/A')
            price = payload.get('price', 0)
            quantity = payload.get('quantity', 0)

            # 颜色标记
            side_style = "green" if side == "BUY" else "red"

            table.add_row(
                ts,
                symbol,
                status,
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

        # 从事件日志获取账户信息
        account = self._get_account_from_events()
        cash = account.get('cash', 0)
        equity = account.get('equity', 0)
        pnl = account.get('pnl', 0)

        lines.append(f"可用资金: ${cash:,.2f}")
        lines.append(f"总权益: ${equity:,.2f}")
        lines.append(f"持仓数量: {len(self._get_positions_from_events())}/{risk_config.get('max_positions', 5)}")
        lines.append(f"仓位比例: {equity/cash*100 if cash > 0 else 0:.1f}%")

        # 计算单日盈亏
        initial_cash = self.config.get('account', {}).get('initial_cash', 100000)
        daily_loss_ratio = pnl / initial_cash if initial_cash > 0 else 0
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

    def _get_positions_from_events(self) -> Dict[str, Dict[str, Any]]:
        """从事件日志获取所有持仓"""
        positions = {}
        for symbol in self.watchlist:
            pos = self._get_position_from_events(symbol)
            if pos:
                positions[symbol] = pos
        return positions

    def _get_symbol_status(self, symbol: str) -> str:
        """获取标的状态

        Args:
            symbol: 标的代码

        Returns:
            状态文本
        """
        # 检查是否有持仓（从事件日志）
        position = self._get_position_from_events(symbol)
        if position and position.get('quantity', 0) > 0:
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
        # 从事件日志获取最新价格
        latest_price = self._get_latest_price(symbol)
        if latest_price:
            return f"${latest_price:.2f}"

        # 如果有持仓，显示成本价
        position = self._get_position_from_events(symbol)
        if position:
            return f"${position.get('avg_price', 0):.2f}"

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


class SimpleStateCallback(StateCallback):
    """简单的状态回调实现（用于测试）"""

    def __init__(self, initial_cash: float = 100000):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.positions.get(symbol)

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.positions

    def get_cash_balance(self) -> float:
        return self.cash

    def get_total_equity(self) -> float:
        equity = self.cash
        for pos in self.positions.values():
            equity += pos.get('market_value', 0)
        return equity

    def get_total_pnl(self) -> float:
        return self.get_total_equity() - self.initial_cash


def run_simple_display():
    """运行简单显示（测试用）"""
    print("\n=== TUI 简单显示演示 ===\n")

    # 创建状态回调
    state_callback = SimpleStateCallback(initial_cash=100000)

    # 模拟持仓
    state_callback.positions['AAPL'] = {
        'symbol': 'AAPL',
        'quantity': 100,
        'avg_price': 150.0,
        'market_value': 15050.0,
        'unrealized_pnl': 50.0
    }
    state_callback.cash = 84950.0

    # 创建TUI
    tui = WatchlistTUI(state_callback=state_callback)

    # 记录事件
    logger = get_event_logger()
    logger.log_event(
        ts=datetime.now(),
        symbol="AAPL",
        stage="order_fill",
        payload={"side": "BUY", "price": 150.0, "quantity": 100},
        reason="订单执行成功"
    )

    # 渲染布局
    layout = tui.render()

    # 打印到控制台
    console = Console()
    console.print(layout)

    print("\n✓ TUI显示完成")


if __name__ == "__main__":
    """自测代码"""
    run_simple_display()
