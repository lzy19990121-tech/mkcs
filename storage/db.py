"""
数据持久化层

使用SQLite存储交易记录、K线数据和持仓快照
"""

import sqlite3
import json
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from core.models import Bar, Trade, Position


class TradeDB:
    """交易数据库"""

    def __init__(self, db_path: str = "trading.db"):
        """初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        cursor = self.conn.cursor()

        # 创建交易记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                commission TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建K线数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open TEXT NOT NULL,
                high TEXT NOT NULL,
                low TEXT NOT NULL,
                close TEXT NOT NULL,
                volume INTEGER NOT NULL,
                interval TEXT NOT NULL,
                UNIQUE(symbol, timestamp, interval)
            )
        """)

        # 创建持仓快照表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_price TEXT NOT NULL,
                market_value TEXT NOT NULL,
                unrealized_pnl TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol ON bars(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bars_timestamp ON bars(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_date ON positions(snapshot_date)")

        self.conn.commit()

    def save_trade(self, trade: Trade) -> None:
        """保存交易记录

        Args:
            trade: 交易对象
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, symbol, side, price, quantity, timestamp, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id,
            trade.symbol,
            trade.side,
            str(trade.price),
            trade.quantity,
            trade.timestamp.isoformat(),
            str(trade.commission)
        ))
        self.conn.commit()

    def save_bar(self, bar: Bar) -> None:
        """保存K线数据

        Args:
            bar: K线对象
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO bars
            (symbol, timestamp, open, high, low, close, volume, interval)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bar.symbol,
            bar.timestamp.isoformat(),
            str(bar.open),
            str(bar.high),
            str(bar.low),
            str(bar.close),
            bar.volume,
            bar.interval
        ))
        self.conn.commit()

    def save_position(self, position: Position, snapshot_date: date) -> None:
        """保存持仓快照

        Args:
            position: 持仓对象
            snapshot_date: 快照日期
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO positions
            (symbol, quantity, avg_price, market_value, unrealized_pnl, snapshot_date)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            position.symbol,
            position.quantity,
            str(position.avg_price),
            str(position.market_value),
            str(position.unrealized_pnl),
            snapshot_date.isoformat()
        ))
        self.conn.commit()

    def get_trades(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbol: Optional[str] = None
    ) -> List[Trade]:
        """查询交易记录

        Args:
            start: 开始时间
            end: 结束时间
            symbol: 标的代码（可选）

        Returns:
            交易记录列表
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        trades = []
        for row in rows:
            trade = Trade(
                trade_id=row["trade_id"],
                symbol=row["symbol"],
                side=row["side"],
                price=Decimal(row["price"]),
                quantity=row["quantity"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                commission=Decimal(row["commission"])
            )
            trades.append(trade)

        return trades

    def get_bars(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        interval: Optional[str] = None
    ) -> List[Bar]:
        """查询K线数据

        Args:
            symbol: 标的代码
            start: 开始时间
            end: 结束时间
            interval: 时间周期（可选）

        Returns:
            K线数据列表
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM bars WHERE symbol = ?"
        params = [symbol]

        if start:
            query += " AND timestamp >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND timestamp <= ?"
            params.append(end.isoformat())
        if interval:
            query += " AND interval = ?"
            params.append(interval)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        bars = []
        for row in rows:
            bar = Bar(
                symbol=row["symbol"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=row["volume"],
                interval=row["interval"]
            )
            bars.append(bar)

        return bars

    def get_positions(self, snapshot_date: Optional[date] = None) -> List[Position]:
        """查询持仓快照

        Args:
            snapshot_date: 快照日期（可选，默认最新）

        Returns:
            持仓列表
        """
        cursor = self.conn.cursor()

        if snapshot_date:
            query = "SELECT * FROM positions WHERE snapshot_date = ?"
            params = [snapshot_date.isoformat()]
        else:
            query = "SELECT * FROM positions ORDER BY snapshot_date DESC LIMIT 100"
            params = []

        cursor.execute(query, params)
        rows = cursor.fetchall()

        positions = []
        for row in rows:
            position = Position(
                symbol=row["symbol"],
                quantity=row["quantity"],
                avg_price=Decimal(row["avg_price"]),
                market_value=Decimal(row["market_value"]),
                unrealized_pnl=Decimal(row["unrealized_pnl"])
            )
            positions.append(position)

        return positions

    def get_trade_count(self, symbol: Optional[str] = None) -> int:
        """获取交易次数

        Args:
            symbol: 标的代码（可选）

        Returns:
            交易次数
        """
        cursor = self.conn.cursor()
        if symbol:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE symbol = ?", (symbol,))
        else:
            cursor.execute("SELECT COUNT(*) FROM trades")
        return cursor.fetchone()[0]

    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


if __name__ == "__main__":
    """自测代码"""
    import tempfile
    import os

    print("=== TradeDB 测试 ===\n")

    # 创建临时数据库
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        temp_db = f.name

    try:
        # 初始化数据库
        db = TradeDB(temp_db)
        print("1. 数据库初始化成功")

        # 测试保存交易
        trade1 = Trade(
            trade_id="T001",
            symbol="AAPL",
            side="BUY",
            price=Decimal("150.00"),
            quantity=100,
            timestamp=datetime(2024, 1, 1, 10, 30),
            commission=Decimal("1.00")
        )
        db.save_trade(trade1)
        print("2. 保存交易记录成功")

        # 测试保存K线
        bar1 = Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1),
            open=Decimal("150.0"),
            high=Decimal("155.0"),
            low=Decimal("149.0"),
            close=Decimal("154.0"),
            volume=1000000,
            interval="1d"
        )
        db.save_bar(bar1)
        print("3. 保存K线数据成功")

        # 测试保存持仓
        position1 = Position(
            symbol="AAPL",
            quantity=100,
            avg_price=Decimal("150.00"),
            market_value=Decimal("15400.00"),
            unrealized_pnl=Decimal("400.00")
        )
        db.save_position(position1, date(2024, 1, 1))
        print("4. 保存持仓快照成功")

        # 测试查询交易
        trades = db.get_trades(symbol="AAPL")
        print(f"\n5. 查询交易记录: 找到 {len(trades)} 条")
        for trade in trades:
            print(f"   {trade.trade_id}: {trade.side} {trade.quantity} "
                  f"{trade.symbol} @ {trade.price}")

        # 测试查询K线
        bars = db.get_bars("AAPL")
        print(f"\n6. 查询K线数据: 找到 {len(bars)} 条")
        for bar in bars:
            print(f"   {bar.timestamp.strftime('%Y-%m-%d')}: {bar.close}")

        # 测试查询持仓
        positions = db.get_positions(snapshot_date=date(2024, 1, 1))
        print(f"\n7. 查询持仓快照: 找到 {len(positions)} 个")
        for pos in positions:
            print(f"   {pos.symbol}: {pos.quantity} 股, 成本 {pos.avg_price}")

        # 测试统计
        count = db.get_trade_count(symbol="AAPL")
        print(f"\n8. AAPL交易次数: {count}")

        # 测试批量保存
        print("\n9. 批量保存测试:")
        for i in range(5):
            trade = Trade(
                trade_id=f"T{i+100:03d}",
                symbol="GOOGL",
                side="BUY" if i % 2 == 0 else "SELL",
                price=Decimal(f"14{i}.00"),
                quantity=100 + i * 10,
                timestamp=datetime(2024, 1, i + 1, 10, 30),
                commission=Decimal("1.00")
            )
            db.save_trade(trade)

        googl_trades = db.get_trades(symbol="GOOGL")
        print(f"   GOOGL 交易记录: {len(googl_trades)} 条")

        # 测试时间范围查询
        print("\n10. 时间范围查询测试:")
        start_time = datetime(2024, 1, 2)
        end_time = datetime(2024, 1, 4)
        ranged_trades = db.get_trades(start=start_time, end=end_time)
        print(f"   2024-01-02 到 2024-01-04 的交易: {len(ranged_trades)} 条")

        db.close()
        print("\n✓ 所有测试通过")

    finally:
        # 清理临时文件
        if os.path.exists(temp_db):
            os.remove(temp_db)
