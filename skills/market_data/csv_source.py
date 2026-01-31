"""
CSV 文件数据源

用于 replay_real 模式，从 CSV 文件读取历史数据
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

from core.models import Bar
from skills.market_data.base import MarketDataSource


class CSVDataSource(MarketDataSource):
    """CSV 文件数据源 - 用于 replay_real 模式"""

    def __init__(
        self,
        data_dir: str,
        symbol_map: Optional[Dict[str, str]] = None,
        date_format: str = "%Y-%m-%d"
    ):
        """初始化 CSV 数据源

        Args:
            data_dir: CSV 文件目录
            symbol_map: 符号到文件名的映射，如 {"AAPL": "aapl_2022_2024.csv"}
            date_format: 日期列解析格式
        """
        self.data_dir = Path(data_dir)
        self.symbol_map = symbol_map or {}
        self.date_format = date_format
        self._cache: Dict[str, List[Bar]] = {}
        self._file_hashes: Dict[str, str] = {}

    def _get_file_path(self, symbol: str) -> Path:
        """获取符号对应的文件路径"""
        if symbol in self.symbol_map:
            filename = self.symbol_map[symbol]
        else:
            filename = f"{symbol.lower()}.csv"
        return self.data_dir / filename

    def _load_csv(self, symbol: str) -> List[Bar]:
        """加载 CSV 文件并转换为 Bar 列表"""
        if symbol in self._cache:
            return self._cache[symbol]

        file_path = self._get_file_path(symbol)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {file_path}")

        # 读取 CSV
        df = pd.read_csv(file_path)

        # 标准化列名
        df.columns = df.columns.str.lower().str.strip()

        # 解析日期列
        date_cols = ['date', 'timestamp', 'datetime', 'time']
        date_col = None
        for col in date_cols:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            raise ValueError(f"CSV 文件缺少日期列: {file_path}")

        df[date_col] = pd.to_datetime(df[date_col])

        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 文件缺少列 {missing}: {file_path}")

        # 按日期排序
        df = df.sort_values(date_col)

        # 转换为 Bar 对象
        bars = []
        for _, row in df.iterrows():
            bar = Bar(
                symbol=symbol,
                timestamp=row[date_col].to_pydatetime(),
                open=Decimal(str(row['open'])),
                high=Decimal(str(row['high'])),
                low=Decimal(str(row['low'])),
                close=Decimal(str(row['close'])),
                volume=int(row['volume']),
                interval="1d"  # 默认日线
            )
            bars.append(bar)

        self._cache[symbol] = bars
        return bars

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取 K 线数据

        Args:
            symbol: 标的代码
            start: 开始时间
            end: 结束时间
            interval: 时间周期（目前只支持 1d）

        Returns:
            K 线数据列表
        """
        all_bars = self._load_csv(symbol)

        # 过滤日期范围
        filtered = [
            b for b in all_bars
            if start <= b.timestamp <= end
        ]

        return filtered

    def get_bars_until(
        self,
        symbol: str,
        end: datetime,
        interval: str
    ) -> List[Bar]:
        """获取截至某个时间点的 K 线数据

        Args:
            symbol: 标的代码
            end: 截止时间
            interval: 时间周期

        Returns:
            K 线数据列表
        """
        all_bars = self._load_csv(symbol)

        # 过滤日期范围
        filtered = [b for b in all_bars if b.timestamp <= end]

        return filtered

    def get_quote(self, symbol: str):
        """获取实时报价

        CSV 数据源不支持实时报价
        """
        raise NotImplementedError("CSV 数据源不支持实时报价")

    def get_data_hash(self, symbol: str) -> Optional[str]:
        """获取数据文件的哈希值

        Args:
            symbol: 标的代码

        Returns:
            文件哈希值
        """
        from utils.hash import compute_data_hash

        file_path = self._get_file_path(symbol)
        if not file_path.exists():
            return None

        return compute_data_hash(str(file_path))

    def get_all_hashes(self) -> Dict[str, str]:
        """获取所有数据文件的哈希值"""
        hashes = {}
        for symbol in self.symbol_map.keys():
            hash_value = self.get_data_hash(symbol)
            if hash_value:
                hashes[symbol] = hash_value
        return hashes

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()


class MultiCSVDataSource(CSVDataSource):
    """多文件 CSV 数据源"""

    def __init__(
        self,
        symbol_files: Dict[str, str],
        date_format: str = "%Y-%m-%d"
    ):
        """初始化多文件数据源

        Args:
            symbol_files: 符号到文件路径的映射
            date_format: 日期格式
        """
        # 提取目录和文件名映射
        paths = {s: Path(p) for s, p in symbol_files.items()}

        # 检查是否在同一目录
        dirs = set(p.parent for p in paths.values())
        if len(dirs) > 1:
            raise ValueError("所有 CSV 文件必须在同一目录")

        data_dir = list(dirs)[0]
        symbol_map = {s: p.name for s, p in paths.items()}

        super().__init__(str(data_dir), symbol_map, date_format)


if __name__ == "__main__":
    """自测代码"""
    print("=== CSVDataSource 测试 ===\n")

    # 创建测试 CSV 文件
    import tempfile
    import os

    test_dir = tempfile.mkdtemp()
    test_file = os.path.join(test_dir, "aapl.csv")

    # 生成测试数据
    test_data = []
    base_date = datetime(2024, 1, 1)
    base_price = 150.0

    for i in range(30):
        date = base_date + pd.Timedelta(days=i)
        if date.weekday() >= 5:  # 跳过周末
            continue

        price = base_price + (i * 0.5) + (i % 3 - 1) * 2
        test_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": price - 1,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 1000000 + i * 1000
        })

    df = pd.DataFrame(test_data)
    df.to_csv(test_file, index=False)

    print(f"1. 创建测试数据: {len(test_data)} 条记录")

    # 测试数据源
    print("\n2. 加载 CSV 数据源:")
    source = CSVDataSource(
        data_dir=test_dir,
        symbol_map={"AAPL": "aapl.csv"}
    )

    # 测试获取数据
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 15)
    bars = source.get_bars("AAPL", start, end, "1d")
    print(f"   获取到 {len(bars)} 根 K 线")

    if bars:
        print(f"   第一根: {bars[0].timestamp.date()} 开={bars[0].open} 收={bars[0].close}")
        print(f"   最后一根: {bars[-1].timestamp.date()} 开={bars[-1].open} 收={bars[-1].close}")

    # 测试 get_bars_until
    print("\n3. 测试 get_bars_until:")
    bars_until = source.get_bars_until("AAPL", datetime(2024, 1, 10), "1d")
    print(f"   截至 2024-01-10 有 {len(bars_until)} 根 K 线")

    # 测试数据哈希
    print("\n4. 数据文件哈希:")
    data_hash = source.get_data_hash("AAPL")
    print(f"   哈希: {data_hash}")

    # 测试缓存
    print("\n5. 缓存测试:")
    bars2 = source.get_bars("AAPL", start, end, "1d")
    print(f"   再次获取相同数据: {len(bars2)} 根 K 线")

    # 清理
    import shutil
    shutil.rmtree(test_dir)

    print("\n✓ 所有测试通过")
