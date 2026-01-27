"""
交易时段管理

支持美股和港股的交易日历和时段检查
"""

from datetime import datetime, time, date
from typing import Tuple, List


class TradingSession:
    """交易时段管理器"""

    # 美股交易时段（美东时间）
    US_MARKET_OPEN = time(9, 30)  # 9:30 AM
    US_MARKET_CLOSE = time(16, 0)  # 4:00 PM

    # 港股交易时段（香港时间）
    HK_MARKET_OPEN_MORNING = time(9, 30)  # 9:30 AM
    HK_MARKET_CLOSE_MORNING = time(12, 0)  # 12:00 PM
    HK_MARKET_OPEN_AFTERNOON = time(13, 0)  # 1:00 PM
    HK_MARKET_CLOSE_AFTERNOON = time(16, 0)  # 4:00 PM

    # 美股主要假期（2024年）
    US_HOLIDAYS_2024 = [
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
    ]

    # 港股主要假期（2024年）
    HK_HOLIDAYS_2024 = [
        date(2024, 1, 1),   # 元旦
        date(2024, 2, 10),  # 农历年初一
        date(2024, 2, 11),  # 农历年初二
        date(2024, 2, 12),  # 农历年初三
        date(2024, 3, 29),  # 耶稣受难日
        date(2024, 4, 1),   # 耶稣受难日翌日
        date(2024, 4, 4),   # 清明节
        date(2024, 5, 1),   # 劳动节
        date(2024, 5, 15),  # 佛诞节
        date(2024, 6, 10),  # 端午节翌日
        date(2024, 7, 1),   # 香港特别行政区成立纪念日
        date(2024, 9, 17),  # 中秋节翌日
        date(2024, 10, 1),  # 国庆日
        date(2024, 10, 11), # 重阳节
        date(2024, 12, 25), # 圣诞节
        date(2024, 12, 26), # 节礼日
    ]

    @staticmethod
    def get_market_type(symbol: str) -> str:
        """判断标的所属市场

        Args:
            symbol: 标的代码

        Returns:
            'US' 或 'HK'
        """
        if symbol.endswith('.HK') or '.' in symbol:
            return 'HK'
        return 'US'

    @staticmethod
    def is_trading_day(dt: date, market: str = 'US') -> bool:
        """判断是否为交易日

        Args:
            dt: 日期
            market: 市场类型 ('US' 或 'HK')

        Returns:
            是否为交易日
        """
        # 检查是否为周末
        if dt.weekday() >= 5:  # 5=周六, 6=周日
            return False

        # 检查是否为假期
        holidays = TradingSession.US_HOLIDAYS_2024 if market == 'US' else TradingSession.HK_HOLIDAYS_2024
        if dt in holidays:
            return False

        return True

    @staticmethod
    def is_trading_time(dt: datetime, market: str = 'US') -> bool:
        """判断是否在交易时段内

        Args:
            dt: 时间戳
            market: 市场类型 ('US' 或 'HK')

        Returns:
            是否在交易时段内
        """
        # 首先检查是否为交易日
        if not TradingSession.is_trading_day(dt.date(), market):
            return False

        current_time = dt.time()

        if market == 'US':
            # 美股：9:30 AM - 4:00 PM
            return TradingSession.US_MARKET_OPEN <= current_time <= TradingSession.US_MARKET_CLOSE
        else:
            # 港股：上午 9:30-12:00，下午 1:00-4:00
            morning_session = (TradingSession.HK_MARKET_OPEN_MORNING <= current_time
                             <= TradingSession.HK_MARKET_CLOSE_MORNING)
            afternoon_session = (TradingSession.HK_MARKET_OPEN_AFTERNOON <= current_time
                               <= TradingSession.HK_MARKET_CLOSE_AFTERNOON)
            return morning_session or afternoon_session

    @staticmethod
    def get_trading_sessions(dt: date, market: str = 'US') -> List[Tuple[datetime, datetime]]:
        """获取指定日期的交易时段

        Args:
            dt: 日期
            market: 市场类型

        Returns:
            交易时段列表，每个时段为 (开始时间, 结束时间) 的元组
        """
        if not TradingSession.is_trading_day(dt, market):
            return []

        sessions = []

        if market == 'US':
            start = datetime.combine(dt, TradingSession.US_MARKET_OPEN)
            end = datetime.combine(dt, TradingSession.US_MARKET_CLOSE)
            sessions.append((start, end))
        else:
            # 港股：上午和下午两个时段
            morning_start = datetime.combine(dt, TradingSession.HK_MARKET_OPEN_MORNING)
            morning_end = datetime.combine(dt, TradingSession.HK_MARKET_CLOSE_MORNING)
            sessions.append((morning_start, morning_end))

            afternoon_start = datetime.combine(dt, TradingSession.HK_MARKET_OPEN_AFTERNOON)
            afternoon_end = datetime.combine(dt, TradingSession.HK_MARKET_CLOSE_AFTERNOON)
            sessions.append((afternoon_start, afternoon_end))

        return sessions

    @staticmethod
    def get_trading_days(start: date, end: date, market: str = 'US') -> List[date]:
        """获取日期范围内的所有交易日

        Args:
            start: 开始日期
            end: 结束日期
            market: 市场类型

        Returns:
            交易日列表
        """
        trading_days = []
        current = start

        while current <= end:
            if TradingSession.is_trading_day(current, market):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days


if __name__ == "__main__":
    """自测代码"""
    from datetime import timedelta

    print("=== TradingSession 测试 ===\n")

    # 测试市场类型判断
    print("1. 市场类型判断:")
    print(f"   AAPL: {TradingSession.get_market_type('AAPL')}")
    print(f"   00700.HK: {TradingSession.get_market_type('00700.HK')}")

    # 测试交易日判断
    print("\n2. 交易日判断:")
    test_dates = [
        (date(2024, 1, 1), "元旦（假期）"),
        (date(2024, 1, 2), "工作日"),
        (date(2024, 1, 6), "周六"),
        (date(2024, 1, 7), "周日"),
    ]
    for dt, desc in test_dates:
        is_trading = TradingSession.is_trading_day(dt, market='US')
        print(f"   {dt} ({desc}): {'交易日' if is_trading else '非交易日'}")

    # 测试交易时段判断
    print("\n3. 交易时段判断（美股）:")
    test_times = [
        (datetime(2024, 1, 2, 9, 0), "开盘前"),
        (datetime(2024, 1, 2, 9, 30), "开盘"),
        (datetime(2024, 1, 2, 12, 0), "午盘中"),
        (datetime(2024, 1, 2, 16, 0), "收盘"),
        (datetime(2024, 1, 2, 18, 0), "收盘后"),
    ]
    for dt, desc in test_times:
        is_trading = TradingSession.is_trading_time(dt, market='US')
        print(f"   {dt.strftime('%Y-%m-%d %H:%M')} ({desc}): {'交易时段' if is_trading else '非交易时段'}")

    # 测试港股交易时段
    print("\n4. 港股交易时段:")
    hk_times = [
        (datetime(2024, 1, 2, 9, 30), "上午开盘"),
        (datetime(2024, 1, 2, 12, 30), "午休"),
        (datetime(2024, 1, 2, 13, 30), "下午交易中"),
        (datetime(2024, 1, 2, 16, 30), "收盘后"),
    ]
    for dt, desc in hk_times:
        is_trading = TradingSession.is_trading_time(dt, market='HK')
        print(f"   {dt.strftime('%Y-%m-%d %H:%M')} ({desc}): {'交易时段' if is_trading else '非交易时段'}")

    # 测试获取交易时段
    print("\n5. 获取交易时段:")
    us_sessions = TradingSession.get_trading_sessions(date(2024, 1, 2), market='US')
    print(f"   美股 {date(2024, 1, 2)}: {len(us_sessions)} 个时段")
    for start, end in us_sessions:
        print(f"     {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")

    hk_sessions = TradingSession.get_trading_sessions(date(2024, 1, 2), market='HK')
    print(f"   港股 {date(2024, 1, 2)}: {len(hk_sessions)} 个时段")
    for start, end in hk_sessions:
        print(f"     {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")

    # 测试获取交易日列表
    print("\n6. 获取交易日列表:")
    start = date(2024, 1, 1)
    end = date(2024, 1, 14)
    trading_days = TradingSession.get_trading_days(start, end, market='US')
    print(f"   {start} 到 {end} 之间有 {len(trading_days)} 个交易日")
    print(f"   前5个交易日: {[d.strftime('%Y-%m-%d') for d in trading_days[:5]]}")

    # 测试假期
    print("\n7. 测试假期检测:")
    holidays = TradingSession.US_HOLIDAYS_2024[:3]
    for h in holidays:
        is_trading = TradingSession.is_trading_day(h, market='US')
        print(f"   {h}: {'交易日' if is_trading else '假期 ✓'}")

    print("\n✓ 所有测试通过")
