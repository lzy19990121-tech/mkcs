"""
数据库初始化脚本

创建 SQLite 数据库和表
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from web.db.models import Base, Annotation, SellRange, TradeRecord, Watchlist

# 数据库路径
DATA_DIR = 'data'
DB_PATH = os.path.join(DATA_DIR, 'trading.db')


def get_engine():
    """获取数据库引擎"""
    os.makedirs(DATA_DIR, exist_ok=True)
    db_url = f'sqlite:///{DB_PATH}'
    return create_engine(db_url, echo=False)


def init_database(seed: bool = True):
    """初始化数据库"""
    engine = get_engine()

    # 创建所有表
    Base.metadata.create_all(engine)

    # 添加种子数据
    if seed:
        session = get_session()
        try:
            seed_watchlist(session)
        finally:
            session.close()

    print(f"✅ 数据库初始化完成: {DB_PATH}")
    print("   已创建表: annotations, sell_ranges, trade_records, watchlist")

    return engine


def get_session():
    """获取数据库会话"""
    engine = get_engine()
    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)


def seed_watchlist(session):
    """添加默认观察列表"""
    default_symbols = [
        ('AAPL', '苹果公司', 1),
        ('GOOGL', '谷歌', 2),
        ('MSFT', '微软', 3),
        ('AMZN', '亚马逊', 4),
        ('NVDA', '英伟达', 5),
        ('TSLA', '特斯拉', 6),
        ('META', 'Meta', 7),
        ('SPY', '标普500 ETF', 8),
    ]

    for symbol, name, order in default_symbols:
        exists = session.query(Watchlist).filter_by(symbol=symbol).first()
        if not exists:
            watchlist = Watchlist(
                symbol=symbol,
                display_name=name,
                sort_order=order
            )
            session.add(watchlist)

    session.commit()
    print("✅ 默认观察列表已添加")


if __name__ == '__main__':
    engine = init_database()
    session = get_session()

    try:
        seed_watchlist(session)
    finally:
        session.close()
