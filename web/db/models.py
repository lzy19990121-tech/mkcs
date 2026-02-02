"""
数据库模型定义 - SQLAlchemy ORM

用于存储用户标注和交易记录
"""

from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Enum as SQLEnum, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MarkerType(str, Enum):
    """标注类型"""
    BUY = "buy"
    SELL = "sell"
    ENTRY = "entry"  # 入场点
    EXIT = "exit"   # 出场点


class OrderStatus(str, Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Annotation(Base):
    """用户图表标注（买卖点）"""

    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    marker_type = Column(SQLEnum(MarkerType), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=True)  # 可选：计划交易数量
    timestamp = Column(DateTime, nullable=False)  # 标注在图表上的时间点
    notes = Column(Text, nullable=True)  # 用户备注
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 索引
    __table_args__ = (
        Index('idx_annotations_symbol', 'symbol'),
    )


class SellRange(Base):
    """卖出区间（价格范围框）"""

    __tablename__ = "sell_ranges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    start_time = Column(DateTime, nullable=False)  # 区间开始时间
    end_time = Column(DateTime, nullable=False)    # 区间结束时间
    target_price = Column(Float, nullable=True)    # 目标价格（可选）
    notes = Column(Text, nullable=True)  # 用户备注
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 索引
    __table_args__ = (
        Index('idx_sell_ranges_symbol', 'symbol'),
    )


class TradeRecord(Base):
    """交易记录（用户手动记录或系统成交）"""

    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    commission = Column(Float, default=0.0)
    realized_pnl = Column(Float, nullable=True)  # 实现盈亏（平仓时填写）
    notes = Column(Text, nullable=True)

    # 时间戳
    trade_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联到标注（可选）
    annotation_id = Column(Integer, ForeignKey('annotations.id'), nullable=True)

    # 索引
    __table_args__ = (
        Index('idx_trade_records_symbol', 'symbol'),
        Index('idx_trade_records_time', 'trade_time'),
    )


class Watchlist(Base):
    """观察列表"""

    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, unique=True)
    display_name = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 索引
    __table_args__ = (
        Index('idx_watchlist_order', 'sort_order'),
    )
