"""
健康监控模块

监控paper模式运行的健康状态
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """健康指标"""
    # 数据延迟
    data_lag_seconds: float = 0.0

    # 错误计数
    error_count: int = 0
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    last_error_message: str = ""

    # 订单状态
    orders_pending: int = 0
    orders_last_hour: int = 0

    # 心跳
    last_heartbeat: Optional[datetime] = None
    last_check_time: Optional[datetime] = None

    # 运行时间
    started_at: Optional[datetime] = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "data_lag_seconds": self.data_lag_seconds,
            "error_count": self.error_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "last_error_message": self.last_error_message,
            "orders_pending": self.orders_pending,
            "orders_last_hour": self.orders_last_hour,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "uptime_seconds": self.uptime_seconds
        }


class HealthMonitor:
    """健康监控器

    监控paper模式的运行状态，提供健康检查和告警
    """

    def __init__(
        self,
        max_consecutive_errors: int = 5,
        max_data_lag_seconds: float = 300.0,
        heartbeat_interval_seconds: int = 60
    ):
        """初始化健康监控器

        Args:
            max_consecutive_errors: 最大连续错误数
            max_data_lag_seconds: 最大数据延迟（秒）
            heartbeat_interval_seconds: 心跳间隔（秒）
        """
        self.max_consecutive_errors = max_consecutive_errors
        self.max_data_lag_seconds = max_data_lag_seconds
        self.heartbeat_interval = heartbeat_interval_seconds

        self.metrics = HealthMetrics(
            started_at=datetime.now(),
            last_heartbeat=datetime.now()
        )

        # 错误历史（最近100个）
        self.error_history: deque = deque(maxlen=100)

        # 数据延迟历史（最近100个）
        self.data_lag_history: deque = deque(maxlen=100)

    def record_heartbeat(self):
        """记录心跳"""
        self.metrics.last_heartbeat = datetime.now()
        self.metrics.last_check_time = datetime.now()

        # 更新运行时间
        if self.metrics.started_at:
            self.metrics.uptime_seconds = (
                datetime.now() - self.metrics.started_at
            ).total_seconds()

    def record_error(self, error: Exception):
        """记录错误"""
        self.metrics.error_count += 1
        self.metrics.consecutive_errors += 1
        self.metrics.last_error_time = datetime.now()
        self.metrics.last_error_message = str(error)[:100]

        self.error_history.append({
            "time": datetime.now().isoformat(),
            "error": str(error)[:200],
            "type": type(error).__name__
        })

        logger.error(f"错误记录: {error}")

    def record_success(self):
        """记录成功操作"""
        self.metrics.consecutive_errors = 0

    def record_data_lag(self, lag_seconds: float):
        """记录数据延迟"""
        self.metrics.data_lag_seconds = lag_seconds
        self.data_lag_history.append({
            "time": datetime.now().isoformat(),
            "lag_seconds": lag_seconds
        })

    def update_orders_status(self, pending: int, last_hour: int):
        """更新订单状态"""
        self.metrics.orders_pending = pending
        self.metrics.orders_last_hour = last_hour

    def is_healthy(self) -> bool:
        """检查系统是否健康"""
        # 检查连续错误
        if self.metrics.consecutive_errors >= self.max_consecutive_errors:
            logger.warning(f"系统不健康: 连续错误 {self.metrics.consecutive_errors}")
            return False

        # 检查数据延迟
        if self.metrics.data_lag_seconds > self.max_data_lag_seconds:
            logger.warning(f"系统不健康: 数据延迟 {self.metrics.data_lag_seconds}s")
            return False

        # 检查心跳
        if self.metrics.last_heartbeat:
            heartbeat_age = (datetime.now() - self.metrics.last_heartbeat).total_seconds()
            if heartbeat_age > self.heartbeat_interval * 3:
                logger.warning(f"系统不健康: 心跳超时 {heartbeat_age}s")
                return False

        return True

    def get_status(self) -> Dict:
        """获取健康状态"""
        return {
            "healthy": self.is_healthy(),
            "metrics": self.metrics.to_dict(),
            "summary": self._get_summary()
        }

    def _get_summary(self) -> str:
        """获取状态摘要"""
        issues = []

        if self.metrics.consecutive_errors > 0:
            issues.append(f"连续错误: {self.metrics.consecutive_errors}")

        if self.metrics.data_lag_seconds > 60:
            issues.append(f"数据延迟: {self.metrics.data_lag_seconds:.0f}s")

        if self.metrics.orders_pending > 10:
            issues.append(f"挂单过多: {self.metrics.orders_pending}")

        if not issues:
            return "✅ 系统正常"

        return "⚠️ " + ", ".join(issues)


class RetryManager:
    """重试管理器

    处理断线重试和限频保护
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        rate_limit_calls: int = 10,
        rate_limit_period: float = 60.0
    ):
        """初始化重试管理器

        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟（秒）
            max_delay: 最大延迟（秒）
            rate_limit_calls: 限频调用次数
            rate_limit_period: 限频周期（秒）
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period

        # 调用历史
        self.call_history: deque = deque()

    def retry_with_backoff(self, func, *args, **kwargs):
        """带退避的重试

        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数

        Returns:
            函数返回值
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"重试成功 (第{attempt + 1}次尝试)")
                return result

            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"尝试失败 (第{attempt + 1}次)，"
                        f"{delay:.1f}秒后重试: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"达到最大重试次数: {e}")

        raise last_error

    def check_rate_limit(self) -> bool:
        """检查是否超过限频

        Returns:
            True 表示允许调用，False 表示需要限频
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.rate_limit_period)

        # 清理旧记录
        self.call_history = deque([
            t for t in self.call_history
            if t > cutoff
        ])

        # 检查调用次数
        if len(self.call_history) >= self.rate_limit_calls:
            logger.warning(f"限频触发: {len(self.call_history)}/{self.rate_limit_calls}")
            return False

        self.call_history.append(now)
        return True


if __name__ == "__main__":
    """测试代码"""
    print("=== HealthMonitor 测试 ===\n")

    monitor = HealthMonitor()

    # 测试心跳
    monitor.record_heartbeat()
    print(f"健康状态: {monitor.get_status()}")

    # 测试错误记录
    try:
        raise ValueError("测试错误")
    except Exception as e:
        monitor.record_error(e)

    monitor.record_success()
    print(f"健康状态: {monitor.get_status()}")

    # 测试重试
    retry_manager = RetryManager(max_retries=3)

    call_count = [0]

    def failing_func():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("模拟断线")
        return "success"

    result = retry_manager.retry_with_backoff(failing_func)
    print(f"\n重试结果: {result}")
    print(f"调用次数: {call_count[0]}")

    print("\n✓ 测试通过")
