#!/usr/bin/env python3
"""
Live Trading Environment Verification Script
实时交易环境验证脚本

验证所有必需的依赖、配置和功能
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class LiveEnvVerifier:
    """环境验证器"""

    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, name: str, passed: bool, message: str = "", critical: bool = True):
        """记录检查结果"""
        self.checks.append({
            "name": name,
            "passed": passed,
            "message": message,
            "critical": critical
        })

        if passed:
            self.passed += 1
            logger.info(f"  ✅ {name}")
            if message:
                logger.info(f"     {message}")
        else:
            if critical:
                self.failed += 1
                logger.error(f"  ❌ {name}")
            else:
                self.warnings += 1
                logger.warning(f"  ⚠️  {name}")
            if message:
                logger.info(f"     {message}")

    def verify_all(self) -> bool:
        """运行所有验证"""
        logger.info("=" * 60)
        logger.info("🔍 Live Trading Environment Verification")
        logger.info("=" * 60)
        logger.info(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Python 版本
        logger.info("\n1️⃣  Python 版本")
        version = sys.version_info
        self.check(
            "Python 版本 >= 3.9",
            version.major == 3 and version.minor >= 9,
            f"当前: {version.major}.{version.minor}.{version.micro}"
        )

        # 2. 核心依赖包
        logger.info("\n2️⃣  核心依赖包")
        self._verify_dependencies()

        # 3. 项目模块
        logger.info("\n3️⃣  项目模块")
        self._verify_project_modules()

        # 4. 配置文件
        logger.info("\n4️⃣  配置文件")
        self._verify_configurations()

        # 5. 数据源连接
        logger.info("\n5️⃣  数据源连接")
        self._verify_data_source()

        # 6. SPL-7a 组件
        logger.info("\n6️⃣  SPL-7a 组件")
        self._verify_spl7a()

        # 7. LiveTrader 初始化
        logger.info("\n7️⃣  LiveTrader 初始化")
        self._verify_live_trader()

        # 8. RiskMonitor 功能
        logger.info("\n8️⃣  RiskMonitor 功能")
        self._verify_risk_monitor()

        # 打印总结
        self._print_summary()

        return self.failed == 0

    def _verify_dependencies(self):
        """验证依赖包"""
        dependencies = [
            ("dataclasses", "数据结构（内置于 Python 3.7+）"),
            ("yfinance", "Yahoo Finance 数据源"),
            ("pandas", "数据处理"),
            ("numpy", "数值计算"),
            ("pytz", "时区处理"),
        ]

        for module, description in dependencies:
            try:
                if module == "dataclasses":
                    import dataclasses
                    self.check(f"{module} ({description})", True, "已安装")
                else:
                    __import__(module)
                    self.check(f"{module} ({description})", True, "已安装")
            except ImportError:
                self.check(
                    f"{module} ({description})",
                    False,
                    "未安装，请运行: pip install -r requirements-live.txt",
                    critical=True
                )

    def _verify_project_modules(self):
        """验证项目模块"""
        modules = [
            ("core.models", "核心数据模型"),
            ("skills.market_data.yahoo_source", "Yahoo Finance 数据源"),
            ("skills.strategy.moving_average", "MA 策略"),
            ("skills.risk.basic_risk", "基础风控"),
            ("broker.paper", "Paper Broker"),
            ("agent.live_runner", "Live Trader"),
            ("analysis.online.risk_monitor", "SPL-7a RiskMonitor"),
        ]

        for module_path, description in modules:
            try:
                module = __import__(module_path, fromlist=[''])
                self.check(f"{module_path} ({description})", True, "可导入")
            except ImportError as e:
                self.check(
                    f"{module_path} ({description})",
                    False,
                    f"导入失败: {e}",
                    critical=True
                )

    def _verify_configurations(self):
        """验证配置文件"""
        configs = [
            ("config/live/live_config.yaml", "Live 交易配置"),
            ("config/live/alerts.yaml", "告警规则配置"),
        ]

        for config_path, description in configs:
            path = Path(config_path)
            exists = path.exists()

            if not exists:
                # 尝试创建目录
                path.parent.mkdir(parents=True, exist_ok=True)

            self.check(
                f"{config_path} ({description})",
                exists,
                "存在" if exists else "不存在（将使用默认配置）",
                critical=False
            )

            # 尝试解析 YAML
            if exists:
                try:
                    import yaml
                    with open(path) as f:
                        yaml.safe_load(f)
                    self.check(f"{config_path} (YAML 语法)", True, "语法正确")
                except Exception as e:
                    self.check(
                        f"{config_path} (YAML 语法)",
                        False,
                        f"语法错误: {e}",
                        critical=True
                    )

    def _verify_data_source(self):
        """验证数据源"""
        try:
            from skills.market_data.yahoo_source import YahooFinanceSource

            source = YahooFinanceSource(enable_cache=True)

            # 测试获取报价
            try:
                quote = source.get_quote("AAPL")
                self.check(
                    "Yahoo Finance 数据连接",
                    True,
                    f"AAPL 最新价: ${quote.bid_price}"
                )
            except Exception as e:
                self.check(
                    "Yahoo Finance 数据连接",
                    False,
                    f"连接失败: {e}\n请检查网络或使用 VPN",
                    critical=True
                )

        except ImportError as e:
            self.check(
                "Yahoo Finance 数据源",
                False,
                f"导入失败: {e}",
                critical=True
            )

    def _verify_spl7a(self):
        """验证 SPL-7a 组件"""
        components = [
            ("analysis.online.risk_monitor", "RiskMonitor"),
            ("analysis.online.risk_metrics_collector", "RiskMetricsCollector"),
            ("analysis.online.risk_state_machine", "RiskStateMachine"),
            ("analysis.online.trend_detector", "TrendDetector"),
            ("analysis.online.alerting", "AlertingManager"),
            ("analysis.online.postmortem_generator", "PostMortemGenerator"),
            ("analysis.online.risk_event_store", "RiskEventStore"),
        ]

        for module_path, class_name in components:
            try:
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
                self.check(f"{class_name}", True, "可导入")
            except (ImportError, AttributeError) as e:
                self.check(
                    f"{class_name}",
                    False,
                    f"导入失败: {e}",
                    critical=True
                )

    def _verify_live_trader(self):
        """验证 LiveTrader 初始化"""
        try:
            from agent.live_runner import LiveTrader, LiveTradingConfig, TradingMode

            config = LiveTradingConfig(
                mode=TradingMode.PAPER,
                symbols=["AAPL"],
                interval="5m",
                enable_risk_monitor=False  # 暂时禁用以验证基础功能
            )

            trader = LiveTrader(config=config)

            self.check(
                "LiveTrader 初始化",
                True,
                "可创建实例"
            )

        except Exception as e:
            self.check(
                "LiveTrader 初始化",
                False,
                f"初始化失败: {e}",
                critical=True
            )

    def _verify_risk_monitor(self):
        """验证 RiskMonitor 功能"""
        try:
            from analysis.online.risk_monitor import RiskMonitor

            monitor = RiskMonitor(
                strategy_id="test_strategy",
                symbols=["AAPL"],
                output_dir="data/test_verification"
            )

            self.check(
                "RiskMonitor 初始化",
                True,
                "可创建实例"
            )

            # 测试生成快照
            from core.models import Bar
            now = datetime.now()
            bar = Bar(
                symbol="AAPL",
                timestamp=now,
                open=Decimal("175.0"),
                high=Decimal("176.0"),
                low=Decimal("174.0"),
                close=Decimal("175.5"),
                volume=1000000,
                interval="1m"
            )

            signal = monitor.pre_decision_hook(
                symbol="AAPL",
                bar=bar,
                position=None,
                context={}
            )

            self.check(
                "RiskMonitor pre_decision_hook",
                signal is not None,
                "可生成风险信号"
            )

            # 检查快照文件
            snapshot_file = Path("data/test_verification/snapshots.jsonl")
            if snapshot_file.exists():
                with open(snapshot_file) as f:
                    content = f.read()
                    self.check(
                        "RiskMonitor 快照输出",
                        len(content) > 0,
                        f"快照文件已生成 ({len(content)} bytes)"
                    )

            # 测试关闭
            monitor.shutdown()

            self.check(
                "RiskMonitor shutdown",
                True,
                "可正常关闭"
            )

        except Exception as e:
            self.check(
                "RiskMonitor 功能",
                False,
                f"测试失败: {e}",
                critical=True
            )

    def _print_summary(self):
        """打印总结"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 验证总结")
        logger.info("=" * 60)
        logger.info(f"✅ 通过: {self.passed}")
        logger.info(f"⚠️  警告: {self.warnings}")
        logger.info(f"❌ 失败: {self.failed}")

        if self.failed == 0:
            logger.info("\n🎉 所有验证通过！Live Trading 环境就绪")
            logger.info("\n启动命令:")
            logger.info("  python scripts/run_live_with_monitoring.py \\")
            logger.info("      --config config/live/live_config.yaml \\")
            logger.info("      --symbols AAPL MSFT --cash 100000")
            return
        else:
            logger.info("\n⛔ 存在阻断性问题，请先解决上述失败项")
            logger.info("\n常见解决方案:")
            logger.info("  1. 安装依赖: pip install -r requirements-live.txt")
            logger.info("  2. 检查网络连接（Yahoo Finance 需要外网）")
            logger.info("  3. 检查配置文件语法")
            logger.info("  4. 检查目录权限")

        logger.info("=" * 60)


def main():
    """主函数"""
    verifier = LiveEnvVerifier()
    success = verifier.verify_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
