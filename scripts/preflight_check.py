"""
实时交易预检脚本

在启动实时交易前检查所有必需项
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class PreflightChecker:
    """预检检查器"""

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

    def run_all_checks(self):
        """运行所有检查"""
        logger.info("=" * 60)
        logger.info("🔍 实时交易预检")
        logger.info("=" * 60)
        logger.info(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

        # 1. Python 环境检查
        self._check_python_environment()

        # 2. 依赖包检查
        self._check_dependencies()

        # 3. 目录结构检查
        self._check_directories()

        # 4. 数据源检查
        self._check_data_source()

        # 5. SPL-7a 组件检查
        self._check_spl7a_components()

        # 6. 配置文件检查
        self._check_configurations()

        # 7. 时区检查
        self._check_timezone()

        # 打印总结
        self._print_summary()

        return self.failed == 0

    def _check_python_environment(self):
        """检查 Python 环境"""
        logger.info("1️⃣  Python 环境检查")
        version = sys.version_info
        self.check(
            "Python 版本",
            version.major == 3 and version.minor >= 9,
            f"当前: {version.major}.{version.minor}.{version.micro}",
            critical=True
        )

    def _check_dependencies(self):
        """检查依赖包"""
        logger.info("\n2️⃣  依赖包检查")

        dependencies = [
            ("yfinance", "Yahoo Finance 数据源"),
            ("pandas", "数据处理"),
            ("numpy", "数值计算"),
            ("dataclasses", "数据结构（内置于 Python 3.7+）"),
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
                    "未安装，请运行: pip install " + module,
                    critical=True
                )

    def _check_directories(self):
        """检查目录结构"""
        logger.info("\n3️⃣  目录结构检查")

        required_dirs = [
            ("data", "数据存储目录"),
            ("runs", "回测结果目录"),
            ("analysis/online", "SPL-7a 在线监控模块"),
            ("analysis/counterfactual", "SPL-7b 反事实分析模块"),
        ]

        for dir_path, description in required_dirs:
            path = Path(dir_path)
            exists = path.exists()
            if not exists:
                path.mkdir(parents=True, exist_ok=True)

            self.check(
                f"{dir_path} ({description})",
                True,
                "已存在" if exists else "已创建"
            )

    def _check_data_source(self):
        """检查数据源"""
        logger.info("\n4️⃣  数据源检查")

        try:
            from skills.market_data.yahoo_source import YahooFinanceSource

            source = YahooFinanceSource(enable_cache=True)

            # 尝试获取 AAPL 最新价格
            try:
                quote = source.get_quote("AAPL")
                self.check(
                    "Yahoo Finance 数据连接",
                    True,
                    f"AAPL 最新价: ${quote.bid_price}",
                    critical=True
                )
            except Exception as e:
                self.check(
                    "Yahoo Finance 数据连接",
                    False,
                    f"连接失败: {e}",
                    critical=True
                )

        except ImportError as e:
            self.check(
                "Yahoo Finance 数据源",
                False,
                f"导入失败: {e}",
                critical=True
            )

    def _check_spl7a_components(self):
        """检查 SPL-7a 组件"""
        logger.info("\n5️⃣  SPL-7a 组件检查")

        components = [
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

    def _check_configurations(self):
        """检查配置文件"""
        logger.info("\n6️⃣  配置文件检查")

        configs = [
            ("config/online_metrics.yaml", "在线监控指标配置"),
            ("config/alerting_rules.yaml", "告警规则配置"),
        ]

        for config_path, description in configs:
            path = Path(config_path)
            self.check(
                f"{config_path} ({description})",
                path.exists(),
                "存在" if path.exists() else "不存在（将使用默认配置）",
                critical=False
            )

    def _check_timezone(self):
        """检查时区"""
        logger.info("\n7️⃣  时区检查")

        from datetime import datetime
        import pytz

        try:
            # 检查本地时区
            local_tz = datetime.now().astimezone().tzinfo
            logger.info(f"  本地时区: {local_tz}")

            # 检查美东时区（美股市场）
            eastern = pytz.timezone('America/New_York')
            eastern_time = datetime.now(eastern)
            logger.info(f"  美东时间: {eastern_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

            # 检查是否在交易时段
            hour = eastern_time.hour
            minute = eastern_time.minute
            current_time = hour + minute / 60

            # 美股交易时间: 9:30 - 16:00 ET
            market_open = 9.5
            market_close = 16.0

            if market_open <= current_time <= market_close:
                self.check(
                    "市场时段",
                    True,
                    "✅ 当前在美股交易时段内！可以启动",
                    critical=False
                )
            else:
                self.check(
                    "市场时段",
                    True,
                    f"⚠️  当前不在交易时段（{eastern_time.strftime('%H:%M')} ET），交易器将等待开盘",
                    critical=False
                )

        except ImportError:
            self.check(
                "时区检查",
                True,
                "pytz 未安装，跳过（不影响运行）",
                critical=False
            )

    def _print_summary(self):
        """打印检查总结"""
        logger.info("\n" + "=" * 60)
        logger.info("📋 预检总结")
        logger.info("=" * 60)
        logger.info(f"✅ 通过: {self.passed}")
        logger.info(f"⚠️  警告: {self.warnings}")
        logger.info(f"❌ 失败: {self.failed}")

        if self.failed == 0:
            logger.info("\n🎉 所有检查通过！可以启动实时交易")
            logger.info("\n启动命令:")
            logger.info("  python scripts/run_live_with_monitoring.py --symbols AAPL MSFT --cash 100000")
        else:
            logger.info("\n⛔ 存在阻断性问题，请先解决上述失败项")
            logger.info("\n常见解决方案:")
            logger.info("  1. 安装依赖: pip install yfinance pandas numpy")
            logger.info("  2. 检查网络连接（Yahoo Finance 需要外网）")
            logger.info("  3. 检查目录权限")

        logger.info("=" * 60)


def main():
    """主函数"""
    checker = PreflightChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
