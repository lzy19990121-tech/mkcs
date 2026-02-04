import React, { useEffect, useRef, useCallback, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { Space, Button, Segmented } from 'antd';
import { PlusOutlined } from '@ant-design/icons';

// 时间周期层级，用于智能切换
const INTERVAL_HIERARCHY = ['1m', '5m', '15m', '1h', '4h', '1d', '1wk'];
// 每个时间周期对应的大致秒数
const INTERVAL_SECONDS = {
  '1m': 60,
  '5m': 300,
  '15m': 900,
  '1h': 3600,
  '4h': 14400,
  '1d': 86400,
  '1wk': 604800,
};

// 切换阈值：当可见K线数量小于此值时，切换到更小周期
const SWITCH_THRESHOLD = {
  '1wk': 15,   // 显示少于15周时，切换到日线
  '1d': 20,    // 显示少于20天时，切换到4小时
  '4h': 25,    // 显示少于25根4小时K线时，切换到1小时
  '1h': 30,    // 显示少于30根1小时K线时，切换到15分钟
  '15m': 35,   // 显示少于35根15分钟K线时，切换到5分钟
  '5m': 40,    // 显示少于40根5分钟K线时，切换到1分钟
  '1m': 999,   // 1分钟是最小周期，不再切换
};

// 反向切换阈值：当可见K线数量超过此值时，切换到更大周期
const SWITCH_UP_THRESHOLD = {
  '1m': 200,   // 显示超过200根1分钟K线时，切换到5分钟
  '5m': 150,   // 显示超过150根5分钟K线时，切换到15分钟
  '15m': 120,  // 显示超过120根15分钟K线时，切换到1小时
  '1h': 100,   // 显示超过100根1小时K线时，切换到4小时
  '4h': 80,    // 显示超过80根4小时K线时，切换到日线
  '1d': 60,    // 显示超过60天时，切换到周线
  '1wk': 999,  // 周线是最大周期，不再切换
};

const CHART_OPTIONS = {
  layout: {
    background: { type: ColorType.Solid, color: '#161b22' },
    textColor: '#8b949e',
  },
  grid: {
    vertLines: { color: '#21262d' },
    horzLines: { color: '#21262d' },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: {
      color: '#00d4aa',
      width: 1,
      style: 2,
    },
    horLine: {
      color: '#00d4aa',
      width: 1,
      style: 2,
    },
  },
  rightPriceScale: {
    borderColor: '#30363d',
  },
  timeScale: {
    borderColor: '#30363d',
    timeVisible: true,
    secondsVisible: false,
  },
  handleScroll: {
    mouseWheel: true,
    pressedMouseMove: true,
    horzTouchDrag: true,
    vertTouchDrag: true,
  },
  handleScale: {
    mouseWheel: true,
    pinch: true,
    axisPressedMouseMove: true,
  },
};

// 将时间戳转换为 lightweight-charts 格式
function formatTimeForChart(timestamp, interval) {
  const date = new Date(timestamp);

  // 对于日线及以上，只使用日期字符串 YYYY-MM-DD
  if (interval === '1d' || interval === '1wk' || interval === '1mo') {
    return date.toISOString().split('T')[0];
  }

  // 对于日内数据，使用 Unix 时间戳（秒）
  // lightweight-charts 支持 UNIX timestamp
  return Math.floor(date.getTime() / 1000);
}

// 辅助函数：去重并排序图表数据
function deduplicateAndSortData(data) {
  const dataMap = new Map();
  data.forEach(item => {
    dataMap.set(item.time, item);
  });
  return Array.from(dataMap.values()).sort((a, b) => {
    if (typeof a.time === 'string') {
      return a.time.localeCompare(b.time);
    }
    return a.time - b.time;
  });
}

// 辅助函数：从 bars 创建指定值的数据点（去重并排序）
function createDataPointsFromBars(bars, interval, value) {
  const dataMap = new Map();
  bars.forEach(bar => {
    const time = formatTimeForChart(bar.timestamp, interval);
    dataMap.set(time, { time, value });
  });
  return Array.from(dataMap.values()).sort((a, b) => {
    if (typeof a.time === 'string') {
      return a.time.localeCompare(b.time);
    }
    return a.time - b.time;
  });
}

function CandlestickChart({
  symbol,
  bars = [],
  maData = {},
  markers = [],
  sellRanges = [],
  onAddMarker,
  interval = '1d',
  onIntervalChange,
}) {
  console.log('[CandlestickChart] Rendered with:', { symbol, interval, barsLength: bars?.length });

  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const maSeriesRef = useRef({});
  const sellRangeSeriesRef = useRef([]);
  const initialLoadRef = useRef(false);
  const autoSwitchEnabledRef = useRef(false); // 暂时禁用自动切换
  const pendingSwitchRef = useRef(null); // 待处理的切换
  const lastVisibleBarsRef = useRef(0); // 上次可见的K线数量
  const [activeMA, setActiveMA] = useState(['ma5', 'ma20', 'ma50']);
  const [showSellRanges, setShowSellRanges] = useState(true);
  const [localInterval, setLocalInterval] = useState(interval);

  // 同步外部 interval 变化到本地状态
  useEffect(() => {
    console.log('[CandlestickChart] Sync interval from props:', interval);
    setLocalInterval(interval);
    initialLoadRef.current = false;  // interval 变化时重置加载状态
  }, [symbol, interval]);

  // 切换到新的时间周期
  const switchInterval = useCallback((newInterval) => {
    if (newInterval === localInterval) return;

    console.log('[CandlestickChart] Switching interval to:', newInterval);
    setLocalInterval(newInterval);

    // 通知父组件
    if (onIntervalChange) {
      onIntervalChange(newInterval);
    }
  }, [localInterval, onIntervalChange]);

  // 初始化图表（只在组件挂载时运行一次）
  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      ...CHART_OPTIONS,
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight || 400,
    });

    chartRef.current = chart;

    // 添加蜡烛图系列
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#3fb950',
      downColor: '#f85149',
      borderUpColor: '#3fb950',
      borderDownColor: '#f85149',
      wickUpColor: '#3fb950',
      wickDownColor: '#f85149',
    });
    candleSeriesRef.current = candleSeries;

    // 处理点击事件
    chart.subscribeClick((param) => {
      if (param.point && param.time) {
        const price = candleSeries.coordinateToPrice(param.point.y);
        if (price && onAddMarker) {
          onAddMarker({
            time: param.time,
            price: price,
          });
        }
      }
    });

    // 响应式
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, []);

  // 监听可见范围变化，实现智能时间周期切换（单独的 effect）
  useEffect(() => {
    if (!chartRef.current) return;

    // 当 symbol 改变时，清理旧数据并重置
    console.log('[CandlestickChart] Setting up for symbol:', symbol, 'with bars:', bars?.length);

    // 只在有足够数据时才启用自动切换
    if (!bars || bars.length < 10) {
      console.log('[CandlestickChart] Not enough data for auto-switch yet');
      return;
    }

    console.log('[CandlestickChart] Auto-switch enabled with', bars.length, 'bars');

    const handleVisibleRangeChange = () => {
      if (!autoSwitchEnabledRef.current) return;
      if (!bars || bars.length === 0) return;
      // 至少要有10根K线才考虑自动切换
      if (bars.length < 10) return;

      const timeScale = chartRef.current.timeScale();
      const visibleRange = timeScale.getVisibleRange();

      if (!visibleRange) return;

      // 计算可见的K线数量
      const { from, to } = visibleRange;
      let visibleCount = 0;

      for (const bar of bars) {
        const barTime = formatTimeForChart(bar.timestamp, localInterval);
        if (barTime >= from && barTime <= to) {
          visibleCount++;
        }
      }

      // 防抖：如果K线数量变化不大，不触发切换
      if (Math.abs(visibleCount - lastVisibleBarsRef.current) < 3) {
        return;
      }
      lastVisibleBarsRef.current = visibleCount;

      // 根据可见K线数量决定是否切换周期
      const currentIndex = INTERVAL_HIERARCHY.indexOf(localInterval);

      if (visibleCount < SWITCH_THRESHOLD[localInterval]) {
        // 切换到更小的周期
        if (currentIndex > 0) {
          const targetInterval = INTERVAL_HIERARCHY[currentIndex - 1];
          switchInterval(targetInterval);
        }
      } else if (visibleCount > SWITCH_UP_THRESHOLD[localInterval]) {
        // 切换到更大的周期
        if (currentIndex < INTERVAL_HIERARCHY.length - 1) {
          const targetInterval = INTERVAL_HIERARCHY[currentIndex + 1];
          switchInterval(targetInterval);
        }
      }
    };

    // 使用防抖处理缩放事件
    let rangeChangeTimeout;
    const debouncedRangeChange = () => {
      clearTimeout(rangeChangeTimeout);
      rangeChangeTimeout = setTimeout(handleVisibleRangeChange, 300);
    };

    chartRef.current.timeScale().subscribeVisibleLogicalRangeChange(debouncedRangeChange);

    return () => {
      clearTimeout(rangeChangeTimeout);
    };
  }, [bars, localInterval, switchInterval]);

  // 清理旧的卖出区间 series
  useEffect(() => {
    return () => {
      // 当组件卸载或symbol变化时，清理所有卖出区间 series
      if (sellRangeSeriesRef.current && chartRef.current) {
        sellRangeSeriesRef.current.forEach(series => {
          try {
            if (series && chartRef.current) {
              chartRef.current.removeSeries(series);
            }
          } catch (e) {
            // 忽略已删除或不存在的 series
          }
        });
        sellRangeSeriesRef.current = [];
      }
    };
  }, [symbol]);

  // 更新 K 线数据
  useEffect(() => {
    if (!candleSeriesRef.current) return;
    if (!bars || bars.length === 0) {
      console.log('[CandlestickChart] No bars data available', { symbol, bars, localInterval });
      return;
    }

    console.log('[CandlestickChart] Updating chart with bars:', bars.length, 'bars');

    // 处理数据：去重（按时间）并排序
    const chartDataMap = new Map();
    bars.forEach((bar) => {
      const time = formatTimeForChart(bar.timestamp, localInterval);
      // 如果有重复时间，保留最后的数据
      chartDataMap.set(time, {
        time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      });
    });

    // 转为数组并按时间排序
    const chartData = Array.from(chartDataMap.values()).sort((a, b) => {
      if (typeof a.time === 'string') {
        return a.time.localeCompare(b.time);
      }
      return a.time - b.time;
    });

    candleSeriesRef.current.setData(chartData);

    // 只在初始加载时调整时间范围，之后保持用户的缩放状态
    if (chartRef.current && !initialLoadRef.current) {
      chartRef.current.timeScale().fitContent();
      initialLoadRef.current = true;
    }
  }, [bars, localInterval]);

  // 更新 MA 线
  useEffect(() => {
    if (!candleSeriesRef.current) return;

    // 移除旧的 MA 线
    Object.values(maSeriesRef.current).forEach((series) => {
      try {
        if (series && chartRef.current) {
          chartRef.current.removeSeries(series);
        }
      } catch (e) {
        // 忽略已删除或不存在的 series
      }
    });
    maSeriesRef.current = {};

    // 添加新的 MA 线
    const colors = {
      ma5: '#00d4aa',
      ma10: '#00a8e8',
      ma20: '#f59e0b',
      ma50: '#6366f1',
    };

    activeMA.forEach((maKey) => {
      if (maData[maKey] && bars.length > 0) {
        const maSeries = chartRef.current.addLineSeries({
          color: colors[maKey] || '#00d4aa',
          lineWidth: 2,
          priceLineVisible: false,
        });

        // MA 数据也需要去重和排序
        const maDataMap = new Map();
        bars.forEach((bar, index) => {
          const time = formatTimeForChart(bar.timestamp, localInterval);
          const value = maData[maKey][index] || bar.close;
          if (value !== null) {
            maDataMap.set(time, { time, value });
          }
        });
        const maDataPoints = Array.from(maDataMap.values()).sort((a, b) => {
          if (typeof a.time === 'string') {
            return a.time.localeCompare(b.time);
          }
          return a.time - b.time;
        });

        maSeries.setData(maDataPoints);
        maSeriesRef.current[maKey] = maSeries;
      }
    });
  }, [maData, activeMA, bars, localInterval]);

  // 更新标注标记
  useEffect(() => {
    if (!candleSeriesRef.current || markers.length === 0) return;

    const markerData = markers
      .filter((m) => m.timestamp && m.price)
      .map((m) => {
        const time = formatTimeForChart(m.timestamp, localInterval);
        let color = '#00d4aa';
        let shape = 'arrowUp';

        switch (m.marker_type) {
          case 'buy':
          case 'entry':
            color = '#3fb950';
            shape = 'arrowUp';
            break;
          case 'sell':
          case 'exit':
            color = '#f85149';
            shape = 'arrowDown';
            break;
          default:
            color = '#00d4aa';
        }

        return {
          time,
          position: shape === 'arrowUp' ? 'belowBar' : 'aboveBar',
          color,
          shape,
          text: `${m.marker_type?.toUpperCase()} @ ${m.price}`,
        };
      });

    candleSeriesRef.current.setMarkers(markerData);
  }, [markers, localInterval]);

  // 更新卖出区间（策略信号矩形区域）
  useEffect(() => {
    // 清理旧的卖出区间 series
    if (sellRangeSeriesRef.current && chartRef.current) {
      sellRangeSeriesRef.current.forEach(series => {
        try {
          if (series && chartRef.current) {
            chartRef.current.removeSeries(series);
          }
        } catch (e) {
          // 忽略已删除或不存在的 series
        }
      });
      sellRangeSeriesRef.current = [];
    }

    if (!chartRef.current || !showSellRanges || sellRanges.length === 0 || !bars.length) return;

    // 只处理策略信号（有 stop_loss 的）
    const strategyRanges = sellRanges.filter(range => range.is_strategy_signal && range.stop_loss);

    if (strategyRanges.length === 0) return;

    strategyRanges.forEach((range) => {
      const stopLoss = range.stop_loss;
      const targetPrice = range.target_price;

      if (!stopLoss || !targetPrice) return;

      // 创建填充区域（使用 area series）- 去重并排序
      const areaSeries = chartRef.current.addAreaSeries({
        topColor: 'rgba(82, 196, 26, 0.2)',
        bottomColor: 'rgba(82, 196, 26, 0.05)',
        lineColor: 'transparent',
        lineWidth: 0,
        priceLineVisible: false,
        lastValueVisible: false,
        priceScaleId: '',
      });

      const areaData = createDataPointsFromBars(bars, localInterval, targetPrice);
      areaSeries.setData(areaData);

      // 添加目标价格线（实线，绿色）- 去重并排序
      const targetLine = chartRef.current.addLineSeries({
        color: '#52c41a',
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        lastValueText: `目标 $${targetPrice.toFixed(2)}`,
        priceFormat: {
          type: 'price',
          precision: 2,
          minMove: 0.01,
        },
      });

      const targetLineData = createDataPointsFromBars(bars, localInterval, targetPrice);
      targetLine.setData(targetLineData);

      // 添加止损价格线（虚线，红色）- 去重并排序
      const stopLossLine = chartRef.current.addLineSeries({
        color: '#f5222d',
        lineWidth: 2,
        lineStyle: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        lastValueText: `止损 $${stopLoss.toFixed(2)}`,
      });

      const stopLossLineData = createDataPointsFromBars(bars, localInterval, stopLoss);
      stopLossLine.setData(stopLossLineData);

      // 保存引用以便清理
      sellRangeSeriesRef.current.push(areaSeries, targetLine, stopLossLine);
    });
  }, [sellRanges, showSellRanges, bars, localInterval]);

  // MA 切换
  const handleMAChange = (value) => {
    // 确保 value 始终是数组
    setActiveMA(Array.isArray(value) ? value : [value]);
  };

  // 手动切换时间周期（禁用自动切换一段时间）
  const handleIntervalChange = (value) => {
    autoSwitchEnabledRef.current = false;
    setLocalInterval(value);
    if (onIntervalChange) {
      onIntervalChange(value);
    }
    // 3秒后恢复自动切换
    setTimeout(() => {
      autoSwitchEnabledRef.current = true;
    }, 3000);
  };

  return (
    <div style={{ width: '100%' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 8,
        }}
      >
        <Space>
          <span style={{ color: '#8b949e', fontSize: 12 }}>
            点击图表添加标注 | 滚轮缩放 | 拖拽平移
          </span>
        </Space>
        <Space>
          <Segmented
            size="small"
            options={[
              { label: 'MA5', value: 'ma5' },
              { label: 'MA20', value: 'ma20' },
              { label: 'MA50', value: 'ma50' },
            ]}
            value={activeMA}
            onChange={handleMAChange}
            multiple
          />
          <Button
            size="small"
            type={showSellRanges ? 'primary' : 'default'}
            onClick={() => setShowSellRanges(!showSellRanges)}
          >
            卖出区间
          </Button>
          {onIntervalChange && (
            <Segmented
              size="small"
              options={[
                { label: '1分', value: '1m' },
                { label: '5分', value: '5m' },
                { label: '15分', value: '15m' },
                { label: '1时', value: '1h' },
                { label: '4时', value: '4h' },
                { label: '日K', value: '1d' },
                { label: '周K', value: '1wk' },
              ]}
              value={localInterval}
              onChange={handleIntervalChange}
            />
          )}
        </Space>
      </div>
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: 400,
          borderRadius: 8,
          overflow: 'hidden',
          pointerEvents: 'auto',
        }}
      />
    </div>
  );
}

export default CandlestickChart;
