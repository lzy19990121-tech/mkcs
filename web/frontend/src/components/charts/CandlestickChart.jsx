import React, { useEffect, useRef, useCallback, useState } from 'react';
import { createChart, ColorType, CrosshairMode } from 'lightweight-charts';
import { Space, Button, Segmented } from 'antd';
import { PlusOutlined } from '@ant-design/icons';

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
    horzLine: {
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
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candleSeriesRef = useRef(null);
  const maSeriesRef = useRef({});
  const sellRangeSeriesRef = useRef([]);
  const initialLoadRef = useRef(false);
  const [activeMA, setActiveMA] = useState(['ma5', 'ma20', 'ma50']);
  const [showSellRanges, setShowSellRanges] = useState(true);

  // 重置初始加载标记当 symbol 或 interval 改变时
  useEffect(() => {
    initialLoadRef.current = false;
  }, [symbol, interval]);

  // 初始化图表
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

  // 清理旧的卖出区间 series
  useEffect(() => {
    return () => {
      // 当组件卸载或symbol变化时，清理所有卖出区间 series
      if (sellRangeSeriesRef.current && chartRef.current) {
        sellRangeSeriesRef.current.forEach(series => {
          if (chartRef.current) {
            chartRef.current.removeSeries(series);
          }
        });
        sellRangeSeriesRef.current = [];
      }
    };
  }, [symbol]);

  // 更新 K 线数据
  useEffect(() => {
    if (!candleSeriesRef.current || bars.length === 0) return;

    const chartData = bars.map((bar) => ({
      time: bar.timestamp.split('T')[0], // 简化时间格式
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
    }));

    candleSeriesRef.current.setData(chartData);

    // 只在初始加载时调整时间范围，之后保持用户的缩放状态
    if (chartRef.current && !initialLoadRef.current) {
      chartRef.current.timeScale().fitContent();
      initialLoadRef.current = true;
    }
  }, [bars]);

  // 更新 MA 线
  useEffect(() => {
    if (!candleSeriesRef.current) return;

    // 移除旧的 MA 线
    Object.values(maSeriesRef.current).forEach((series) => {
      if (chartRef.current) {
        chartRef.current.removeSeries(series);
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

        const maDataPoints = bars.map((bar, index) => ({
          time: bar.timestamp.split('T')[0],
          value: maData[maKey][index] || bar.close,
        })).filter((p) => p.value !== null);

        maSeries.setData(maDataPoints);
        maSeriesRef.current[maKey] = maSeries;
      }
    });
  }, [maData, activeMA, bars]);

  // 更新标注标记
  useEffect(() => {
    if (!candleSeriesRef.current || markers.length === 0) return;

    const markerData = markers
      .filter((m) => m.timestamp && m.price)
      .map((m) => {
        const time = m.timestamp.split('T')[0];
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
  }, [markers]);

  // 更新卖出区间（策略信号矩形区域）
  useEffect(() => {
    // 清理旧的卖出区间 series
    if (sellRangeSeriesRef.current && chartRef.current) {
      sellRangeSeriesRef.current.forEach(series => {
        try {
          chartRef.current.removeSeries(series);
        } catch (e) {
          // 忽略已删除的series
        }
      });
      sellRangeSeriesRef.current = [];
    }

    if (!chartRef.current || !showSellRanges || sellRanges.length === 0 || !bars.length) return;

    // 只处理策略信号（有 stop_loss 的）
    const strategyRanges = sellRanges.filter(range => range.is_strategy_signal && range.stop_loss);

    if (strategyRanges.length === 0) return;

    // 获取时间范围
    const firstBarTime = bars[0].timestamp.split('T')[0];
    const lastBarTime = bars[bars.length - 1].timestamp.split('T')[0];

    strategyRanges.forEach((range) => {
      const stopLoss = range.stop_loss;
      const targetPrice = range.target_price;

      if (!stopLoss || !targetPrice) return;

      // 创建填充区域（使用 area series）
      // 我们需要创建一个从止损价到目标价的区域
      const areaSeries = chartRef.current.addAreaSeries({
        topColor: 'rgba(82, 196, 26, 0.2)',        // 浅绿色，透明度20%
        bottomColor: 'rgba(82, 196, 26, 0.05)',     // 更浅的绿色，透明度5%
        lineColor: 'transparent',                   // 隐藏边线
        lineWidth: 0,
        priceLineVisible: false,
        lastValueVisible: false,
        priceScaleId: '',  // 使用默认价格刻度
      });

      // 创建区域数据
      // 上边界是目标价
      const areaData = bars.map(bar => ({
        time: bar.timestamp.split('T')[0],
        value: targetPrice
      }));

      areaSeries.setData(areaData);

      // 添加目标价格线（实线，绿色）
      const targetLine = chartRef.current.addLineSeries({
        color: '#52c41a',              // 绿色
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

      const targetLineData = bars.map(bar => ({
        time: bar.timestamp.split('T')[0],
        value: targetPrice
      }));
      targetLine.setData(targetLineData);

      // 添加止损价格线（虚线，红色）
      const stopLossLine = chartRef.current.addLineSeries({
        color: '#f5222d',              // 红色
        lineWidth: 2,
        lineStyle: 2,                  // 虚线
        priceLineVisible: false,
        lastValueVisible: true,
        lastValueText: `止损 $${stopLoss.toFixed(2)}`,
      });

      const stopLossLineData = bars.map(bar => ({
        time: bar.timestamp.split('T')[0],
        value: stopLoss
      }));
      stopLossLine.setData(stopLossLineData);

      // 保存引用以便清理
      sellRangeSeriesRef.current.push(areaSeries, targetLine, stopLossLine);
    });
  }, [sellRanges, showSellRanges, bars]);

  // MA 切换
  const handleMAChange = (value) => {
    // 确保 value 始终是数组
    setActiveMA(Array.isArray(value) ? value : [value]);
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
                { label: '15分', value: '15m' },
                { label: '日K', value: '1d' },
              ]}
              value={interval}
              onChange={onIntervalChange}
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
