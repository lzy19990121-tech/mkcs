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
  const [activeMA, setActiveMA] = useState(['ma5', 'ma20']);
  const [showSellRanges, setShowSellRanges] = useState(true);

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

    // 调整时间范围
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
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

  // 更新卖出区间
  useEffect(() => {
    if (!chartRef.current || !showSellRanges || sellRanges.length === 0) return;

    // 卖出区间使用背景矩形实现
    // 这里简化处理，实际可以使用 TradingView 的高级功能
    sellRanges.forEach((range) => {
      if (range.start_time && range.end_time && range.target_price) {
        // 可以添加水平线标记目标价格
        const startTime = range.start_time.split('T')[0];
        const endTime = range.end_time.split('T')[0];

        // 添加目标价格线
        const targetLine = chartRef.current.addLineSeries({
          color: '#f59e0b',
          lineWidth: 2,
          lineStyle: 2, // 虚线
          priceLineVisible: false,
        });

        targetLine.setData([
          { time: startTime, value: range.target_price },
          { time: endTime, value: range.target_price },
        ]);
      }
    });
  }, [sellRanges, showSellRanges]);

  // MA 切换
  const handleMAChange = (value) => {
    setActiveMA(value);
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
            allowClear
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
        }}
      />
    </div>
  );
}

export default CandlestickChart;
