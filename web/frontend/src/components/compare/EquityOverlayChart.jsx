import React, { useEffect, useRef } from 'react';
import { Card } from 'antd';

/**
 * EquityOverlayChart - 权益曲线对比图组件
 *
 * 叠加显示两条权益曲线进行对比
 */
function EquityOverlayChart({ leftData, rightData, leftLabel, rightLabel, height = 400 }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesLeftRef = useRef(null);
  const seriesRightRef = useRef(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // 动态导入 Lightweight Charts
    let chart = null;
    let seriesLeft = null;
    let seriesRight = null;

    import('lightweight-charts').then(({ createChart, LineData, Time }) => {
      chart = createChart(chartContainerRef.current, {
        width: chartContainerRef.current.clientWidth,
        height,
        layout: {
          background: { color: '#0d1117' },
          textColor: '#8b949e',
        },
        grid: {
          vertLines: { color: '#21262d' },
          horzLines: { color: '#21262d' },
        },
        crosshair: {
          mode: 1,
        },
        rightPriceScale: {
          borderColor: '#30363d',
        },
        timeScale: {
          borderColor: '#30363d',
          timeVisible: true,
          secondsVisible: false,
        },
      });

      // 创建左线（绿色）
      seriesLeft = chart.addLineSeries({
        color: '#00d4aa',
        lineWidth: 2,
        title: leftLabel || 'Left',
      });

      // 创建右线（橙色）
      seriesRight = chart.addLineSeries({
        color: '#ff9800',
        lineWidth: 2,
        title: rightLabel || 'Right',
      });

      // 转换左数据
      if (leftData && leftData.length > 0) {
        const lineData = leftData.map((point) => {
          let timestamp;
          try {
            timestamp = Math.floor(new Date(point.timestamp).getTime() / 1000);
          } catch {
            timestamp = Math.floor(Date.now() / 1000);
          }
          return {
            time: timestamp,
            value: point.equity,
          };
        }).filter(point => point.time > 0);
        seriesLeft.setData(lineData);
      }

      // 转换右数据
      if (rightData && rightData.length > 0) {
        const lineData = rightData.map((point) => {
          let timestamp;
          try {
            timestamp = Math.floor(new Date(point.timestamp).getTime() / 1000);
          } catch {
            timestamp = Math.floor(Date.now() / 1000);
          }
          return {
            time: timestamp,
            value: point.equity,
          };
        }).filter(point => point.time > 0);
        seriesRight.setData(lineData);
      }

      chart.timeScale().fitContent();

      chartRef.current = chart;
      seriesLeftRef.current = seriesLeft;
      seriesRightRef.current = seriesRight;
    });

    return () => {
      if (chart) {
        chart.remove();
      }
    };
  }, [leftData, rightData, leftLabel, rightLabel, height]);

  // 响应式调整
  useEffect(() => {
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 图例
  const legend = (
    <div style={{ display: 'flex', gap: 24, padding: '8px 16px', background: '#161b22' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ width: 12, height: 2, background: '#00d4aa' }} />
        <span style={{ color: '#c9d1d9', fontSize: 12 }}>{leftLabel || 'Left'}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ width: 12, height: 2, background: '#ff9800' }} />
        <span style={{ color: '#c9d1d9', fontSize: 12 }}>{rightLabel || 'Right'}</span>
      </div>
    </div>
  );

  return (
    <Card
      bordered={false}
      style={{
        background: '#161b22',
        borderRadius: 8,
      }}
      bodyStyle={{ padding: 0 }}
    >
      {legend}
      <div ref={chartContainerRef} style={{ position: 'relative' }} />
    </Card>
  );
}

export default EquityOverlayChart;
