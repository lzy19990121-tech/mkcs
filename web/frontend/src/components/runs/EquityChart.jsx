import React, { useEffect, useRef, useState } from 'react';
import { Card, Spin } from 'antd';

/**
 * EquityChart - 权益曲线图组件
 *
 * 使用 Lightweight Charts 渲染权益曲线
 */
function EquityChart({ data, loading = false, height = 400 }) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const [tooltipContent, setTooltipContent] = useState(null);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    // 动态导入 Lightweight Charts
    let chart = null;
    let series = null;

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
          vertLine: {
            color: '#00d4aa',
            width: 1,
            style: 2,
            labelBackgroundColor: '#00d4aa',
          },
          horzLine: {
            color: '#00d4aa',
            width: 1,
            style: 2,
            labelBackgroundColor: '#00d4aa',
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
      });

      // 转换数据格式
      const lineData = data.map((point) => {
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

      series = chart.addLineSeries({
        color: '#00d4aa',
        lineWidth: 2,
        title: '权益',
      });

      series.setData(lineData);
      chart.timeScale().fitContent();

      // 订阅十字光标移动事件
      chart.subscribeCrosshairMove((param) => {
        if (param.time) {
          const dataPoint = param.seriesData.get(series);
          if (dataPoint) {
            const date = new Date(param.time * 1000);
            setTooltipContent({
              time: date.toLocaleString('zh-CN'),
              value: dataPoint.value.toFixed(2),
            });
          }
        } else {
          setTooltipContent(null);
        }
      });

      chartRef.current = chart;
      seriesRef.current = series;
    });

    // 清理
    return () => {
      if (chart) {
        chart.remove();
      }
    };
  }, [data, height]);

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

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!data || data.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>
        暂无权益曲线数据
      </div>
    );
  }

  // 计算统计信息
  const startEquity = data[0]?.equity || 0;
  const endEquity = data[data.length - 1]?.equity || 0;
  const maxEquity = Math.max(...data.map(d => d.equity));
  const minEquity = Math.min(...data.map(d => d.equity));
  const totalReturn = startEquity > 0 ? ((endEquity - startEquity) / startEquity * 100) : 0;
  const maxDrawdown = maxEquity > 0 ? ((maxEquity - minEquity) / maxEquity * 100) : 0;

  return (
    <div>
      {/* 统计信息卡片 */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <Card
          size="small"
          style={{
            flex: 1,
            background: '#161b22',
            border: '1px solid #30363d',
          }}
        >
          <div style={{ fontSize: 12, color: '#8b949e' }}>起始权益</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#c9d1d9' }}>
            ${startEquity.toFixed(2)}
          </div>
        </Card>
        <Card
          size="small"
          style={{
            flex: 1,
            background: '#161b22',
            border: '1px solid #30363d',
          }}
        >
          <div style={{ fontSize: 12, color: '#8b949e' }}>最终权益</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#c9d1d9' }}>
            ${endEquity.toFixed(2)}
          </div>
        </Card>
        <Card
          size="small"
          style={{
            flex: 1,
            background: '#161b22',
            border: '1px solid #30363d',
          }}
        >
          <div style={{ fontSize: 12, color: '#8b949e' }}>总收益率</div>
          <div
            style={{
              fontSize: 18,
              fontWeight: 'bold',
              color: totalReturn >= 0 ? '#52c41a' : '#ff4d4f',
            }}
          >
            {totalReturn >= 0 ? '+' : ''}{totalReturn.toFixed(2)}%
          </div>
        </Card>
        <Card
          size="small"
          style={{
            flex: 1,
            background: '#161b22',
            border: '1px solid #30363d',
          }}
        >
          <div style={{ fontSize: 12, color: '#8b949e' }}>最大回撤</div>
          <div style={{ fontSize: 18, fontWeight: 'bold', color: '#ff4d4f' }}>
            -{maxDrawdown.toFixed(2)}%
          </div>
        </Card>
      </div>

      {/* 图表容器 */}
      <Card
        bordered={false}
        style={{
          background: '#161b22',
          borderRadius: 8,
        }}
        bodyStyle={{ padding: 16 }}
      >
        {tooltipContent && (
          <div
            style={{
              position: 'absolute',
              top: 16,
              left: 16,
              background: 'rgba(22, 27, 34, 0.9)',
              padding: '8px 12px',
              borderRadius: 4,
              border: '1px solid #30363d',
              zIndex: 10,
              fontSize: 12,
              color: '#c9d1d9',
            }}
          >
            <div>时间: {tooltipContent.time}</div>
            <div>权益: ${tooltipContent.value}</div>
          </div>
        )}
        <div ref={chartContainerRef} style={{ position: 'relative' }} />
      </Card>
    </div>
  );
}

export default EquityChart;
