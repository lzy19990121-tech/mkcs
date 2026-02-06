import React, { useMemo, useState } from 'react';
import { Card, Select, Row, Col, Statistic, List, Tag, Typography } from 'antd';
import { BarChartOutlined, ClockCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;
const { Option } = Select;

/**
 * RiskRejectsChart - 风控拒绝统计图表组件
 *
 * 展示 TopN 拒绝原因统计和时间线视图
 */
function RiskRejectsChart({ rejects }) {
  const [topN, setTopN] = useState(10);
  const [timelineOrder, setTimelineOrder] = useState('desc'); // 'desc' = 最新在前

  // 统计拒绝原因
  const reasonStats = useMemo(() => {
    if (!rejects || rejects.length === 0) return [];

    const stats = {};
    rejects.forEach(item => {
      // 简化原因，取主要部分
      const reasonKey = item.reason?.split('|')[0]?.trim() || item.reason || 'Unknown';
      stats[reasonKey] = (stats[reasonKey] || 0) + 1;
    });

    return Object.entries(stats)
      .map(([reason, count]) => ({ reason, count }))
      .sort((a, b) => b.count - a.count);
  }, [rejects]);

  // TopN 数据
  const topReasons = useMemo(() => {
    return reasonStats.slice(0, topN);
  }, [reasonStats, topN]);

  // 时间线数据
  const timelineData = useMemo(() => {
    if (!rejects || rejects.length === 0) return [];

    return [...rejects].sort((a, b) => {
      const timeA = new Date(a.timestamp || 0).getTime();
      const timeB = new Date(b.timestamp || 0).getTime();
      return timelineOrder === 'desc' ? timeB - timeA : timeA - timeB;
    });
  }, [rejects, timelineOrder]);

  // 格式化日期时间
  const formatDateTime = (dateStr) => {
    if (!dateStr) return '-';
    try {
      return new Date(dateStr).toLocaleString('zh-CN');
    } catch {
      return dateStr;
    }
  };

  // 获取动作颜色
  const getActionColor = (action) => {
    const actionUpper = (action || '').toUpperCase();
    switch (actionUpper) {
      case 'BUY': return 'green';
      case 'SELL': return 'red';
      case 'REDUCE': return 'orange';
      case 'PAUSE': return 'blue';
      case 'DISABLE': return 'purple';
      default: return 'default';
    }
  };

  // 计算最大计数（用于比例显示）
  const maxCount = topReasons.length > 0 ? topReasons[0].count : 1;

  if (!rejects || rejects.length === 0) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>
        暂无风控拒绝记录
      </div>
    );
  }

  return (
    <div>
      <Row gutter={[16, 16]}>
        {/* TopN 拒绝原因统计 */}
        <Col xs={24} lg={14}>
          <Card
            title={
              <span style={{ color: '#c9d1d9' }}>
                <BarChartOutlined /> Top {topN} 拒绝原因
              </span>
            }
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
            headStyle={{ borderBottom: '1px solid #30363d' }}
            extra={
              <Select
                value={topN}
                onChange={setTopN}
                style={{ width: 80 }}
              >
                <Option value={5}>5</Option>
                <Option value={10}>10</Option>
                <Option value={20}>20</Option>
                <Option value={50}>50</Option>
              </Select>
            }
          >
            <div style={{ maxHeight: 400, overflowY: 'auto' }}>
              {topReasons.map((item, index) => (
                <div
                  key={index}
                  style={{
                    marginBottom: 16,
                    borderBottom: '1px solid #21262d',
                    paddingBottom: 12,
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: 4,
                    }}
                  >
                    <Text
                      ellipsis
                      style={{ color: '#c9d1d9', maxWidth: '70%' }}
                      title={item.reason}
                    >
                      {index + 1}. {item.reason}
                    </Text>
                    <Text style={{ color: '#00d4aa', fontWeight: 'bold' }}>
                      {item.count} 次
                    </Text>
                  </div>
                  {/* 进度条 */}
                  <div
                    style={{
                      height: 6,
                      background: '#21262d',
                      borderRadius: 3,
                      overflow: 'hidden',
                    }}
                  >
                    <div
                      style={{
                        height: '100%',
                        width: `${(item.count / maxCount) * 100}%`,
                        background: index === 0 ? '#ff4d4f' : '#00d4aa',
                        borderRadius: 3,
                        transition: 'width 0.3s',
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* 总计 */}
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #30363d' }}>
              <Statistic
                title={<span style={{ color: '#8b949e' }}>总拒绝次数</span>}
                value={rejects.length}
                valueStyle={{ color: '#c9d1d9' }}
              />
            </div>
          </Card>
        </Col>

        {/* 时间线视图 */}
        <Col xs={24} lg={10}>
          <Card
            title={
              <span style={{ color: '#c9d1d9' }}>
                <ClockCircleOutlined /> 时间线
              </span>
            }
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
            headStyle={{ borderBottom: '1px solid #30363d' }}
            extra={
              <Select
                value={timelineOrder}
                onChange={setTimelineOrder}
                style={{ width: 100 }}
              >
                <Option value="desc">最新</Option>
                <Option value="asc">最早</Option>
              </Select>
            }
          >
            <div style={{ maxHeight: 450, overflowY: 'auto' }}>
              <List
                size="small"
                dataSource={timelineData.slice(0, 50)}
                renderItem={(item) => (
                  <List.Item
                    style={{
                      borderBottom: '1px solid #21262d',
                      padding: '8px 0',
                    }}
                  >
                    <div style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <Tag color={getActionColor(item.action)} style={{ margin: 0 }}>
                          {item.action || '-'}
                        </Tag>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {formatDateTime(item.timestamp)}
                        </Text>
                      </div>
                      <Text
                        type="secondary"
                        ellipsis
                        style={{ fontSize: 12, display: 'block' }}
                        title={item.reason}
                      >
                        {item.reason}
                      </Text>
                      {item.symbol && (
                        <Text style={{ fontSize: 11, color: '#8b949e' }}>
                          股票: {item.symbol}
                        </Text>
                      )}
                    </div>
                  </List.Item>
                )}
              />
              {timelineData.length > 50 && (
                <div style={{ textAlign: 'center', padding: 8, color: '#8b949e', fontSize: 12 }}>
                  仅显示最近 50 条记录
                </div>
              )}
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default RiskRejectsChart;
