import React, { useMemo, useState } from 'react';
import { Table, Tag, Select, Space, Typography, Tooltip } from 'antd';

const { Text } = Typography;
const { Option } = Select;

/**
 * RiskRejectsTable - 风控拒绝记录表组件
 *
 * 显示被风控拒绝的交易记录
 */
function RiskRejectsTable({ rejects, loading = false }) {
  const [symbolFilter, setSymbolFilter] = useState('');
  const [actionFilter, setActionFilter] = useState('all');
  const [reasonFilter, setReasonFilter] = useState('');

  // 获取所有唯一的股票代码
  const uniqueSymbols = useMemo(() => {
    if (!rejects) return [];
    const symbols = [...new Set(rejects.map(r => r.symbol))].sort();
    return symbols;
  }, [rejects]);

  // 获取所有唯一的原因（简化显示）
  const uniqueReasons = useMemo(() => {
    if (!rejects) return [];
    const reasons = [...new Set(rejects.map(r => {
      // 简化原因文本，取主要部分
      const parts = r.reason?.split('|')[0] || r.reason;
      return parts?.trim().substring(0, 50);
    }))].filter(Boolean).sort();
    return reasons;
  }, [rejects]);

  // 过滤后的数据
  const filteredData = useMemo(() => {
    if (!rejects) return [];

    return rejects.filter(item => {
      // 按股票代码过滤
      if (symbolFilter && item.symbol !== symbolFilter) return false;

      // 按动作过滤
      if (actionFilter !== 'all' && item.action !== actionFilter) return false;

      // 按原因过滤
      if (reasonFilter) {
        const reasonText = item.reason?.toLowerCase() || '';
        if (!reasonText.includes(reasonFilter.toLowerCase())) return false;
      }

      return true;
    });
  }, [rejects, symbolFilter, actionFilter, reasonFilter]);

  // 格式化日期时间
  const formatDateTime = (dateStr) => {
    if (!dateStr) return '-';
    try {
      return new Date(dateStr).toLocaleString('zh-CN');
    } catch {
      return dateStr;
    }
  };

  // 获取动作标签颜色
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

  // 表格列定义
  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 180,
      render: formatDateTime,
    },
    {
      title: '股票',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
    },
    {
      title: '动作',
      dataIndex: 'action',
      key: 'action',
      width: 100,
      render: (action) => (
        <Tag color={getActionColor(action)}>
          {action || '-'}
        </Tag>
      ),
    },
    {
      title: '拒绝原因',
      dataIndex: 'reason',
      key: 'reason',
      render: (reason) => (
        <Tooltip title={reason}>
          <Text
            ellipsis
            style={{ maxWidth: 400, display: 'inline-block' }}
          >
            {reason || '-'}
          </Text>
        </Tooltip>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      width: 100,
      align: 'right',
      render: (confidence) => {
        if (confidence === undefined || confidence === null) return '-';
        const conf = typeof confidence === 'number' ? confidence : parseFloat(confidence);
        const percentage = (conf * 100).toFixed(1);
        return <Text type={conf > 0.8 ? 'success' : conf > 0.5 ? 'warning' : 'secondary'}>
          {percentage}%
        </Text>;
      },
    },
  ];

  // 统计信息
  const stats = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;

    const byAction = {};
    filteredData.forEach(item => {
      const action = item.action || 'UNKNOWN';
      byAction[action] = (byAction[action] || 0) + 1;
    });

    return {
      total: filteredData.length,
      byAction,
    };
  }, [filteredData]);

  return (
    <div>
      {/* 筛选器 */}
      <Space style={{ marginBottom: 16, flexWrap: 'wrap' }}>
        <Select
          placeholder="选择股票"
          style={{ width: 150 }}
          value={symbolFilter || undefined}
          onChange={setSymbolFilter}
          allowClear
        >
          {uniqueSymbols.map(symbol => (
            <Option key={symbol} value={symbol}>{symbol}</Option>
          ))}
        </Select>

        <Select
          placeholder="动作类型"
          style={{ width: 120 }}
          value={actionFilter}
          onChange={setActionFilter}
        >
          <Option value="all">全部</Option>
          <Option value="BUY">买入</Option>
          <Option value="SELL">卖出</Option>
          <Option value="REDUCE">减仓</Option>
          <Option value="PAUSE">暂停</Option>
          <Option value="DISABLE">禁用</Option>
        </Select>

        <Select
          placeholder="拒绝原因"
          style={{ width: 200 }}
          value={reasonFilter || undefined}
          onChange={setReasonFilter}
          allowClear
          showSearch
        >
          {uniqueReasons.map(reason => (
            <Option key={reason} value={reason}>
              {reason.length > 30 ? reason.substring(0, 30) + '...' : reason}
            </Option>
          ))}
        </Select>

        {stats && (
          <Space style={{ marginLeft: 'auto' }}>
            <Text type="secondary">共 {stats.total} 条拒绝</Text>
            {Object.entries(stats.byAction).map(([action, count]) => (
              <Text key={action} type="secondary">
                {action}: {count}
              </Text>
            ))}
          </Space>
        )}
      </Space>

      {/* 表格 */}
      <Table
        rowKey={(record, index) => `${record.timestamp}-${record.symbol}-${record.action}-${index}`}
        columns={columns}
        dataSource={filteredData}
        loading={loading}
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showTotal: (total) => `共 ${total} 条`,
          style: { color: '#8b949e' },
        }}
        size="small"
        style={{
          background: '#0d1117',
          borderRadius: 6,
        }}
      />
    </div>
  );
}

export default RiskRejectsTable;
