import React, { useState, useMemo } from 'react';
import { Table, Tag, Select, Input, Space, Typography } from 'antd';

const { Text } = Typography;
const { Option } = Select;

/**
 * TradesTable - 交易记录表组件
 *
 * 显示交易记录，支持筛选和排序
 */
function TradesTable({ trades, loading = false }) {
  const [symbolFilter, setSymbolFilter] = useState('');
  const [sideFilter, setSideFilter] = useState('all');
  const [pnlFilter, setPnlFilter] = useState('all');
  const [sortField, setSortField] = useState(null);
  const [sortOrder, setSortOrder] = useState(null);

  // 获取所有唯一的股票代码
  const uniqueSymbols = useMemo(() => {
    if (!trades) return [];
    const symbols = [...new Set(trades.map(t => t.symbol))].sort();
    return symbols;
  }, [trades]);

  // 过滤和排序后的数据
  const filteredData = useMemo(() => {
    if (!trades) return [];

    let filtered = [...trades];

    // 按股票代码过滤
    if (symbolFilter) {
      filtered = filtered.filter(t => t.symbol === symbolFilter);
    }

    // 按买卖方向过滤
    if (sideFilter !== 'all') {
      filtered = filtered.filter(t => t.side === sideFilter);
    }

    // 按盈亏过滤
    if (pnlFilter !== 'all') {
      filtered = filtered.filter(t => {
        if (t.pnl === undefined || t.pnl === null) return false;
        if (pnlFilter === 'profit') return t.pnl > 0;
        if (pnlFilter === 'loss') return t.pnl < 0;
        if (pnlFilter === 'flat') return t.pnl === 0;
        return true;
      });
    }

    // 排序
    if (sortField) {
      filtered.sort((a, b) => {
        let aVal = a[sortField];
        let bVal = b[sortField];

        if (sortField === 'timestamp') {
          aVal = new Date(aVal).getTime();
          bVal = new Date(bVal).getTime();
        }

        if (aVal === bVal) return 0;
        if (aVal === undefined || aVal === null) return 1;
        if (bVal === undefined || bVal === null) return -1;

        if (typeof aVal === 'string') {
          return sortOrder === 'ascend'
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal);
        }

        return sortOrder === 'ascend' ? aVal - bVal : bVal - aVal;
      });
    }

    return filtered;
  }, [trades, symbolFilter, sideFilter, pnlFilter, sortField, sortOrder]);

  // 格式化日期时间
  const formatDateTime = (dateStr) => {
    if (!dateStr) return '-';
    try {
      return new Date(dateStr).toLocaleString('zh-CN');
    } catch {
      return dateStr;
    }
  };

  // 格式化盈亏
  const formatPnl = (pnl) => {
    if (pnl === undefined || pnl === null) return <Text type="secondary">-</Text>;
    const formatted = typeof pnl === 'number' ? pnl : parseFloat(pnl);
    const color = formatted > 0 ? '#52c41a' : formatted < 0 ? '#ff4d4f' : '#8b949e';
    return (
      <span style={{ color, fontWeight: formatted !== 0 ? 'bold' : 'normal' }}>
        {formatted >= 0 ? '+' : ''}${formatted.toFixed(2)}
      </span>
    );
  };

  // 表格列定义
  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 180,
      sorter: true,
      sortOrder: sortField === 'timestamp' ? sortOrder : null,
      render: formatDateTime,
    },
    {
      title: '股票',
      dataIndex: 'symbol',
      key: 'symbol',
      width: 100,
      sorter: true,
      sortOrder: sortField === 'symbol' ? sortOrder : null,
    },
    {
      title: '方向',
      dataIndex: 'side',
      key: 'side',
      width: 80,
      render: (side) => (
        <Tag color={side === 'BUY' ? 'green' : 'red'}>
          {side === 'BUY' ? '买入' : '卖出'}
        </Tag>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      width: 100,
      align: 'right',
      render: (price) => `$${typeof price === 'number' ? price.toFixed(2) : price}`,
    },
    {
      title: '数量',
      dataIndex: 'quantity',
      key: 'quantity',
      width: 80,
      align: 'right',
    },
    {
      title: '手续费',
      dataIndex: 'commission',
      key: 'commission',
      width: 100,
      align: 'right',
      render: (commission) => `$${(typeof commission === 'number' ? commission : parseFloat(commission || 0)).toFixed(2)}`,
    },
    {
      title: '盈亏',
      dataIndex: 'pnl',
      key: 'pnl',
      width: 120,
      align: 'right',
      sorter: true,
      sortOrder: sortField === 'pnl' ? sortOrder : null,
      render: formatPnl,
    },
  ];

  // 统计信息
  const stats = useMemo(() => {
    if (!filteredData || filteredData.length === 0) return null;

    const withPnl = filteredData.filter(t => t.pnl !== undefined && t.pnl !== null);
    const totalPnl = withPnl.reduce((sum, t) => sum + (typeof t.pnl === 'number' ? t.pnl : parseFloat(t.pnl || 0)), 0);
    const winCount = withPnl.filter(t => t.pnl > 0).length;
    const lossCount = withPnl.filter(t => t.pnl < 0).length;
    const winRate = withPnl.length > 0 ? (winCount / withPnl.length * 100) : 0;

    return {
      total: filteredData.length,
      totalPnl,
      winCount,
      lossCount,
      winRate,
    };
  }, [filteredData]);

  const handleTableChange = (pagination, filters, sorter) => {
    setSortField(sorter.field);
    setSortOrder(sorter.order);
  };

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
          placeholder="买卖方向"
          style={{ width: 120 }}
          value={sideFilter}
          onChange={setSideFilter}
        >
          <Option value="all">全部</Option>
          <Option value="BUY">买入</Option>
          <Option value="SELL">卖出</Option>
        </Select>

        <Select
          placeholder="盈亏状态"
          style={{ width: 120 }}
          value={pnlFilter}
          onChange={setPnlFilter}
        >
          <Option value="all">全部</Option>
          <Option value="profit">盈利</Option>
          <Option value="loss">亏损</Option>
          <Option value="flat">平盘</Option>
        </Select>

        {stats && (
          <Space style={{ marginLeft: 'auto' }}>
            <Text type="secondary">共 {stats.total} 笔</Text>
            <Text type="secondary">|</Text>
            <Text type="secondary">
              盈: {stats.winCount} / 亏: {stats.lossCount}
            </Text>
            <Text type="secondary">|</Text>
            <Text style={{ color: stats.totalPnl >= 0 ? '#52c41a' : '#ff4d4f' }}>
              总盈亏: {stats.totalPnl >= 0 ? '+' : ''}${stats.totalPnl.toFixed(2)}
            </Text>
            <Text type="secondary">|</Text>
            <Text type="secondary">
              胜率: {stats.winRate.toFixed(1)}%
            </Text>
          </Space>
        )}
      </Space>

      {/* 表格 */}
      <Table
        rowKey={(record, index) => `${record.timestamp}-${record.symbol}-${index}`}
        columns={columns}
        dataSource={filteredData}
        loading={loading}
        onChange={handleTableChange}
        pagination={{
          pageSize: 50,
          showSizeChanger: true,
          showTotal: (total) => `共 ${total} 条`,
          style: { color: '#8b949e' },
        }}
        size="small"
        style={{
          background: '#0d1117',
          borderRadius: 6,
        }}
        rowClassName={(record) => {
          if (record.pnl === undefined || record.pnl === null) return '';
          return record.pnl > 0 ? 'row-profit' : record.pnl < 0 ? 'row-loss' : '';
        }}
      />

      <style>{`
        .row-profit {
          background: rgba(82, 196, 26, 0.05) !important;
        }
        .row-loss {
          background: rgba(255, 77, 79, 0.05) !important;
        }
      `}</style>
    </div>
  );
}

export default TradesTable;
