import React, { useState, useEffect, useCallback } from 'react';
import { Table, Card, Input, Space, Button, Tag, Typography, message } from 'antd';
import { Link, useNavigate } from 'react-router-dom';
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons';
import { backtestsAPI } from '../services/api';

const { Title, Text } = Typography;

/**
 * RunsList - 回测研究列表页
 *
 * 显示所有回测运行结果，支持搜索和排序
 */
function RunsList() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [runs, setRuns] = useState([]);
  const [filteredRuns, setFilteredRuns] = useState([]);
  const [searchText, setSearchText] = useState('');
  const [selectedRowKeys, setSelectedRowKeys] = useState([]);

  // 加载回测列表
  const loadRuns = useCallback(async () => {
    setLoading(true);
    try {
      const data = await backtestsAPI.list();
      setRuns(data || []);
      setFilteredRuns(data || []);
    } catch (error) {
      if (!error.canceled) {
        message.error('加载回测列表失败: ' + (error.message || '未知错误'));
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  // 搜索过滤
  useEffect(() => {
    if (!searchText) {
      setFilteredRuns(runs);
    } else {
      const filtered = runs.filter(run =>
        run.id?.toLowerCase().includes(searchText.toLowerCase()) ||
        run.strategy?.toLowerCase().includes(searchText.toLowerCase())
      );
      setFilteredRuns(filtered);
    }
  }, [searchText, runs]);

  // 格式化百分比
  const formatPercent = (val) => {
    if (val === undefined || val === null) return '-';
    const formatted = typeof val === 'number' ? val : parseFloat(val);
    const color = formatted >= 0 ? '#52c41a' : '#ff4d4f';
    return (
      <span style={{ color }}>
        {formatted >= 0 ? '+' : ''}{formatted.toFixed(2)}%
      </span>
    );
  };

  // 格式化金额
  const formatMoney = (val) => {
    if (val === undefined || val === null) return '-';
    const formatted = typeof val === 'number' ? val : parseFloat(val);
    return `$${formatted.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  // 格式化日期
  const formatDate = (dateStr) => {
    if (!dateStr) return '-';
    try {
      return new Date(dateStr).toLocaleString('zh-CN');
    } catch {
      return dateStr;
    }
  };

  // 表格列定义
  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 200,
      ellipsis: true,
      render: (id, record) => (
        <Link to={`/runs/${id}`} style={{ color: '#00d4aa', fontWeight: 500 }}>
          {id}
        </Link>
      ),
    },
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      width: 150,
      render: (strategy) => strategy ? <Tag color="blue">{strategy}</Tag> : '-',
    },
    {
      title: '日期',
      dataIndex: 'date',
      key: 'date',
      width: 180,
      sorter: (a, b) => new Date(a.date || 0) - new Date(b.date || 0),
      render: formatDate,
    },
    {
      title: '区间',
      key: 'period',
      width: 200,
      render: (_, record) => (
        <Text type="secondary" style={{ fontSize: 12 }}>
          {record.start_date || '-'} 至 {record.end_date || '-'}
        </Text>
      ),
    },
    {
      title: '最终权益',
      dataIndex: 'final_equity',
      key: 'final_equity',
      width: 120,
      align: 'right',
      sorter: (a, b) => (a.final_equity || 0) - (b.final_equity || 0),
      render: formatMoney,
    },
    {
      title: '总收益率',
      dataIndex: 'total_return',
      key: 'total_return',
      width: 120,
      align: 'right',
      sorter: (a, b) => (a.total_return || 0) - (b.total_return || 0),
      render: formatPercent,
    },
    {
      title: '操作',
      key: 'action',
      width: 100,
      render: (_, record) => (
        <Link to={`/runs/${record.id}`}>
          <Button type="link" size="small" style={{ color: '#00d4aa' }}>
            详情
          </Button>
        </Link>
      ),
    },
  ];

  // 行选择配置
  const rowSelection = {
    selectedRowKeys,
    onChange: setSelectedRowKeys,
    type: 'checkbox',
    getCheckboxProps: (record) => ({
      disabled: !record.id,
    }),
  };

  // 比较按钮点击
  const handleCompare = () => {
    if (selectedRowKeys.length !== 2) {
      message.warning('请选择两条回测记录进行对比');
      return;
    }
    navigate(`/compare?l=${selectedRowKeys[0]}&r=${selectedRowKeys[1]}`);
  };

  return (
    <div style={{ padding: '0 0 24px' }}>
      <Card
        bordered={false}
        style={{
          background: '#161b22',
          borderRadius: 8,
          marginBottom: 16,
        }}
      >
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Title level={4} style={{ margin: 0, color: '#c9d1d9' }}>
            回测研究
          </Title>
          <Space>
            {selectedRowKeys.length === 2 && (
              <Button type="primary" onClick={handleCompare}>
                对比选中
              </Button>
            )}
            <Button
              icon={<ReloadOutlined />}
              onClick={loadRuns}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        </Space>
      </Card>

      <Card
        bordered={false}
        style={{
          background: '#161b22',
          borderRadius: 8,
        }}
      >
        <Space style={{ marginBottom: 16, width: '100%', justifyContent: 'space-between' }}>
          <Input
            placeholder="搜索回测 ID 或策略名称..."
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            style={{ width: 300 }}
            allowClear
          />
          <Text type="secondary">
            共 {filteredRuns.length} 条记录
          </Text>
        </Space>

        <Table
          rowKey="id"
          columns={columns}
          dataSource={filteredRuns}
          loading={loading}
          rowSelection={rowSelection}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 条`,
            style: { color: '#8b949e' },
          }}
          style={{
            background: '#0d1117',
            borderRadius: 6,
          }}
          rowClassName={(record) => (record.total_return >= 0 ? 'row-positive' : 'row-negative')}
        />
      </Card>

      <style>{`
        .ant-table {
          background: #0d1117 !important;
          color: #c9d1d9 !important;
        }
        .ant-table-thead > tr > th {
          background: #161b22 !important;
          color: #c9d1d9 !important;
          border-bottom: 1px solid #30363d !important;
        }
        .ant-table-tbody > tr > td {
          background: #0d1117 !important;
          border-bottom: 1px solid #21262d !important;
          color: #c9d1d9 !important;
        }
        .ant-table-tbody > tr:hover > td {
          background: #161b22 !important;
        }
        .row-positive {
          background: rgba(82, 196, 26, 0.05) !important;
        }
        .row-negative {
          background: rgba(255, 77, 79, 0.05) !important;
        }
        .ant-table-wrapper .ant-pagination {
          color: #8b949e !important;
        }
        .ant-pagination-item {
          background: #161b22 !important;
          border-color: #30363d !important;
        }
        .ant-pagination-item a {
          color: #8b949e !important;
        }
        .ant-pagination-item-active {
          background: #00d4aa !important;
          border-color: #00d4aa !important;
        }
        .ant-pagination-item-active a {
          color: #fff !important;
        }
        .ant-input {
          background: #0d1117 !important;
          border-color: #30363d !important;
          color: #c9d1d9 !important;
        }
        .ant-input:hover,
        .ant-input:focus {
          border-color: #00d4aa !important;
        }
        .ant-card {
          background: #161b22 !important;
          border-color: #30363d !important;
        }
      `}</style>
    </div>
  );
}

export default RunsList;
