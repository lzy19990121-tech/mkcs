import React, { useMemo, useState } from 'react';
import { Table, Card, Button, Typography, Input, Space } from 'antd';
import { DownloadOutlined, SearchOutlined } from '@ant-design/icons';

const { Text } = Typography;

/**
 * CSVViewer - CSV 查看器组件
 *
 * 显示 CSV 数据的表格预览
 */
function CSVViewer({ content, filename, loading = false, onDownload }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [pagination, setPagination] = useState({ current: 1, pageSize: 20 });

  // 解析 CSV 内容
  const { headers, rows } = useMemo(() => {
    if (!content) return { headers: [], rows: [] };

    const lines = content.trim().split('\n');
    if (lines.length === 0) return { headers: [], rows: [] };

    // 解析 CSV（简单实现，不支持带逗号的引号内容）
    const parseLine = (line) => {
      return line.split(',').map(cell => cell.trim());
    };

    const headers = parseLine(lines[0]);
    const rows = lines.slice(1).map((line, index) => {
      const values = parseLine(line);
      return { key: index, ...Object.fromEntries(headers.map((h, i) => [h, values[i]])) };
    });

    return { headers, rows };
  }, [content]);

  // 过滤数据
  const filteredRows = useMemo(() => {
    if (!searchTerm) return rows;

    const term = searchTerm.toLowerCase();
    return rows.filter(row => {
      return Object.values(row).some(val =>
        String(val).toLowerCase().includes(term)
      );
    });
  }, [rows, searchTerm]);

  // 生成表格列
  const columns = useMemo(() => {
    return headers.map(header => ({
      title: header,
      dataIndex: header,
      key: header,
      width: 150,
      ellipsis: true,
      render: (value) => {
        if (value === undefined || value === null || value === '') {
          return <Text type="secondary">-</Text>;
        }
        return String(value);
      },
    }));
  }, [headers]);

  // 处理下载
  const handleDownload = () => {
    if (onDownload) {
      onDownload();
    } else if (content) {
      // 默认下载行为
      const blob = new Blob([content], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename || 'data.csv';
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        加载中...
      </div>
    );
  }

  if (!content) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>
        暂无内容
      </div>
    );
  }

  // 数据太大时提示下载
  if (rows.length > 1000) {
    return (
      <Card
        bordered={false}
        style={{
          background: '#161b22',
          borderRadius: 8,
        }}
      >
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Text type="secondary">
            此文件包含 {rows.length} 行数据，建议下载后查看
          </Text>
          <div style={{ marginTop: 16 }}>
            <Button
              type="primary"
              icon={<DownloadOutlined />}
              onClick={handleDownload}
            >
              下载 CSV 文件
            </Button>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card
      bordered={false}
      style={{
        background: '#161b22',
        borderRadius: 8,
      }}
      bodyStyle={{ padding: 16 }}
    >
      <Space style={{ marginBottom: 16, width: '100%', justifyContent: 'space-between' }}>
        <Space>
          <Text type="secondary">共 {filteredRows.length} 行</Text>
          {rows.length > 100 && (
            <Text type="secondary">|</Text>
          )}
          {rows.length > 100 && (
            <Text type="secondary">显示前 100 行</Text>
          )}
        </Space>
        <Space>
          <Input
            placeholder="搜索..."
            prefix={<SearchOutlined />}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ width: 250 }}
            allowClear
          />
          <Button
            icon={<DownloadOutlined />}
            onClick={handleDownload}
          >
            下载
          </Button>
        </Space>
      </Space>

      <Table
        columns={columns}
        dataSource={filteredRows.slice(0, 100)}
        pagination={{
          current: pagination.current,
          pageSize: pagination.pageSize,
          total: Math.min(filteredRows.length, 100),
          showSizeChanger: true,
          showTotal: (total) => `共 ${total} 条`,
          onChange: (page, pageSize) => setPagination({ current: page, pageSize }),
        }}
        size="small"
        scroll={{ x: 'max-content' }}
        style={{
          background: '#0d1117',
          borderRadius: 6,
        }}
      />
    </Card>
  );
}

export default CSVViewer;
