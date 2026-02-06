import React, { useState } from 'react';
import { Card, Typography, Button, Input, Space } from 'antd';
import { SearchOutlined } from '@ant-design/icons';

const { Text } = Typography;

/**
 * JSONViewer - JSON 查看器组件
 *
 * 展示 JSON 数据，支持搜索和折叠
 */
function JSONViewer({ data, loading = false }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [expanded, setExpanded] = useState(true);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        加载中...
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>
        暂无数据
      </div>
    );
  }

  // 渲染 JSON 值
  const renderValue = (value, depth = 0) => {
    const indent = depth * 16;

    if (value === null) {
      return <span style={{ color: '#ff7b72' }}>null</span>;
    }

    if (typeof value === 'boolean') {
      return <span style={{ color: '#79c0ff' }}>{value.toString()}</span>;
    }

    if (typeof value === 'number') {
      return <span style={{ color: '#79c0ff' }}>{value}</span>;
    }

    if (typeof value === 'string') {
      // 检查是否包含搜索词
      const parts = value.split(new RegExp(`(${searchTerm})`, 'gi'));
      return (
        <span style={{ color: '#a5d6ff' }}>
          "
          {parts.map((part, i) => (
            searchTerm && part.toLowerCase() === searchTerm.toLowerCase() ? (
              <mark key={i} style={{ background: '#ffd33d', color: '#000' }}>{part}</mark>
            ) : (
              <span key={i}>{part}</span>
            )
          ))}
          "
        </span>
      );
    }

    if (Array.isArray(value)) {
      if (value.length === 0) {
        return <span>[]</span>;
      }

      return (
        <div>
          <span style={{ color: '#ff7b72' }}>[</span>
          <div style={{ marginLeft: indent + 16 }}>
            {value.map((item, index) => (
              <div key={index}>
                {renderValue(item, depth + 1)}
                {index < value.length - 1 && <span style={{ color: '#8b949e' }}>,</span>}
              </div>
            ))}
          </div>
          <span style={{ color: '#ff7b72' }}>]</span>
        </div>
      );
    }

    if (typeof value === 'object') {
      const keys = Object.keys(value);
      if (keys.length === 0) {
        return <span>{'{}'}</span>;
      }

      return (
        <div>
          <span style={{ color: '#ff7b72' }}>{'{'}</span>
          <div style={{ marginLeft: indent + 16 }}>
            {keys.map((key, index) => (
              <div key={key}>
                <span style={{ color: '#7ee787' }}>"{key}"</span>
                <span style={{ color: '#8b949e' }}>: </span>
                {renderValue(value[key], depth + 1)}
                {index < keys.length - 1 && <span style={{ color: '#8b949e' }}>,</span>}
              </div>
            ))}
          </div>
          <span style={{ color: '#ff7b72' }}>{'}'}</span>
        </div>
      );
    }

    return String(value);
  };

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
        <Input
          placeholder="搜索..."
          prefix={<SearchOutlined />}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{ width: 300 }}
          allowClear
        />
        <Button
          size="small"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? '全部折叠' : '全部展开'}
        </Button>
      </Space>

      <div
        style={{
          background: '#0d1117',
          padding: 16,
          borderRadius: 6,
          fontFamily: 'monospace',
          fontSize: 13,
          overflowX: 'auto',
          maxHeight: 600,
          overflowY: 'auto',
        }}
      >
        {renderValue(data)}
      </div>
    </Card>
  );
}

export default JSONViewer;
