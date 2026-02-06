import React, { useState, useEffect } from 'react';
import { Card, Typography, Divider } from 'antd';

const { Text, Paragraph } = Typography;

/**
 * MarkdownViewer - Markdown 渲染组件
 *
 * 渲染 Markdown 内容并支持目录跳转
 * 不依赖外部库，使用简单的正则解析
 */
function MarkdownViewer({ content, loading = false }) {
  const [toc, setToc] = useState([]);

  useEffect(() => {
    if (!content) return;

    // 提取标题生成目录
    const lines = content.split('\n');
    const headings = [];

    lines.forEach((line, index) => {
      const match = line.match(/^(#{1,3})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const text = match[2].trim();
        const id = `heading-${index}`;
        headings.push({ level, text, id, index });
      }
    });

    setToc(headings);
  }, [content]);

  // 简单的 Markdown 转 HTML
  const renderMarkdown = (text) => {
    if (!text) return null;

    const lines = text.split('\n');
    const elements = [];
    let inCodeBlock = false;
    let codeContent = [];
    let inList = false;
    let listItems = [];

    const flushList = () => {
      if (inList) {
        elements.push(
          <ul key={`list-${elements.length}`} style={{ marginLeft: 24, marginBottom: 16 }}>
            {listItems}
          </ul>
        );
        listItems = [];
        inList = false;
      }
    };

    lines.forEach((line, index) => {
      // 代码块处理
      if (line.trim().startsWith('```')) {
        if (inCodeBlock) {
          elements.push(
            <pre key={`code-${index}`} style={{
              background: '#0d1117',
              padding: 16,
              borderRadius: 6,
              overflowX: 'auto',
              border: '1px solid #30363d',
              marginBottom: 16
            }}>
              <code style={{ color: '#c9d1d9', fontSize: 13 }}>
                {codeContent.join('\n')}
              </code>
            </pre>
          );
          codeContent = [];
          inCodeBlock = false;
        } else {
          flushList();
          inCodeBlock = true;
        }
        return;
      }

      if (inCodeBlock) {
        codeContent.push(line);
        return;
      }

      // 标题处理
      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
      if (headingMatch) {
        flushList();
        const level = headingMatch[1].length;
        const text = headingMatch[2].trim();
        const id = `heading-${index}`;
        const HeadingTag = `h${level}`;

        const fontSize = { 1: '24px', 2: '20px', 3: '16px', 4: '14px', 5: '13px', 6: '12px' }[level] || '14px';

        elements.push(
          React.createElement(HeadingTag, {
            key: index,
            id: id,
            style: {
              color: '#c9d1d9',
              fontSize,
              marginTop: level === 1 ? 0 : 24,
              marginBottom: 16,
              borderBottom: '1px solid #30363d',
              paddingBottom: 8
            }
          }, text)
        );
        return;
      }

      // 列表处理
      const listMatch = line.match(/^[\s]*[-*]\s+(.+)$/);
      if (listMatch) {
        inList = true;
        const text = listMatch[1];
        // 处理可能的加粗和代码
        const processedText = text
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          .replace(/`(.+?)`/g, '<code style="background:#21262d;padding:2px 6px;border-radius:3px;color:#00d4aa;">$1</code>');

        listItems.push(
          <li key={index} dangerouslySetInnerHTML={{ __html: processedText }} style={{ marginBottom: 4 }} />
        );
        return;
      }

      // 分隔线
      if (line.trim() === '---') {
        flushList();
        elements.push(<Divider key={index} style={{ borderColor: '#30363d', margin: '16px 0' }} />);
        return;
      }

      // 普通段落
      if (line.trim()) {
        flushList();
        // 处理内联格式
        let processedText = line
          // 加粗
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
          // 斜体
          .replace(/\*(.+?)\*/g, '<em>$1</em>')
          // 行内代码
          .replace(/`(.+?)`/g, '<code style="background:#21262d;padding:2px 6px;border-radius:3px;color:#00d4aa;">$1</code>')
          // 链接
          .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" style="color:#00d4aa;">$1</a>');

        elements.push(
          <Paragraph key={index} style={{ color: '#c9d1d9', marginBottom: 16 }} dangerouslySetInnerHTML={{ __html: processedText }} />
        );
      }
    });

    flushList();
    return elements;
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>
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

  return (
    <div style={{ display: 'flex', gap: 24 }}>
      {/* 目录 */}
      {toc.length > 0 && (
        <div style={{ width: 200, flexShrink: 0 }}>
          <Card
            title="目录"
            size="small"
            style={{
              background: '#0d1117',
              border: '1px solid #30363d',
              position: 'sticky',
              top: 16,
            }}
            headStyle={{ borderBottom: '1px solid #30363d', color: '#c9d1d9' }}
            bodyStyle={{ padding: '8px 0' }}
          >
            <div style={{ maxHeight: 'calc(100vh - 200px)', overflowY: 'auto' }}>
              {toc.map((item) => (
                <div
                  key={item.index}
                  style={{
                    paddingLeft: `${(item.level - 1) * 12}px`,
                    marginBottom: 4,
                  }}
                >
                  <a
                    href={`#${item.id}`}
                    style={{
                      color: '#8b949e',
                      textDecoration: 'none',
                      fontSize: item.level === 1 ? 14 : 13,
                      display: 'block',
                      padding: '2px 8px',
                      borderRadius: 4,
                    }}
                    onMouseEnter={(e) => {
                      e.target.style.background = '#21262d';
                      e.target.style.color = '#00d4aa';
                    }}
                    onMouseLeave={(e) => {
                      e.target.style.background = 'transparent';
                      e.target.style.color = '#8b949e';
                    }}
                  >
                    {item.text}
                  </a>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Markdown 内容 */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <Card
          bordered={false}
          style={{
            background: '#161b22',
            borderRadius: 8,
          }}
          bodyStyle={{
            padding: 24,
            color: '#c9d1d9',
            lineHeight: 1.6,
          }}
        >
          <div className="markdown-content">
            {renderMarkdown(content)}
          </div>
        </Card>
      </div>
    </div>
  );
}

export default MarkdownViewer;
