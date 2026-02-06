import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Tree, Card, Typography, Spin, message, Button, Space, Empty, Tabs } from 'antd';
import {
  FileOutlined,
  FolderOutlined,
  FileMarkdownOutlined,
  FileTextOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  FileExcelOutlined,
  CodeOutlined,
  DownloadOutlined,
  FileSearchOutlined,
} from '@ant-design/icons';
import { backtestsAPI } from '../../services/api';
import MarkdownViewer from './MarkdownViewer';
import JSONViewer from './JSONViewer';
import CSVViewer from './CSVViewer';

const { Text } = Typography;
const { TabPane } = Tabs;

// 报告文件优先级
const REPORT_FILES = [
  'risk_report.md',
  'deep_risk_report.md',
  'report.md',
  'summary.json'
];

/**
 * ArtifactsBrowser - 文件浏览器组件
 *
 * 浏览和预览回测目录中的文件，自动加载报告
 */
function ArtifactsBrowser({ backtestId }) {
  const [loading, setLoading] = useState(false);
  const [artifacts, setArtifacts] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [contentLoading, setContentLoading] = useState(false);
  const [reportFile, setReportFile] = useState(null);
  const [reportContent, setReportContent] = useState(null);
  const [activeTab, setActiveTab] = useState('report');
  const hasAutoLoadedReport = useRef(false);

  // 查找报告文件
  const findReportFile = useCallback((items) => {
    const flattenFiles = (list, prefix = '') => {
      let files = [];
      list.forEach(item => {
        const path = prefix ? `${prefix}/${item.name}` : item.name;
        if (item.type === 'directory' && item.children) {
          files = files.concat(flattenFiles(item.children, path));
        } else if (item.type === 'file') {
          files.push({ ...item, fullPath: path });
        }
      });
      return files;
    };

    const allFiles = flattenFiles(items);

    // 按优先级查找报告文件
    for (const reportName of REPORT_FILES) {
      const found = allFiles.find(f => f.name === reportName || f.fullPath.endsWith(reportName));
      if (found) {
        return found;
      }
    }

    return null;
  }, []);

  // 加载报告文件
  const loadReport = useCallback(async () => {
    if (!backtestId || hasAutoLoadedReport.current) return;

    setContentLoading(true);
    try {
      const data = await backtestsAPI.listArtifacts(backtestId);

      // 查找报告文件
      const report = findReportFile(data);
      if (report) {
        setReportFile(report);
        // 加载报告内容
        const content = await backtestsAPI.getArtifact(backtestId, report.fullPath || report.path);
        setReportContent(content);
        hasAutoLoadedReport.current = true;
      }

      // 转换为树形结构
      const treeData = convertToTreeData(data);
      setArtifacts(treeData);
    } catch (error) {
      if (!error.canceled) {
        console.error('加载报告失败:', error);
      }
    } finally {
      setContentLoading(false);
    }
  }, [backtestId, findReportFile]);

  useEffect(() => {
    loadReport();
  }, [loadReport]);

  // 加载文件清单（用于文件浏览器标签）
  const loadArtifacts = useCallback(async () => {
    if (!backtestId) return;

    setLoading(true);
    try {
      const data = await backtestsAPI.listArtifacts(backtestId);
      const treeData = convertToTreeData(data);
      setArtifacts(treeData);
    } catch (error) {
      if (!error.canceled) {
        message.error('加载文件清单失败');
      }
    } finally {
      setLoading(false);
    }
  }, [backtestId]);

  // 将扁平化的文件列表转换为树形结构
  const convertToTreeData = (items) => {
    const tree = [];

    items.forEach(item => {
      if (item.type === 'directory') {
        tree.push({
          title: item.name,
          key: item.path,
          selectable: false,
          icon: <FolderOutlined style={{ color: '#54aeff' }} />,
          children: item.children ? convertToTreeData(item.children) : [],
        });
      } else {
        tree.push({
          title: item.name,
          key: item.path,
          selectable: true,
          icon: getFileIcon(item.name),
          isLeaf: true,
          data: item,
        });
      }
    });

    return tree;
  };

  // 获取文件图标
  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();

    switch (ext) {
      case 'md':
        return <FileMarkdownOutlined style={{ color: '#54aeff' }} />;
      case 'txt':
      case 'log':
        return <FileTextOutlined style={{ color: '#8b949e' }} />;
      case 'pdf':
        return <FilePdfOutlined style={{ color: '#ff7b72' }} />;
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'gif':
      case 'svg':
        return <FileImageOutlined style={{ color: '#a5d6ff' }} />;
      case 'csv':
      case 'xlsx':
        return <FileExcelOutlined style={{ color: '#3fb950' }} />;
      case 'json':
        return <CodeOutlined style={{ color: '#79c0ff' }} />;
      default:
        return <FileOutlined style={{ color: '#8b949e' }} />;
    }
  };

  // 加载文件内容
  const loadFileContent = useCallback(async (filepath) => {
    if (!backtestId || !filepath) return;

    setContentLoading(true);
    setFileContent(null);

    try {
      const data = await backtestsAPI.getArtifact(backtestId, filepath);
      setFileContent(data);
    } catch (error) {
      if (!error.canceled) {
        message.error('加载文件内容失败');
      }
    } finally {
      setContentLoading(false);
    }
  }, [backtestId]);

  // 处理文件选择
  const handleSelect = (selectedKeys, info) => {
    if (selectedKeys.length > 0) {
      const filepath = selectedKeys[0];
      const fileData = info.node.data;
      setSelectedFile({ path: filepath, ...fileData });
      loadFileContent(filepath);
      setActiveTab('files');
    } else {
      setSelectedFile(null);
      setFileContent(null);
    }
  };

  // 处理下载
  const handleDownload = (filepath) => {
    if (!backtestId) return;

    const path = filepath || selectedFile?.path;
    if (!path) return;

    const url = `/api/backtest/${backtestId}/artifact/${path}?download=true`;
    window.open(url, '_blank');
  };

  // 渲染文件内容
  const renderFileContent = () => {
    if (contentLoading) {
      return (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
        </div>
      );
    }

    if (!fileContent) {
      return (
        <Empty
          description="请从左侧选择文件查看"
          style={{ color: '#8b949e', marginTop: 40 }}
        />
      );
    }

    const { type, content: data, file } = fileContent;
    const filename = selectedFile?.path || file || '';

    switch (type) {
      case 'text':
      case 'markdown':
        return <MarkdownViewer content={data} />;

      case 'json':
        return <JSONViewer data={data} />;

      case 'csv':
        return <CSVViewer content={data} filename={filename} onDownload={() => handleDownload(filename)} />;

      case 'binary':
        return (
          <Card
            bordered={false}
            style={{
              background: '#161b22',
              borderRadius: 8,
            }}
          >
            <Text type="secondary">
              此文件类型不支持预览
            </Text>
            <div style={{ marginTop: 16 }}>
              <Button
                type="primary"
                icon={<DownloadOutlined />}
                onClick={() => handleDownload(filename)}
              >
                下载文件
              </Button>
            </div>
          </Card>
        );

      default:
        return (
          <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>
            不支持的文件类型
          </div>
        );
    }
  };

  // 渲染报告内容
  const renderReport = () => {
    if (contentLoading && !reportContent) {
      return (
        <div style={{ textAlign: 'center', padding: 40 }}>
          <Spin size="large" />
        </div>
      );
    }

    if (!reportContent) {
      return (
        <Empty
          description="未找到报告文件"
          style={{ color: '#8b949e', marginTop: 40 }}
        >
          <Text type="secondary" style={{ fontSize: 12 }}>
            支持的报告文件: {REPORT_FILES.join(', ')}
          </Text>
        </Empty>
      );
    }

    const { type, content: data, file } = reportContent;

    switch (type) {
      case 'text':
      case 'markdown':
        return <MarkdownViewer content={data} />;

      case 'json':
        return <JSONViewer data={data} />;

      default:
        return (
          <Card
            bordered={false}
            style={{
              background: '#161b22',
              borderRadius: 8,
            }}
          >
            <Text type="secondary">不支持的报告类型: {type}</Text>
          </Card>
        );
    }
  };

  return (
    <div style={{ height: 600 }}>
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        style={{
          background: 'transparent',
        }}
        tabBarStyle={{
          color: '#8b949e',
          borderBottom: '1px solid #30363d',
          marginBottom: 16,
        }}
      >
        <TabPane
          tab={
            <span>
              <FileSearchOutlined />
              报告 {reportFile && `(${reportFile.name})`}
            </span>
          }
          key="report"
        >
          <div style={{ height: 500 }}>
            {renderReport()}
          </div>
        </TabPane>

        <TabPane
          tab={
            <span>
              <FolderOutlined />
              文件浏览器
            </span>
          }
          key="files"
        >
          <div style={{ display: 'flex', gap: 16, height: 500 }}>
            {/* 文件树 */}
            <Card
              title="文件"
              size="small"
              style={{
                width: 280,
                flexShrink: 0,
                background: '#161b22',
                border: '1px solid #30363d',
              }}
              headStyle={{ borderBottom: '1px solid #30363d', color: '#c9d1d9' }}
              bodyStyle={{ padding: 8, overflowY: 'auto', maxHeight: 450 }}
            >
              {loading ? (
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <Spin size="small" />
                </div>
              ) : artifacts.length > 0 ? (
                <Tree
                  showIcon
                  defaultExpandAll
                  onSelect={handleSelect}
                  treeData={artifacts}
                  style={{
                    background: 'transparent',
                    color: '#c9d1d9',
                  }}
                />
              ) : (
                <Empty
                  description="暂无文件"
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  style={{ color: '#8b949e' }}
                />
              )}
            </Card>

            {/* 文件内容 */}
            <Card
              title={
                <Space>
                  <Text style={{ color: '#c9d1d9' }}>
                    {selectedFile?.path || '预览'}
                  </Text>
                  {selectedFile && (
                    <Button
                      size="small"
                      icon={<DownloadOutlined />}
                      onClick={() => handleDownload()}
                    >
                      下载
                    </Button>
                  )}
                </Space>
              }
              style={{
                flex: 1,
                background: '#161b22',
                border: '1px solid #30363d',
              }}
              headStyle={{ borderBottom: '1px solid #30363d', color: '#c9d1d9' }}
              bodyStyle={{
                padding: 0,
                overflow: 'auto',
                maxHeight: 450,
              }}
            >
              {renderFileContent()}
            </Card>
          </div>
        </TabPane>
      </Tabs>
    </div>
  );
}

export default ArtifactsBrowser;
