import api from '../services/api';

/**
 * Backtests API
 *
 * 提供回测结果相关的 API 调用
 */

export const backtestsAPI = {
  /**
   * 获取所有回测列表
   * @param {AbortSignal} signal - 取消信号
   * @returns {Promise<Array>} 回测摘要列表
   */
  list: (signal) => api.get('/backtests', signal ? { signal } : undefined),

  /**
   * 获取单个回测详情
   * @param {string} id - 回测 ID
   * @param {AbortSignal} signal - 取消信号
   * @returns {Promise<Object>} 回测详情
   */
  get: (id, signal) => api.get(`/backtest/${id}`, signal ? { signal } : undefined),

  /**
   * 获取回测性能指标
   * @param {string} id - 回测 ID
   * @param {AbortSignal} signal - 取消信号
   * @returns {Promise<Object>} 性能指标
   */
  getMetrics: (id, signal) => api.get(`/backtest/${id}/metrics`, signal ? { signal } : undefined),

  /**
   * 获取回测文件清单
   * @param {string} id - 回测 ID
   * @param {AbortSignal} signal - 取消信号
   * @returns {Promise<Array>} 文件清单
   */
  listArtifacts: (id, signal) => api.get(`/backtest/${id}/artifacts`, signal ? { signal } : undefined),

  /**
   * 获取文件内容
   * @param {string} id - 回测 ID
   * @param {string} filepath - 文件路径
   * @param {AbortSignal} signal - 取消信号
   * @returns {Promise<string|Object>} 文件内容
   */
  getArtifact: (id, filepath, signal) => api.get(`/backtest/${id}/artifact/${filepath}`, signal ? { signal } : undefined),
};

export default backtestsAPI;
