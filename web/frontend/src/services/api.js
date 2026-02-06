import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // 忽略取消请求的错误（包括 CancelToken 和 AbortController）
    const isCanceled = axios.isCancel(error) ||
                        error.code === 'ERR_CANCELED' ||
                        error.code === 'ECONNABORTED' ||
                        error.message?.includes('canceled') ||
                        error.message?.includes('aborted');

    if (isCanceled) {
      // 静默处理，不记录任何日志
      return Promise.reject({ canceled: true });
    }
    console.error('API Error:', error.response?.data || error.message);
    throw error;
  }
);

// ============ Stocks API ============

export const stocksAPI = {
  // 获取观察列表
  list: (signal) => api.get('/stocks', signal ? { signal } : undefined),

  // 获取单个股票信息
  get: (symbol, signal) => api.get(`/stocks/${symbol}`, signal ? { signal } : undefined),

  // 获取 K 线数据
  getBars: (symbol, params = {}, signal) => api.get(`/stocks/${symbol}/bars`, {
    params,
    signal: signal || undefined,
  }),

  // 获取实时报价
  getQuote: (symbol, signal) => api.get(`/stocks/${symbol}/quote`, signal ? { signal } : undefined),

  // 获取最新价格
  getPrice: (symbol, signal) => api.get(`/stocks/${symbol}/price`, signal ? { signal } : undefined),

  // 获取策略信号和卖出区间
  getSignals: (symbol, params = {}, signal) => api.get(`/stocks/${symbol}/signals`, {
    params,
    signal: signal || undefined,
  }),
};

// ============ Orders API ============

export const ordersAPI = {
  // 获取订单/成交
  list: (signal) => api.get('/orders', signal ? { signal } : undefined),

  // 提交订单
  submit: (data, signal) => api.post('/orders', data, signal ? { signal } : undefined),

  // 撤销订单
  cancel: (orderId, signal) => api.delete(`/orders/${orderId}`, signal ? { signal } : undefined),

  // 获取交易历史
  getHistory: (params = {}, signal) => api.get('/orders/history', {
    params,
    signal: signal || undefined,
  }),
};

// ============ Annotations API ============

export const annotationsAPI = {
  // 获取标注
  getMarkers: (symbol, signal) => api.get(`/annotations/${symbol}/markers`, signal ? { signal } : undefined),

  // 添加标注
  addMarker: (symbol, data, signal) => api.post(`/annotations/${symbol}/markers`, data, signal ? { signal } : undefined),

  // 更新标注
  updateMarker: (symbol, markerId, data, signal) => api.put(`/annotations/${symbol}/markers/${markerId}`, data, signal ? { signal } : undefined),

  // 删除标注
  deleteMarker: (symbol, markerId, signal) => api.delete(`/annotations/${symbol}/markers/${markerId}`, signal ? { signal } : undefined),

  // 获取卖出区间
  getRanges: (symbol, signal) => api.get(`/annotations/${symbol}/ranges`, signal ? { signal } : undefined),

  // 添加卖出区间
  addRange: (symbol, data, signal) => api.post(`/annotations/${symbol}/ranges`, data, signal ? { signal } : undefined),

  // 更新卖出区间
  updateRange: (symbol, rangeId, data, signal) => api.put(`/annotations/${symbol}/ranges/${rangeId}`, data, signal ? { signal } : undefined),

  // 删除卖出区间
  deleteRange: (symbol, rangeId, signal) => api.delete(`/annotations/${symbol}/ranges/${rangeId}`, signal ? { signal } : undefined),
};

// ============ Risk API ============

export const riskAPI = {
  // 获取风控状态
  getStatus: (signal) => api.get('/risk/status', signal ? { signal } : undefined),

  // 获取持仓
  getPositions: (signal) => api.get('/risk/positions', signal ? { signal } : undefined),

  // 获取绩效
  getPerformance: (signal) => api.get('/risk/performance', signal ? { signal } : undefined),

  // 获取交易器状态
  getTraderStatus: (signal) => api.get('/trader/status', signal ? { signal } : undefined),

  // 控制交易器
  controlTrader: (action, config = {}, signal) => api.post('/trader/control', { action, ...config }, signal ? { signal } : undefined),

  // 获取账户汇总
  getAccountSummary: (signal) => api.get('/account/summary', signal ? { signal } : undefined),
};

// ============ Health API ============

export const healthAPI = {
  check: () => api.get('/health'),
};

// ============ Backtests API ============

export const backtestsAPI = {
  // 获取所有回测列表
  list: (signal) => api.get('/backtests', signal ? { signal } : undefined),

  // 获取单个回测详情
  get: (id, signal) => api.get(`/backtest/${id}`, signal ? { signal } : undefined),

  // 获取回测性能指标
  getMetrics: (id, signal) => api.get(`/backtest/${id}/metrics`, signal ? { signal } : undefined),

  // 获取回测文件清单
  listArtifacts: (id, signal) => api.get(`/backtest/${id}/artifacts`, signal ? { signal } : undefined),

  // 获取文件内容
  getArtifact: (id, filepath, signal) => api.get(`/backtest/${id}/artifact/${filepath}`, signal ? { signal } : undefined),
};

export default api;
