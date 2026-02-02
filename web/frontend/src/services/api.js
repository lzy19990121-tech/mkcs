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
    console.error('API Error:', error.response?.data || error.message);
    throw error;
  }
);

// ============ Stocks API ============

export const stocksAPI = {
  // 获取观察列表
  list: () => api.get('/stocks'),

  // 获取单个股票信息
  get: (symbol) => api.get(`/stocks/${symbol}`),

  // 获取 K 线数据
  getBars: (symbol, params = {}) => api.get(`/stocks/${symbol}/bars`, { params }),

  // 获取实时报价
  getQuote: (symbol) => api.get(`/stocks/${symbol}/quote`),

  // 获取最新价格
  getPrice: (symbol) => api.get(`/stocks/${symbol}/price`),

  // 获取策略信号和卖出区间
  getSignals: (symbol, params = {}) => api.get(`/stocks/${symbol}/signals`, { params }),
};

// ============ Orders API ============

export const ordersAPI = {
  // 获取订单/成交
  list: () => api.get('/orders'),

  // 提交订单
  submit: (data) => api.post('/orders', data),

  // 撤销订单
  cancel: (orderId) => api.delete(`/orders/${orderId}`),

  // 获取交易历史
  getHistory: (params = {}) => api.get('/orders/history', { params }),
};

// ============ Annotations API ============

export const annotationsAPI = {
  // 获取标注
  getMarkers: (symbol) => api.get(`/annotations/${symbol}/markers`),

  // 添加标注
  addMarker: (symbol, data) => api.post(`/annotations/${symbol}/markers`, data),

  // 更新标注
  updateMarker: (symbol, markerId, data) => api.put(`/annotations/${symbol}/markers/${markerId}`, data),

  // 删除标注
  deleteMarker: (symbol, markerId) => api.delete(`/annotations/${symbol}/markers/${markerId}`),

  // 获取卖出区间
  getRanges: (symbol) => api.get(`/annotations/${symbol}/ranges`),

  // 添加卖出区间
  addRange: (symbol, data) => api.post(`/annotations/${symbol}/ranges`, data),

  // 更新卖出区间
  updateRange: (symbol, rangeId, data) => api.put(`/annotations/${symbol}/ranges/${rangeId}`, data),

  // 删除卖出区间
  deleteRange: (symbol, rangeId) => api.delete(`/annotations/${symbol}/ranges/${rangeId}`),
};

// ============ Risk API ============

export const riskAPI = {
  // 获取风控状态
  getStatus: () => api.get('/risk/status'),

  // 获取持仓
  getPositions: () => api.get('/risk/positions'),

  // 获取绩效
  getPerformance: () => api.get('/risk/performance'),

  // 获取交易器状态
  getTraderStatus: () => api.get('/trader/status'),

  // 控制交易器
  controlTrader: (action, config = {}) => api.post('/trader/control', { action, ...config }),

  // 获取账户汇总
  getAccountSummary: () => api.get('/account/summary'),
};

// ============ Health API ============

export const healthAPI = {
  check: () => api.get('/health'),
};

export default api;
