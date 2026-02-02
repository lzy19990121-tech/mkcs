import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

const useMarketStore = create(
  subscribeWithSelector((set, get) => ({
    // 观察列表
    watchlist: [],
    watchlistLoading: false,

    // 股票数据
    quotes: {},
    bars: {},
    maData: {},

    // 持仓
    positions: {},

    // 绩效
    performance: null,

    // 交易历史
    trades: [],

    // 风控状态
    riskStatus: null,

    // 交易器状态
    traderStatus: null,

    // 实时数据
    realtimePrices: {},

    // 信号
    latestSignal: null,

    // 事件流
    events: [],

    // ============ Actions ============

    setWatchlist: (watchlist) => set({ watchlist }),

    setWatchlistLoading: (loading) => set({ watchlistLoading: loading }),

    updateQuote: (symbol, quote) => set((state) => ({
      quotes: { ...state.quotes, [symbol]: quote }
    })),

    setBars: (symbol, bars) => set((state) => ({
      bars: { ...state.bars, [symbol]: bars }
    })),

    setMAData: (symbol, maData) => set((state) => ({
      maData: { ...state.maData, [symbol]: maData }
    })),

    setPositions: (positions) => set({ positions }),

    updatePosition: (symbol, position) => set((state) => ({
      positions: { ...state.positions, [symbol]: position }
    })),

    setPerformance: (performance) => set({ performance }),

    setTrades: (trades) => set({ trades }),

    addTrade: (trade) => set((state) => ({
      trades: [trade, ...state.trades]
    })),

    setRiskStatus: (riskStatus) => set({ riskStatus }),

    setTraderStatus: (traderStatus) => set({ traderStatus }),

    updateRealtimePrice: (symbol, price) => set((state) => ({
      realtimePrices: { ...state.realtimePrices, [symbol]: price }
    })),

    setLatestSignal: (signal) => set({ latestSignal: signal }),

    addEvent: (event) => set((state) => ({
      events: [
        { ...event, id: Date.now(), timestamp: new Date().toISOString() },
        ...state.events.slice(0, 99)  // 保留最近100条
      ]
    })),

    clearEvents: () => set({ events: [] }),

    // ============ Computed ============

    getSymbolData: (symbol) => {
      const state = get();
      return {
        quote: state.quotes[symbol],
        bars: state.bars[symbol],
        maData: state.maData[symbol],
        position: state.positions[symbol],
        realtimePrice: state.realtimePrices[symbol],
      };
    },

    getWatchlistWithPrices: () => {
      const state = get();
      return state.watchlist.map((item) => ({
        ...item,
        quote: state.quotes[item.symbol] || state.realtimePrices[item.symbol],
        position: state.positions[item.symbol],
      }));
    },
  }))
);

export default useMarketStore;
