import { useEffect, useCallback } from 'react';
import { stocksAPI, riskAPI } from '../services/api';
import useMarketStore from '../stores/marketStore';

export function useWatchlist() {
  const watchlist = useMarketStore((state) => state.watchlist);
  const setWatchlist = useMarketStore((state) => state.setWatchlist);
  const setWatchlistLoading = useMarketStore((state) => state.setWatchlistLoading);
  const updateQuote = useMarketStore((state) => state.updateQuote);

  const fetchWatchlist = useCallback(async () => {
    setWatchlistLoading(true);
    try {
      const data = await stocksAPI.list();
      setWatchlist(data);
      return data;
    } catch (error) {
      console.error('Failed to fetch watchlist:', error);
      return [];
    } finally {
      setWatchlistLoading(false);
    }
  }, [setWatchlist, setWatchlistLoading]);

  const refreshQuotes = useCallback(async (symbols) => {
    for (const symbol of symbols) {
      try {
        const quote = await stocksAPI.getQuote(symbol);
        updateQuote(symbol, quote);
      } catch (error) {
        console.error(`Failed to fetch quote for ${symbol}:`, error);
      }
    }
  }, [updateQuote]);

  useEffect(() => {
    fetchWatchlist();
  }, [fetchWatchlist]);

  return {
    watchlist,
    fetchWatchlist,
    refreshQuotes,
  };
}

export function useBars(symbol, interval = '1d', days = 90, maPeriods = [5, 20, 50]) {
  const bars = useMarketStore((state) => state.bars[symbol]);
  const maData = useMarketStore((state) => state.maData[symbol]);
  const setBars = useMarketStore((state) => state.setBars);
  const setMAData = useMarketStore((state) => state.setMAData);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  // 使用 ref 来追踪当前参数，避免依赖数组问题
  const paramsRef = React.useRef({ symbol, interval, days, maPeriods });

  // 使用 ref 存储 fetchBars，避免依赖变化
  const fetchBarsRef = React.useRef(async () => {
    const params = paramsRef.current;
    if (!params.symbol) return;

    const maString = params.maPeriods.join(',');
    setLoading(true);
    setError(null);

    try {
      const data = await stocksAPI.getBars(params.symbol, {
        interval: params.interval,
        days: params.days,
        ma: maString
      });
      setBars(params.symbol, data.bars);
      setMAData(params.symbol, data.ma);
    } catch (err) {
      setError(err);
      console.error(`Failed to fetch bars for ${params.symbol}:`, err);
    } finally {
      setLoading(false);
    }
  });

  // 稳定的 fetchBars 函数
  const fetchBars = React.useCallback(() => {
    fetchBarsRef.current();
  }, []);

  // 更新 ref 并触发 fetch 当参数改变时
  useEffect(() => {
    paramsRef.current = { symbol, interval, days, maPeriods };
    fetchBars();
  }, [symbol, interval, days, maPeriods.join(',')]); // 使用 join(',') 避免数组引用问题

  return { bars, maData, loading, error, refetch: fetchBars };
}

import React from 'react';

export function useOrders() {
  const positions = useMarketStore((state) => state.positions);
  const trades = useMarketStore((state) => state.trades);
  const setPositions = useMarketStore((state) => state.setPositions);
  const setTrades = useMarketStore((state) => state.setTrades);

  const fetchOrders = useCallback(async () => {
    try {
      const data = await stocksAPI.list();
      setPositions(data.positions || {});
      // 合并系统成交和手动记录
      const allTrades = [...(data.trades || [])];
      setTrades(allTrades);
    } catch (error) {
      console.error('Failed to fetch orders:', error);
    }
  }, [setPositions, setTrades]);

  useEffect(() => {
    fetchOrders();
    const interval = setInterval(fetchOrders, 30000); // 每30秒刷新
    return () => clearInterval(interval);
  }, [fetchOrders]);

  return { positions, trades, refetch: fetchOrders };
}

export function usePerformance() {
  const performance = useMarketStore((state) => state.performance);
  const setPerformance = useMarketStore((state) => state.setPerformance);
  const riskStatus = useMarketStore((state) => state.riskStatus);
  const setRiskStatus = useMarketStore((state) => state.setRiskStatus);

  const fetchPerformance = useCallback(async () => {
    try {
      const perf = await riskAPI.getPerformance();
      setPerformance(perf);
    } catch (error) {
      console.error('Failed to fetch performance:', error);
    }
  }, [setPerformance]);

  const fetchRiskStatus = useCallback(async () => {
    try {
      const status = await riskAPI.getStatus();
      setRiskStatus(status);
    } catch (error) {
      console.error('Failed to fetch risk status:', error);
    }
  }, [setRiskStatus]);

  useEffect(() => {
    fetchPerformance();
    fetchRiskStatus();
    const interval = setInterval(() => {
      fetchPerformance();
      fetchRiskStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchPerformance, fetchRiskStatus]);

  return { performance, riskStatus, refetch: () => {
    fetchPerformance();
    fetchRiskStatus();
  }};
}

export default {
  useWatchlist,
  useBars,
  useOrders,
  usePerformance,
};
