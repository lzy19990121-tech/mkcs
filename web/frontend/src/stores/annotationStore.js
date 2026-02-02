import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

const useAnnotationStore = create(
  subscribeWithSelector((set, get) => ({
    // 标注数据
    markers: {},
    sellRanges: {},

    // 加载状态
    loading: false,

    // ============ Actions ============

    setLoading: (loading) => set({ loading }),

    setMarkers: (symbol, markers) => set((state) => ({
      markers: { ...state.markers, [symbol]: markers }
    })),

    addMarker: (symbol, marker) => set((state) => {
      const symbolMarkers = state.markers[symbol] || [];
      return {
        markers: { ...state.markers, [symbol]: [...symbolMarkers, marker] }
      };
    }),

    updateMarker: (symbol, markerId, updates) => set((state) => {
      const symbolMarkers = state.markers[symbol] || [];
      return {
        markers: {
          ...state.markers,
          [symbol]: symbolMarkers.map((m) =>
            m.id === markerId ? { ...m, ...updates } : m
          )
        }
      };
    }),

    removeMarker: (symbol, markerId) => set((state) => {
      const symbolMarkers = state.markers[symbol] || [];
      return {
        markers: {
          ...state.markers,
          [symbol]: symbolMarkers.filter((m) => m.id !== markerId)
        }
      };
    }),

    setSellRanges: (symbol, ranges) => set((state) => ({
      sellRanges: { ...state.sellRanges, [symbol]: ranges }
    })),

    addSellRange: (symbol, range) => set((state) => {
      const symbolRanges = state.sellRanges[symbol] || [];
      return {
        sellRanges: { ...state.sellRanges, [symbol]: [...symbolRanges, range] }
      };
    }),

    updateSellRange: (symbol, rangeId, updates) => set((state) => {
      const symbolRanges = state.sellRanges[symbol] || [];
      return {
        sellRanges: {
          ...state.sellRanges,
          [symbol]: symbolRanges.map((r) =>
            r.id === rangeId ? { ...r, ...updates } : r
          )
        }
      };
    }),

    removeSellRange: (symbol, rangeId) => set((state) => {
      const symbolRanges = state.sellRanges[symbol] || [];
      return {
        sellRanges: {
          ...state.sellRanges,
          [symbol]: symbolRanges.filter((r) => r.id !== rangeId)
        }
      };
    }),

    // ============ Computed ============

    getSymbolAnnotations: (symbol) => {
      const state = get();
      return {
        markers: state.markers[symbol] || [],
        sellRanges: state.sellRanges[symbol] || [],
      };
    },
  }))
);

export default useAnnotationStore;
