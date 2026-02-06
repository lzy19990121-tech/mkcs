# MKCS Web API Contract

This document defines the TypeScript interfaces for the MKCS Web UI API.

## Backtest Summary

```typescript
interface BacktestSummary {
  id: string;                  // Unique identifier (directory name)
  date: string;                // Backtest execution date (ISO format)
  start_date: string;          // Backtest start date
  end_date: string;            // Backtest end date
  final_equity: number;        // Final equity value
  total_return: number;        // Total return percentage
  strategy: string;            // Strategy name/identifier
  path?: string;               // Full path to backtest directory
}
```

## Backtest Detail

```typescript
interface BacktestDetail {
  summary: BacktestSummary;
  equity_curve: EquityPoint[];
  trades: Trade[];
  risk_rejects: RiskReject[];
}

interface EquityPoint {
  timestamp: string;           // ISO format timestamp
  equity: number;              // Equity value at this point
  cash?: number;               // Cash balance (optional)
}
```

## Trade

```typescript
interface Trade {
  timestamp: string;           // ISO format timestamp
  symbol: string;              // Trading symbol
  side: 'BUY' | 'SELL';        // Trade side
  price: number;               // Execution price
  quantity: number;            // Quantity traded
  commission: number;          // Commission fee
  pnl?: number;                // Profit/Loss (for closing trades)
}
```

## Risk Reject

```typescript
interface RiskReject {
  timestamp: string;           // ISO format timestamp
  symbol: string;              // Symbol that triggered rejection
  action: string;              // Action that was rejected (e.g., 'BUY', 'SELL')
  reason: string;              // Rejection reason
  confidence?: number;         // Confidence score (0-1)
}
```

## Risk Event

```typescript
interface RiskEvent {
  id: string;                  // Unique event ID
  timestamp: string;           // ISO format timestamp
  symbol?: string;             // Related symbol (optional)
  action: string;              // Action taken (e.g., 'REDUCE', 'PAUSE', 'DISABLE')
  reason: string;              // Reason for the action
  metrics_snapshot?: {         // System metrics at the time of event
    equity?: number;
    drawdown?: number;
    position_count?: number;
  };
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
}
```

## Rule Version

```typescript
interface RuleVersion {
  id: string;                  // Version ID
  name: string;                // Rule name
  version: number;             // Version number
  created_at: string;          // Creation timestamp
  is_active: boolean;          // Whether this version is currently active
  rules: Rule[];
}

interface Rule {
  name: string;                // Rule name/identifier
  condition: string;           // Condition expression
  threshold: number;           // Threshold value
  action: string;              // Action to take when triggered
  priority: number;            // Priority level
  scope: string[];             // Applicable symbols/universe
}
```

## Artifact

```typescript
interface Artifact {
  name: string;                // File/directory name
  type: 'file' | 'directory';  // Artifact type
  path: string;                // Relative path from backtest root
  size?: number;               // File size in bytes (for files)
}
```

## Compare Result

```typescript
interface CompareResult {
  left: BacktestDetail;
  right: BacktestDetail;
  diff: {
    return_diff: number;
    sharpe_diff: number;
    mdd_diff: number;
    trades_diff: number;
  };
}
```

## API Endpoints

### Backtests

- `GET /api/backtests` - List all backtests (returns `BacktestSummary[]`)
- `GET /api/backtest/:id` - Get backtest detail (returns `BacktestDetail`)
- `GET /api/backtest/:id/artifacts` - List artifacts (returns `Artifact[]`)
- `GET /api/backtest/:id/artifact/:filepath` - Get artifact content

### Risk

- `GET /api/risk/status` - Get current risk status
- `GET /api/risk/events?since=<timestamp>` - Get risk events (returns `RiskEvent[]`)

### Rules

- `GET /api/rules/versions` - List rule versions (returns `RuleVersion[]`)
- `GET /api/rules/:vid` - Get rule version detail
- `POST /api/rules/apply` - Apply a rule version to paper trading

### Health

- `GET /api/health` - Health check
