-- Database initialization for market making strategy
-- Tables for trade logging, risk metrics, and audit trail

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trades table
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trade_id VARCHAR(64) UNIQUE NOT NULL,
    instrument VARCHAR(32) NOT NULL,
    side VARCHAR(8) NOT NULL CHECK (side IN ('buy', 'sell')),
    quantity DECIMAL(18, 8) NOT NULL,
    price DECIMAL(18, 8) NOT NULL,
    notional DECIMAL(18, 8) NOT NULL,
    fee DECIMAL(18, 8) DEFAULT 0,
    strategy VARCHAR(64) NOT NULL,
    is_hedge BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for time-series queries
CREATE INDEX idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX idx_trades_instrument ON trades(instrument);
CREATE INDEX idx_trades_strategy ON trades(strategy);

-- Portfolio snapshots
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_value DECIMAL(18, 8) NOT NULL,
    cash DECIMAL(18, 8) NOT NULL,
    delta DECIMAL(18, 8) NOT NULL,
    gamma DECIMAL(18, 8),
    vega DECIMAL(18, 8),
    theta DECIMAL(18, 8),
    positions JSONB NOT NULL,
    pnl_24h DECIMAL(18, 8),
    pnl_total DECIMAL(18, 8),
    drawdown DECIMAL(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp DESC);

-- Risk events log
CREATE TABLE IF NOT EXISTS risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(64) NOT NULL,
    severity VARCHAR(16) NOT NULL CHECK (severity IN ('info', 'warning', 'critical')),
    description TEXT NOT NULL,
    old_state VARCHAR(32),
    new_state VARCHAR(32),
    metrics JSONB DEFAULT '{}',
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(64),
    acknowledged_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_risk_events_timestamp ON risk_events(timestamp DESC);
CREATE INDEX idx_risk_events_severity ON risk_events(severity);
CREATE INDEX idx_risk_events_acknowledged ON risk_events(acknowledged);

-- Circuit breaker history
CREATE TABLE IF NOT EXISTS circuit_breaker_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_type VARCHAR(64) NOT NULL,
    old_state VARCHAR(32) NOT NULL,
    new_state VARCHAR(32) NOT NULL,
    trigger_value DECIMAL(18, 8),
    threshold DECIMAL(18, 8),
    portfolio_value DECIMAL(18, 8),
    duration_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_circuit_breaker_history_timestamp ON circuit_breaker_history(timestamp DESC);

-- Regime detection history
CREATE TABLE IF NOT EXISTS regime_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    regime VARCHAR(32) NOT NULL,
    regime_probability DECIMAL(10, 4),
    switch_probability DECIMAL(10, 4),
    volatility_estimate DECIMAL(18, 8),
    features JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_regime_history_timestamp ON regime_history(timestamp DESC);

-- Hedging activity log
CREATE TABLE IF NOT EXISTS hedge_activity (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    trigger_reason VARCHAR(128) NOT NULL,
    target_delta DECIMAL(18, 8) NOT NULL,
    current_delta DECIMAL(18, 8) NOT NULL,
    hedge_size DECIMAL(18, 8) NOT NULL,
    hedge_price DECIMAL(18, 8),
    expected_cost DECIMAL(18, 8),
    executed BOOLEAN DEFAULT FALSE,
    execution_time_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hedge_activity_timestamp ON hedge_activity(timestamp DESC);
CREATE INDEX idx_hedge_activity_executed ON hedge_activity(executed);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name VARCHAR(64) NOT NULL,
    metric_value DECIMAL(18, 8) NOT NULL,
    labels JSONB DEFAULT '{}',
    window_seconds INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_performance_metrics_name_timestamp ON performance_metrics(metric_name, timestamp DESC);

-- Create views for common queries
CREATE OR REPLACE VIEW daily_pnl AS
SELECT
    DATE(timestamp) as date,
    SUM(CASE WHEN side = 'sell' THEN notional ELSE -notional END) - SUM(fee) as gross_pnl,
    COUNT(*) as trade_count,
    SUM(fee) as total_fees
FROM trades
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Create materialized view for risk dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS risk_summary AS
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_events,
    COUNT(*) FILTER (WHERE severity = 'warning') as warning_events,
    AVG(drawdown) as avg_drawdown,
    MAX(drawdown) as max_drawdown
FROM risk_events
LEFT JOIN portfolio_snapshots ON DATE_TRUNC('hour', risk_events.timestamp) = DATE_TRUNC('hour', portfolio_snapshots.timestamp)
GROUP BY DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- Refresh materialized view every 5 minutes
SELECT cron.schedule('refresh-risk-summary', '*/5 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY risk_summary');
