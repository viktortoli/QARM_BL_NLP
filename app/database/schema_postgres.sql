-- Table 1: Stock Metadata
CREATE TABLE IF NOT EXISTS stock_metadata (
    ticker VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap DECIMAL(15, 2),  -- in billions USD
    currency VARCHAR(10) DEFAULT 'USD',
    exchange VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Daily Price Data
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    close DECIMAL(12, 4) NOT NULL,  -- Close price
    PRIMARY KEY (ticker, date)
);

-- Table 3: Market Cap History
CREATE TABLE IF NOT EXISTS market_cap_history (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    market_cap DECIMAL(15, 2) NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- Table 4: Data Quality Log
CREATE TABLE IF NOT EXISTS data_quality_log (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date_from DATE,
    date_to DATE,
    days_expected INTEGER,
    days_found INTEGER,
    completeness_pct DECIMAL(5, 2),
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    issues TEXT
);

-- Indexes (perf)
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_market_cap_date ON market_cap_history(date);
CREATE INDEX IF NOT EXISTS idx_metadata_sector ON stock_metadata(sector);

-- Table 5: Sector ETF Prices
CREATE TABLE IF NOT EXISTS sector_etf_prices (
    etf_ticker VARCHAR(10) NOT NULL,
    sector_name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    close DECIMAL(12, 4) NOT NULL,
    PRIMARY KEY (etf_ticker, date)
);

-- Index for sector ETF queries
CREATE INDEX IF NOT EXISTS idx_sector_etf_date ON sector_etf_prices(date);
CREATE INDEX IF NOT EXISTS idx_sector_etf_ticker_date ON sector_etf_prices(etf_ticker, date DESC);

-- View: ETF prices and 30-day returns
CREATE OR REPLACE VIEW sector_returns AS
SELECT 
    s1.etf_ticker,
    s1.sector_name,
    s1.close as current_price,
    s1.date as latest_date,
    s2.close as price_30d_ago,
    s2.date as date_30d_ago,
    ROUND((s1.close - s2.close) / s2.close * 100, 2) as return_30d_pct
FROM sector_etf_prices s1
LEFT JOIN sector_etf_prices s2 ON s1.etf_ticker = s2.etf_ticker 
    AND s2.date = (
        SELECT MAX(date) FROM sector_etf_prices 
        WHERE etf_ticker = s1.etf_ticker 
        AND date <= s1.date - INTERVAL '30 days'
    )
WHERE s1.date = (SELECT MAX(date) FROM sector_etf_prices WHERE etf_ticker = s1.etf_ticker);

-- View: Latest price for each ticker
CREATE OR REPLACE VIEW latest_prices AS
SELECT
    ticker,
    date as last_updated,
    close as current_price
FROM daily_prices dp1
WHERE date = (
    SELECT MAX(date)
    FROM daily_prices dp2
    WHERE dp2.ticker = dp1.ticker
);

-- View: Data coverage
CREATE OR REPLACE VIEW data_coverage AS
SELECT
    ticker,
    MIN(date) as first_date,
    MAX(date) as last_date,
    COUNT(*) as trading_days,
    (MAX(date) - MIN(date)) as calendar_days,
    ROUND(COUNT(*) * 100.0 / NULLIF((MAX(date) - MIN(date)) / 7 * 5, 0), 2) as completeness_pct
FROM daily_prices
GROUP BY ticker;

-- ============================================================
-- SENTIMENT ANALYSIS TABLES
-- ============================================================

-- Table 6: News Articles
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    url TEXT UNIQUE NOT NULL,
    source VARCHAR(255),
    published_at TIMESTAMP NOT NULL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 7: Sentiment Scores (individual article scores from FinBERT)
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL PRIMARY KEY,
    news_article_id INTEGER NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    sentiment_score DECIMAL(6, 4) NOT NULL,  -- FinBERT score: -1.0 to +1.0
    confidence DECIMAL(5, 4) NOT NULL,       -- FinBERT confidence: 0.0 to 1.0
    sentiment_label VARCHAR(20) NOT NULL,    -- 'positive', 'neutral', 'negative'
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 8: Ticker Sentiment Summary (pre-computed weighted aggregates)
CREATE TABLE IF NOT EXISTS ticker_sentiment_summary (
    ticker VARCHAR(10) PRIMARY KEY,
    weighted_sentiment_score DECIMAL(6, 4) NOT NULL,
    simple_avg_score DECIMAL(6, 4) NOT NULL,
    
    -- Article statistics
    article_count INTEGER NOT NULL,
    positive_count INTEGER NOT NULL,
    neutral_count INTEGER NOT NULL,
    negative_count INTEGER NOT NULL,
    avg_confidence DECIMAL(5, 4) NOT NULL,
    oldest_article_date TIMESTAMP,
    newest_article_date TIMESTAMP,
    lambda_decay DECIMAL(5, 3) DEFAULT 0.1,
    lookback_days INTEGER DEFAULT 7,         
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Distribution 
    positive_pct DECIMAL(5, 2),
    neutral_pct DECIMAL(5, 2),
    negative_pct DECIMAL(5, 2)
);

-- Indexes for sentiment tables
CREATE INDEX IF NOT EXISTS idx_news_articles_ticker ON news_articles(ticker);
CREATE INDEX IF NOT EXISTS idx_news_articles_published ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_ticker_date ON news_articles(ticker, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_ticker ON sentiment_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_article ON sentiment_scores(news_article_id);

-- View: Latest sentiment for each ticker (quickk)
CREATE OR REPLACE VIEW latest_sentiment AS
SELECT
    tss.ticker,
    tss.weighted_sentiment_score,
    tss.simple_avg_score,
    tss.article_count,
    tss.avg_confidence,
    tss.positive_pct,
    tss.neutral_pct,
    tss.negative_pct,
    tss.newest_article_date,
    tss.last_updated,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - tss.newest_article_date)) / 3600 as hours_since_latest_article
FROM ticker_sentiment_summary tss;
