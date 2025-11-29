"""Sentiment cache manager for PostgreSQL"""
import os
import psycopg2
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from contextlib import contextmanager


class SentimentCache:
    """Reads pre-computed sentiment data from PostgreSQL"""

    def __init__(self, db_path=None):
        """Initialize connection"""
        self.postgres_url = os.getenv('DATABASE_URL')
        
        # Also check streamlit secrets if env var not set
        if not self.postgres_url:
            try:
                import streamlit as st
                if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
                    self.postgres_url = st.secrets['DATABASE_URL']
            except:
                pass
        
        if not self.postgres_url:
            raise ValueError("DATABASE_URL environment variable not set")

    @contextmanager
    def get_connection(self):
        """Get database connection"""
        conn = psycopg2.connect(self.postgres_url)
        try:
            yield conn
        finally:
            conn.close()

    def get_ticker_sentiment_summary(self, ticker: str, days: int = 7) -> Optional[Dict]:
        """Get weighted sentiment summary for ticker"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # First try to get from pre-computed summary table
            cursor.execute("""
                SELECT
                    ticker,
                    weighted_sentiment_score,
                    simple_avg_score,
                    article_count,
                    positive_count,
                    neutral_count,
                    negative_count,
                    avg_confidence,
                    oldest_article_date,
                    newest_article_date,
                    lambda_decay,
                    lookback_days,
                    last_updated,
                    positive_pct,
                    neutral_pct,
                    negative_pct
                FROM ticker_sentiment_summary
                WHERE ticker = %s
            """, (ticker,))

            row = cursor.fetchone()
            
            if row:
                cursor.close()
                
                newest_article_date = row[9]
                last_updated = row[12]
                
                hours_old = 0
                article_hours_old = 0
                now = datetime.now()
                
                if last_updated:
                    hours_old = (now - last_updated).total_seconds() / 3600
                if newest_article_date:
                    article_hours_old = (now - newest_article_date).total_seconds() / 3600

                return {
                    'ticker': row[0],
                    'avg_sentiment_score': float(row[1]),  # This is the WEIGHTED score
                    'simple_avg_score': float(row[2]),
                    'avg_confidence': float(row[7]),
                    'article_count': row[3],
                    'positive_count': row[4],
                    'neutral_count': row[5],
                    'negative_count': row[6],
                    'distribution': {
                        'positive': float(row[13]) if row[13] else 0,
                        'neutral': float(row[14]) if row[14] else 0,
                        'negative': float(row[15]) if row[15] else 0
                    },
                    'last_updated': last_updated.isoformat() if last_updated else None,
                    'last_article_date': newest_article_date.isoformat() if newest_article_date else None,
                    'hours_old': hours_old,
                    'article_hours_old': article_hours_old,
                    'is_stale': hours_old > 24,
                    'lambda_decay': float(row[10]) if row[10] else 0.1,
                    'lookback_days': row[11] if row[11] else 7,
                    'is_weighted': True
                }
            
            # Fallback: compute from raw sentiment_scores if summary doesn't exist
            cursor.execute("""
                SELECT
                    COUNT(*) as article_count,
                    AVG(ss.sentiment_score) as avg_sentiment_score,
                    AVG(ss.confidence) as avg_confidence,
                    SUM(CASE WHEN ss.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN ss.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN ss.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                    MAX(na.published_at) as last_article_date,
                    MAX(ss.analyzed_at) as last_updated
                FROM sentiment_scores ss
                JOIN news_articles na ON ss.news_article_id = na.id
                WHERE ss.ticker = %s
                  AND na.published_at >= %s
            """, (ticker, datetime.now() - timedelta(days=days)))

            row = cursor.fetchone()
            cursor.close()

            if not row or row[0] == 0:
                return None

            article_count = row[0]
            avg_sentiment = row[1] if row[1] is not None else 0
            avg_confidence = row[2] if row[2] is not None else 0
            positive_count = row[3] or 0
            negative_count = row[4] or 0
            neutral_count = row[5] or 0
            last_article_date = row[6]
            last_updated = row[7]

            hours_old = 0
            article_hours_old = 0
            now = datetime.now()

            if last_updated:
                hours_old = (now - last_updated).total_seconds() / 3600
            if last_article_date:
                article_hours_old = (now - last_article_date).total_seconds() / 3600

            # Calculate distribution percentages
            distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
            if article_count > 0:
                distribution = {
                    'positive': (positive_count / article_count) * 100,
                    'neutral': (neutral_count / article_count) * 100,
                    'negative': (negative_count / article_count) * 100
                }

            return {
                'ticker': ticker,
                'avg_sentiment_score': float(avg_sentiment),  # Simple avg (fallback)
                'simple_avg_score': float(avg_sentiment),
                'avg_confidence': float(avg_confidence),
                'article_count': article_count,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'distribution': distribution,
                'last_updated': last_updated.isoformat() if last_updated else None,
                'last_article_date': last_article_date.isoformat() if last_article_date else None,
                'hours_old': hours_old,
                'article_hours_old': article_hours_old,
                'is_stale': hours_old > 24,
                'is_weighted': False  
            }

    def get_most_recent_article_date(self, ticker: str) -> Optional[str]:
        """Get most recent article date"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT published_at
                FROM news_articles
                WHERE ticker = %s
                ORDER BY published_at DESC
                LIMIT 1
            """, (ticker,))

            row = cursor.fetchone()
            cursor.close()

            return row[0].isoformat() if row else None

    def has_recent_data(self, ticker: str, max_age_hours: int = 24) -> bool:
        """Check if recent sentiment data exists"""
        summary = self.get_ticker_sentiment_summary(ticker)

        if not summary:
            return False

        return summary['article_hours_old'] < max_age_hours

    def get_bulk_sentiment_summary(self, tickers: list, days: int = 7) -> Dict[str, Dict]:
        """Get sentiment summaries for multiple tickers"""
        if not tickers:
            return {}

        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now()

            placeholders = ','.join(['%s'] * len(tickers))
            cursor.execute(f"""
                SELECT
                    ticker,
                    weighted_sentiment_score,
                    simple_avg_score,
                    article_count,
                    positive_count,
                    neutral_count,
                    negative_count,
                    avg_confidence,
                    oldest_article_date,
                    newest_article_date,
                    lambda_decay,
                    lookback_days,
                    last_updated,
                    positive_pct,
                    neutral_pct,
                    negative_pct
                FROM ticker_sentiment_summary
                WHERE ticker IN ({placeholders})
            """, tuple(tickers))

            rows = cursor.fetchall()
            results = {}
            found_tickers = set()

            for row in rows:
                ticker = row[0]
                found_tickers.add(ticker)
                
                newest_article_date = row[9]
                last_updated = row[12]
                
                hours_old = 0
                article_hours_old = 0
                
                if last_updated:
                    hours_old = (now - last_updated).total_seconds() / 3600
                if newest_article_date:
                    article_hours_old = (now - newest_article_date).total_seconds() / 3600

                results[ticker] = {
                    'ticker': ticker,
                    'avg_sentiment_score': float(row[1]),  # WEIGHTED score
                    'simple_avg_score': float(row[2]),
                    'avg_confidence': float(row[7]),
                    'article_count': row[3],
                    'positive_count': row[4],
                    'neutral_count': row[5],
                    'negative_count': row[6],
                    'distribution': {
                        'positive': float(row[13]) if row[13] else 0,
                        'neutral': float(row[14]) if row[14] else 0,
                        'negative': float(row[15]) if row[15] else 0
                    },
                    'last_updated': last_updated.isoformat() if last_updated else None,
                    'last_article_date': newest_article_date.isoformat() if newest_article_date else None,
                    'hours_old': hours_old,
                    'article_hours_old': article_hours_old,
                    'is_stale': hours_old > 24,
                    'is_weighted': True
                }

            # Fallback: get remaining tickers from raw sentiment_scores
            missing_tickers = [t for t in tickers if t not in found_tickers]
            
            if missing_tickers:
                cutoff_date = now - timedelta(days=days)
                placeholders = ','.join(['%s'] * len(missing_tickers))
                
                cursor.execute(f"""
                    SELECT
                        ss.ticker,
                        COUNT(*) as article_count,
                        AVG(ss.sentiment_score) as avg_sentiment_score,
                        AVG(ss.confidence) as avg_confidence,
                        SUM(CASE WHEN ss.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                        SUM(CASE WHEN ss.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                        SUM(CASE WHEN ss.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                        MAX(na.published_at) as last_article_date,
                        MAX(ss.analyzed_at) as last_updated
                    FROM sentiment_scores ss
                    JOIN news_articles na ON ss.news_article_id = na.id
                    WHERE ss.ticker IN ({placeholders})
                      AND na.published_at >= %s
                    GROUP BY ss.ticker
                """, (*missing_tickers, cutoff_date))

                for row in cursor.fetchall():
                    ticker = row[0]
                    article_count = row[1]
                    avg_sentiment = row[2] if row[2] is not None else 0
                    avg_confidence = row[3] if row[3] is not None else 0
                    positive_count = row[4] or 0
                    negative_count = row[5] or 0
                    neutral_count = row[6] or 0
                    last_article_date = row[7]
                    last_updated = row[8]

                    hours_old = 0
                    article_hours_old = 0

                    if last_updated:
                        hours_old = (now - last_updated).total_seconds() / 3600
                    if last_article_date:
                        article_hours_old = (now - last_article_date).total_seconds() / 3600

                    distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
                    if article_count > 0:
                        distribution = {
                            'positive': (positive_count / article_count) * 100,
                            'neutral': (neutral_count / article_count) * 100,
                            'negative': (negative_count / article_count) * 100
                        }

                    results[ticker] = {
                        'ticker': ticker,
                        'avg_sentiment_score': float(avg_sentiment),
                        'simple_avg_score': float(avg_sentiment),
                        'avg_confidence': float(avg_confidence),
                        'article_count': article_count,
                        'positive_count': positive_count,
                        'negative_count': negative_count,
                        'neutral_count': neutral_count,
                        'distribution': distribution,
                        'last_updated': last_updated.isoformat() if last_updated else None,
                        'last_article_date': last_article_date.isoformat() if last_article_date else None,
                        'hours_old': hours_old,
                        'article_hours_old': article_hours_old,
                        'is_stale': hours_old > 24,
                        'is_weighted': False
                    }

            cursor.close()
            return results
    def get_weighted_scores_for_optimization(self, tickers: List[str]) -> Dict[str, float]:
        """Get weighted sentiment scores for optimization"""
        summaries = self.get_bulk_sentiment_summary(tickers)
        return {
            ticker: data['avg_sentiment_score']
            for ticker, data in summaries.items()
        }
