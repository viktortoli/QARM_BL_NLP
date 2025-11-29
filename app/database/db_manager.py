"""Database manager for stock data (PostgreSQL only)"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import warnings
import os
import time
import psycopg2
import psycopg2.extras
from urllib.parse import quote_plus, urlparse, urlunparse
warnings.filterwarnings('ignore')

# Retry configuration for Render's flaky SSL connections
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


class DatabaseManager:
    """Database manager for stock data (PostgreSQL)"""

    def __init__(self):
        """Initialize PostgreSQL database connection"""
        # Check environment variable first, then Streamlit secrets
        self.postgres_url = os.getenv('DATABASE_URL')
        
        # Try Streamlit secrets if env var not set
        if not self.postgres_url:
            try:
                import streamlit as st
                self.postgres_url = st.secrets.get('DATABASE_URL')
            except Exception:
                pass

        if not self.postgres_url:
            raise ValueError("DATABASE_URL not set. PostgreSQL connection required.")

        # Fix postgres:// to postgresql:// for psycopg2
        if self.postgres_url.startswith('postgres://'):
            self.postgres_url = self.postgres_url.replace('postgres://', 'postgresql://', 1)

        # Handle encoding issues with DATABASE_URL
        # Parse URL and re-encode password with proper URL encoding
        try:
            parsed = urlparse(self.postgres_url)
            # Re-encode password to handle special characters
            if parsed.password:
                # Reconstruct URL with properly encoded password
                safe_password = quote_plus(parsed.password)
                netloc = f"{parsed.username}:{safe_password}@{parsed.hostname}"
                if parsed.port:
                    netloc += f":{parsed.port}"

                self.postgres_url = urlunparse((
                    parsed.scheme,
                    netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
                print("[OK] DATABASE_URL password re-encoded for UTF-8 safety")
        except Exception as e:
            print(f"[WARNING] DATABASE_URL encoding issue: {e}")
            print("[INFO] Will try to use URL as-is")

        # Ensure sslmode is set for Render PostgreSQL
        if 'sslmode' not in self.postgres_url:
            separator = '&' if '?' in self.postgres_url else '?'
            self.postgres_url = f"{self.postgres_url}{separator}sslmode=require"

        print("[OK] Using PostgreSQL")

    def get_connection(self):
        """Get PostgreSQL database connection with retry logic"""
        # Force URL to be clean ASCII string to avoid psycopg2 UTF-8 decoding issues
        clean_url = str(self.postgres_url).encode('ascii', errors='ignore').decode('ascii')
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                conn = psycopg2.connect(
                    clean_url,
                    connect_timeout=30,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=10,
                    keepalives_count=5
                )
                return conn
            except psycopg2.OperationalError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    print(f"[WARNING] Connection attempt {attempt + 1} failed, retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
        
        raise last_error

    # ========== STOCK METADATA OPERATIONS ==========

    def insert_stock_metadata(self, ticker: str, name: str, sector: str, market_cap: Optional[float] = None):
        """Insert stock metadata"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO stock_metadata (ticker, name, sector, market_cap, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (ticker) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    market_cap = EXCLUDED.market_cap,
                    updated_at = EXCLUDED.updated_at
            """, (ticker, name, sector, market_cap, datetime.now()))
            conn.commit()
        finally:
            cursor.close()
            conn.close()

    def bulk_insert_metadata(self, metadata_df: pd.DataFrame):
        """Bulk insert stock metadata"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            updated_at = datetime.now()
            inserted = 0
            
            for _, row in metadata_df.iterrows():
                values = [
                    row.get('ticker'),
                    row.get('name'),
                    row.get('sector'),
                    row.get('market_cap'),
                    row.get('currency', 'USD'),
                    row.get('exchange'),
                    row.get('industry'),
                    updated_at
                ]
                
                cursor.execute("""
                    INSERT INTO stock_metadata 
                    (ticker, name, sector, market_cap, currency, exchange, industry, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker) DO UPDATE SET
                        name = EXCLUDED.name,
                        sector = EXCLUDED.sector,
                        market_cap = EXCLUDED.market_cap,
                        currency = EXCLUDED.currency,
                        exchange = EXCLUDED.exchange,
                        industry = EXCLUDED.industry,
                        updated_at = EXCLUDED.updated_at
                """, values)
                inserted += 1
            
            conn.commit()
            print(f"[OK] Inserted/updated {inserted} stock metadata records")
        finally:
            cursor.close()
            conn.close()

    def get_all_tickers(self) -> List[str]:
        """Get all tickers"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT ticker FROM stock_metadata ORDER BY ticker")
            return [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

    def get_stock_metadata(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Get stock metadata"""
        conn = self.get_connection()
        try:
            if ticker:
                query = "SELECT * FROM stock_metadata WHERE ticker = %s"
                df = pd.read_sql_query(query, conn, params=(ticker,))
            else:
                query = "SELECT * FROM stock_metadata ORDER BY ticker"
                df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    # ========== PRICE DATA OPERATIONS ==========

    def insert_price_data(self, ticker: str, price_df: pd.DataFrame):
        """Insert price data"""
        if price_df.empty:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            price_df = price_df.copy()
            price_df['ticker'] = ticker
            
            if not isinstance(price_df.index, pd.DatetimeIndex):
                price_df.index = pd.to_datetime(price_df.index)
            price_df['date'] = price_df.index.strftime('%Y-%m-%d')
            
            rows = 0
            for _, row in price_df.iterrows():
                cursor.execute("""
                    INSERT INTO daily_prices (ticker, date, close)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker, date) DO NOTHING
                """, (row['ticker'], row['date'], row.get('close')))
                rows += 1
            
            conn.commit()
            print(f"[OK] Inserted {rows} price records for {ticker}")
        finally:
            cursor.close()
            conn.close()

    def bulk_insert_prices(self, prices_dict: dict):
        """Bulk insert prices"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            total_rows = 0
            for ticker, price_df in prices_dict.items():
                if price_df.empty:
                    continue
                
                price_df = price_df.copy()
                price_df['ticker'] = ticker
                
                if not isinstance(price_df.index, pd.DatetimeIndex):
                    price_df.index = pd.to_datetime(price_df.index)
                price_df['date'] = price_df.index.strftime('%Y-%m-%d')
                
                rows = 0
                for _, row in price_df.iterrows():
                    cursor.execute("""
                        INSERT INTO daily_prices (ticker, date, close)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (ticker, date) DO NOTHING
                    """, (row['ticker'], row['date'], row.get('close')))
                    rows += 1
                
                total_rows += rows
            
            conn.commit()
            print(f"[OK] Bulk inserted {total_rows} price records")
        finally:
            cursor.close()
            conn.close()

    def get_price_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get price data"""
        conn = self.get_connection()
        try:
            query = "SELECT date, close FROM daily_prices WHERE ticker = %s"
            params = [ticker]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            if not df.empty:
                df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    def get_prices_multiple(self, tickers: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get prices for multiple tickers"""
        conn = self.get_connection()
        try:
            placeholders = ','.join(['%s'] * len(tickers))
            query = f"SELECT ticker, date, close FROM daily_prices WHERE ticker IN ({placeholders})"
            params = tickers.copy()
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            
            if not df.empty:
                df_pivot = df.pivot(index='date', columns='ticker', values='close')
                return df_pivot
            
            return pd.DataFrame()
        finally:
            conn.close()

    def get_latest_date(self, ticker: Optional[str] = None) -> Optional[datetime]:
        """Get latest date"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if ticker:
                cursor.execute("SELECT MAX(date) FROM daily_prices WHERE ticker = %s", (ticker,))
            else:
                cursor.execute("SELECT MAX(date) FROM daily_prices")
            
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
        finally:
            cursor.close()
            conn.close()

    # ========== MARKET CAP OPERATIONS ==========

    def insert_market_cap_history(self, ticker: str, market_cap_df: pd.DataFrame):
        """Insert market cap history"""
        if market_cap_df.empty:
            return
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            market_cap_df = market_cap_df.copy()
            market_cap_df['ticker'] = ticker
            
            if not isinstance(market_cap_df.index, pd.DatetimeIndex):
                market_cap_df.index = pd.to_datetime(market_cap_df.index)
            market_cap_df['date'] = market_cap_df.index.strftime('%Y-%m-%d')
            
            rows = 0
            for _, row in market_cap_df.iterrows():
                cursor.execute("""
                    INSERT INTO market_cap_history (ticker, date, market_cap)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker, date) DO NOTHING
                """, (row['ticker'], row['date'], row.get('market_cap')))
                rows += 1
            
            conn.commit()
            print(f"[OK] Inserted {rows} market cap records for {ticker}")
        finally:
            cursor.close()
            conn.close()

    def get_market_cap_history(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Get market cap history"""
        conn = self.get_connection()
        try:
            query = "SELECT date, market_cap FROM market_cap_history WHERE ticker = %s"
            params = [ticker]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['date'])
            if not df.empty:
                df.set_index('date', inplace=True)
            return df
        finally:
            conn.close()

    # ========== UTILITY METHODS ==========

    def get_data_coverage(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """Get coverage summary"""
        conn = self.get_connection()
        try:
            if ticker:
                query = "SELECT * FROM data_coverage WHERE ticker = %s"
                df = pd.read_sql_query(query, conn, params=(ticker,))
            else:
                query = "SELECT * FROM data_coverage ORDER BY completeness_pct DESC"
                df = pd.read_sql_query(query, conn)
            return df
        finally:
            conn.close()

    def check_data_quality(self, ticker: str, expected_days: int = 1260) -> dict:
        """Check data quality"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT MIN(date), MAX(date), COUNT(*)
                FROM daily_prices
                WHERE ticker = %s
            """, (ticker,))
            
            result = cursor.fetchone()
            if not result or not result[0]:
                return {
                    'ticker': ticker,
                    'status': 'NO_DATA',
                    'days_found': 0,
                    'days_expected': expected_days,
                    'completeness_pct': 0.0
                }
            
            first_date, last_date, days_found = result
            days_expected = min(expected_days, (last_date - first_date).days)
            completeness_pct = (days_found / days_expected) * 100 if days_expected > 0 else 0
            
            return {
                'ticker': ticker,
                'status': 'OK' if completeness_pct >= 90 else 'INCOMPLETE',
                'first_date': first_date,
                'last_date': last_date,
                'days_found': days_found,
                'days_expected': days_expected,
                'completeness_pct': completeness_pct
            }
        finally:
            cursor.close()
            conn.close()

    def delete_ticker(self, ticker: str):
        """Delete ticker data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM daily_prices WHERE ticker = %s", (ticker,))
            cursor.execute("DELETE FROM market_cap_history WHERE ticker = %s", (ticker,))
            cursor.execute("DELETE FROM stock_metadata WHERE ticker = %s", (ticker,))
            conn.commit()
            print(f"[OK] Deleted all data for {ticker}")
        finally:
            cursor.close()
            conn.close()

    def get_database_stats(self) -> dict:
        """Get database statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM stock_metadata")
            stats['total_stocks'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM daily_prices")
            stats['total_prices'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM daily_prices")
            result = cursor.fetchone()
            stats['date_range'] = {
                'start': result[0] if result[0] else None,
                'end': result[1] if result[1] else None
            }
            
            return stats
        finally:
            cursor.close()
            conn.close()

    def vacuum_database(self):
        """Optimize database"""
        conn = self.get_connection()
        old_isolation_level = conn.isolation_level
        conn.set_isolation_level(0)
        cursor = conn.cursor()
        try:
            cursor.execute("VACUUM ANALYZE")
            print("[OK] Database optimized")
        finally:
            cursor.close()
            conn.set_isolation_level(old_isolation_level)
            conn.close()

    # ========== SECTOR ETF OPERATIONS ==========
    
    def get_sector_returns(self) -> pd.DataFrame:
        """Get 30-day returns for sector ETFs"""
        conn = self.get_connection()
        try:
            query = """
                SELECT 
                    s1.etf_ticker,
                    s1.sector_name,
                    s1.close as current_price,
                    s1.date as latest_date,
                    s2.close as price_30d_ago,
                    ROUND(((s1.close - s2.close) / s2.close * 100)::numeric, 2) as return_30d_pct
                FROM sector_etf_prices s1
                LEFT JOIN sector_etf_prices s2 ON s1.etf_ticker = s2.etf_ticker 
                    AND s2.date = (
                        SELECT MAX(date) FROM sector_etf_prices 
                        WHERE etf_ticker = s1.etf_ticker 
                        AND date <= s1.date - INTERVAL '30 days'
                    )
                WHERE s1.date = (SELECT MAX(date) FROM sector_etf_prices WHERE etf_ticker = s1.etf_ticker)
                ORDER BY return_30d_pct DESC
            """
            
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            print(f"[WARNING] Error fetching sector returns: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_sector_etf_history(self, etf_ticker: str, days: int = 30) -> pd.DataFrame:
        """Get sector ETF history"""
        conn = self.get_connection()
        try:
            query = f"""
                SELECT date, close
                FROM sector_etf_prices
                WHERE etf_ticker = %s
                AND date >= CURRENT_DATE - INTERVAL '{days} days'
                ORDER BY date
            """
            
            df = pd.read_sql_query(query, conn, params=(etf_ticker,))
            return df
        finally:
            conn.close()
    
    def insert_sector_etf_price(self, etf_ticker: str, sector_name: str, date, close: float):
        """Insert sector ETF price"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO sector_etf_prices (etf_ticker, sector_name, date, close)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (etf_ticker, date)
                DO UPDATE SET close = EXCLUDED.close, sector_name = EXCLUDED.sector_name
            """, (etf_ticker, sector_name, date, close))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
