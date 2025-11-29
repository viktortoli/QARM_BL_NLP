"""Data loader for S&P 500 constituents from database"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional
import warnings
import sys
from pathlib import Path

_app_dir = Path(__file__).parent.parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    class st:
        @staticmethod
        def cache_resource(func):
            return func
        @staticmethod
        def error(msg):
            print(f"ERROR: {msg}")

warnings.filterwarnings('ignore')

from database.db_manager import DatabaseManager


class DataLoader:
    """Load S&P 500 stock data from database"""

    def __init__(self):
        """Initialize database connection"""
        self._sp500_cache = None

        try:
            self._db = DatabaseManager()
            print("[OK] Database connection established")
        except Exception as e:
            error_msg = f"[ERROR] Database connection failed: {e}\nPlease run: python scripts/fetch_prices.py"
            print(error_msg)
            st.error("Database not found. Please populate database first.")
            raise RuntimeError("Database required for production mode")

        self._load_sp500_from_local_file()

    def _load_sp500_from_local_file(self):
        """Load S&P 500 constituents from Parquet file"""
        parquet_path = 'app/imports/sp500_constituents.parquet'

        try:
            import os
            if not os.path.exists(parquet_path):
                error_msg = f"Parquet file not found at: {os.path.abspath(parquet_path)}"
                print(f"[ERROR] {error_msg}")
                print(f"[INFO] Current working directory: {os.getcwd()}")
                print(f"[INFO] Please ensure sp500_constituents.parquet is in app/imports/")
                st.error(f"Required file missing: {parquet_path}")
                self._sp500_cache = pd.DataFrame(columns=['ticker', 'name', 'sector'])
                return

            print("[INFO] Loading from Parquet...")
            df = pd.read_parquet(parquet_path)
            print(f"[OK] Loaded {len(df)} stocks from Parquet in milliseconds!")

            df.columns = df.columns.str.lower()

            required_cols = ['ticker', 'name', 'sector']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                print(f"[ERROR] {error_msg}")
                print(f"[INFO] Available columns: {list(df.columns)}")
                st.error(f"Invalid Parquet file structure: {error_msg}")
                self._sp500_cache = pd.DataFrame(columns=['ticker', 'name', 'sector'])
                return

            df['ticker'] = df['ticker'].str.strip()

            initial_count = len(df)
            df = df.dropna(subset=['ticker'])

            googl_count = (df['ticker'] == 'GOOGL').sum()
            if googl_count > 0:
                df = df[df['ticker'] != 'GOOGL']
                print(f"[INFO] Filtered out GOOGL (using GOOG only)")
            if len(df) < initial_count:
                print(f"[WARNING] Removed {initial_count - len(df)} rows with missing tickers")

            df = df.sort_values('ticker').reset_index(drop=True)

            self._sp500_cache = df[['ticker', 'name', 'sector']].copy()

            print(f"[SUCCESS] Initialized with {len(self._sp500_cache)} stocks")
            
        except FileNotFoundError as e:
            error_msg = f"Parquet file not found: {parquet_path}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] {e}")
            st.error(f"Required file missing: {parquet_path}\nPlease ensure the Parquet file exists.")
            self._sp500_cache = pd.DataFrame(columns=['ticker', 'name', 'sector'])
            
        except Exception as e:
            error_msg = f"Failed to load S&P 500 from Parquet: {type(e).__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            print(traceback.format_exc())
            st.error(f"Cannot load S&P 500 data: {e}\nPlease check the Parquet file.")
            self._sp500_cache = pd.DataFrame(columns=['ticker', 'name', 'sector'])

    def get_sp500_constituents(self) -> pd.DataFrame:
        """Get S&P 500 constituents with prices and market caps"""
        if self._sp500_cache is None or len(self._sp500_cache) == 0:
            error_msg = "Stock cache is empty"
            print(f"[ERROR] {error_msg}")
            st.error("Stock data not loaded. Please check Parquet file.")
            return pd.DataFrame()

        print(f"[INFO] Loading current prices and market caps from database...")

        try:
            metadata_df = self._db.get_stock_metadata()

            if metadata_df.empty:
                error_msg = "No metadata in database - market caps required"
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/update_database.py market-caps")
                return pd.DataFrame()

            df = self._sp500_cache.merge(
                metadata_df[['ticker', 'market_cap']],
                on='ticker',
                how='left'
            )

            try:
                conn = self._db.get_connection()

                latest_prices_query = """
                    SELECT DISTINCT ON (ticker) ticker, close as price
                    FROM daily_prices
                    ORDER BY ticker, date DESC
                """

                latest_prices_df = pd.read_sql_query(latest_prices_query, conn)
                conn.close()

                df = df.merge(
                    latest_prices_df[['ticker', 'price']],
                    on='ticker',
                    how='left'
                )

                if df['price'].isna().all():
                    error_msg = "No price data in database"
                    print(f"[ERROR] {error_msg}")
                    st.error(f"{error_msg}\nPlease run: python scripts/fetch_prices.py")
                    return pd.DataFrame()

                prices_found = (~df['price'].isna()).sum()
                print(f"[OK] Loaded {len(df)} stocks with {prices_found} prices from database")

            except Exception as e:
                error_msg = f"Failed to load prices: {e}"
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/fetch_prices.py")
                return pd.DataFrame()

            df['market_cap'] = df['market_cap'].fillna(0.0)
            df['price'] = df['price'].fillna(0.0)

            df = df.sort_values('market_cap', ascending=False).reset_index(drop=True)

            return df

        except Exception as e:
            error_msg = f"Database error: {e}"
            print(f"[ERROR] {error_msg}")
            st.error(f"{error_msg}\nEnsure database is populated.")
            return pd.DataFrame()

    def get_historical_prices(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        periods: int = 252
    ) -> pd.DataFrame:
        """Get historical price data from database"""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=int(periods * 1.5))

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"[INFO] Loading {len(tickers)} tickers from database...")
        print(f"   Period: {start_str} to {end_str}")

        try:
            prices = self._db.get_prices_multiple(tickers, start_str, end_str)

            if prices.empty:
                error_msg = "No data found in database for the specified period."
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/fetch_prices.py")
                raise ValueError(error_msg)

            if len(prices) < 50:
                error_msg = f"Insufficient data: only {len(prices)} days found (need at least 50)"
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/fetch_prices.py")
                raise ValueError(error_msg)

            missing_tickers = [t for t in tickers if t not in prices.columns]
            if missing_tickers:
                error_msg = f"Database missing data for {len(missing_tickers)} tickers: {', '.join(missing_tickers[:5])}"
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/fetch_prices.py")
                raise ValueError(error_msg)

            prices = prices.ffill().bfill()

            print(f"[OK] Loaded {len(prices)} days from database (ultra-fast!)")
            return prices

        except Exception as e:
            error_msg = f"Database query failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            st.error(f"{error_msg}\nEnsure database is populated with: python scripts/fetch_prices.py")
            raise

    def get_market_caps(self, tickers: List[str]) -> np.ndarray:
        """Get market capitalizations from database"""
        print(f"[INFO] Loading market caps from database...")

        try:
            metadata_df = self._db.get_stock_metadata()

            if metadata_df.empty:
                error_msg = "No metadata found in database"
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/update_database.py market-caps")
                raise ValueError(error_msg)

            # Get market caps from database
            db_market_caps = {}
            for _, row in metadata_df.iterrows():
                if row['ticker'] in tickers and row['market_cap']:
                    db_market_caps[row['ticker']] = row['market_cap']

            # Check if we have all tickers
            missing = [t for t in tickers if t not in db_market_caps]
            if missing:
                error_msg = f"Missing market caps for {len(missing)} tickers: {', '.join(missing[:5])}"
                print(f"[ERROR] {error_msg}")
                st.error(f"{error_msg}\nPlease run: python scripts/update_database.py market-caps")
                raise ValueError(error_msg)

            market_caps = [db_market_caps[t] for t in tickers]
            print(f"[OK] Loaded {len(market_caps)} market caps from database")
            return np.array(market_caps)

        except Exception as e:
            error_msg = f"Database query failed: {str(e)}"
            print(f"[ERROR] {error_msg}")
            st.error(f"{error_msg}\nEnsure database has market caps: python scripts/update_database.py market-caps")
            raise

    def get_sectors(self, tickers: List[str]) -> List[str]:
        """Get sectors for tickers"""
        if self._sp500_cache is None:
            return ['Unknown'] * len(tickers)

        sectors = []
        for ticker in tickers:
            sector_row = self._sp500_cache[self._sp500_cache['ticker'] == ticker]
            if len(sector_row) > 0:
                sectors.append(sector_row.iloc[0]['sector'])
            else:
                sectors.append('Unknown')

        return sectors

    def get_available_sectors(self) -> List[str]:
        """Get available sectors"""
        if self._sp500_cache is None or len(self._sp500_cache) == 0:
            return []

        return sorted(self._sp500_cache['sector'].unique().tolist())

    def filter_by_sector(self, sector: str) -> List[str]:
        """Get tickers in sector"""
        if self._sp500_cache is None:
            return []

        filtered = self._sp500_cache[self._sp500_cache['sector'] == sector]
        return filtered['ticker'].tolist()

    def search_stocks(self, query: str) -> pd.DataFrame:
        """Search stocks by ticker or name"""
        df = self.get_sp500_constituents()

        if df.empty or not query:
            return df

        query = query.lower()
        mask = (
            df['ticker'].str.lower().str.contains(query, na=False) |
            df['name'].str.lower().str.contains(query, na=False)
        )

        return df[mask]


# Singleton instance with Streamlit resource caching
@st.cache_resource
def get_data_loader() -> DataLoader:
    """Get singleton DataLoader instance"""
    return DataLoader()
