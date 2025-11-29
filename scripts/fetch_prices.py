"""
Fetch prices from yfinance, Fama-French factors, and calculate factor loadings
"""

import os
import sys
from datetime import datetime, timedelta
import warnings
import psycopg2
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import zipfile

warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

SECTOR_ETFS = {
    'SPY': 'S&P 500',
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}


def log(message):
    """Print timestamped log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_db_connection():
    """Get PostgreSQL connection"""
    postgres_url = os.getenv('DATABASE_URL')

    if not postgres_url:
        log("ERROR: DATABASE_URL not set!")
        sys.exit(1)

    if postgres_url.startswith('postgres://'):
        postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)

    return psycopg2.connect(postgres_url)


def ensure_sector_etf_table(conn):
    """Create sector_etf_prices table if it doesn't exist"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sector_etf_prices (
                etf_ticker VARCHAR(10) NOT NULL,
                sector_name VARCHAR(100) NOT NULL,
                date DATE NOT NULL,
                close DECIMAL(12, 4) NOT NULL,
                PRIMARY KEY (etf_ticker, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_etf_date ON sector_etf_prices(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sector_etf_ticker_date ON sector_etf_prices(etf_ticker, date DESC)")
        conn.commit()
        log("Sector ETF table ready")
    except Exception as e:
        log(f"Note: {e}")
        conn.rollback()
    finally:
        cursor.close()


def get_all_tickers(conn):
    """Get all tickers from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tickers


def fetch_latest_prices(tickers):
    """Fetch latest prices for all tickers using yfinance"""
    log(f"Fetching latest prices for {len(tickers)} tickers...")

    all_data = []
    failed = []

    batch_size = 50
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        log(f"  Batch {batch_num}/{total_batches} ({len(batch)} tickers)")

        try:
            data = yf.download(
                batch,
                period="5d",
                progress=False,
                group_by='ticker',
                auto_adjust=True,
                threads=True
            )

            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]

                    if ticker_data.empty:
                        failed.append(ticker)
                        continue

                    latest_date = ticker_data.index[-1].date()
                    latest_close = ticker_data['Close'].iloc[-1]

                    all_data.append({
                        'ticker': ticker,
                        'date': latest_date,
                        'close': float(latest_close)
                    })

                except Exception:
                    failed.append(ticker)
                    continue

        except Exception as e:
            log(f"  Error fetching batch {batch_num}: {e}")
            failed.extend(batch)

    log(f"Successfully fetched {len(all_data)} prices")
    if failed:
        log(f"Failed: {len(failed)} tickers")

    return all_data


def update_database(conn, price_data):
    """Update database with new price data"""
    log(f"Updating database with {len(price_data)} records...")

    cursor = conn.cursor()
    inserted = 0

    for data in price_data:
        try:
            cursor.execute("""
                INSERT INTO daily_prices (ticker, date, close)
                VALUES (%s, %s, %s)
                ON CONFLICT (ticker, date)
                DO UPDATE SET close = EXCLUDED.close
            """, (data['ticker'], data['date'], data['close']))
            inserted += 1

        except Exception as e:
            log(f"  Error updating {data['ticker']}: {e}")
            continue

    conn.commit()
    cursor.close()

    log(f"Database updated: {inserted} price records")


def fetch_sector_etf_prices():
    """Fetch sector ETF prices (45 days) and SPY benchmark (5 years)"""
    log(f"Fetching {len(SECTOR_ETFS)} ETFs (sectors + benchmarks)...")
    
    all_data = []
    
    # Fetch SPY with 5 years of data for portfolio comparison
    try:
        log("  Fetching SPY (5 years for portfolio comparison)...")
        spy_data = yf.download(
            'SPY',
            period="5y",
            progress=False,
            auto_adjust=True
        )
        
        if not spy_data.empty:
            for idx in range(len(spy_data)):
                date = spy_data.index[idx].date()
                close = float(spy_data['Close'].iloc[idx])
                all_data.append({
                    'etf_ticker': 'SPY',
                    'sector_name': 'S&P 500',
                    'date': date,
                    'close': close
                })
            log(f"  SPY: {len(spy_data)} days fetched")
        else:
            log("  Warning: No data for SPY")
    except Exception as e:
        log(f"  Error fetching SPY: {e}")
    
    # Fetch sector ETFs with 45 days of data
    sector_tickers = [t for t in SECTOR_ETFS.keys() if t != 'SPY']
    
    try:
        log(f"  Fetching {len(sector_tickers)} sector ETFs (45 days)...")
        data = yf.download(
            sector_tickers,
            period="45d",
            progress=False,
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
        
        for etf_ticker in sector_tickers:
            sector_name = SECTOR_ETFS[etf_ticker]
            try:
                if len(sector_tickers) == 1:
                    etf_data = data
                else:
                    etf_data = data[etf_ticker]
                
                if etf_data.empty:
                    log(f"  Warning: No data for {etf_ticker}")
                    continue
                
                # Get all available dates
                for idx in range(len(etf_data)):
                    date = etf_data.index[idx].date()
                    close = float(etf_data['Close'].iloc[idx])
                    
                    all_data.append({
                        'etf_ticker': etf_ticker,
                        'sector_name': sector_name,
                        'date': date,
                        'close': close
                    })
                    
            except Exception as e:
                log(f"  Error processing {etf_ticker}: {e}")
                continue
                
    except Exception as e:
        log(f"Error fetching sector ETFs: {e}")
    
    log(f"Fetched {len(all_data)} sector ETF price records")
    return all_data


def update_sector_etf_database(conn, etf_data):
    """Update database with sector ETF price data"""
    if not etf_data:
        log("No sector ETF data to update")
        return
        
    log(f"Updating sector ETF table with {len(etf_data)} records...")
    
    cursor = conn.cursor()
    inserted = 0
    
    for data in etf_data:
        try:
            cursor.execute("""
                INSERT INTO sector_etf_prices (etf_ticker, sector_name, date, close)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (etf_ticker, date)
                DO UPDATE SET close = EXCLUDED.close, sector_name = EXCLUDED.sector_name
            """, (data['etf_ticker'], data['sector_name'], data['date'], data['close']))
            inserted += 1
            
        except Exception as e:
            log(f"  Error updating {data['etf_ticker']}: {e}")
            continue
    
    conn.commit()
    cursor.close()
    
    log(f"Sector ETF table updated: {inserted} records")


def ensure_factor_table(conn):
    """Create factor_returns table if it doesn't exist"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_returns (
                factor_name VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                return DECIMAL(12, 6) NOT NULL,
                PRIMARY KEY (factor_name, date)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_date ON factor_returns(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_factor_name_date ON factor_returns(factor_name, date DESC)")
        conn.commit()
        log("Factor returns table ready")
    except Exception as e:
        log(f"Note: {e}")
        conn.rollback()
    finally:
        cursor.close()


def ensure_factor_loadings_table(conn):
    """Create factor_loadings table if it doesn't exist"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS factor_loadings (
                ticker VARCHAR(10) NOT NULL,
                factor_name VARCHAR(50) NOT NULL,
                loading DECIMAL(12, 6) NOT NULL,
                r_squared DECIMAL(12, 6),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, factor_name)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_loadings_ticker ON factor_loadings(ticker)")
        conn.commit()
        log("Factor loadings table ready")
    except Exception as e:
        log(f"Note: {e}")
        conn.rollback()
    finally:
        cursor.close()


def fetch_fama_french_factors():
    """Fetch Fama-French 5-Factor Model daily data from Kenneth French Data Library"""
    log("Fetching Fama-French 5-Factor daily data...")

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Extract CSV from ZIP
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, skiprows=3)

        df = df[df.iloc[:, 0].astype(str).str.match(r'^\d{8}$')]
        df.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0

        df = df.dropna()
        df = df.tail(1260)  # Last 5 years

        log(f"  Fetched {len(df)} daily factor observations ({df['date'].min()} to {df['date'].max()})")
        return df

    except Exception as e:
        log(f"  Error fetching factors: {e}")
        return pd.DataFrame()


def update_factor_database(conn, factor_data):
    """Update database with factor returns (optimized batch insert)"""
    if factor_data.empty:
        log("  No factor data to update")
        return

    log(f"  Updating factor returns table...")

    cursor = conn.cursor()

    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

    # Prepare batch insert data
    log(f"  Preparing batch data for {len(factor_data)} days × {len(factor_names)} factors...")
    batch_data = []
    for _, row in factor_data.iterrows():
        date = row['date'].date()
        for factor_name in factor_names:
            factor_return = row[factor_name]
            batch_data.append((factor_name, date, factor_return))

    log(f"  Prepared {len(batch_data)} records, starting batch insert...")

    # Batch insert using executemany
    try:
        import time
        start_time = time.time()

        cursor.executemany("""
            INSERT INTO factor_returns (factor_name, date, return)
            VALUES (%s, %s, %s)
            ON CONFLICT (factor_name, date)
            DO UPDATE SET return = EXCLUDED.return
        """, batch_data)

        log(f"  Batch insert completed, committing...")
        conn.commit()

        elapsed = time.time() - start_time
        inserted = len(batch_data)
        log(f"  Factor returns updated: {inserted} records in {elapsed:.1f}s")

    except Exception as e:
        log(f"  Error batch updating factors: {e}")
        conn.rollback()
    finally:
        cursor.close()


def calculate_factor_loadings(conn, tickers):
    """Calculate and update factor loadings for all tickers"""
    import time
    total_start = time.time()

    log("Calculating factor loadings...")
    log(f"  Total tickers to process: {len(tickers)}")

    # Load factor returns (last 3 years for regression)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=3 * 365)

    cursor = conn.cursor()

    # Check if factor returns exist
    cursor.execute("SELECT COUNT(*) FROM factor_returns WHERE date >= %s", (start_date,))
    factor_count = cursor.fetchone()[0]

    if factor_count == 0:
        log("  No factor returns found. Skipping factor loadings.")
        cursor.close()
        return

    log(f"  Loading factor returns from {start_date} to {end_date} ({factor_count} records)...")

    # Load factor returns
    cursor.execute("""
        SELECT date, factor_name, return
        FROM factor_returns
        WHERE date >= %s AND date <= %s
        ORDER BY date, factor_name
    """, (start_date, end_date))

    factor_rows = cursor.fetchall()

    factor_data = {}
    for date, factor_name, return_val in factor_rows:
        if date not in factor_data:
            factor_data[date] = {}
        factor_data[date][factor_name] = float(return_val)

    factor_df = pd.DataFrame.from_dict(factor_data, orient='index')
    factor_df.index = pd.to_datetime(factor_df.index)
    factor_df.sort_index(inplace=True)

    log(f"  Loaded {len(factor_df)} days of factor data")

    # Fetch ALL ticker prices in one query (optimization)
    log(f"  Loading all ticker prices from {start_date} to {end_date}...")
    cursor.execute("""
        SELECT ticker, date, close
        FROM daily_prices
        WHERE date >= %s AND date <= %s
        ORDER BY ticker, date
    """, (start_date, end_date))

    all_price_rows = cursor.fetchall()
    log(f"  Loaded {len(all_price_rows)} price records for all tickers")

    ticker_prices = {}
    for ticker, date, close in all_price_rows:
        if ticker not in ticker_prices:
            ticker_prices[ticker] = []
        ticker_prices[ticker].append((date, close))

    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    successful = 0
    failed = 0
    batch_loadings = []

    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            log(f"  Progress: {i + 1}/{len(tickers)} ({(i + 1)/len(tickers)*100:.1f}%)")

        price_rows = ticker_prices.get(ticker, [])

        if not price_rows or len(price_rows) < 30:
            failed += 1
            continue

        prices = pd.DataFrame(price_rows, columns=['date', 'close'])
        prices['close'] = prices['close'].astype(float)  # Convert Decimal to float
        prices.set_index('date', inplace=True)
        returns = prices['close'].pct_change().dropna()

        aligned_data = pd.DataFrame({
            'ticker_return': returns,
            **{f: factor_df[f] for f in factor_names if f in factor_df.columns}
        }).dropna()

        if 'RF' in factor_df.columns:
            aligned_data['RF'] = factor_df['RF']
            aligned_data = aligned_data.dropna()

        if len(aligned_data) < 30:
            failed += 1
            continue

        y = aligned_data['ticker_return'] - aligned_data.get('RF', 0)

        X = aligned_data[factor_names].values
        X = np.column_stack([np.ones(len(X)), X])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            batch_loadings.append((ticker, 'alpha', float(beta[0]), float(r_squared)))
            for j, factor_name in enumerate(factor_names):
                batch_loadings.append((ticker, factor_name, float(beta[j + 1]), float(r_squared)))

            successful += 1

        except np.linalg.LinAlgError:
            failed += 1
            continue

    # Batch insert all loadings
    if batch_loadings:
        log(f"  Starting batch insert of {len(batch_loadings)} loading records...")
        insert_start = time.time()

        cursor.executemany("""
            INSERT INTO factor_loadings (ticker, factor_name, loading, r_squared)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (ticker, factor_name)
            DO UPDATE SET
                loading = EXCLUDED.loading,
                r_squared = EXCLUDED.r_squared,
                last_updated = CURRENT_TIMESTAMP
        """, batch_loadings)

        log(f"  Batch insert completed in {time.time() - insert_start:.1f}s, committing...")

    conn.commit()
    cursor.close()

    total_time = time.time() - total_start
    log(f"  Factor loadings completed: {successful} successful, {failed} failed in {total_time:.1f}s")


def main():
    """Main execution function"""
    log("=" * 60)
    log("Starting price fetch job")
    log("=" * 60)

    try:
        conn = get_db_connection()
        log("Connected to PostgreSQL")

        # Fetch stock prices
        tickers = get_all_tickers(conn)
        log(f"Found {len(tickers)} tickers in database")

        price_data = fetch_latest_prices(tickers)

        if price_data:
            update_database(conn, price_data)
        else:
            log("No price data to update")

        # Fetch sector ETF prices
        log("-" * 40)
        log("Fetching sector ETF data...")

        # Ensure table exists
        ensure_sector_etf_table(conn)

        sector_etf_data = fetch_sector_etf_prices()

        if sector_etf_data:
            update_sector_etf_database(conn, sector_etf_data)

        # Fetch Fama-French factors
        log("-" * 40)
        log("Fetching Fama-French factors...")

        # Ensure tables exist
        ensure_factor_table(conn)
        ensure_factor_loadings_table(conn)

        factor_data = fetch_fama_french_factors()

        if not factor_data.empty:
            update_factor_database(conn, factor_data)

        # Calculate factor loadings
        log("-" * 40)
        log("Updating factor loadings...")
        calculate_factor_loadings(conn, tickers)

        conn.close()

        log("=" * 60)
        log("Price fetch job completed successfully")
        log("=" * 60)

        sys.exit(0)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
