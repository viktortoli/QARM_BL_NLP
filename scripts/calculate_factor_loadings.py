"""
Calculate factor loadings for all tickers using regression
"""
import os
import sys
from datetime import datetime, timedelta
import warnings
import psycopg2
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def get_all_tickers(conn):
    """Get all tickers from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker")
    tickers = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tickers


def load_stock_returns(conn, ticker, start_date, end_date):
    """Load stock returns from database"""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, close
        FROM daily_prices
        WHERE ticker = %s AND date >= %s AND date <= %s
        ORDER BY date
    """, (ticker, start_date, end_date))

    rows = cursor.fetchall()
    cursor.close()

    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows, columns=['date', 'close'])
    df.set_index('date', inplace=True)

    returns = df['close'].pct_change().dropna()

    return returns


def load_factor_returns(conn, start_date, end_date):
    """Load factor returns from database"""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT date, factor_name, return
        FROM factor_returns
        WHERE date >= %s AND date <= %s
        ORDER BY date, factor_name
    """, (start_date, end_date))

    rows = cursor.fetchall()
    cursor.close()

    if not rows:
        return pd.DataFrame()

    data = {}
    for date, factor_name, return_val in rows:
        if date not in data:
            data[date] = {}
        data[date][factor_name] = float(return_val)

    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    return df


def calculate_factor_loadings(ticker_returns, factor_returns):
    """
    Calculate factor loadings using regression

    Returns:
        Dict with factor loadings and R-squared
    """
    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    aligned_data = pd.DataFrame({
        'ticker_return': ticker_returns,
        **{f: factor_returns[f] for f in factor_names if f in factor_returns.columns}
    }).dropna()

    # Also need RF for excess returns
    if 'RF' in factor_returns.columns:
        aligned_data['RF'] = factor_returns['RF']
        aligned_data = aligned_data.dropna()

    if len(aligned_data) < 30:
        return {}

    y = aligned_data['ticker_return'] - aligned_data.get('RF', 0)

    X = aligned_data[factor_names].values
    X = np.column_stack([np.ones(len(X)), X])

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        loadings = {
            'alpha': beta[0],
            **{factor_names[i]: beta[i + 1] for i in range(len(factor_names))},
            'R-squared': r_squared
        }

        return loadings

    except np.linalg.LinAlgError:
        return {}


def save_loadings(conn, ticker, loadings):
    """Save factor loadings to database"""
    if not loadings:
        return

    cursor = conn.cursor()

    try:
        r_squared = loadings.get('R-squared', None)

        for factor_name, loading_value in loadings.items():
            if factor_name == 'R-squared':
                continue

            cursor.execute("""
                INSERT INTO factor_loadings (ticker, factor_name, loading, r_squared)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (ticker, factor_name)
                DO UPDATE SET
                    loading = EXCLUDED.loading,
                    r_squared = EXCLUDED.r_squared,
                    last_updated = CURRENT_TIMESTAMP
            """, (ticker, factor_name, loading_value, r_squared))

        conn.commit()

    except Exception as e:
        log(f"  Error saving loadings for {ticker}: {e}")
        conn.rollback()

    finally:
        cursor.close()


def main():
    """Main execution"""
    log("=" * 60)
    log("Starting factor loadings calculation")
    log("=" * 60)

    try:
        conn = get_db_connection()
        log("Connected to PostgreSQL")

        # Get tickers
        tickers = get_all_tickers(conn)
        log(f"Found {len(tickers)} tickers")

        # Load factor returns (last 3 years for regression)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=3 * 365)

        log(f"Loading factor returns from {start_date} to {end_date}")
        factor_returns = load_factor_returns(conn, start_date, end_date)

        if factor_returns.empty:
            log("ERROR: No factor returns found. Run fetch_factors.py first!")
            sys.exit(1)

        log(f"Loaded {len(factor_returns)} days of factor data")

        # Calculate loadings for each ticker
        successful = 0
        failed = 0

        for i, ticker in enumerate(tickers, 1):
            if i % 10 == 0:
                log(f"Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")

            # Load stock returns
            ticker_returns = load_stock_returns(conn, ticker, start_date, end_date)

            if ticker_returns.empty:
                failed += 1
                continue

            # Calculate loadings
            loadings = calculate_factor_loadings(ticker_returns, factor_returns)

            if not loadings:
                failed += 1
                continue

            # Save to database
            save_loadings(conn, ticker, loadings)
            successful += 1

            # Log for first few tickers
            if i <= 5:
                log(f"  {ticker}: Beta={loadings.get('Mkt-RF', 0):.2f}, "
                    f"SMB={loadings.get('SMB', 0):.2f}, "
                    f"HML={loadings.get('HML', 0):.2f}, "
                    f"R²={loadings.get('R-squared', 0):.2f}")

        conn.close()

        log("=" * 60)
        log(f"Factor loadings calculation completed")
        log(f"  Successful: {successful}")
        log(f"  Failed: {failed}")
        log("=" * 60)

        sys.exit(0)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
