"""
Fetch Fama-French factor returns from Kenneth French Data Library
"""
import os
import sys
from datetime import datetime
import warnings
import psycopg2
import pandas as pd
import requests
from io import BytesIO
import zipfile

warnings.filterwarnings('ignore')


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

    # Fix postgres:// to postgresql://
    if postgres_url.startswith('postgres://'):
        postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)

    return psycopg2.connect(postgres_url)


def ensure_factor_table(conn):
    """Create factor_returns table"""
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
    """Create factor_loadings table"""
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
    """
    Fetch Fama-French 5-Factor daily data.
    Returns DataFrame with: date, Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    log("Fetching Fama-French 5-Factor daily data...")

    # Fama-French 5-Factor Daily Data
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                # Read CSV, skip header rows
                df = pd.read_csv(f, skiprows=3)

        df = df[df.iloc[:, 0].astype(str).str.match(r'^\d{8}$')]

        df.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0

        df = df.dropna()
        df = df.tail(1260)

        log(f"Fetched {len(df)} daily factor observations")
        log(f"Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    except Exception as e:
        log(f"Error fetching Fama-French factors: {e}")
        raise


def update_factor_database(conn, factor_data):
    """Update database with factor returns"""
    log(f"Updating factor returns table...")

    cursor = conn.cursor()
    inserted = 0

    factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

    for _, row in factor_data.iterrows():
        date = row['date'].date()

        for factor_name in factor_names:
            factor_return = row[factor_name]

            try:
                cursor.execute("""
                    INSERT INTO factor_returns (factor_name, date, return)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (factor_name, date)
                    DO UPDATE SET return = EXCLUDED.return
                """, (factor_name, date, factor_return))
                inserted += 1

            except Exception as e:
                log(f"  Error updating {factor_name} on {date}: {e}")
                continue

    conn.commit()
    cursor.close()

    log(f"Factor returns updated: {inserted} records")


def main():
    """Main execution function"""
    log("=" * 60)
    log("Starting factor fetch job")
    log("=" * 60)

    try:
        conn = get_db_connection()
        log("Connected to PostgreSQL")

        # Ensure tables exist
        ensure_factor_table(conn)
        ensure_factor_loadings_table(conn)

        # Fetch Fama-French factors
        factor_data = fetch_fama_french_factors()

        if not factor_data.empty:
            update_factor_database(conn, factor_data)
        else:
            log("No factor data to update")

        conn.close()

        log("=" * 60)
        log("Factor fetch job completed successfully")
        log("=" * 60)

        sys.exit(0)

    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
