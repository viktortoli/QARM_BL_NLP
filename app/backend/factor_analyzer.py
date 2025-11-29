"""Factor analysis using Fama-French factors"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import psycopg2
import os


class FactorAnalyzer:
    """Fama-French factor analyzer"""

    def __init__(self):
        """Initialize analyzer"""
        self.factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    def get_db_connection(self):
        """Get database connection"""
        postgres_url = os.getenv('DATABASE_URL')
        if not postgres_url:
            raise ValueError("DATABASE_URL environment variable not set")
        if postgres_url.startswith('postgres://'):
            postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)
        return psycopg2.connect(postgres_url)

    def load_factor_returns(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Load factor returns from database"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT date, factor_name, return
                FROM factor_returns
                WHERE date >= %s AND date <= %s
                ORDER BY date, factor_name
            """, (start_date, end_date))

            rows = cursor.fetchall()

            if not rows:
                return pd.DataFrame()

            data = {'date': [], **{f: [] for f in self.factor_names}}

            current_date = None
            current_factors = {}

            for date, factor_name, return_val in rows:
                if current_date is None:
                    current_date = date

                if date != current_date:
                    # Save previous date's data
                    data['date'].append(current_date)
                    for f in self.factor_names:
                        data[f].append(current_factors.get(f, np.nan))

                    current_date = date
                    current_factors = {}

                current_factors[factor_name] = float(return_val)

            # Save last date
            if current_date:
                data['date'].append(current_date)
                for f in self.factor_names:
                    data[f].append(current_factors.get(f, np.nan))

            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)

            return df

        finally:
            cursor.close()
            conn.close()

    def calculate_factor_loadings(
        self,
        ticker: str,
        ticker_returns: pd.Series,
        factor_returns: pd.DataFrame,
        risk_free_rate: pd.Series
    ) -> Dict[str, float]:
        """Calculate factor loadings using regression"""

        aligned_data = pd.DataFrame({
            'ticker_return': ticker_returns,
            'rf': risk_free_rate,
            **{f: factor_returns[f] for f in self.factor_names if f in factor_returns.columns}
        }).dropna()

        if len(aligned_data) < 30:
            print(f"[WARNING] Insufficient data for {ticker} factor regression ({len(aligned_data)} obs)")
            return {}

        y = aligned_data['ticker_return'] - aligned_data['rf']

        X = aligned_data[self.factor_names].values
        X = np.column_stack([np.ones(len(X)), X])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            loadings = {
                'alpha': beta[0],
                **{self.factor_names[i]: beta[i + 1] for i in range(len(self.factor_names))},
                'R-squared': r_squared,
                'observations': len(aligned_data)
            }

            return loadings

        except np.linalg.LinAlgError as e:
            print(f"[ERROR] Regression failed for {ticker}: {e}")
            return {}

    def calculate_portfolio_exposure(
        self,
        weights: Dict[str, float],
        factor_loadings: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate portfolio-level factor exposures"""
        portfolio_exposure = {f: 0.0 for f in self.factor_names}
        portfolio_exposure['alpha'] = 0.0

        for ticker, weight in weights.items():
            if ticker not in factor_loadings:
                continue

            loadings = factor_loadings[ticker]

            portfolio_exposure['alpha'] += weight * loadings.get('alpha', 0.0)

            for factor in self.factor_names:
                portfolio_exposure[factor] += weight * loadings.get(factor, 0.0)

        return portfolio_exposure

    def save_loadings_to_db(self, ticker: str, loadings: Dict[str, float]):
        """Save loadings to database"""
        if not loadings:
            return

        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            r_squared = loadings.get('R-squared', None)

            for factor in self.factor_names + ['alpha']:
                if factor in loadings:
                    loading_value = loadings[factor]

                    cursor.execute("""
                        INSERT INTO factor_loadings (ticker, factor_name, loading, r_squared)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (ticker, factor_name)
                        DO UPDATE SET
                            loading = EXCLUDED.loading,
                            r_squared = EXCLUDED.r_squared,
                            last_updated = CURRENT_TIMESTAMP
                    """, (ticker, factor, loading_value, r_squared))

            conn.commit()

        except Exception as e:
            print(f"[ERROR] Failed to save loadings for {ticker}: {e}")
            conn.rollback()

        finally:
            cursor.close()
            conn.close()

    def load_loadings_from_db(self, tickers: List[str]) -> Dict[str, Dict[str, float]]:
        """Load factor loadings from database"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            placeholders = ','.join(['%s'] * len(tickers))
            cursor.execute(f"""
                SELECT ticker, factor_name, loading, r_squared
                FROM factor_loadings
                WHERE ticker IN ({placeholders})
            """, tickers)

            rows = cursor.fetchall()

            loadings_dict = {}
            for ticker, factor_name, loading, r_squared in rows:
                if ticker not in loadings_dict:
                    loadings_dict[ticker] = {'R-squared': r_squared}

                loadings_dict[ticker][factor_name] = float(loading)

            return loadings_dict

        finally:
            cursor.close()
            conn.close()

    def get_factor_interpretation(self, factor_name: str, loading: float) -> str:
        """Get human-readable factor interpretation"""
        interpretations = {
            'Mkt-RF': {
                'positive': 'moves with the market',
                'negative': 'moves against the market (hedging)',
                'neutral': 'market-neutral'
            },
            'SMB': {
                'positive': 'tilted toward small-cap stocks',
                'negative': 'tilted toward large-cap stocks',
                'neutral': 'size-neutral'
            },
            'HML': {
                'positive': 'tilted toward value stocks',
                'negative': 'tilted toward growth stocks',
                'neutral': 'value/growth neutral'
            },
            'RMW': {
                'positive': 'tilted toward profitable stocks',
                'negative': 'tilted toward weak profitability',
                'neutral': 'profitability-neutral'
            },
            'CMA': {
                'positive': 'tilted toward conservative investment',
                'negative': 'tilted toward aggressive investment',
                'neutral': 'investment-neutral'
            }
        }

        if factor_name not in interpretations:
            return ''

        if abs(loading) < 0.1:
            category = 'neutral'
        elif loading > 0:
            category = 'positive'
        else:
            category = 'negative'

        return interpretations[factor_name][category]
