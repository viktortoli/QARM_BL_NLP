"""Black-Litterman view generator from sentiment data"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import rankdata, norm


class BLViewGenerator:
    """Generate Black-Litterman views from sentiment"""

    def __init__(
        self,
        scaling_factor: float = 0.10,
        min_significance: float = 0.005,
        min_articles: int = 10
    ):
        """Initialize view generator"""
        self.scaling_factor = scaling_factor
        self.min_significance = min_significance
        self.min_articles = min_articles

    def generate_views_ranked(
        self,
        sentiment_data: Dict[str, Dict],
        tickers: List[str],
        returns_matrix: Optional[np.ndarray] = None,
        volatilities: Optional[np.ndarray] = None,
        tau: float = 0.025,
        equilibrium_returns: Optional[np.ndarray] = None,
        kappa: float = 0.15,
        sentiment_threshold: float = 0.3,
        normalize_sentiment: bool = True
    ) -> Dict:
        """Generate views by scaling sentiment with volatility"""
        # Validate inputs
        if volatilities is None or len(volatilities) != len(tickers):
            raise ValueError("volatilities array is required and must match tickers length")

        if equilibrium_returns is None or len(equilibrium_returns) != len(tickers):
            raise ValueError("equilibrium_returns array is required and must match tickers length")

        raw_sentiments = {}
        for ticker in tickers:
            if ticker in sentiment_data:
                ticker_sentiment = sentiment_data[ticker]
                article_count = ticker_sentiment.get('article_count', 0)
                if article_count >= self.min_articles:
                    raw_sentiments[ticker] = ticker_sentiment['avg_sentiment_score']

        if normalize_sentiment and len(raw_sentiments) > 2:
            sentiment_values = list(raw_sentiments.values())
            sentiment_mean = np.mean(sentiment_values)
            sentiment_std = np.std(sentiment_values)

            if sentiment_std < 1e-6:
                print(f"[WARNING] Sentiment std={sentiment_std:.6f} too low, skipping normalization")
                normalized_sentiments = raw_sentiments.copy()
            else:
                normalized_sentiments = {
                    ticker: (score - sentiment_mean) / sentiment_std
                    for ticker, score in raw_sentiments.items()
                }
                print(f"[INFO] Applied z-score normalization: mean={sentiment_mean:.3f}, std={sentiment_std:.3f}")
        else:
            normalized_sentiments = raw_sentiments.copy()
            if normalize_sentiment:
                print(f"[INFO] Skipping normalization (only {len(raw_sentiments)} tickers with sentiment)")

        P_matrix = []
        Q_vector = []
        Omega_vector = []
        descriptions = []
        tickers_with_views = []

        N = len(tickers)

        print(f"[DEBUG] Kappa: {kappa}, Sentiment threshold: {sentiment_threshold}")

        for ticker in tickers:
            if ticker not in normalized_sentiments:
                continue

            ticker_sentiment = sentiment_data[ticker]
            article_count = ticker_sentiment.get('article_count', 0)

            sentiment_score = normalized_sentiments[ticker]
            raw_sentiment_score = raw_sentiments[ticker]

            effective_threshold = 0.5 if normalize_sentiment else sentiment_threshold
            if abs(sentiment_score) < effective_threshold:
                continue

            ticker_idx = tickers.index(ticker)
            sigma_i = volatilities[ticker_idx]
            pi_i = equilibrium_returns[ticker_idx]

            Q_value = pi_i + kappa * sentiment_score * sigma_i

            P_row = [0.0] * N
            P_row[ticker_idx] = 1.0

            article_adjustment = 1.0 / np.sqrt(article_count / 10.0)
            omega = tau * sigma_i**2 * article_adjustment

            distribution = ticker_sentiment.get('distribution', {})
            max_pct = max(
                distribution.get('positive', 0),
                distribution.get('neutral', 0),
                distribution.get('negative', 0)
            )
            consensus = max_pct / 100.0

            P_matrix.append(P_row)
            Q_vector.append(Q_value)
            Omega_vector.append(omega)

            sentiment_contribution = kappa * sentiment_score * sigma_i

            if normalize_sentiment:
                print(f"[DEBUG] {ticker}: raw_sentiment={raw_sentiment_score:+.2f}, "
                      f"z_score={sentiment_score:+.2f}, "
                      f"equilibrium={pi_i*100:.2f}%, vol={sigma_i*100:.2f}%, "
                      f"sentiment_contrib={sentiment_contribution*100:+.2f}%, "
                      f"view_return={Q_value*100:+.2f}%, omega={omega:.6f}")
            else:
                print(f"[DEBUG] {ticker}: sentiment={sentiment_score:+.2f}, "
                      f"equilibrium={pi_i*100:.2f}%, vol={sigma_i*100:.2f}%, "
                      f"sentiment_contrib={sentiment_contribution*100:+.2f}%, "
                      f"view_return={Q_value*100:+.2f}%, omega={omega:.6f}")

            direction = "positive" if sentiment_score > 0 else "negative"
            if normalize_sentiment:
                descriptions.append(
                    f"{ticker}: {Q_value*100:+.2f}% expected return "
                    f"(baseline: {pi_i*100:+.2f}%, sentiment adj: {sentiment_contribution*100:+.2f}%, "
                    f"{article_count} articles, {direction} z={abs(sentiment_score):.2f})"
                )
            else:
                descriptions.append(
                    f"{ticker}: {Q_value*100:+.2f}% expected return "
                    f"(baseline: {pi_i*100:+.2f}%, sentiment adj: {sentiment_contribution*100:+.2f}%, "
                    f"{article_count} articles, {direction} {abs(sentiment_score):.2f})"
                )

            tickers_with_views.append(ticker)

        return {
            'P': P_matrix,
            'Q': Q_vector,
            'Omega': Omega_vector,
            'descriptions': descriptions,
            'tickers_with_views': tickers_with_views
        }

    def generate_views_absolute(
        self,
        sentiment_data: Dict[str, Dict],
        tickers: List[str]
    ) -> Dict:
        """Generate absolute views from sentiment"""
        valid_tickers = [
            t for t in tickers
            if t in sentiment_data
            and sentiment_data[t].get('article_count', 0) >= self.min_articles
        ]

        if not valid_tickers:
            return {
                'P': [],
                'Q': [],
                'Omega': [],
                'descriptions': [],
                'tickers_with_views': []
            }

        P_matrix = []
        Q_vector = []
        Omega_vector = []
        descriptions = []
        tickers_with_views = []

        N = len(tickers)

        for ticker in valid_tickers:
            sentiment_score = sentiment_data[ticker]['avg_sentiment_score']
            expected_return = sentiment_score * self.scaling_factor

            if abs(expected_return) < self.min_significance:
                continue

            ticker_idx = tickers.index(ticker)
            P_row = [0.0] * N
            P_row[ticker_idx] = 1.0

            P_matrix.append(P_row)
            Q_vector.append(expected_return)

            omega = self._calculate_omega(sentiment_data[ticker])
            Omega_vector.append(omega)

            direction = "positive" if expected_return > 0 else "negative"
            descriptions.append(
                f"{ticker}: {expected_return*100:+.2f}% ({direction} sentiment: {sentiment_score:+.3f}, "
                f"articles: {sentiment_data[ticker]['article_count']})"
            )

            tickers_with_views.append(ticker)

        return {
            'P': P_matrix,
            'Q': Q_vector,
            'Omega': Omega_vector,
            'descriptions': descriptions,
            'tickers_with_views': tickers_with_views
        }

    def generate_views_vs_market(
        self,
        sentiment_data: Dict[str, Dict],
        tickers: List[str]
    ) -> Dict:
        """Generate views vs portfolio average sentiment"""
        valid_tickers = [
            t for t in tickers
            if t in sentiment_data
            and sentiment_data[t].get('article_count', 0) >= self.min_articles
        ]

        if not valid_tickers:
            return {
                'P': [],
                'Q': [],
                'Omega': [],
                'descriptions': [],
                'market_avg_sentiment': 0.0,
                'tickers_with_views': []
            }

        market_avg_sentiment = sum(
            sentiment_data[t]['avg_sentiment_score']
            for t in valid_tickers
        ) / len(valid_tickers)

        P_matrix = []
        Q_vector = []
        Omega_vector = []
        descriptions = []
        tickers_with_views = []

        N = len(tickers)

        for ticker in valid_tickers:
            sentiment_score = sentiment_data[ticker]['avg_sentiment_score']
            outperformance = (sentiment_score - market_avg_sentiment) * self.scaling_factor

            if abs(outperformance) < self.min_significance:
                continue

            ticker_idx = tickers.index(ticker)
            P_row = [-1.0 / N] * N
            P_row[ticker_idx] = 1.0 - 1.0 / N

            P_matrix.append(P_row)
            Q_vector.append(outperformance)

            omega = self._calculate_omega(sentiment_data[ticker])
            Omega_vector.append(omega)

            direction = "outperform" if outperformance > 0 else "underperform"
            descriptions.append(
                f"{ticker} will {direction} market by {abs(outperformance)*100:.2f}% "
                f"(sentiment: {sentiment_score:+.3f}, articles: {sentiment_data[ticker]['article_count']})"
            )

            tickers_with_views.append(ticker)

        return {
            'P': P_matrix,
            'Q': Q_vector,
            'Omega': Omega_vector,
            'descriptions': descriptions,
            'market_avg_sentiment': market_avg_sentiment,
            'tickers_with_views': tickers_with_views
        }

    def _calculate_omega(self, sentiment_summary: Dict) -> float:
        """Calculate view uncertainty based on data quality"""
        distribution = sentiment_summary.get('distribution', {})
        max_percentage = max(
            distribution.get('positive', 0),
            distribution.get('neutral', 0),
            distribution.get('negative', 0)
        )
        consensus = max_percentage / 100.0

        article_count = sentiment_summary.get('article_count', 0)
        article_factor = min(article_count / 100.0, 1.0)

        omega = (1.0 - consensus) * (1.0 - article_factor) * 0.01

        return max(omega, 0.0001)

    def validate_views(
        self,
        P: List[List[float]],
        Q: List[float],
        Omega: List[float]
    ) -> Tuple[bool, str]:
        """Validate generated views"""
        if not P or not Q or not Omega:
            return True, ""

        if len(P) != len(Q) or len(P) != len(Omega):
            return False, "Dimension mismatch: P, Q, and Omega must have same length"

        row_length = len(P[0])
        if not all(len(row) == row_length for row in P):
            return False, "P matrix rows have inconsistent lengths"

        for i, row in enumerate(P):
            row_sum = sum(row)
            if abs(row_sum) > 1e-10:
                return False, f"P matrix row {i} does not sum to zero (sum={row_sum:.6f})"

        if any(omega <= 0 for omega in Omega):
            return False, "Omega values must be positive"

        return True, ""

    def get_scaling_factor_presets(self) -> Dict[str, float]:
        """Get preset scaling factors"""
        return {
            'Conservative': 0.05,
            'Moderate': 0.10,
            'Aggressive': 0.15
        }
