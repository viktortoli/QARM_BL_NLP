"""Portfolio analysis calculations"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns from prices"""
    return prices.pct_change().dropna()


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    shrinkage: bool = True
) -> np.ndarray:
    """Calculate covariance matrix with optional Ledoit-Wolf shrinkage"""
    if shrinkage:
        try:
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf()
            cov = lw.fit(returns.values).covariance_

            shrinkage_intensity = lw.shrinkage_
            print(f"[INFO] Applied Ledoit-Wolf shrinkage (intensity: {shrinkage_intensity:.3f})")

        except ImportError:
            print("[WARNING] sklearn not available, using sample covariance")
            cov = returns.cov().values
    else:
        cov = returns.cov().values

    if annualize:
        cov = cov * 252

    return cov


def calculate_expected_returns(returns: pd.DataFrame, method: str = 'historical') -> np.ndarray:
    """Calculate expected returns"""
    if method == 'historical':
        return returns.mean().values * 252

    return returns.mean().values * 252


def calculate_portfolio_performance(
    weights: np.ndarray,
    returns: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None
) -> dict:
    """Calculate historical portfolio performance"""
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    total_return = cumulative_returns.iloc[-1] - 1
    n_periods = len(portfolio_returns)
    annualized_return = (1 + total_return) ** (252 / n_periods) - 1
    annualized_volatility = portfolio_returns.std() * np.sqrt(252)

    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

    max_drawdown = calculate_max_drawdown(cumulative_returns)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'cumulative_returns': cumulative_returns,
        'portfolio_returns': portfolio_returns
    }


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calculate_risk_contribution(
    weights: np.ndarray,
    cov_matrix: np.ndarray
) -> np.ndarray:
    """Calculate risk contribution of each asset"""
    portfolio_variance = weights @ cov_matrix @ weights
    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / portfolio_variance

    return risk_contrib


def calculate_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    num_portfolios: int = 100,
    risk_free_rate: float = 0.02
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate efficient frontier"""
    from scipy.optimize import minimize

    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_guess = np.array([1/n_assets] * n_assets)

    min_var_result = minimize(
        portfolio_variance,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    min_return = np.dot(min_var_result.x, expected_returns)
    max_return = np.max(expected_returns)

    target_returns = np.linspace(min_return, max_return, num_portfolios)
    results = np.zeros((3, num_portfolios))

    for i, target_return in enumerate(target_returns):
        constraints_with_return = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, expected_returns) - target_return}
        ]

        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_with_return,
            options={'maxiter': 1000}
        )

        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
        else:
            if i > 0:
                results[:, i] = results[:, i-1]

    return results[0], results[1], results[2]


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix"""
    return returns.corr()


def calculate_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series
) -> float:
    """Calculate beta relative to market"""
    covariance = np.cov(asset_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)

    return covariance / market_variance


def calculate_capm_return(
    beta: float,
    market_return: float,
    risk_free_rate: float = 0.02
) -> float:
    """Calculate expected return using CAPM"""
    return risk_free_rate + beta * (market_return - risk_free_rate)


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize returns"""
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize volatility"""
    return returns.std() * np.sqrt(periods_per_year)
