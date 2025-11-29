"""Black-Litterman model implementation"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class BlackLittermanModel:
    """Black-Litterman portfolio optimization model"""

    def __init__(
        self,
        cov_matrix: np.ndarray,
        market_caps: np.ndarray,
        risk_free_rate: float = 0.02,
        risk_aversion: float = 2.5,
        tau: float = 0.025
    ):
        """Initialize Black-Litterman model"""
        self.cov_matrix = cov_matrix
        self.market_caps = market_caps
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Calculate market equilibrium weights
        self.market_weights = self._calculate_market_weights()

        # Calculate implied equilibrium returns (Pi)
        self.prior_returns = self._calculate_implied_returns()

    def _calculate_market_weights(self) -> np.ndarray:
        """Calculate market cap weights"""
        return self.market_caps / np.sum(self.market_caps)

    def _calculate_implied_returns(self) -> np.ndarray:
        """Calculate equilibrium returns: delta * Sigma * w_mkt"""
        return self.risk_aversion * self.cov_matrix @ self.market_weights

    def calculate_posterior(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate posterior returns and covariance"""
        if Omega is None:
            Omega = np.diag(np.diag(P @ (self.tau * self.cov_matrix) @ P.T))

        tau_sigma = self.tau * self.cov_matrix
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(Omega)

        M_inv = tau_sigma_inv + P.T @ omega_inv @ P
        M = np.linalg.inv(M_inv)

        posterior_returns = M @ (tau_sigma_inv @ self.prior_returns + P.T @ omega_inv @ Q)
        posterior_cov = self.cov_matrix + M

        return posterior_returns, posterior_cov

    def optimize_portfolio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None,
        constraints: Optional[dict] = None,
        tickers: Optional[list] = None,
        ticker_sectors: Optional[dict] = None,
        sector_constraints: Optional[dict] = None
    ) -> np.ndarray:
        """Calculate optimal portfolio weights with optional constraints"""
        from scipy.optimize import minimize

        if cov_matrix is None:
            cov_matrix = self.cov_matrix

        n_assets = len(expected_returns)

        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)

        opt_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        if constraints:
            min_w = constraints.get('min_weight') or 0.0
            max_w = constraints.get('max_weight') or 1.0
            bounds = tuple((min_w, max_w) for _ in range(n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(n_assets))

        if sector_constraints and tickers and ticker_sectors:
            sector_indices = {}
            for i, ticker in enumerate(tickers):
                sector = ticker_sectors.get(ticker, 'Unknown')
                if sector not in sector_indices:
                    sector_indices[sector] = []
                sector_indices[sector].append(i)

            for sector, limits in sector_constraints.items():
                if sector not in sector_indices:
                    continue

                indices = sector_indices[sector]
                sector_max = limits.get('max')
                sector_min = limits.get('min')

                if sector_max is not None:
                    opt_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=indices, mx=sector_max: mx - sum(w[i] for i in idx)
                    })

                if sector_min is not None and sector_min > 0:
                    opt_constraints.append({
                        'type': 'ineq',
                        'fun': lambda w, idx=indices, mn=sector_min: sum(w[i] for i in idx) - mn
                    })

        initial_guess = np.array([1/n_assets] * n_assets)

        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'maxiter': 2000, 'ftol': 1e-9}
        )

        if not result.success:
            print(f"[WARNING] Optimization did not converge: {result.message}")
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=opt_constraints,
                options={'maxiter': 3000, 'ftol': 1e-6}
            )
            if not result.success:
                print(f"[WARNING] Fallback optimization also failed, using equal weights")
                return initial_guess

        return result.x

    def get_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None
    ) -> dict:
        """Calculate portfolio performance metrics"""
        if cov_matrix is None:
            cov_matrix = self.cov_matrix

        portfolio_return = weights @ expected_returns
        portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility

        return {
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'weights': weights
        }


def create_pick_matrix(view_assets: list, n_assets: int, view_type: str = 'absolute') -> np.ndarray:
    """Create pick matrix for views"""
    P_row = np.zeros(n_assets)

    if view_type == 'absolute':
        P_row[view_assets[0]] = 1.0
    elif view_type == 'relative':
        P_row[view_assets[0]] = 1.0
        P_row[view_assets[1]] = -1.0

    return P_row
