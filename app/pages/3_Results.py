"""
Results Page
Display Black-Litterman portfolio optimization results and analysis
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    load_css, init_session_state, display_page_header, render_sidebar_navigation,
    create_pick_matrix_from_views, create_omega_matrix, format_percentage
)
from components.charts import (
    create_portfolio_pie_chart, create_weights_comparison_bar_chart,
    create_correlation_heatmap, create_risk_contribution_chart,
    create_returns_comparison_chart
)
from components.metrics import (
    display_portfolio_summary_table,
    display_comparison_table, display_sector_allocation_table
)
from backend.data_loader import get_data_loader
from backend.bl_model import BlackLittermanModel
from backend.calculations import (
    calculate_returns, calculate_covariance_matrix,
    calculate_correlation_matrix, calculate_risk_contribution
)

st.set_page_config(
    page_title="Results & Analysis",
    page_icon="📊",
    layout="wide"
)

load_css()
init_session_state()
render_sidebar_navigation()


def run_optimization():
    """Run Black-Litterman optimization"""

    data_loader = get_data_loader()

    # Get selected tickers (filter out GOOGL - we use GOOG only)
    tickers = [t for t in st.session_state.selected_tickers if t != 'GOOGL']

    # Get historical prices (5 years = 1260 trading days)
    prices = data_loader.get_historical_prices(
        tickers,
        periods=1260
    )

    # Calculate returns and covariance
    returns = calculate_returns(prices)
    cov_matrix = calculate_covariance_matrix(returns)

    # Get market caps for benchmark
    market_caps = data_loader.get_market_caps(tickers)

    # Initialize BL model
    bl_model = BlackLittermanModel(
        cov_matrix=cov_matrix,
        market_caps=market_caps,
        risk_free_rate=st.session_state.get('risk_free_rate', 0.02),
        risk_aversion=st.session_state.get('risk_aversion', 2.5),
        tau=st.session_state.get('tau', 0.025)
    )

    # Get prior returns
    prior_returns = bl_model.prior_returns
    market_weights = bl_model.market_weights

    # Setup constraints first (needed for both with/without views optimization)
    constraints = {}
    if st.session_state.get('max_weight'):
        constraints['max_weight'] = st.session_state.max_weight
    if st.session_state.get('min_weight'):
        constraints['min_weight'] = st.session_state.min_weight

    # Get ticker-sector mapping and per-sector constraints
    ticker_sectors = st.session_state.get('ticker_sectors', {})
    sector_constraints = st.session_state.get('sector_constraints', {})

    # Process views - check manual views OR regenerate sentiment views fresh
    has_manual_views = bool(st.session_state.views)

    # Check if sentiment mode is active (don't use saved views)
    use_sentiment_views = (
        'view_mode' in st.session_state
        and st.session_state.view_mode == "Auto (Sentiment)"
    )

    if use_sentiment_views and not has_manual_views:
        # Regenerate sentiment views fresh (don't use cached)
        from backend.sentiment_cache import SentimentCache
        from backend.bl_view_generator import BLViewGenerator

        sentiment_cache = SentimentCache()

        # Load sentiment data for all tickers in ONE query (fast!)
        sentiment_data = sentiment_cache.get_bulk_sentiment_summary(tickers, days=7)

        if sentiment_data:
            # Generate views fresh using current data
            volatilities = np.std(returns, axis=0) * np.sqrt(252)  # Annualize volatilities
            generator = BLViewGenerator(min_articles=3)

            auto_views = generator.generate_views_ranked(
                sentiment_data,
                tickers,
                volatilities=volatilities,
                tau=st.session_state.get('tau', 0.05),
                equilibrium_returns=prior_returns,
                kappa=st.session_state.get('sentiment_kappa', 0.15),  # Use value from slider
                sentiment_threshold=0.3,
                normalize_sentiment=False  # Use raw sentiment [-1, 1] without z-score normalization
            )

            if auto_views['P']:
                P = np.array(auto_views['P'])
                Q = np.array(auto_views['Q'])
                omega = np.diag(auto_views['Omega'])

                # Calculate posterior WITH views
                posterior_returns, posterior_cov = bl_model.calculate_posterior(P, Q, omega)

                # Also optimize WITHOUT views for comparison
                bl_no_views_weights = bl_model.optimize_portfolio(
                    prior_returns,
                    cov_matrix,
                    constraints if constraints else None,
                    tickers=tickers,
                    ticker_sectors=ticker_sectors,
                    sector_constraints=sector_constraints
                )
            else:
                # No views generated (sentiment too moderate)
                posterior_returns = prior_returns
                posterior_cov = cov_matrix
                bl_no_views_weights = None
        else:
            # No sentiment data available
            posterior_returns = prior_returns
            posterior_cov = cov_matrix
            bl_no_views_weights = None
    elif has_manual_views:
        # Use manual views
        P, Q = create_pick_matrix_from_views(st.session_state.views, tickers)
        confidences = [v['confidence'] for v in st.session_state.views]

        # Create Omega matrix
        omega = create_omega_matrix(P, cov_matrix, bl_model.tau, confidences)

        # Calculate posterior WITH views
        posterior_returns, posterior_cov = bl_model.calculate_posterior(P, Q, omega)

        # Also optimize WITHOUT views for comparison
        bl_no_views_weights = bl_model.optimize_portfolio(
            prior_returns,
            cov_matrix,
            constraints if constraints else None,
            tickers=tickers,
            ticker_sectors=ticker_sectors,
            sector_constraints=sector_constraints
        )
    else:
        # No views, use prior
        posterior_returns = prior_returns
        posterior_cov = cov_matrix
        bl_no_views_weights = None  # No comparison needed when no views

    # Optimize WITH views (or with prior if no views)
    bl_weights = bl_model.optimize_portfolio(
        posterior_returns,
        posterior_cov,
        constraints if constraints else None,
        tickers=tickers,
        ticker_sectors=ticker_sectors,
        sector_constraints=sector_constraints
    )

    # Debug: Print weight comparison and returns analysis
    print("\n" + "="*80)
    print("BLACK-LITTERMAN ANALYSIS")
    print("="*80)
    print("\n1. EXPECTED RETURNS: Prior vs Posterior")
    print("-" * 80)
    for i, ticker in enumerate(tickers):
        prior_r = prior_returns[i] * 100
        post_r = posterior_returns[i] * 100
        diff_r = post_r - prior_r
        print(f"{ticker:6s} | Prior: {prior_r:+6.2f}% | Posterior: {post_r:+6.2f}% | Diff: {diff_r:+6.2f}%")

    print("\n2. PORTFOLIO WEIGHTS: Market Cap vs BL (No Views) vs BL (With Views)")
    print("-" * 80)
    if bl_no_views_weights is not None:
        for i, ticker in enumerate(tickers):
            mkt_w = market_weights[i] * 100
            bl_no_views_w = bl_no_views_weights[i] * 100
            bl_w = bl_weights[i] * 100
            print(f"{ticker:6s} | Market: {mkt_w:5.2f}% | BL(No): {bl_no_views_w:5.2f}% | BL(Views): {bl_w:5.2f}%")
    else:
        for i, ticker in enumerate(tickers):
            mkt_w = market_weights[i] * 100
            bl_w = bl_weights[i] * 100
            diff_w = bl_w - mkt_w
            print(f"{ticker:6s} | Market: {mkt_w:6.2f}% | BL: {bl_w:6.2f}% | Diff: {diff_w:+6.2f}%")
    print("="*80 + "\n")

    # Calculate metrics
    # Use posterior returns for all portfolios to ensure consistent comparison
    bl_metrics = bl_model.get_portfolio_metrics(bl_weights, posterior_returns, posterior_cov)
    market_metrics = bl_model.get_portfolio_metrics(market_weights, posterior_returns, posterior_cov)

    # Calculate metrics for BL without views if available
    if bl_no_views_weights is not None:
        bl_no_views_metrics = bl_model.get_portfolio_metrics(bl_no_views_weights, prior_returns, cov_matrix)
    else:
        bl_no_views_metrics = None

    # Risk contribution
    risk_contrib = calculate_risk_contribution(bl_weights, posterior_cov)

    # Correlation matrix
    corr_matrix = calculate_correlation_matrix(returns)

    # Get sectors
    sectors = data_loader.get_sectors(tickers)

    # Get current prices
    constituents_df = data_loader.get_sp500_constituents()
    price_dict = dict(zip(constituents_df['ticker'], constituents_df['price']))
    prices_array = np.array([price_dict[t] for t in tickers])

    # Store results
    results = {
        'tickers': tickers,
        'bl_weights': bl_weights,
        'market_weights': market_weights,
        'bl_no_views_weights': bl_no_views_weights,
        'prior_returns': prior_returns,
        'posterior_returns': posterior_returns,
        'bl_metrics': bl_metrics,
        'market_metrics': market_metrics,
        'bl_no_views_metrics': bl_no_views_metrics,
        'risk_contribution': risk_contrib,
        'correlation_matrix': corr_matrix,
        'sectors': sectors,
        'prices': prices_array,
        'cov_matrix': cov_matrix
    }

    st.session_state.bl_results = results

    return results


def main():
    display_page_header(
        "Portfolio Results & Analysis",
        "Black-Litterman optimization results and comprehensive portfolio analysis",
        "📊"
    )

    # Check prerequisites
    if not st.session_state.selected_tickers:
        st.warning("⚠️ No stocks selected. Please select stocks first.")
        if st.button("Go to Stock Selection"):
            st.switch_page("pages/1_Stock_Selection.py")
        return

    # Only run optimization if results don't exist or user explicitly requests it
    if 'bl_results' not in st.session_state or st.session_state.bl_results is None:
        with st.spinner("Optimizing portfolio..."):
            results = run_optimization()
            st.session_state.bl_results = results
    else:
        results = st.session_state.bl_results

    # Extract results first (needed for sidebar)
    tickers = results['tickers']
    bl_weights = results['bl_weights']
    market_weights = results['market_weights']
    bl_no_views_weights = results.get('bl_no_views_weights')
    prior_returns = results['prior_returns']
    posterior_returns = results['posterior_returns']
    bl_metrics = results['bl_metrics']
    market_metrics = results['market_metrics']
    bl_no_views_metrics = results.get('bl_no_views_metrics')
    sectors = results['sectors']

    # Sidebar
    with st.sidebar:
        # View mode indicator
        has_manual_views = bool(st.session_state.views)
        use_sentiment = (
            'view_mode' in st.session_state
            and st.session_state.view_mode == "Auto (Sentiment)"
        )

        if use_sentiment and not has_manual_views:
            kappa_val = st.session_state.get('sentiment_kappa', 0.15)
            st.info(f"**Auto Mode**: Sentiment-driven views\n\n"
                   f"Sentiment Impact (kappa): **{kappa_val:.2f}**")
        elif has_manual_views:
            st.info(f"**Manual Mode**: {len(st.session_state.views)} custom view(s)")
        else:
            st.info("**No Views**: Market equilibrium")

        st.markdown("---")

        # Export Position Details
        st.markdown("### Export")

        # Create position details dataframe for export
        position_df = pd.DataFrame({
            'Ticker': tickers,
            'Weight (%)': bl_weights * 100,
            'Expected Return (%)': posterior_returns * 100,
            'Sector': sectors
        })

        if results.get('prices') is not None:
            position_df['Price ($)'] = results['prices']

        if st.session_state.get('portfolio_amount'):
            portfolio_amount = st.session_state.portfolio_amount
            position_df['Allocation ($)'] = bl_weights * portfolio_amount
            position_df['Shares'] = (bl_weights * portfolio_amount / results['prices']).astype(int)

        # Sort by weight descending
        position_df = position_df.sort_values('Weight (%)', ascending=False)

        # Export button
        csv_data = position_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Position Details (CSV)",
            data=csv_data,
            file_name="bl_portfolio_positions.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Determine benchmark to use for comparisons
    # If we have BL without views, use that as benchmark; otherwise use market cap
    if bl_no_views_weights is not None and bl_no_views_metrics is not None:
        benchmark_weights = bl_no_views_weights
        benchmark_metrics = bl_no_views_metrics
        benchmark_name = "BL (No Views)"
    else:
        benchmark_weights = market_weights
        benchmark_metrics = market_metrics
        benchmark_name = "Market Cap"

    # Main metrics - Historical Performance vs SPY
    st.markdown("## Portfolio Performance Metrics")

    # Load data for historical performance comparison
    data_loader = get_data_loader()
    prices = data_loader.get_historical_prices(tickers, periods=1260)
    returns = calculate_returns(prices)

    # Calculate historical portfolio performance
    portfolio_returns = (returns * bl_weights).sum(axis=1)

    # Load SPY data for comparison (from sector_etf_prices table)
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        spy_df = db.get_sector_etf_history('SPY', days=1260)
        
        if not spy_df.empty and len(spy_df) >= 20:
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df = spy_df.set_index('date').sort_index()
            spy_returns = spy_df['close'].pct_change().dropna()

            # Align dates
            common_dates = portfolio_returns.index.intersection(spy_returns.index)
            portfolio_returns_aligned = portfolio_returns.loc[common_dates]
            spy_returns_aligned = spy_returns.loc[common_dates]

            # Calculate cumulative returns
            portfolio_cumulative = (1 + portfolio_returns_aligned).cumprod() - 1
            spy_cumulative = (1 + spy_returns_aligned).cumprod() - 1

            # Calculate metrics
            portfolio_total_return = portfolio_cumulative.iloc[-1]
            spy_total_return = spy_cumulative.iloc[-1]

            portfolio_annualized = (1 + portfolio_total_return) ** (252 / len(common_dates)) - 1
            spy_annualized = (1 + spy_total_return) ** (252 / len(common_dates)) - 1

            portfolio_vol = portfolio_returns_aligned.std() * np.sqrt(252)
            spy_vol = spy_returns_aligned.std() * np.sqrt(252)

            portfolio_sharpe = portfolio_annualized / portfolio_vol if portfolio_vol > 0 else 0
            spy_sharpe = spy_annualized / spy_vol if spy_vol > 0 else 0

            # Display comparison metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "BL Portfolio Return (Ann.)",
                    format_percentage(portfolio_annualized),
                    delta=format_percentage(portfolio_annualized - spy_annualized),
                    help="Annualized return of BL portfolio vs SPY"
                )

            with col2:
                st.metric(
                    "BL Portfolio Volatility (Ann.)",
                    format_percentage(portfolio_vol),
                    delta=format_percentage(portfolio_vol - spy_vol),
                    delta_color='inverse',
                    help="Annualized volatility of BL portfolio vs SPY"
                )

            with col3:
                st.metric(
                    "BL Portfolio Sharpe Ratio",
                    f"{portfolio_sharpe:.3f}",
                    delta=f"{portfolio_sharpe - spy_sharpe:+.3f}",
                    help="Risk-adjusted return of BL portfolio vs SPY"
                )

            # Show cumulative returns chart
            st.markdown("### Historical Performance: BL Portfolio vs SPY")
            st.caption("*Indicative performance only. Past returns do not guarantee future results.*")

            comparison_df = pd.DataFrame({
                'BL Portfolio': portfolio_cumulative * 100,
                'SPY': spy_cumulative * 100
            })

            import plotly.graph_objects as go
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=comparison_df.index,
                y=comparison_df['BL Portfolio'],
                mode='lines',
                name='BL Portfolio',
                line=dict(color='#4CAF50', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=comparison_df.index,
                y=comparison_df['SPY'],
                mode='lines',
                name='SPY',
                line=dict(color='#2196F3', width=2)
            ))

            fig.update_layout(
                title='Cumulative Returns (%)',
                xaxis_title='Date',
                yaxis_title='Cumulative Return (%)',
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            spy_count = len(spy_df) if not spy_df.empty else 0
            st.warning(f"SPY data insufficient for comparison ({spy_count} days available, need at least 20). Run `python scripts/fetch_prices.py` to fetch more data.")
    except Exception as e:
        st.warning(f"Could not load SPY comparison: {e}")

    # Tabs for different analyses
    tabs = st.tabs([
        "📊 Portfolio Allocation",
        "📈 Performance Analysis",
        "⚠️ Risk Analysis",
        "🧬 Factor Exposure",
        "🔍 Detailed Holdings",
        "📥 Export"
    ])

    # Tab 1: Portfolio Allocation
    with tabs[0]:
        st.markdown("### Portfolio Weights")

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig_pie = create_portfolio_pie_chart(tickers, bl_weights, "Black-Litterman Allocation")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart comparison
            fig_bar = create_weights_comparison_bar_chart(
                tickers, bl_weights, benchmark_weights,
                f"BL Portfolio vs {benchmark_name}",
                benchmark_label=benchmark_name
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # Sector allocation
        st.markdown("### Sector Allocation")
        display_sector_allocation_table(tickers, bl_weights, sectors)

        # Returns comparison
        st.markdown("---")
        st.markdown("### Expected Returns: Impact of Views")

        fig_returns = create_returns_comparison_chart(
            tickers, prior_returns, posterior_returns,
            "Prior (Equilibrium) vs Posterior (BL) Returns"
        )
        st.plotly_chart(fig_returns, use_container_width=True)

    # Tab 2: Performance Analysis
    with tabs[1]:
        # Performance comparison table
        st.markdown("### Portfolio Comparison")

        comparison_df = pd.DataFrame({
            'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
            'BL Portfolio': [
                format_percentage(bl_metrics['return']),
                format_percentage(bl_metrics['volatility']),
                f"{bl_metrics['sharpe_ratio']:.3f}"
            ],
            benchmark_name: [
                format_percentage(benchmark_metrics['return']),
                format_percentage(benchmark_metrics['volatility']),
                f"{benchmark_metrics['sharpe_ratio']:.3f}"
            ],
            'Difference': [
                format_percentage(bl_metrics['return'] - benchmark_metrics['return']),
                format_percentage(bl_metrics['volatility'] - benchmark_metrics['volatility']),
                f"{bl_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio']:+.3f}"
            ]
        })

        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Tab 3: Risk Analysis
    with tabs[2]:
        st.markdown("### Risk Contribution")

        fig_risk = create_risk_contribution_chart(
            tickers, results['risk_contribution'],
            "Risk Contribution by Asset"
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("---")

        # Correlation heatmap
        st.markdown("### Correlation Matrix")

        # Show top correlated assets
        n_display = min(20, len(tickers))
        corr_subset = results['correlation_matrix'].iloc[:n_display, :n_display]

        fig_corr = create_correlation_heatmap(corr_subset, "Asset Correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        # Diversification metrics
        st.markdown("### Diversification Analysis")

        col1, col2, col3 = st.columns(3)

        # Effective number of stocks
        herfindahl = np.sum(bl_weights ** 2)
        effective_n = 1 / herfindahl

        with col1:
            st.metric(
                "Effective N",
                f"{effective_n:.1f}",
                help="Effective number of stocks (1/sum(w²))"
            )

        with col2:
            concentration = np.sum(np.sort(bl_weights)[-5:])
            st.metric(
                "Top 5 Concentration",
                format_percentage(concentration),
                help="Weight in top 5 holdings"
            )

        with col3:
            n_significant = np.sum(bl_weights > 0.01)
            st.metric(
                "Holdings > 1%",
                f"{n_significant}",
                help="Number of positions above 1%"
            )

    # Tab 4: Factor Exposure
    with tabs[3]:
        st.markdown("### Factor Exposure Analysis")
        st.caption("Fama-French 5-Factor Model")

        try:
            from backend.factor_analyzer import FactorAnalyzer

            factor_analyzer = FactorAnalyzer()

            # Load factor loadings from database
            factor_loadings = factor_analyzer.load_loadings_from_db(tickers)

            if not factor_loadings:
                st.warning("⚠️ No factor loadings available. Run the factor calculation script:")
                st.code("python scripts/fetch_factors.py\npython scripts/calculate_factor_loadings.py", language="bash")
            else:
                # Calculate portfolio factor exposures
                weights_dict = dict(zip(tickers, bl_weights))
                portfolio_exposure = factor_analyzer.calculate_portfolio_exposure(weights_dict, factor_loadings)

                # Display portfolio-level exposures
                st.markdown("#### Portfolio Factor Exposures")

                col1, col2, col3 = st.columns(3)

                with col1:
                    beta = portfolio_exposure.get('Mkt-RF', 0)
                    st.metric(
                        "Market Beta",
                        f"{beta:.3f}",
                        help="Sensitivity to market movements (1.0 = moves with market)"
                    )
                    beta_interp = factor_analyzer.get_factor_interpretation('Mkt-RF', beta)
                    if beta_interp:
                        st.caption(beta_interp)

                with col2:
                    smb = portfolio_exposure.get('SMB', 0)
                    st.metric(
                        "Size (SMB)",
                        f"{smb:+.3f}",
                        help="Small cap vs large cap tilt"
                    )
                    smb_interp = factor_analyzer.get_factor_interpretation('SMB', smb)
                    if smb_interp:
                        st.caption(smb_interp)

                with col3:
                    hml = portfolio_exposure.get('HML', 0)
                    st.metric(
                        "Value (HML)",
                        f"{hml:+.3f}",
                        help="Value vs growth tilt"
                    )
                    hml_interp = factor_analyzer.get_factor_interpretation('HML', hml)
                    if hml_interp:
                        st.caption(hml_interp)

                col4, col5 = st.columns(2)

                with col4:
                    rmw = portfolio_exposure.get('RMW', 0)
                    st.metric(
                        "Profitability (RMW)",
                        f"{rmw:+.3f}",
                        help="Robust vs weak profitability"
                    )
                    rmw_interp = factor_analyzer.get_factor_interpretation('RMW', rmw)
                    if rmw_interp:
                        st.caption(rmw_interp)

                with col5:
                    cma = portfolio_exposure.get('CMA', 0)
                    st.metric(
                        "Investment (CMA)",
                        f"{cma:+.3f}",
                        help="Conservative vs aggressive investment"
                    )
                    cma_interp = factor_analyzer.get_factor_interpretation('CMA', cma)
                    if cma_interp:
                        st.caption(cma_interp)

                st.markdown("---")

                # Stock-level factor loadings
                st.markdown("#### Individual Stock Factor Loadings")

                # Create dataframe with loadings
                loadings_data = []
                for ticker in tickers:
                    if ticker in factor_loadings:
                        loading = factor_loadings[ticker]
                        loadings_data.append({
                            'Ticker': ticker,
                            'Weight': format_percentage(weights_dict[ticker]),
                            'Beta': f"{loading.get('Mkt-RF', 0):.2f}",
                            'SMB': f"{loading.get('SMB', 0):+.2f}",
                            'HML': f"{loading.get('HML', 0):+.2f}",
                            'RMW': f"{loading.get('RMW', 0):+.2f}",
                            'CMA': f"{loading.get('CMA', 0):+.2f}",
                            'R²': f"{loading.get('R-squared', 0):.2f}"
                        })

                if loadings_data:
                    loadings_df = pd.DataFrame(loadings_data)
                    st.dataframe(loadings_df, use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # Factor exposure breakdown chart
                    st.markdown("#### Factor Exposure Breakdown")

                    import plotly.graph_objects as go

                    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
                    exposures = [portfolio_exposure.get(f, 0) for f in factors]

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=factors,
                        y=exposures,
                        marker_color=['#1f77b4' if e >= 0 else '#d62728' for e in exposures],
                        text=[f"{e:+.3f}" for e in exposures],
                        textposition='outside'
                    ))

                    fig.update_layout(
                        title="Portfolio Factor Exposures",
                        xaxis_title="Factor",
                        yaxis_title="Exposure",
                        showlegend=False,
                        height=400,
                        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No factor loadings available for selected stocks.")

        except ImportError as e:
            st.error(f"Factor analysis module not available: {e}")
        except Exception as e:
            st.error(f"Error loading factor exposures: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Tab 5: Detailed Holdings
    with tabs[4]:
        st.markdown("### Complete Portfolio Holdings")

        display_comparison_table(
            tickers, bl_weights, market_weights,
            posterior_returns, prior_returns,
            top_n=len(tickers)
        )

        st.markdown("---")

        st.markdown("### Position Details")

        display_portfolio_summary_table(
            tickers, bl_weights, posterior_returns,
            sectors, results['prices'],
            portfolio_amount=st.session_state.get('portfolio_amount', 100000)
        )

    # Tab 6: Export
    with tabs[5]:
        st.markdown("### Export Results")

        # Prepare export data
        export_df = pd.DataFrame({
            'Ticker': tickers,
            'BL_Weight': bl_weights,
            'Market_Weight': market_weights,
            'Sector': sectors,
            'Prior_Return': prior_returns,
            'Posterior_Return': posterior_returns,
            'Price': results['prices']
        })

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Download Portfolio Data**")

            csv = export_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="bl_portfolio.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            st.markdown("**Download Metrics**")

            metrics_df = pd.DataFrame({
                'Metric': ['Expected Return', 'Volatility', 'Sharpe Ratio'],
                'BL Portfolio': [
                    bl_metrics['return'],
                    bl_metrics['volatility'],
                    bl_metrics['sharpe_ratio']
                ],
                benchmark_name: [
                    benchmark_metrics['return'],
                    benchmark_metrics['volatility'],
                    benchmark_metrics['sharpe_ratio']
                ]
            })

            csv_metrics = metrics_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv_metrics,
                file_name="bl_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")

        st.markdown("**Preview Export Data**")
        st.dataframe(export_df, use_container_width=True)

    # Navigation
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back to Views & Params", use_container_width=True):
            st.switch_page("pages/2_Views_Configuration.py")

    with col2:
        if st.button("Back to Home", use_container_width=True):
            st.switch_page("main.py")

    with col3:
        if st.button("New Portfolio", use_container_width=True):
            st.session_state.selected_tickers = []
            st.session_state.views = []
            st.session_state.bl_results = None
            st.success("Session reset!")
            st.switch_page("main.py")


if __name__ == "__main__":
    main()
