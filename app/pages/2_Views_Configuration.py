"""
Views & Parameters Configuration Page
Configure investor views and Black-Litterman model parameters
"""

import streamlit as st
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import (
    load_css, init_session_state, render_sidebar_navigation,
    add_view, remove_view, clear_all_views, validate_view
)
from components.sentiment import display_sentiment_table
from backend.sentiment_cache import SentimentCache
from backend.bl_view_generator import BLViewGenerator
from backend.data_loader import get_data_loader
from backend.calculations import calculate_returns

st.set_page_config(
    page_title="Views & Parameters",
    page_icon="🎯",
    layout="wide"
)

load_css()
init_session_state()
render_sidebar_navigation(show_news_feed=True)


@st.cache_data(ttl=43200, show_spinner=False)
def load_sentiment_data(tickers):
    """Load sentiment data for selected tickers using bulk query (fast!)"""
    try:
        cache = SentimentCache()
        # Use bulk query - single DB call for all tickers instead of N calls
        return cache.get_bulk_sentiment_summary(list(tickers), days=7)
    except Exception as e:
        print(f"[ERROR] load_sentiment_data failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_prior_returns(tickers_tuple):
    """Load equilibrium (prior) expected returns for selected tickers"""
    try:
        from backend.bl_model import BlackLittermanModel

        tickers = list(tickers_tuple)
        data_loader = get_data_loader()

        # Get historical prices (5 years)
        prices = data_loader.get_historical_prices(tickers, periods=1260)
        market_caps = data_loader.get_market_caps(tickers)

        # Calculate returns and covariance with Ledoit-Wolf shrinkage
        returns = calculate_returns(prices)
        from backend.calculations import calculate_covariance_matrix
        cov_matrix = calculate_covariance_matrix(returns, annualize=True, shrinkage=True)

        # Create BL model
        bl_model = BlackLittermanModel(
            cov_matrix=cov_matrix,
            market_caps=market_caps,
            risk_aversion=st.session_state.get('risk_aversion', 2.5),
            tau=st.session_state.get('tau', 0.025)
        )

        # Return dict mapping ticker -> expected return
        return dict(zip(tickers, bl_model.prior_returns))
    except Exception as e:
        print(f"[ERROR] load_prior_returns failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    st.markdown("## 🎯 Views & Parameters")

    # Clear cached optimization results - any change here means user wants to re-optimize
    if 'bl_results' in st.session_state:
        st.session_state.bl_results = None

    # Check if stocks are selected
    if not st.session_state.selected_tickers:
        st.warning("No stocks selected. Please select stocks first.")
        if st.button("Go to Stock Selection"):
            st.switch_page("pages/1_Stock_Selection.py")
        return

    # Filter out GOOGL from selected tickers (we use GOOG only)
    if 'GOOGL' in st.session_state.selected_tickers:
        st.session_state.selected_tickers = [t for t in st.session_state.selected_tickers if t != 'GOOGL']
        st.rerun()

    # TWO COLUMN LAYOUT
    left_col, right_col = st.columns([3, 2], gap="medium")

    # ========== LEFT COLUMN: Views & Parameters ==========
    with left_col:
        # COMPARTMENT 1: Market Views
        with st.container(border=True):
            st.markdown("### 📊 Market Views")

            view_mode = st.radio(
                "Mode",
                ["Manual", "Auto (Sentiment)"],
                horizontal=True,
                label_visibility="collapsed"
            )

            # Save view mode to session state so Results page can detect it
            st.session_state.view_mode = view_mode

            if view_mode == "Auto (Sentiment)":
                st.caption("Generate views from sentiment analysis")

                sentiment_data = load_sentiment_data(tuple(st.session_state.selected_tickers))

                if not sentiment_data:
                    st.info("No sentiment data. Cannot generate views.")
                else:
                    sufficient_data = sum(
                        1 for data in sentiment_data.values()
                        if data.get('article_count', 0) >= 10
                    )

                    if sufficient_data == 0:
                        st.warning("Need at least 10 articles per ticker for reliable views.")
                    else:
                        st.caption(f"{sufficient_data}/{len(st.session_state.selected_tickers)} stocks with sufficient news articles data")

                        # Kappa slider for sentiment impact
                        kappa = st.slider(
                            "Sentiment Impact (κ)",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.15,
                            step=0.05,
                            help="Controls how much sentiment affects expected returns. "
                                 "view = equilibrium + kappa * sentiment * volatility. "
                                 "Higher values = stronger sentiment influence. "
                                 "Recommended range: 0.05-0.25 for moderate impact."
                        )

                        # Store in session state for Results page
                        st.session_state.sentiment_kappa = kappa

                        if st.button("Generate Views", use_container_width=True, type="primary"):
                            try:
                                from backend.bl_model import BlackLittermanModel

                                # Load historical data
                                data_loader = get_data_loader()
                                tickers = st.session_state.selected_tickers

                                # Get historical prices and market caps (5 years = 1260 trading days)
                                prices = data_loader.get_historical_prices(
                                    tickers,
                                    periods=1260
                                )
                                market_caps = data_loader.get_market_caps(tickers)

                                # Calculate returns and covariance with Ledoit-Wolf shrinkage
                                returns = calculate_returns(prices)
                                volatilities = np.std(returns, axis=0) * np.sqrt(252)  # Annualize volatilities
                                from backend.calculations import calculate_covariance_matrix
                                cov_matrix = calculate_covariance_matrix(returns, annualize=True, shrinkage=True)

                                # Create BL model to get equilibrium returns
                                bl_model = BlackLittermanModel(
                                    cov_matrix=cov_matrix,
                                    market_caps=market_caps,
                                    risk_aversion=st.session_state.get('risk_aversion', 2.5),
                                    tau=st.session_state.get('tau', 0.025)
                                )

                                # Get equilibrium returns (prior)
                                equilibrium_returns = bl_model.prior_returns

                                # Generate views using sentiment scaling
                                generator = BLViewGenerator(
                                    min_articles=10  # Restored to academic standard
                                )

                                result = generator.generate_views_ranked(
                                    sentiment_data,
                                    tickers,
                                    volatilities=volatilities,
                                    tau=st.session_state.get('tau', 0.025),
                                    equilibrium_returns=equilibrium_returns,
                                    kappa=kappa,  # Use slider value
                                    sentiment_threshold=0.3,  # Only strong sentiment
                                    normalize_sentiment=False  # Use raw sentiment [-1, 1] without z-score normalization
                                )

                                if result['P']:
                                    st.session_state.auto_generated_views = result
                                    st.rerun()
                                else:
                                    st.warning("No significant views generated (sentiment too moderate).")
                            except Exception as e:
                                st.error(f"Failed: {e}")
                                import traceback
                                traceback.print_exc()

                # Show generated views (preview only - will regenerate fresh on optimization)
                if 'auto_generated_views' in st.session_state and st.session_state.auto_generated_views:
                    views_result = st.session_state.auto_generated_views
                    st.markdown(f"**Preview: Generated Views** ({len(views_result['descriptions'])})")
                    st.caption("These views will be regenerated fresh when you run optimization")

                    for desc in views_result['descriptions']:
                        ticker = desc.split(':')[0]
                        st.caption(f"• {desc}")

                    if st.button("Clear Preview", use_container_width=True):
                        del st.session_state.auto_generated_views
                        st.rerun()

            else:
                # MANUAL MODE
                st.caption("Enter views manually")

                view_type = st.radio("Type", ["Absolute", "Relative"], horizontal=True, label_visibility="collapsed")
                view_type_key = 'absolute' if view_type == "Absolute" else 'relative'

                # Load prior returns for displaying current expected returns
                prior_returns_dict = load_prior_returns(tuple(st.session_state.selected_tickers))

                # Asset selection
                if view_type_key == 'absolute':
                    selected_asset = st.selectbox("Asset", st.session_state.selected_tickers)
                    view_assets = [selected_asset]

                    # Show current expected return for selected asset
                    if prior_returns_dict and selected_asset in prior_returns_dict:
                        current_return = prior_returns_dict[selected_asset] * 100  # Convert to percentage
                        st.caption(f"Current expected return: **{current_return:.2f}%**")
                else:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        asset_1 = st.selectbox("Outperform", st.session_state.selected_tickers, key="rel_1")
                    with col_b:
                        asset_2_options = [t for t in st.session_state.selected_tickers if t != asset_1]
                        asset_2 = st.selectbox("vs", asset_2_options, key="rel_2")
                    view_assets = [asset_1, asset_2]

                # Return and confidence - different labels for absolute vs relative
                col_ret, col_conf = st.columns(2)
                with col_ret:
                    if view_type_key == 'absolute':
                        expected_return = st.number_input(
                            "Expected Return (%)",
                            -50.0, 100.0, 10.0, 1.0,
                            help="Expected annual return for this asset"
                        )
                    else:
                        expected_return = st.number_input(
                            "Outperformance (%)",
                            -50.0, 100.0, 5.0, 1.0,
                            help=f"How much {asset_1} will outperform {asset_2} annually"
                        )
                with col_conf:
                    confidence_pct = st.slider("Confidence", 0, 100, 50, 5, format="%d%%")

                if st.button("Add View", use_container_width=True, type="primary"):
                    is_valid, error_msg = validate_view(view_assets, view_type_key)
                    if is_valid:
                        add_view(view_assets, view_type_key, expected_return, confidence_pct / 100)
                        st.rerun()
                    else:
                        st.error(error_msg)

                # Display manual views
                if st.session_state.views:
                    st.markdown(f"**Active Views** ({len(st.session_state.views)})")
                    for i, view in enumerate(st.session_state.views):
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            if view['type'] == 'absolute':
                                st.caption(f"{view['assets'][0]}: {view['expected_return']:.1f}% ({view['confidence']:.0%})")
                            else:
                                st.caption(f"{view['assets'][0]} > {view['assets'][1]}: {view['expected_return']:.1f}% ({view['confidence']:.0%})")
                        with col2:
                            if st.button("×", key=f"rm_{i}", use_container_width=True):
                                remove_view(i)
                                st.rerun()
                else:
                    st.info("No views. Market equilibrium will be used.")

        # COMPARTMENT 2: Model Parameters
        with st.container(border=True):
            st.markdown("### ⚙️ Model Parameters")

            # Quick presets - compact inline buttons (model parameters only, no constraints)
            st.caption("Presets:")
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            with preset_col1:
                if st.button("Conservative", use_container_width=True, key="preset_cons"):
                    st.session_state.tau = 0.01
                    st.session_state.risk_aversion = 3.5
                    st.rerun()
            with preset_col2:
                if st.button("Moderate", use_container_width=True, key="preset_mod"):
                    st.session_state.tau = 0.025
                    st.session_state.risk_aversion = 2.5
                    st.rerun()
            with preset_col3:
                if st.button("Aggressive", use_container_width=True, key="preset_agg"):
                    st.session_state.tau = 0.05
                    st.session_state.risk_aversion = 2.0
                    st.rerun()

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                tau = st.slider(
                    "Tau",
                    0.001, 0.100, st.session_state.get('tau', 0.025), 0.001,
                    format="%.3f",
                    help="Prior uncertainty. Higher = more weight to views"
                )
                st.session_state.tau = tau

                # Fixed 5-year historical period (1260 trading days)
                st.session_state.periods = 1260

            with col2:
                risk_aversion = st.slider(
                    "Risk Aversion",
                    0.5, 5.0, st.session_state.get('risk_aversion', 2.5), 0.1,
                    help="Higher = safer portfolio"
                )
                st.session_state.risk_aversion = risk_aversion

                risk_free_rate = st.number_input(
                    "Risk-Free Rate (%)",
                    0.0, 10.0, 2.0, 0.1
                )
                st.session_state.risk_free_rate = risk_free_rate / 100

    # ========== RIGHT COLUMN: Sentiment & Constraints ==========
    with right_col:
        # COMPARTMENT 4: Sentiment Data
        with st.container(border=True):
            st.markdown("### Sentiment")
            st.caption("Recent news sentiment (7 days)")

            try:
                sentiment_data = load_sentiment_data(tuple(st.session_state.selected_tickers))

                if sentiment_data:
                    display_sentiment_table(sentiment_data, show_distribution=False)
                else:
                    st.info("No sentiment data available for selected tickers.")
            except Exception as e:
                st.error(f"Error loading sentiment: {e}")

        # COMPARTMENT 3: Portfolio Constraints
        with st.container(border=True):
            st.markdown("### 🔒 Portfolio Constraints")

            # Calculate feasible bounds based on number of stocks
            n_stocks = len(st.session_state.selected_tickers)
            max_possible_min = 100.0 / n_stocks if n_stocks > 0 else 100.0

            # Get sectors for selected tickers
            data_loader = get_data_loader()
            sectors = data_loader.get_sectors(st.session_state.selected_tickers)
            unique_sectors = sorted(set(sectors))
            n_sectors = len(unique_sectors)

            # Count stocks per sector
            sector_counts = {}
            for sector in sectors:
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            st.caption("**Stock Constraints**")
            col1, col2 = st.columns(2)

            with col1:
                max_weight = st.number_input(
                    "Max Weight per Stock (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.get('max_weight', 0.0) * 100 if st.session_state.get('max_weight') else 0.0,
                    step=1.0,
                    help="Maximum allocation per stock (0 = no limit)"
                )

            with col2:
                # Calculate safe value that respects max_possible_min to avoid Streamlit error
                saved_min = st.session_state.get('min_weight', 0.0) * 100 if st.session_state.get('min_weight') else 0.0
                safe_min_value = min(saved_min, max_possible_min)  # Cap at max_possible_min

                min_weight = st.number_input(
                    "Min Weight per Stock (%)",
                    min_value=0.0,
                    max_value=max_possible_min,
                    value=safe_min_value,
                    step=0.1,
                    help=f"Minimum allocation per stock (max {max_possible_min:.1f}% for {n_stocks} stocks, 0 = no limit)"
                )

            st.caption(f"**Sector Constraints** ({n_sectors} sectors)")

            # Per-sector constraints in expandable section
            with st.expander("Set limits per sector", expanded=False):
                st.caption("Set specific min/max for each sector (0 = use global default)")

                # Initialize per-sector constraints in session state if not exists
                if 'sector_constraints' not in st.session_state:
                    st.session_state.sector_constraints = {}

                # Column headers
                header_col_name, header_col_min, header_col_max = st.columns([2, 1, 1])
                with header_col_name:
                    st.markdown("**Sector**")
                with header_col_min:
                    st.markdown("**Min %**")
                with header_col_max:
                    st.markdown("**Max %**")

                # Create inputs for each sector
                per_sector_constraints = {}
                for sector in unique_sectors:
                    stocks_count = sector_counts.get(sector, 0)
                    saved_sector_data = st.session_state.sector_constraints.get(sector, {})

                    col_name, col_min, col_max = st.columns([2, 1, 1])
                    with col_name:
                        st.markdown(f"**{sector}** ({stocks_count})")
                    with col_min:
                        # Default min=0 means "no minimum required"
                        saved_min = saved_sector_data.get('min')
                        default_min = saved_min * 100 if saved_min is not None else 0.0
                        sector_min = st.number_input(
                            "Min %",
                            min_value=0.0,
                            max_value=100.0,
                            value=default_min,
                            step=1.0,
                            key=f"sector_min_{sector}",
                            label_visibility="collapsed",
                            help="0% = no minimum required"
                        )
                    with col_max:
                        # Default max=100 means "no limit", max=0 means "exclude sector"
                        saved_max = saved_sector_data.get('max')
                        default_max = saved_max * 100 if saved_max is not None else 100.0
                        sector_max = st.number_input(
                            "Max %",
                            min_value=0.0,
                            max_value=100.0,
                            value=default_max,
                            step=5.0,
                            key=f"sector_max_{sector}",
                            label_visibility="collapsed",
                            help="100% = no limit, 0% = exclude sector"
                        )

                    # Add constraint if max < 100 (limit enforced) or min > 0 (minimum required)
                    if sector_max < 100.0 or sector_min > 0:
                        per_sector_constraints[sector] = {
                            'max': sector_max / 100 if sector_max < 100.0 else None,
                            'min': sector_min / 100 if sector_min > 0 else None
                        }

            # Validate all constraints
            validation_errors = []

            # Stock constraint validation
            if max_weight > 0 and min_weight > 0:
                if min_weight > max_weight:
                    validation_errors.append("Stock: Minimum weight cannot exceed maximum weight")

            if min_weight > 0 and n_stocks > 0:
                total_min = min_weight * n_stocks
                if total_min > 100.0:
                    validation_errors.append(f"Stock: Min weight too high ({n_stocks} × {min_weight:.1f}% = {total_min:.1f}% > 100%)")

            if max_weight > 0 and n_stocks > 0:
                if max_weight < (100.0 / n_stocks):
                    validation_errors.append(f"Stock: Max weight too low (need ≥ {100.0/n_stocks:.1f}%)")

            # Per-sector constraint validation
            total_sector_min = 0.0
            for sector, limits in per_sector_constraints.items():
                sector_max = limits.get('max')
                sector_min = limits.get('min')
                stocks_in_sector = sector_counts.get(sector, 0)

                if sector_max and sector_min and sector_min > sector_max:
                    validation_errors.append(f"{sector}: Min ({sector_min*100:.0f}%) > Max ({sector_max*100:.0f}%)")

                if sector_min:
                    total_sector_min += sector_min

                # Check if per-stock max allows reaching sector min
                if sector_min and max_weight > 0:
                    max_achievable = stocks_in_sector * max_weight
                    if max_achievable < sector_min:
                        validation_errors.append(
                            f"{sector}: {stocks_in_sector} stocks × {max_weight*100:.1f}% = {max_achievable*100:.1f}% < {sector_min*100:.0f}% min"
                        )

            # Check total min sectors doesn't exceed 100%
            if total_sector_min > 1.0:
                validation_errors.append(f"Sector mins total {total_sector_min*100:.1f}% > 100%")

            # Apply constraints only if valid
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.session_state.max_weight = None
                st.session_state.min_weight = None
                st.session_state.sector_constraints = {}
            else:
                # Stock constraints
                st.session_state.max_weight = max_weight / 100 if max_weight > 0 else None
                st.session_state.min_weight = min_weight / 100 if min_weight > 0 else None

                # Per-sector constraints
                st.session_state.sector_constraints = per_sector_constraints

                # Store sector mapping for optimization
                st.session_state.ticker_sectors = dict(zip(st.session_state.selected_tickers, sectors))

                # Show active constraints summary
                active_constraints = []
                if st.session_state.get('max_weight'):
                    active_constraints.append(f"Stock max: {st.session_state.max_weight*100:.1f}%")
                if st.session_state.get('min_weight'):
                    active_constraints.append(f"Stock min: {st.session_state.min_weight*100:.1f}%")

                # Count per-sector constraints
                n_sector_limits = len(per_sector_constraints)
                if n_sector_limits > 0:
                    active_constraints.append(f"{n_sector_limits} sector limits")

                if active_constraints:
                    st.caption(f"✓ Active: {', '.join(active_constraints)}")
                else:
                    st.caption("No constraints applied")

    # Sentiment Analysis Section - Full Width Below
    st.markdown("---")

    # Quick actions at bottom
    col1, col2, _ = st.columns(3)
    with col1:
        if st.session_state.views:
            if st.button("Clear Views", use_container_width=True):
                clear_all_views()
                st.rerun()
    with col2:
        if st.button("Reset Defaults", use_container_width=True):
            st.session_state.tau = 0.025
            st.session_state.risk_aversion = 2.5
            st.session_state.risk_free_rate = 0.02
            st.session_state.max_weight = None
            st.session_state.min_weight = None
            st.session_state.sector_constraints = {}
            st.rerun()

    # Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Stock Selection", use_container_width=True):
            st.switch_page("pages/1_Stock_Selection.py")
    with col2:
        if st.button("Continue to Results →", use_container_width=True, type="primary"):
            st.switch_page("pages/3_Results.py")


if __name__ == "__main__":
    main()
