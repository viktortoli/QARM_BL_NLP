"""
Stock Selection Page
Select S&P 500 constituents for the portfolio
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.helpers import load_css, init_session_state, render_sidebar_navigation
from backend.data_loader import get_data_loader

st.set_page_config(
    page_title="Stock Selection",
    page_icon="📈",
    layout="wide"
)

load_css()
init_session_state()
render_sidebar_navigation(show_sector_returns=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_sp500_data():
    """Load S&P 500 constituents with caching"""
    data_loader = get_data_loader()
    return data_loader.get_sp500_constituents()


@st.cache_data(ttl=3600, show_spinner=False)
def get_sectors_list():
    """Get available sectors with caching"""
    data_loader = get_data_loader()
    return ['All Sectors'] + data_loader.get_available_sectors()


def main():
    # Compact page title
    st.markdown("## 📈 Stock Selection")

    # Load data early (cached, no spinner needed after first load)
    df = load_sp500_data()
    all_df = df.copy()  # Keep original for portfolio display

    # Two column layout: Table on left, Controls on right
    table_col, controls_col = st.columns([2.5, 1], gap="medium")

    # LEFT SIDE: Stock Table
    with table_col:
        # Compact filter controls above table
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            sectors = get_sectors_list()
            selected_sector = st.selectbox(
                "Filter by Sector",
                sectors,
                key="sector_filter"
            )

        with filter_col2:
            search_query = st.text_input(
                "Search Stocks",
                "",
                placeholder="Type ticker or company name..."
            )

        # Apply filters
        if selected_sector != 'All Sectors':
            df = df[df['sector'] == selected_sector]

        if search_query:
            mask = (
                df['ticker'].str.contains(search_query, case=False) |
                df['name'].str.contains(search_query, case=False)
            )
            df = df[mask]

        st.markdown(f"**Available Stocks** ({len(df)})")

        # Main stock table - height matches Add Stocks controls
        st.dataframe(
            df,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "name": st.column_config.TextColumn("Company", width="large"),
                "sector": st.column_config.TextColumn("Sector", width="medium"),
                "market_cap": st.column_config.NumberColumn("Cap ($B)", format="%.1f", width="small"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f", width="small")
            },
            hide_index=True,
            use_container_width=True,
            height=230,
            on_select="rerun",
            selection_mode="multi-row",
            key="stock_table"
        )

    # RIGHT SIDE: Selection Controls
    with controls_col:
        # ZONE 1: Portfolio Settings (compact)
        with st.container(border=True):
            st.markdown("**⚙️ Portfolio Amount**")
            portfolio_amount = st.number_input(
                "Portfolio Amount ($)",
                value=st.session_state.portfolio_amount,
                step=10000.0,
                format="%.0f",
                label_visibility="collapsed",
                key="portfolio_amount_input",
                help=""
            )
            st.session_state.portfolio_amount = portfolio_amount

        # ZONE 2: Quick Add Options (compact)
        with st.container(border=True):
            st.markdown("**➕ Add Stocks**")
            st.caption("Add top N stocks by market cap")

            # Top N
            quick_col1, quick_col2 = st.columns([1.5, 1])
            with quick_col1:
                top_n = st.number_input(
                    "Top N",
                    5, 100, 10,
                    label_visibility="collapsed",
                    key="top_n_input"
                )
            with quick_col2:
                if st.button("Add", use_container_width=True, type="secondary", key="add_top_btn"):
                    top_tickers = df.head(top_n)['ticker'].tolist()
                    st.session_state.selected_tickers = list(set(
                        st.session_state.selected_tickers + top_tickers
                    ))
                    st.rerun()

            # Add selected
            if st.session_state.get("stock_table", {}).get("selection", {}).get("rows"):
                num_selected = len(st.session_state.stock_table["selection"]["rows"])
                if st.button(f"Add {num_selected} Selected", use_container_width=True, type="primary", key="add_selected_btn"):
                    selected_rows = st.session_state.stock_table["selection"]["rows"]
                    selected_tickers = df.iloc[selected_rows]['ticker'].tolist()
                    st.session_state.selected_tickers = list(set(
                        st.session_state.selected_tickers + selected_tickers
                    ))
                    st.rerun()
            else:
                st.button("Add Selected", use_container_width=True, disabled=True, key="add_selected_disabled")

    # BOTTOM: Your Portfolio (full width)
    st.markdown("---")
    st.markdown("### 📊 Your Portfolio")

    if st.session_state.selected_tickers:
        selected_df = all_df[all_df['ticker'].isin(st.session_state.selected_tickers)]

        # Stats row - compact metrics across the width
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns([1, 1, 2, 1, 1])

        with stat_col1:
            st.metric("Stocks", len(st.session_state.selected_tickers))
        with stat_col2:
            st.metric("Sectors", len(selected_df['sector'].unique()))
        with stat_col3:
            # Remove stock dropdown
            ticker_to_remove = st.selectbox(
                "Remove Stock",
                st.session_state.selected_tickers,
                key="remove_select"
            )
        with stat_col4:
            st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Remove", use_container_width=True, key="remove_btn"):
                st.session_state.selected_tickers.remove(ticker_to_remove)
                st.rerun()
        with stat_col5:
            st.markdown("<div style='margin-top: 22px;'></div>", unsafe_allow_html=True)
            if st.button("Clear All", use_container_width=True, type="secondary", key="clear_btn_portfolio"):
                st.session_state.selected_tickers = []
                st.rerun()

        # Selected stocks list - full width with Continue button below
        st.markdown("**Selected Stocks**")
        st.dataframe(
            selected_df[['ticker', 'name', 'sector', 'market_cap', 'price']],
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "name": st.column_config.TextColumn("Company", width="medium"),
                "sector": st.column_config.TextColumn("Sector", width="small"),
                "market_cap": st.column_config.NumberColumn("Cap ($B)", format="%.1f", width="small"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f", width="small")
            },
            hide_index=True,
            use_container_width=True,
            height=150
        )

        # Continue button at bottom
        _, col_continue = st.columns([4, 1])
        with col_continue:
            if st.button("Continue →", use_container_width=True, type="primary", key="continue_btn"):
                st.switch_page("pages/2_Views_Configuration.py")

    else:
        st.info("💡 No stocks selected. Select rows from the table above and click **Add Selected**, or use **Add Top N** for quick selection.")


if __name__ == "__main__":
    main()
