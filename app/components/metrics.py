"""Reusable metrics and table components"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


def display_metric_cards(metrics: List[Dict[str, any]], columns: int = 4):
    """Display metrics in card format"""
    cols = st.columns(columns)

    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            delta = metric.get('delta', None)
            help_text = metric.get('help', None)
            # delta_color: "normal" (green=up), "inverse" (green=down), "off"
            delta_color = metric.get('delta_color', 'normal')

            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )


def display_portfolio_summary_table(
    tickers: List[str],
    weights: np.ndarray,
    expected_returns: np.ndarray,
    sectors: Optional[List[str]] = None,
    prices: Optional[np.ndarray] = None,
    top_n: int = 20,
    portfolio_amount: Optional[float] = None
) -> pd.DataFrame:
    """Create and display portfolio summary table"""
    data = {
        'Ticker': tickers,
        'Weight (%)': weights * 100,
        'Expected Return (%)': expected_returns * 100
    }

    # Add dollar allocation if portfolio amount provided
    if portfolio_amount is not None:
        data['Allocation ($)'] = weights * portfolio_amount
        # Calculate number of shares if prices provided
        if prices is not None:
            data['Shares'] = (weights * portfolio_amount) / prices
            data['Price ($)'] = prices
    elif prices is not None:
        data['Price ($)'] = prices

    if sectors is not None:
        data['Sector'] = sectors

    df = pd.DataFrame(data)

    df = df.sort_values('Weight (%)', ascending=False)

    if len(df) > top_n:
        df_display = df.head(top_n).copy()
        others_row = {
            'Ticker': 'Others',
            'Weight (%)': df.iloc[top_n:]['Weight (%)'].sum(),
            'Expected Return (%)': None,
        }
        # Add allocation sum if present
        if 'Allocation ($)' in df.columns:
            others_row['Allocation ($)'] = df.iloc[top_n:]['Allocation ($)'].sum()
        if 'Shares' in df.columns:
            others_row['Shares'] = None
        if 'Price ($)' in df.columns:
            others_row['Price ($)'] = None
        if sectors is not None:
            others_row['Sector'] = ''

        df_display.loc['Others'] = others_row
    else:
        df_display = df.copy()

    def safe_format_number(value, fmt):
        """Format number or return empty string if not numeric"""
        if pd.isna(value) or value is None:
            return ''
        try:
            return fmt.format(value)
        except (ValueError, TypeError):
            return str(value)

    formatted_df = df_display.copy()

    # Format Weight column
    formatted_df['Weight (%)'] = formatted_df['Weight (%)'].apply(
        lambda x: safe_format_number(x, '{:.2f}')
    )

    # Format Expected Return column
    formatted_df['Expected Return (%)'] = formatted_df['Expected Return (%)'].apply(
        lambda x: safe_format_number(x, '{:.2f}')
    )

    # Format Allocation column if present
    if 'Allocation ($)' in formatted_df.columns:
        formatted_df['Allocation ($)'] = formatted_df['Allocation ($)'].apply(
            lambda x: safe_format_number(x, '${:,.0f}')
        )

    # Format Shares column if present
    if 'Shares' in formatted_df.columns:
        formatted_df['Shares'] = formatted_df['Shares'].apply(
            lambda x: safe_format_number(x, '{:.0f}')
        )

    # Format Price column if present
    if 'Price ($)' in formatted_df.columns:
        formatted_df['Price ($)'] = formatted_df['Price ($)'].apply(
            lambda x: safe_format_number(x, '${:.2f}')
        )

    st.dataframe(
        formatted_df,
        use_container_width=True,
        height=min(len(formatted_df) * 35 + 38, 600)
    )

    return df


def display_comparison_table(
    tickers: List[str],
    bl_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    bl_returns: Optional[np.ndarray] = None,
    prior_returns: Optional[np.ndarray] = None,
    top_n: int = 20
) -> pd.DataFrame:
    """Create comparison table between BL and benchmark"""
    data = {
        'Ticker': tickers,
        'BL Weight (%)': bl_weights * 100,
        'Benchmark Weight (%)': benchmark_weights * 100,
        'Difference (%)': (bl_weights - benchmark_weights) * 100
    }

    if bl_returns is not None:
        data['BL Return (%)'] = bl_returns * 100

    if prior_returns is not None:
        data['Prior Return (%)'] = prior_returns * 100

    df = pd.DataFrame(data)

    df['abs_diff'] = np.abs(df['Difference (%)'])
    df = df.sort_values('abs_diff', ascending=False).drop('abs_diff', axis=1)

    df_display = df.head(top_n)

    def color_difference(val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'

    format_dict = {
        'BL Weight (%)': '{:.2f}',
        'Benchmark Weight (%)': '{:.2f}',
        'Difference (%)': '{:+.2f}'
    }

    if bl_returns is not None:
        format_dict['BL Return (%)'] = '{:.2f}'

    if prior_returns is not None:
        format_dict['Prior Return (%)'] = '{:.2f}'

    styled_df = df_display.style.format(format_dict).applymap(
        color_difference,
        subset=['Difference (%)']
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(len(df_display) * 35 + 38, 600)
    )

    return df


def display_risk_metrics_table(metrics: Dict[str, any]):
    """Display risk metrics table"""
    data = {
        'Metric': [],
        'Value': []
    }

    metric_configs = {
        'return': ('Expected Return', '{:.2%}'),
        'annualized_return': ('Annualized Return', '{:.2%}'),
        'volatility': ('Volatility', '{:.2%}'),
        'annualized_volatility': ('Annualized Volatility', '{:.2%}'),
        'sharpe_ratio': ('Sharpe Ratio', '{:.3f}'),
        'sortino_ratio': ('Sortino Ratio', '{:.3f}'),
        'max_drawdown': ('Maximum Drawdown', '{:.2%}'),
        'var_95': ('Value at Risk (95%)', '{:.2%}'),
        'cvar_95': ('Conditional VaR (95%)', '{:.2%}')
    }

    for key, (label, fmt) in metric_configs.items():
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, float)):
                data['Metric'].append(label)
                data['Value'].append(fmt.format(value))

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True, hide_index=True)


def display_sector_allocation_table(
    tickers: List[str],
    weights: np.ndarray,
    sectors: List[str]
) -> pd.DataFrame:
    """Create sector allocation summary"""
    df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights,
        'Sector': sectors
    })

    sector_summary = df.groupby('Sector')['Weight'].sum().reset_index()
    sector_summary.columns = ['Sector', 'Weight (%)']
    sector_summary['Weight (%)'] = sector_summary['Weight (%)'] * 100
    sector_summary = sector_summary.sort_values('Weight (%)', ascending=False)

    st.dataframe(
        sector_summary.style.format({'Weight (%)': '{:.2f}'}),
        use_container_width=True,
        hide_index=True
    )

    return sector_summary


def display_views_table(views: List[Dict[str, any]]):
    """Display active views table"""
    if not views:
        st.info("No active views. Add views to influence the portfolio allocation.")
        return

    data = {
        'View': [],
        'Type': [],
        'Expected Return (%)': [],
        'Confidence': []
    }

    for i, view in enumerate(views, 1):
        view_type = view.get('type', 'absolute')

        if view_type == 'absolute':
            view_desc = f"{view['assets'][0]}"
        else:  # relative
            view_desc = f"{view['assets'][0]} vs {view['assets'][1]}"

        data['View'].append(view_desc)
        data['Type'].append(view_type.capitalize())
        data['Expected Return (%)'].append(view['expected_return'])
        data['Confidence'].append(view.get('confidence', 0.5))

    df = pd.DataFrame(data)

    st.dataframe(
        df.style.format({
            'Expected Return (%)': '{:.2f}',
            'Confidence': '{:.0%}'
        }),
        use_container_width=True,
        hide_index=True
    )


def create_downloadable_table(
    df: pd.DataFrame,
    filename: str = "portfolio_data.csv"
) -> bytes:
    """Convert DataFrame to CSV"""
    return df.to_csv(index=False).encode('utf-8')


def display_info_box(title: str, content: str, icon: str = "ℹ️"):
    """Display informational box"""
    st.markdown(f"""
    <div style="
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    ">
        <h4 style="margin: 0 0 0.5rem 0; color: #1565c0;">
            {icon} {title}
        </h4>
        <p style="margin: 0; color: #424242;">
            {content}
        </p>
    </div>
    """, unsafe_allow_html=True)
