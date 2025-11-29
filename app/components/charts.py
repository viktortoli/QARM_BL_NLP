"""Reusable chart components using Plotly"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Optional


# Dark mode color scheme
COLORS = {
    'primary': '#58a6ff',
    'secondary': '#79c0ff',
    'success': '#3fb950',
    'danger': '#f85149',
    'warning': '#d29922',
    'info': '#58a6ff',
    'dark': '#0f1419',
    'light': '#e7e9ea',
    'gradient': ['#388bfd', '#58a6ff', '#79c0ff', '#a5d8ff'],
}


def create_portfolio_pie_chart(
    tickers: List[str],
    weights: np.ndarray,
    title: str = "Portfolio Allocation",
    max_slices: int = 15,
    min_weight_pct: float = 1.0
) -> go.Figure:
    """Create portfolio allocation pie chart"""
    threshold = 0.001
    mask = weights > threshold
    filtered_tickers = [t for t, m in zip(tickers, mask) if m]
    filtered_weights = weights[mask]

    sorted_indices = np.argsort(filtered_weights)[::-1]
    sorted_tickers = [filtered_tickers[i] for i in sorted_indices]
    sorted_weights = filtered_weights[sorted_indices]

    display_tickers = []
    display_weights = []
    others_weight = 0.0

    for i, (ticker, weight) in enumerate(zip(sorted_tickers, sorted_weights)):
        # Always show top assets up to max_slices - 1 (leave room for "Others")
        if i < max_slices - 1:
            display_tickers.append(ticker)
            display_weights.append(weight)
        else:
            others_weight += weight

    if others_weight > 0.001:
        display_tickers.append("Others")
        display_weights.append(others_weight)

    display_weights = np.array(display_weights)

    fig = go.Figure(data=[go.Pie(
        labels=display_tickers,
        values=display_weights * 100,  # Convert to percentage
        hole=0.4,
        marker=dict(
            colors=px.colors.sequential.Blues_r,
            line=dict(color='white', width=2)
        ),
        textposition='outside',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Weight: %{value:.2f}%<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        height=500,
        template='plotly_dark'
    )

    return fig


def create_weights_comparison_bar_chart(
    tickers: List[str],
    bl_weights: np.ndarray,
    benchmark_weights: np.ndarray,
    title: str = "Top Holdings: BL vs Benchmark",
    benchmark_label: str = "Benchmark",
    max_assets: int = 20
) -> go.Figure:
    """Create BL vs benchmark comparison chart"""
    df = pd.DataFrame({
        'Ticker': tickers,
        'Black-Litterman': bl_weights * 100,
        benchmark_label: benchmark_weights * 100
    })

    df = df.sort_values('Black-Litterman', ascending=False)

    total_assets = len(df)
    df = df.head(max_assets)

    display_title = title
    if total_assets > max_assets:
        display_title = f"{title} (Top {max_assets} of {total_assets})"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Black-Litterman',
        x=df['Ticker'],
        y=df['Black-Litterman'],
        marker_color=COLORS['primary'],
        hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name=benchmark_label,
        x=df['Ticker'],
        y=df[benchmark_label],
        marker_color=COLORS['secondary'],
        hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=display_title, x=0.5, xanchor='center'),
        xaxis_title="Ticker",
        yaxis_title="Weight (%)",
        barmode='group',
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_efficient_frontier(
    frontier_returns: np.ndarray,
    frontier_volatilities: np.ndarray,
    sharpe_ratios: np.ndarray,
    portfolio_return: float,
    portfolio_volatility: float,
    market_return: Optional[float] = None,
    market_volatility: Optional[float] = None,
    title: str = "Efficient Frontier"
) -> go.Figure:
    """Create efficient frontier visualization"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frontier_volatilities * 100,
        y=frontier_returns * 100,
        mode='markers',
        marker=dict(
            size=8,
            color=sharpe_ratios,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe<br>Ratio"),
            line=dict(width=0.5, color='white')
        ),
        name='Random Portfolios',
        hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[portfolio_volatility * 100],
        y=[portfolio_return * 100],
        mode='markers',
        marker=dict(
            size=20,
            color=COLORS['danger'],
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name='BL Portfolio',
        hovertemplate='<b>Black-Litterman</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))

    if market_return is not None and market_volatility is not None:
        fig.add_trace(go.Scatter(
            x=[market_volatility * 100],
            y=[market_return * 100],
            mode='markers',
            marker=dict(
                size=15,
                color=COLORS['success'],
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            name='Market Portfolio',
            hovertemplate='<b>Market</b><br>Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        hovermode='closest',
        height=600,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix"
) -> go.Figure:
    """Create correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="",
        yaxis_title="",
        height=600,
        template='plotly_dark'
    )

    return fig


def create_risk_contribution_chart(
    tickers: List[str],
    risk_contributions: np.ndarray,
    title: str = "Risk Contribution by Asset"
) -> go.Figure:
    """Create risk contribution chart"""
    df = pd.DataFrame({
        'Ticker': tickers,
        'Risk Contribution': risk_contributions * 100
    })

    df = df.sort_values('Risk Contribution', ascending=False).head(20)

    fig = go.Figure(data=[go.Bar(
        x=df['Ticker'],
        y=df['Risk Contribution'],
        marker_color=COLORS['gradient'],
        hovertemplate='<b>%{x}</b><br>Risk Contribution: %{y:.2f}%<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Ticker",
        yaxis_title="Risk Contribution (%)",
        height=400,
        template='plotly_dark'
    )

    return fig


def create_returns_comparison_chart(
    tickers: List[str],
    prior_returns: np.ndarray,
    posterior_returns: np.ndarray,
    title: str = "Expected Returns: Prior vs Posterior"
) -> go.Figure:
    """Create returns comparison chart"""
    df = pd.DataFrame({
        'Ticker': tickers,
        'Prior (CAPM)': prior_returns * 100,
        'Posterior (BL)': posterior_returns * 100,
        'Change': (posterior_returns - prior_returns) * 100
    })

    df = df.iloc[np.argsort(np.abs(df['Change'].values))[::-1]].head(20)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Prior (CAPM)',
        x=df['Ticker'],
        y=df['Prior (CAPM)'],
        marker_color=COLORS['light'],
        marker_line_color=COLORS['dark'],
        marker_line_width=1.5,
        hovertemplate='<b>%{x}</b><br>Prior: %{y:.2f}%<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        name='Posterior (BL)',
        x=df['Ticker'],
        y=df['Posterior (BL)'],
        marker_color=COLORS['primary'],
        hovertemplate='<b>%{x}</b><br>Posterior: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Ticker",
        yaxis_title="Expected Return (%)",
        barmode='group',
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
