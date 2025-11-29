"""Helper utility functions"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Any, List, Dict


def load_css():
    """Load custom CSS"""
    css_file = Path(__file__).parent / "styles.css"

    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sector_returns():
    """Render sector returns"""
    try:
        from database.db_manager import DatabaseManager
        
        # Cache the sector returns to avoid repeated DB calls
        @st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
        def get_cached_sector_returns():
            db = DatabaseManager()
            return db.get_sector_returns()
        
        sector_returns = get_cached_sector_returns()
        
        if sector_returns.empty:
            st.caption("Sector data not available")
            return
        
        # Modern Bloomberg-style CSS
        st.markdown("""
        <style>
            .sector-header {
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                font-size: 0.9rem;
                font-weight: 600;
                color: #e7e9ea;
                margin-bottom: 0.75rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #30363d;
            }
            .sector-header-mtd {
                font-size: 0.65rem;
                font-weight: 400;
                color: #6e7681;
            }
            .sector-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 6px 8px;
                margin: 3px 0;
                border-radius: 6px;
                background: linear-gradient(135deg, #1c1f26 0%, #22252d 100%);
                border-left: 3px solid;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .sector-row:hover {
                transform: translateX(3px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
            .sector-row.positive {
                border-left-color: #00c853;
            }
            .sector-row.negative {
                border-left-color: #ff1744;
            }
            .sector-row.neutral {
                border-left-color: #6e7681;
            }
            .sector-name {
                font-size: 0.75rem;
                color: #e7e9ea;
                font-weight: 500;
                flex: 1;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                margin-right: 8px;
            }
            .sector-return {
                font-size: 0.7rem;
                font-weight: 700;
                font-family: 'SF Mono', 'Consolas', monospace;
                padding: 2px 6px;
                border-radius: 4px;
                white-space: nowrap;
                min-width: 70px;
                text-align: right;
            }
            .sector-return.positive {
                color: #00c853;
                background: rgba(0, 200, 83, 0.15);
            }
            .sector-return.negative {
                color: #ff1744;
                background: rgba(255, 23, 68, 0.15);
            }
            .sector-return.neutral {
                color: #6e7681;
                background: rgba(110, 118, 129, 0.15);
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sector-header"><span>Sector Performance</span><span class="sector-header-mtd">MTD Returns</span></div>', unsafe_allow_html=True)
        
        # Create Bloomberg-style display
        for _, row in sector_returns.iterrows():
            sector = row['sector_name']
            ret = row['return_30d_pct']
            
            if ret is None:
                continue
            
            # Determine styling class
            if ret > 0:
                css_class = "positive"
                arrow = "▲"
                sign = "+"
            elif ret < 0:
                css_class = "negative"
                arrow = "▼"
                sign = ""
            else:
                css_class = "neutral"
                arrow = "–"
                sign = ""
            
            st.markdown(f"""
            <div class="sector-row {css_class}">
                <span class="sector-name">{sector}</span>
                <span class="sector-return {css_class}">{arrow} {sign}{ret:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.caption(f"Sector data unavailable")


def render_sidebar_navigation(show_sector_returns: bool = False, show_news_feed: bool = False):
    """Render sidebar navigation"""
    with st.sidebar:
        # QARM Logo at top, centered
        import base64
        from pathlib import Path
        
        logo_path = Path(__file__).parent.parent / "imports" / "QARMII_Logo.png"
        if logo_path.exists():
            with open(logo_path, "rb") as img_file:
                logo_base64 = base64.b64encode(img_file.read()).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center; margin-bottom: 1rem;">'
                f'<img src="data:image/png;base64,{logo_base64}" alt="QARM II" style="width: 100px;">'
                f'</div>',
                unsafe_allow_html=True
            )
        
        if st.button("Stock Selection", use_container_width=True, key="nav_stock"):
            st.switch_page("pages/1_Stock_Selection.py")
        
        if st.button("Views Configuration", use_container_width=True, key="nav_views"):
            st.switch_page("pages/2_Views_Configuration.py")
        
        if st.button("Results", use_container_width=True, key="nav_results"):
            st.switch_page("pages/3_Results.py")
        
        st.markdown("---")
        
        # Show sector returns if requested
        if show_sector_returns:
            render_sector_returns()
        
        # Show rolling news feed if requested
        if show_news_feed:
            render_rolling_news_sidebar()


@st.cache_data(ttl=300, show_spinner=False)
def get_recent_headlines(limit: int = 30):
    """Fetch recent news headlines"""
    try:
        import os
        import psycopg2
        
        # Get DATABASE_URL from environment or streamlit secrets
        postgres_url = os.getenv('DATABASE_URL')
        if not postgres_url:
            try:
                if hasattr(st, 'secrets') and 'DATABASE_URL' in st.secrets:
                    postgres_url = st.secrets['DATABASE_URL']
            except:
                pass
        
        if not postgres_url:
            print("[ERROR] get_recent_headlines: DATABASE_URL not set")
            return []
        
        # Fix postgres:// prefix for psycopg2
        if postgres_url.startswith('postgres://'):
            postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)
        
        conn = psycopg2.connect(postgres_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                na.ticker,
                na.headline,
                ss.sentiment_score,
                na.published_at,
                na.source
            FROM news_articles na
            JOIN sentiment_scores ss ON na.id = ss.news_article_id
            WHERE na.published_at >= NOW() - INTERVAL '7 days'
            ORDER BY na.published_at DESC
            LIMIT %s
        """, (limit,))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [
            {
                'ticker': row[0],
                'title': row[1] or '',
                'sentiment': float(row[2]) if row[2] else 0,
                'date': row[3].strftime('%b %d, %H:%M') if row[3] else '',
                'source': row[4] or ''
            }
            for row in rows
        ]
    except Exception as e:
        print(f"[ERROR] get_recent_headlines: {e}")
        import traceback
        traceback.print_exc()
        return []


def render_rolling_news_sidebar():
    """Render rolling news feed"""
    import streamlit.components.v1 as components
    
    headlines = get_recent_headlines(limit=100)
    
    if not headlines:
        st.caption("No recent news available")
        return
    
    st.markdown("### 📰 Live News Feed")
    
    # Build headline cards
    cards_html = ""
    for h in headlines:
        sentiment = h.get('sentiment', 0)
        if sentiment > 0.1:
            border_color = '#00D4AA'
            bg_color = 'rgba(0, 212, 170, 0.1)'
            sentiment_icon = '▲'
        elif sentiment < -0.1:
            border_color = '#FF6B6B'
            bg_color = 'rgba(255, 107, 107, 0.1)'
            sentiment_icon = '▼'
        else:
            border_color = '#666'
            bg_color = 'rgba(100, 100, 100, 0.1)'
            sentiment_icon = '●'
        
        ticker = h.get('ticker', '')
        title = h.get('title', '')[:70] + ('...' if len(h.get('title', '')) > 70 else '')
        date = h.get('date', '')
        source = h.get('source', '')[:12]
        
        cards_html += f'''
            <div style="
                border-left: 3px solid {border_color};
                background: {bg_color};
                padding: 8px 10px;
                margin-bottom: 6px;
                border-radius: 6px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                    <span style="
                        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                        color: white;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 10px;
                        font-weight: 700;
                    ">{ticker}</span>
                    <span style="color: {border_color}; font-size: 11px; font-weight: 600;">
                        {sentiment_icon} {sentiment:+.2f}
                    </span>
                </div>
                <div style="
                    color: #e0e0e0;
                    font-size: 11px;
                    line-height: 1.3;
                    margin: 4px 0;
                ">{title}</div>
                <div style="
                    color: #888;
                    font-size: 9px;
                    display: flex;
                    justify-content: space-between;
                ">
                    <span>{source}</span>
                    <span>{date}</span>
                </div>
            </div>
        '''
    
    # Duplicate for seamless scroll
    all_cards = cards_html + cards_html
    
    # Full HTML with embedded CSS
    full_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .news-container {{
                height: 550px;
                overflow: hidden;
                background: linear-gradient(180deg, #0e1117 0%, #1a1a2e 100%);
                border-radius: 8px;
                padding: 8px;
                position: relative;
            }}
            .news-container::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 20px;
                background: linear-gradient(180deg, #0e1117 0%, transparent 100%);
                z-index: 10;
                pointer-events: none;
            }}
            .news-container::after {{
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 20px;
                background: linear-gradient(0deg, #1a1a2e 0%, transparent 100%);
                z-index: 10;
                pointer-events: none;
            }}
            .news-scroll {{
                animation: scroll-up 60s linear infinite;
            }}
            .news-container:hover .news-scroll {{
                animation-play-state: paused;
            }}
            @keyframes scroll-up {{
                0% {{ transform: translateY(0); }}
                100% {{ transform: translateY(-50%); }}
            }}
        </style>
    </head>
    <body>
        <div class="news-container">
            <div class="news-scroll">
                {all_cards}
            </div>
        </div>
    </body>
    </html>
    '''
    
    components.html(full_html, height=570, scrolling=False)


def init_session_state():
    """Initialize session state"""
    defaults = {
        'selected_tickers': [],
        'views': [],
        'tau': 0.05,  # More conservative (less weight to views)
        'risk_aversion': 3.5,  # More risk-averse (safer portfolio)
        'view_confidence': 0.5,
        'periods': 252,
        'risk_free_rate': 0.02,
        'max_weight': None,  # No default constraint (user must set manually)
        'min_weight': None,
        'bl_results': None,
        'run_optimization': False,
        'portfolio_amount': 100000.0  # Default $100k portfolio
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with decimals"""
    return f"{value:.{decimals}f}"


def add_view(
    assets: List[str],
    view_type: str,
    expected_return: float,
    confidence: float
):
    """Add view to session state"""
    view = {
        'assets': assets,
        'type': view_type,
        'expected_return': expected_return,
        'confidence': confidence
    }

    if 'views' not in st.session_state:
        st.session_state.views = []

    # For absolute views, remove any existing view for the same asset
    if view_type == 'absolute':
        st.session_state.views = [
            v for v in st.session_state.views
            if not (v['type'] == 'absolute' and v['assets'][0] == assets[0])
        ]

    # For relative views, remove any existing view for the same pair (in any order)
    elif view_type == 'relative':
        asset_set = set(assets)
        st.session_state.views = [
            v for v in st.session_state.views
            if not (v['type'] == 'relative' and set(v['assets']) == asset_set)
        ]

    st.session_state.views.append(view)


def remove_view(index: int):
    """Remove view by index"""
    if 'views' in st.session_state and 0 <= index < len(st.session_state.views):
        st.session_state.views.pop(index)


def clear_all_views():
    """Clear all views"""
    st.session_state.views = []


def get_views() -> List[Dict[str, Any]]:
    """Get all active views"""
    return st.session_state.get('views', [])


def validate_view(assets: List[str], view_type: str) -> tuple[bool, str]:
    """Validate view before adding"""
    if not assets:
        return False, "No assets selected"

    if view_type == 'absolute' and len(assets) != 1:
        return False, "Absolute view requires exactly one asset"

    if view_type == 'relative' and len(assets) != 2:
        return False, "Relative view requires exactly two assets"

    return True, ""


def create_pick_matrix_from_views(
    views: List[Dict[str, Any]],
    all_tickers: List[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Create P and Q matrices from views"""
    if not views:
        return np.array([]), np.array([])

    n_assets = len(all_tickers)
    n_views = len(views)

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)

    ticker_to_idx = {ticker: i for i, ticker in enumerate(all_tickers)}

    for i, view in enumerate(views):
        view_type = view['type']
        assets = view['assets']
        Q[i] = view['expected_return'] / 100  # Convert percentage to decimal

        if view_type == 'absolute':
            idx = ticker_to_idx[assets[0]]
            P[i, idx] = 1.0
        elif view_type == 'relative':
            idx1 = ticker_to_idx[assets[0]]
            idx2 = ticker_to_idx[assets[1]]
            P[i, idx1] = 1.0
            P[i, idx2] = -1.0

    return P, Q


def create_omega_matrix(
    P: np.ndarray,
    cov_matrix: np.ndarray,
    tau: float,
    confidences: List[float]
) -> np.ndarray:
    """Create Omega uncertainty matrix"""
    # Default: proportional to variance of view
    omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

    # Adjust by confidence (higher confidence = lower uncertainty)
    for i, conf in enumerate(confidences):
        omega[i, i] = omega[i, i] / max(conf, 0.01)  # Avoid division by zero

    return omega


def display_page_header(title: str, description: str, icon: str = "📊"):
    """Display page header"""
    st.markdown(f"""
    <div class="title-banner">
        <h1>{icon} {title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


def display_divider():
    """Display section divider"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide numbers"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def export_to_excel(data_dict: Dict[str, Any], filename: str = "portfolio_results.xlsx"):
    """Export DataFrames to Excel"""
    from io import BytesIO
    import pandas as pd

    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return output


def calculate_portfolio_value(
    weights: np.ndarray,
    prices: np.ndarray,
    initial_capital: float = 100000
) -> Dict[str, Any]:
    """Calculate shares and portfolio value"""
    allocations = weights * initial_capital
    shares = allocations / prices

    return {
        'weights': weights,
        'allocations': allocations,
        'shares': shares,
        'prices': prices,
        'total_value': initial_capital
    }
