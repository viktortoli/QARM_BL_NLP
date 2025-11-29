"""Sentiment analysis display components"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


def display_sentiment_card(
    ticker: str,
    sentiment_data: Dict,
    show_details: bool = False
):
    """Display sentiment card for ticker"""
    if not sentiment_data:
        st.info(f"No sentiment data available for {ticker}")
        return

    score = sentiment_data.get('avg_sentiment_score', 0)
    article_count = sentiment_data.get('article_count', 0)
    distribution = sentiment_data.get('distribution', {})
    last_updated = sentiment_data.get('last_updated')

    if score > 0.1:
        sentiment_label = "Positive"
        color = "#4caf50"  # Green
        bg_color = "#e8f5e9"
    elif score < -0.1:
        sentiment_label = "Negative"
        color = "#f44336"  # Red
        bg_color = "#ffebee"
    else:
        sentiment_label = "Neutral"
        color = "#ff9800"  # Orange
        bg_color = "#fff3e0"

    card_html = f"""
    <div style="
        background-color: {bg_color};
        border-left: 5px solid {color};
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4 style="margin: 0; color: #424242;">{ticker} Sentiment</h4>
                <p style="margin: 0.5rem 0 0 0; color: {color}; font-size: 1.5rem; font-weight: bold;">
                    {sentiment_label} ({score:+.3f})
                </p>
            </div>
            <div style="text-align: right; color: #757575;">
                <p style="margin: 0; font-size: 0.9rem;">Based on {article_count} article{'s' if article_count != 1 else ''}</p>
            </div>
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

    if show_details and distribution:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Positive",
                f"{distribution.get('positive', 0):.1f}%",
                help="Percentage of positive articles"
            )

        with col2:
            st.metric(
                "Neutral",
                f"{distribution.get('neutral', 0):.1f}%",
                help="Percentage of neutral articles"
            )

        with col3:
            st.metric(
                "Negative",
                f"{distribution.get('negative', 0):.1f}%",
                help="Percentage of negative articles"
            )

        if last_updated:
            update_time = datetime.fromisoformat(last_updated).strftime('%Y-%m-%d %H:%M')
            st.caption(f"Last updated: {update_time}")


def display_sentiment_table(
    sentiment_data: Dict[str, Dict],
    show_distribution: bool = True
) -> pd.DataFrame:
    """Display sentiment table for multiple tickers"""
    if not sentiment_data:
        st.info("No sentiment data available")
        return pd.DataFrame()

    data = {
        'Ticker': [],
        'Sentiment': [],
        'Score': [],
        'Articles': [],
        'Last Article': []
    }

    if show_distribution:
        data['Positive (%)'] = []
        data['Neutral (%)'] = []
        data['Negative (%)'] = []

    for ticker, sentiment in sentiment_data.items():
        score = sentiment.get('avg_sentiment_score', 0)

        if score > 0.1:
            label = "Positive"
        elif score < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        hours_old = sentiment.get('article_hours_old', sentiment.get('hours_old', 0))
        if hours_old < 1:
            time_str = f"{int(hours_old * 60)}m ago"
        elif hours_old < 24:
            time_str = f"{int(hours_old)}h ago"
        else:
            time_str = f"{int(hours_old / 24)}d ago"

        if hours_old > 48:  # Article older than 2 days
            time_str += " ⚠️"

        data['Ticker'].append(ticker)
        data['Sentiment'].append(label)
        data['Score'].append(score)
        data['Articles'].append(sentiment.get('article_count', 0))
        data['Last Article'].append(time_str)

        if show_distribution:
            dist = sentiment.get('distribution', {})
            data['Positive (%)'].append(dist.get('positive', 0))
            data['Neutral (%)'].append(dist.get('neutral', 0))
            data['Negative (%)'].append(dist.get('negative', 0))

    df = pd.DataFrame(data)

    def color_sentiment(val):
        if val == 'Positive':
            return 'color: #4caf50; font-weight: bold'
        elif val == 'Negative':
            return 'color: #f44336; font-weight: bold'
        else:
            return 'color: #ff9800; font-weight: bold'

    def color_score(val):
        if pd.isna(val):
            return ''
        if val > 0.1:
            return 'color: #4caf50'
        elif val < -0.1:
            return 'color: #f44336'
        else:
            return 'color: #ff9800'

    format_dict = {
        'Score': '{:+.3f}',
        'Articles': '{:.0f}'
    }

    if show_distribution:
        format_dict['Positive (%)'] = '{:.1f}'
        format_dict['Neutral (%)'] = '{:.1f}'
        format_dict['Negative (%)'] = '{:.1f}'

    styled_df = df.style.format(format_dict) \
        .applymap(color_sentiment, subset=['Sentiment']) \
        .applymap(color_score, subset=['Score'])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(len(df) * 35 + 38, 600),
        hide_index=True
    )

    return df


def display_news_articles(
    articles: List[Dict],
    max_articles: int = 5,
    show_sentiment: bool = True
):
    """Display news articles with sentiment"""
    if not articles:
        st.info("No news articles available")
        return

    st.subheader(f"Recent News ({len(articles)} article{'s' if len(articles) != 1 else ''})")

    for i, article in enumerate(articles[:max_articles], 1):
        title = article.get('title', 'No title')
        pub_date = article.get('published_date', '')
        source = article.get('source', 'Unknown')
        url = article.get('url', '#')

        try:
            date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            date_str = date_obj.strftime('%Y-%m-%d %H:%M')
        except:
            date_str = pub_date

        sentiment_html = ""
        if show_sentiment and 'sentiment_score' in article:
            score = article['sentiment_score']
            label = article.get('sentiment_label', 'Neutral')

            if score > 0.1:
                color = "#4caf50"
            elif score < -0.1:
                color = "#f44336"
            else:
                color = "#ff9800"

            sentiment_html = f"""
            <span style="color: {color}; font-weight: bold; margin-left: 10px;">
                [{label} {score:+.2f}]
            </span>
            """

        article_html = f"""
        <div style="
            border: 1px solid #e0e0e0;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            background-color: #fafafa;
        ">
            <h5 style="margin: 0 0 0.5rem 0;">
                <a href="{url}" target="_blank" style="color: #1976d2; text-decoration: none;">
                    {i}. {title}
                </a>
                {sentiment_html}
            </h5>
            <p style="margin: 0; color: #757575; font-size: 0.9rem;">
                {source} • {date_str}
            </p>
        </div>
        """

        st.markdown(article_html, unsafe_allow_html=True)


def display_sentiment_summary(
    ticker: str,
    sentiment_data: Dict,
    articles: List[Dict],
    show_all_articles: bool = False
):
    """Display comprehensive sentiment summary"""
    # Display sentiment card
    display_sentiment_card(ticker, sentiment_data, show_details=True)

    st.markdown("---")

    # Display articles
    max_articles = len(articles) if show_all_articles else 5
    display_news_articles(articles, max_articles=max_articles, show_sentiment=True)


def get_sentiment_color(score: float) -> str:
    """Get color for sentiment score"""
    if score > 0.1:
        return "#4caf50"  # Green
    elif score < -0.1:
        return "#f44336"  # Red
    else:
        return "#ff9800"  # Orange


def get_sentiment_label(score: float) -> str:
    """Get label for sentiment score"""
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"
