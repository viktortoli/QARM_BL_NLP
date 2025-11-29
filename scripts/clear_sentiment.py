"""
Clear sentiment data from database.
Usage: python scripts/clear_sentiment.py [TICKER1 TICKER2 ...]
"""

import os
import sys
from datetime import datetime
import psycopg2


def log(message):
    """Print timestamped log"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_db_connection():
    """Get PostgreSQL connection"""
    postgres_url = os.getenv('DATABASE_URL')

    if not postgres_url:
        log("ERROR: DATABASE_URL not set!")
        sys.exit(1)

    # Fix postgres:// to postgresql://
    if postgres_url.startswith('postgres://'):
        postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)

    return psycopg2.connect(postgres_url)


def get_current_stats(conn, tickers=None):
    """Get counts of news articles and sentiment scores"""
    cursor = conn.cursor()
    
    if tickers:
        placeholders = ','.join(['%s'] * len(tickers))
        cursor.execute(f"SELECT COUNT(*) FROM news_articles WHERE ticker IN ({placeholders})", tickers)
        articles = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM sentiment_scores WHERE ticker IN ({placeholders})", tickers)
        scores = cursor.fetchone()[0]
        cursor.execute(f"SELECT COUNT(*) FROM ticker_sentiment_summary WHERE ticker IN ({placeholders})", tickers)
        summaries = cursor.fetchone()[0]
    else:
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        articles = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM sentiment_scores")
        scores = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM ticker_sentiment_summary")
        summaries = cursor.fetchone()[0]
    
    cursor.close()
    return articles, scores, summaries


def clear_sentiment_data(conn, tickers=None):
    """Clear sentiment data. If tickers=None, clears all."""
    cursor = conn.cursor()
    
    try:
        if tickers:
            tickers_str = ', '.join(tickers)
            log(f"Clearing sentiment data for: {tickers_str}")
            
            placeholders = ','.join(['%s'] * len(tickers))
            
            # Delete sentiment scores first (foreign key constraint)
            cursor.execute(f"DELETE FROM sentiment_scores WHERE ticker IN ({placeholders})", tickers)
            scores_deleted = cursor.rowcount
            log(f"  Deleted {scores_deleted} sentiment scores")
            
            # Delete news articles
            cursor.execute(f"DELETE FROM news_articles WHERE ticker IN ({placeholders})", tickers)
            articles_deleted = cursor.rowcount
            log(f"  Deleted {articles_deleted} news articles")
            
            # Delete sentiment summaries
            cursor.execute(f"DELETE FROM ticker_sentiment_summary WHERE ticker IN ({placeholders})", tickers)
            summaries_deleted = cursor.rowcount
            log(f"  Deleted {summaries_deleted} sentiment summaries")
            
        else:
            log("Clearing ALL sentiment data...")
            
            # Delete in order due to foreign key constraints
            cursor.execute("DELETE FROM sentiment_scores")
            scores_deleted = cursor.rowcount
            log(f"  Deleted {scores_deleted} sentiment scores")
            
            cursor.execute("DELETE FROM news_articles")
            articles_deleted = cursor.rowcount
            log(f"  Deleted {articles_deleted} news articles")
            
            cursor.execute("DELETE FROM ticker_sentiment_summary")
            summaries_deleted = cursor.rowcount
            log(f"  Deleted {summaries_deleted} sentiment summaries")
        
        conn.commit()
        log("✓ Sentiment data cleared successfully!")
        
        return articles_deleted, scores_deleted, summaries_deleted
        
    except Exception as e:
        conn.rollback()
        log(f"ERROR: {e}")
        raise


def main():
    """Main execution"""
    log("=" * 60)
    log("Sentiment Database Cleanup Tool")
    log("=" * 60)
    
    # Parse command line arguments for optional ticker filtering
    tickers = None
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    
    try:
        conn = get_db_connection()
        log("Connected to PostgreSQL")
        
        # Show current stats
        articles, scores, summaries = get_current_stats(conn, tickers)
        
        scope = f"tickers: {', '.join(tickers)}" if tickers else "ALL tickers"
        log(f"\nCurrent data ({scope}):")
        log(f"  News articles: {articles}")
        log(f"  Sentiment scores: {scores}")
        log(f"  Ticker summaries: {summaries}")
        
        if articles == 0 and scores == 0 and summaries == 0:
            log("\nNo sentiment data to clear.")
            return
        
        # Confirm before deletion
        log(f"\n⚠️  This will DELETE all sentiment data for {scope}!")
        response = input("Type 'yes' to confirm: ").strip().lower()
        
        if response != 'yes':
            log("Aborted.")
            return
        
        # Clear the data
        log("")
        clear_sentiment_data(conn, tickers)
        
        # Verify
        articles, scores, summaries = get_current_stats(conn, tickers)
        log(f"\nVerification ({scope}):")
        log(f"  News articles: {articles}")
        log(f"  Sentiment scores: {scores}")
        log(f"  Ticker summaries: {summaries}")
        
        conn.close()
        log("\nDone! You can now run fetch_sentiment.py to fetch fresh data.")
        
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
