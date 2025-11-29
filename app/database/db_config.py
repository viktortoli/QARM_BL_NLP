"""Database configuration for PostgreSQL"""

import os
from sqlalchemy import create_engine


def get_database_url():
    """Get PostgreSQL database URL from environment"""
    postgres_url = os.getenv('DATABASE_URL')

    if not postgres_url:
        try:
            import streamlit as st
            postgres_url = st.secrets.get('DATABASE_URL')
        except Exception:
            pass

    if not postgres_url:
        raise ValueError("DATABASE_URL not set. PostgreSQL connection required.")

    if postgres_url.startswith('postgres://'):
        postgres_url = postgres_url.replace('postgres://', 'postgresql://', 1)

    print("[OK] Using PostgreSQL database")
    return postgres_url


def create_db_engine():
    """Create SQLAlchemy engine for PostgreSQL"""
    database_url = get_database_url()

    # Ensure sslmode is set for Render PostgreSQL
    if 'sslmode' not in database_url:
        separator = '&' if '?' in database_url else '?'
        database_url = f"{database_url}{separator}sslmode=require"

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=10,
        connect_args={
            "connect_timeout": 30,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5
        },
        echo=False
    )

    return engine


def is_postgres():
    """Check (only a debug for when we switched from SQLite to PostgreSQL)"""
    return True
