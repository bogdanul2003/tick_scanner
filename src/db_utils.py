import psycopg2
import pandas as pd
from datetime import timedelta

DB_PARAMS = {
    "dbname": "ticks",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

def get_connection():
    return psycopg2.connect(**DB_PARAMS)

def create_table():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_cache (
            symbol TEXT,
            date DATE,
            close FLOAT,
            ema12 FLOAT,
            ema26 FLOAT,
            macd FLOAT,
            signal_line FLOAT,
            PRIMARY KEY (symbol, date)
        )
        """)
        conn.commit()

def fetch_from_cache(symbol, date):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT close, ema12, ema26, macd, signal_line
            FROM stock_cache
            WHERE symbol=%s AND date=%s
        """, (symbol, date))
        return cur.fetchone()

def save_to_cache(symbol, date, row):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO stock_cache (symbol, date, close, ema12, ema26, macd, signal_line)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, date) DO NOTHING
        """, (symbol, date, row['Close'], row['EMA12'], row['EMA26'], row['MACD'], row['Signal_Line']))
        conn.commit()

def get_cached_dates(symbol, start_date, end_date):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT date FROM stock_cache
            WHERE symbol=%s AND date BETWEEN %s AND %s
        """, (symbol, start_date, end_date))
        return set(row[0] for row in cur.fetchall())

def save_bulk_to_cache(symbol, df):
    with get_connection() as conn, conn.cursor() as cur:
        for date, row in df.iterrows():
            cur.execute("""
                INSERT INTO stock_cache (symbol, date, close, ema12, ema26, macd, signal_line)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING
            """, (
                symbol,
                date.date(),
                float(row['Close']) if pd.notnull(row['Close']) else None,
                float(row['EMA12']) if pd.notnull(row['EMA12']) else None,
                float(row['EMA26']) if pd.notnull(row['EMA26']) else None,
                float(row['MACD']) if pd.notnull(row['MACD']) else None,
                float(row['Signal_Line']) if pd.notnull(row['Signal_Line']) else None
            ))
        conn.commit()

def get_missing_dates(symbol, start_date, end_date, cached_dates):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(date) FROM stock_cache WHERE symbol=%s
        """, (symbol,))
        result = cur.fetchone()
        latest_cached_date = result[0] if result and result[0] else None

    if latest_cached_date:
        first_missing_date = latest_cached_date + timedelta(days=1)
    else:
        first_missing_date = start_date

    all_missing_dates = []
    d = first_missing_date
    while d <= end_date:
        if d.weekday() < 5:
            all_missing_dates.append(d)
        d += timedelta(days=1)
    return set(all_missing_dates) - cached_dates

def load_cached_data(symbol):
    with get_connection() as conn, conn.cursor() as cur:
        query = """
            SELECT date, close, ema12, ema26, macd, signal_line
            FROM stock_cache
            WHERE symbol=%s 
            ORDER BY date
        """
        cur.execute(query, (symbol,))
        rows = cur.fetchall()
        if rows:
            cached_data = pd.DataFrame(rows, columns=['date', 'Close', 'EMA12', 'EMA26', 'MACD', 'Signal_Line'])
            cached_data['date'] = pd.to_datetime(cached_data['date'])
            cached_data.set_index('date', inplace=True)
        else:
            cached_data = pd.DataFrame(columns=['Close', 'EMA12', 'EMA26', 'MACD', 'Signal_Line'])
            cached_data.index.name = 'date'
    return cached_data
