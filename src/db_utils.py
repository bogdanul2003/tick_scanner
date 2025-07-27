import psycopg2
from psycopg2 import pool
import pandas as pd
from datetime import timedelta
import pickle

DB_PARAMS = {
    "dbname": "ticks",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

# Create a global connection pool (adjust minconn/maxconn as needed)
CONN_POOL = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=15,
    **DB_PARAMS
)

def get_connection():
    """Get a connection from the pool."""
    return CONN_POOL.getconn()

def put_connection(conn):
    """Return a connection to the pool."""
    CONN_POOL.putconn(conn)

def create_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
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
    finally:
        put_connection(conn)

def fetch_from_cache(symbol, date):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT close, ema12, ema26, macd, signal_line
                FROM stock_cache
                WHERE symbol=%s AND date=%s
            """, (symbol, date))
            return cur.fetchone()
    finally:
        put_connection(conn)

def save_to_cache(symbol, date, row):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_cache (symbol, date, close, ema12, ema26, macd, signal_line)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING
            """, (symbol, date, row['Close'], row['EMA12'], row['EMA26'], row['MACD'], row['Signal_Line']))
            conn.commit()
    finally:
        put_connection(conn)

def save_bulk_to_cache(symbol, df):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
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
    finally:
        put_connection(conn)

def get_missing_dates(symbol, start_date, end_date, cached_dates):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
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
    finally:
        put_connection(conn)

def load_cached_data(symbol):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
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
    finally:
        put_connection(conn)
    return cached_data

def get_cached_dates(symbol, start_date, end_date):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date FROM stock_cache
                WHERE symbol=%s AND date BETWEEN %s AND %s
            """, (symbol, start_date, end_date))
            return set(row[0] for row in cur.fetchall())
    finally:
        put_connection(conn)

def fetch_bulk_from_cache(symbols, start_date, end_date):
    """
    Fetch cached data for a list of symbols and a date interval.
    Returns a dict: {symbol: DataFrame}
    """
    result = {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for symbol in symbols:
                cur.execute("""
                    SELECT date, close, ema12, ema26, macd, signal_line
                    FROM stock_cache
                    WHERE symbol=%s AND date BETWEEN %s AND %s
                    ORDER BY date
                """, (symbol, start_date, end_date))
                rows = cur.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=['date', 'Close', 'EMA12', 'EMA26', 'MACD', 'Signal_Line'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df = pd.DataFrame(columns=['Close', 'EMA12', 'EMA26', 'MACD', 'Signal_Line'])
                    df.index.name = 'date'
                result[symbol] = df
    finally:
        put_connection(conn)
    return result

def create_watchlist_tables():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_symbols (
                watchlist_id INTEGER REFERENCES watchlists(id) ON DELETE CASCADE,
                symbol TEXT NOT NULL,
                PRIMARY KEY (watchlist_id, symbol)
            );
            """)
            conn.commit()
    finally:
        put_connection(conn)

def create_watchlist(name):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO watchlists (name) VALUES (%s)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
            """, (name,))
            result = cur.fetchone()
            conn.commit()
            return result[0] if result else None
    finally:
        put_connection(conn)

def delete_watchlist(name):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM watchlists WHERE name=%s
            """, (name,))
            conn.commit()
    finally:
        put_connection(conn)

def add_symbol_to_watchlist(watchlist_name, symbol):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM watchlists WHERE name=%s", (watchlist_name,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Watchlist '{watchlist_name}' does not exist.")
            watchlist_id = row[0]
            cur.execute("""
                INSERT INTO watchlist_symbols (watchlist_id, symbol)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
            """, (watchlist_id, symbol))
            conn.commit()
    finally:
        put_connection(conn)

def remove_symbol_from_watchlist(watchlist_name, symbol):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM watchlists WHERE name=%s", (watchlist_name,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Watchlist '{watchlist_name}' does not exist.")
            watchlist_id = row[0]
            cur.execute("""
                DELETE FROM watchlist_symbols
                WHERE watchlist_id=%s AND symbol=%s
            """, (watchlist_id, symbol))
            conn.commit()
    finally:
        put_connection(conn)

def get_watchlist_symbols(watchlist_name):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM watchlists WHERE name=%s", (watchlist_name,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Watchlist '{watchlist_name}' does not exist.")
            watchlist_id = row[0]
            cur.execute("""
                SELECT symbol FROM watchlist_symbols
                WHERE watchlist_id=%s
            """, (watchlist_id,))
            return [r[0] for r in cur.fetchall()]
    finally:
        put_connection(conn)
    
def get_all_watchlists():
    from db_utils import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM watchlists ORDER BY name")
            return [row[0] for row in cur.fetchall()]
    finally:
        put_connection(conn)

def get_all_watchlists_with_symbols():
    from db_utils import get_connection
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM watchlists ORDER BY name")
            watchlists = cur.fetchall()
            result = []
            for watchlist_id, name in watchlists:
                cur.execute("SELECT symbol FROM watchlist_symbols WHERE watchlist_id=%s", (watchlist_id,))
                symbols = [row[0] for row in cur.fetchall()]
                result.append({"name": name, "symbols": symbols})
            return result
    finally:
        put_connection(conn)

def create_forecast_util_table():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_util (
                symbol TEXT NOT NULL,
                search_date DATE NOT NULL,
                best_order TEXT NOT NULL,
                best_aic FLOAT NOT NULL,
                window_used INTEGER NOT NULL,
                model_blob BYTEA,
                PRIMARY KEY (symbol, search_date)
            )
            """)
            conn.commit()
    finally:
        put_connection(conn)

def serialize_arima_model(model):
    """Serialize ARIMA model to bytes using pickle."""
    return pickle.dumps(model)

def deserialize_arima_model(blob):
    """Deserialize ARIMA model from bytes."""
    return pickle.loads(blob)

def get_arima_grid_cache(symbol, window_size, cache_days=30, with_model=False):
    """
    Returns dict with keys: best_order, best_aic, window_used, search_date, model_blob (if with_model=True)
    or None if not found or window_used does not match.
    """
    import ast
    from datetime import datetime, timedelta
    today = datetime.now().date()
    cache_valid_since = today - timedelta(days=cache_days)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT best_order, best_aic, window_used, search_date, model_blob
                FROM forecast_util
                WHERE symbol=%s AND search_date >= %s
                ORDER BY search_date DESC
                LIMIT 1
            """, (symbol, cache_valid_since))
            row = cur.fetchone()
            if row and row[2] == window_size:
                result = {
                    "best_order": ast.literal_eval(row[0]) if isinstance(row[0], str) else row[0],
                    "best_aic": row[1],
                    "window_used": row[2],
                    "search_date": row[3]
                }
                if with_model:
                    result["model_blob"] = row[4]
                return result
    finally:
        put_connection(conn)
    return None

def set_arima_grid_cache(symbol, best_order, best_aic, window_used, model=None):
    """
    Save ARIMA grid search result for a symbol and window, including the model as a blob.
    """
    from datetime import datetime
    today = datetime.now().date()
    model_blob = serialize_arima_model(model) if model is not None else None
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO forecast_util (symbol, search_date, best_order, best_aic, window_used, model_blob)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, search_date) DO UPDATE
                SET best_order=EXCLUDED.best_order,
                    best_aic=EXCLUDED.best_aic,
                    window_used=EXCLUDED.window_used,
                    model_blob=EXCLUDED.model_blob
            """, (
                symbol,
                today,
                str(best_order),
                float(best_aic),
                int(window_used),
                model_blob
            ))
            conn.commit()
    finally:
        put_connection(conn)

def get_cached_arima_model(symbol, window_size, cache_days=30):
    """
    Returns the deserialized ARIMA model if found and valid, else None.
    """
    cached = get_arima_grid_cache(symbol, window_size, cache_days=cache_days, with_model=True)
    if cached and cached.get("model_blob"):
        return deserialize_arima_model(cached["model_blob"])
    return None
