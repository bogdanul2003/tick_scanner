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
                ma20 FLOAT,
                ma50 FLOAT,
                macd FLOAT,
                signal_line FLOAT,
                will_become_positive BOOLEAN DEFAULT FALSE,
                ma20_will_be_above_ma50 BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (symbol, date)
            )
            """)
            # Add columns if they don't exist (for existing tables)
            cur.execute("""
            DO $$ 
            BEGIN
                BEGIN
                    ALTER TABLE stock_cache ADD COLUMN will_become_positive BOOLEAN DEFAULT FALSE;
                EXCEPTION
                    WHEN duplicate_column THEN NULL;
                END;
                BEGIN
                    ALTER TABLE stock_cache ADD COLUMN ma20_will_be_above_ma50 BOOLEAN DEFAULT FALSE;
                EXCEPTION
                    WHEN duplicate_column THEN NULL;
                END;
            END $$;
            """)
            conn.commit()
    finally:
        put_connection(conn)

def fetch_from_cache(symbol, date):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT close, ema12, ema26, ma20, ma50, macd, signal_line
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
                INSERT INTO stock_cache (symbol, date, close, ema12, ema26, ma20, ma50, macd, signal_line)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, date) DO NOTHING
            """, (symbol, date, row['Close'], row['EMA12'], row['EMA26'], row.get('MA20'), row.get('MA50'), row['MACD'], row['Signal_Line']))
            conn.commit()
    finally:
        put_connection(conn)

def save_bulk_to_cache(symbol, df):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for date, row in df.iterrows():
                cur.execute("""
                    INSERT INTO stock_cache (symbol, date, close, ema12, ema26, ma20, ma50, macd, signal_line)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date) DO NOTHING
                """, (
                    symbol,
                    date.date(),
                    float(row['Close']) if pd.notnull(row['Close']) else None,
                    float(row['EMA12']) if pd.notnull(row['EMA12']) else None,
                    float(row['EMA26']) if pd.notnull(row['EMA26']) else None,
                    float(row['MA20']) if 'MA20' in row and pd.notnull(row['MA20']) else None,
                    float(row['MA50']) if 'MA50' in row and pd.notnull(row['MA50']) else None,
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
                SELECT date, close, ema12, ema26, ma20, ma50, macd, signal_line
                FROM stock_cache
                WHERE symbol=%s 
                ORDER BY date
            """
            cur.execute(query, (symbol,))
            rows = cur.fetchall()
            if rows:
                cached_data = pd.DataFrame(rows, columns=['date', 'Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                cached_data.set_index('date', inplace=True)
            else:
                cached_data = pd.DataFrame(columns=['Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
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
                    SELECT date, close, ema12, ema26, ma20, ma50, macd, signal_line
                    FROM stock_cache
                    WHERE symbol=%s AND date BETWEEN %s AND %s
                    ORDER BY date
                """, (symbol, start_date, end_date))
                rows = cur.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=['date', 'Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df = pd.DataFrame(columns=['Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
                    df.index.name = 'date'
                result[symbol] = df
    finally:
        put_connection(conn)
    return result

def fetch_latest_bulk_from_cache(symbols):
    """
    Fetch cached data for a list of symbols for the most recent date available in DB.
    Returns a dict: {symbol: DataFrame}
    """
    result = {}
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Find the most recent date in stock_cache
            cur.execute("SELECT MAX(date) FROM stock_cache")
            latest_date_row = cur.fetchone()
            latest_date = latest_date_row[0] if latest_date_row else None
            if not latest_date:
                # No data in cache
                for symbol in symbols:
                    df = pd.DataFrame(columns=['Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
                    df.index.name = 'date'
                    result[symbol] = df
                return result
            for symbol in symbols:
                cur.execute("""
                    SELECT date, close, ema12, ema26, ma20, ma50, macd, signal_line
                    FROM stock_cache
                    WHERE symbol=%s AND date=%s
                    ORDER BY date
                """, (symbol, latest_date))
                rows = cur.fetchall()
                if rows:
                    df = pd.DataFrame(rows, columns=['date', 'Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                else:
                    df = pd.DataFrame(columns=['Close', 'EMA12', 'EMA26', 'MA20', 'MA50', 'MACD', 'Signal_Line'])
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
            # Create enum type if not exists
            cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_type_enum') THEN
                    CREATE TYPE model_type_enum AS ENUM ('MACD_MODEL', 'MA20_MODEL', 'MA50_MODEL');
                END IF;
            END$$;
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS forecast_util (
                symbol TEXT NOT NULL,
                search_date DATE NOT NULL,
                best_order TEXT NOT NULL,
                best_aic FLOAT NOT NULL,
                window_used INTEGER NOT NULL,
                model_blob BYTEA,
                model_type model_type_enum NOT NULL,
                PRIMARY KEY (symbol, search_date, model_type)
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

def get_arima_grid_cache(symbol, window_size, cache_days=30, with_model=False, model_type=None):
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
            query = """
                SELECT best_order, best_aic, window_used, search_date, model_blob
                FROM forecast_util
                WHERE symbol=%s AND search_date >= %s
            """
            params = [symbol, cache_valid_since]
            if model_type is not None:
                query += " AND model_type=%s"
                params.append(model_type)
            query += """
                ORDER BY search_date DESC
                LIMIT 1
            """
            cur.execute(query, tuple(params))
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

def set_arima_grid_cache(symbol, best_order, best_aic, window_used, model=None, model_type=None):
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
                INSERT INTO forecast_util (symbol, search_date, best_order, best_aic, window_used, model_blob, model_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, search_date, model_type) DO UPDATE
                SET best_order=EXCLUDED.best_order,
                    best_aic=EXCLUDED.best_aic,
                    window_used=EXCLUDED.window_used,
                    model_blob=EXCLUDED.model_blob,
                    model_type=EXCLUDED.model_type
            """, (
                symbol,
                today,
                str(best_order),
                float(best_aic),
                int(window_used),
                model_blob,
                model_type
            ))
            conn.commit()
    finally:
        put_connection(conn)

def get_cached_arima_model(symbol, window_size, cache_days=30, model_type=None):
    """
    Returns the deserialized ARIMA model if found and valid, else None.
    """
    cached = get_arima_grid_cache(symbol, window_size, cache_days=cache_days, with_model=True, model_type=model_type)
    if cached and cached.get("model_blob"):
        return deserialize_arima_model(cached["model_blob"])
    return None

def create_symbol_picks_table():
    """
    Create the symbol_picks table to store filtered symbols for a watchlist.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS symbol_picks (
                watchlist_name TEXT NOT NULL,
                filter_results JSONB NOT NULL,
                applied_date DATE NOT NULL,
                PRIMARY KEY (watchlist_name, applied_date)
            )
            """)
            conn.commit()
    finally:
        put_connection(conn)

def save_symbol_picks(watchlist_name, filter_results, applied_date):
    """
    Save or update filter results for a watchlist on a specific date.
    filter_results should be a dict: {filter_name: [symbols]}
    Overwrites the row if one already exists for the given watchlist_name and applied_date.
    """
    import json
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO symbol_picks (watchlist_name, filter_results, applied_date)
                VALUES (%s, %s, %s)
                ON CONFLICT (watchlist_name, applied_date) DO UPDATE
                SET filter_results = EXCLUDED.filter_results
            """, (
                watchlist_name,
                json.dumps(filter_results),
                applied_date
            ))
            conn.commit()
    finally:
        put_connection(conn)

def get_symbol_picks(watchlist_name, applied_date):
    """
    Retrieve filter results for a watchlist and date.
    Returns a dict {filter_name: [symbols]} or None if not found.
    """
    import json
    conn = get_connection()
    print(f"Fetching symbol picks for {watchlist_name} on {applied_date}")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT filter_results FROM symbol_picks
                WHERE watchlist_name=%s AND applied_date=%s
            """, (watchlist_name, applied_date))
            row = cur.fetchone()
            if row:
                filter_results = row[0]
                # If already a dict (psycopg2 with jsonb), return as is; else, parse as JSON string
                if isinstance(filter_results, dict):
                    return filter_results
                return json.loads(filter_results)
    finally:
        put_connection(conn)
    return None

def create_symbol_properties_table():
    """
    Create the symbol_properties table with unique symbol and company_name columns.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS symbol_properties (
                symbol TEXT PRIMARY KEY,
                company_name TEXT
            )
            """)
            conn.commit()
    finally:
        put_connection(conn)

def upsert_symbol_property(symbol, company_name):
    """
    Insert or update a symbol and its company name.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO symbol_properties (symbol, company_name)
                VALUES (%s, %s)
                ON CONFLICT (symbol) DO UPDATE
                SET company_name = EXCLUDED.company_name
            """, (symbol, company_name))
            conn.commit()
    finally:
        put_connection(conn)

def get_company_name(symbol):
    """
    Get the company name for a given symbol.
    Returns the company name string or None if not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT company_name FROM symbol_properties WHERE symbol=%s
            """, (symbol,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        put_connection(conn)

def get_all_symbol_properties():
    """
    Get all symbol/company_name pairs as a list of dicts.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT symbol, company_name FROM symbol_properties ORDER BY symbol
            """)
            return [{"symbol": row[0], "company_name": row[1]} for row in cur.fetchall()]
    finally:
        put_connection(conn)

def delete_symbol_property(symbol):
    """
    Delete a symbol from the symbol_properties table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM symbol_properties WHERE symbol=%s
            """, (symbol,))
            conn.commit()
    finally:
        put_connection(conn)

def get_company_names(symbols):
    """
    Given a list of symbols, return a dict {symbol: company_name} for those present in the table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, company_name FROM symbol_properties WHERE symbol = ANY(%s)",
                (symbols,)
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        put_connection(conn)

def cache_macd_positive_forecast(symbol, forecast_date, will_become_positive):
    """
    Cache the will_become_positive result for a symbol and forecast_date in stock_cache table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_cache (symbol, date, will_become_positive)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE
                SET will_become_positive = EXCLUDED.will_become_positive
            """, (symbol, forecast_date, will_become_positive))
            conn.commit()
    finally:
        put_connection(conn)

def cache_ma20_above_ma50_forecast(symbol, forecast_date, ma20_will_be_above_ma50):
    """
    Cache the ma20_will_be_above_ma50 result for a symbol and forecast_date in stock_cache table.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO stock_cache (symbol, date, ma20_will_be_above_ma50)
                VALUES (%s, %s, %s)
                ON CONFLICT (symbol, date) DO UPDATE
                SET ma20_will_be_above_ma50 = EXCLUDED.ma20_will_be_above_ma50
            """, (symbol, forecast_date, ma20_will_be_above_ma50))
            conn.commit()
    finally:
        put_connection(conn)

def get_cached_macd_positive_forecast(symbol, forecast_date):
    """
    Get cached will_become_positive result for a symbol and forecast_date from stock_cache table.
    Returns boolean or None if not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT will_become_positive FROM stock_cache
                WHERE symbol=%s AND date=%s
            """, (symbol, forecast_date))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        put_connection(conn)

def get_cached_ma20_above_ma50_forecast(symbol, forecast_date):
    """
    Get cached ma20_will_be_above_ma50 result for a symbol and forecast_date from stock_cache table.
    Returns boolean or None if not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ma20_will_be_above_ma50 FROM stock_cache
                WHERE symbol=%s AND date=%s
            """, (symbol, forecast_date))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        put_connection(conn)
