import yfinance as yf
import psycopg2
import pandas as pd  # <-- Add this import
from datetime import datetime, timedelta

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
    # Find the latest cached date for the symbol
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT MAX(date) FROM stock_cache WHERE symbol=%s
        """, (symbol,))
        result = cur.fetchone()
        latest_cached_date = result[0] if result and result[0] else None

    # Determine the first missing date (day after latest cached date or start_date)
    if latest_cached_date:
        first_missing_date = latest_cached_date + timedelta(days=1)
    else:
        first_missing_date = start_date

    # Generate all missing weekdays from first_missing_date to end_date (inclusive)
    all_missing_dates = []
    d = first_missing_date
    while d <= end_date:
        if d.weekday() < 5:  # Monday=0, ..., Friday=4
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

pd.set_option('display.float_format', '{:.8f}'.format)
create_table()
cache_duration_in_days = 365
symbol = "OKLO"
today = datetime.now().date()
start_date = today - timedelta(days=cache_duration_in_days)
end_date = today

cached_dates = get_cached_dates(symbol, start_date, end_date)

print(f"Cached dates for {symbol} from {start_date} to {end_date}: {len(cached_dates)} dates found")

if len(cached_dates) >= cache_duration_in_days:
    print(f"Loaded {cache_duration_in_days} days from cache")
else:
    missing_dates = get_missing_dates(symbol, start_date, end_date, cached_dates)
    print(f"Missing dates: {len(missing_dates)} days {missing_dates}")
    if missing_dates:
        # First, load existing cached data
        cached_data = load_cached_data(symbol)
        
        # Find the earliest and latest missing dates to fetch
        if missing_dates:
            earliest_missing = min(missing_dates)
            latest_missing = max(missing_dates)
            
            # Add buffer days for accurate EMA calculation
            lookback_buffer = 10  # Buffer for accurate EMA calculation
            fetch_start = earliest_missing - timedelta(days=lookback_buffer)
            fetch_end = latest_missing + timedelta(days=1)  # +1 for inclusive range
            
            # Fetch only the missing date range from Yahoo Finance
            print(f"Fetching data from {fetch_start} to {fetch_end} for {symbol}")
            yf_ticker = yf.Ticker(symbol)
            new_data = yf_ticker.history(start=fetch_start, end=fetch_end, interval="1d")
            
            # After fetching and processing data
            if not new_data.empty:
                # Ensure we only use the 'Close' column from new_data for the merge
                new_data_close = new_data[['Close']].copy()
                # Make index tz-naive for new_data_close
                if new_data_close.index.tz is not None:
                    new_data_close.index = new_data_close.index.tz_convert(None)
                # Make index tz-naive for cached_data
                if not cached_data.empty and cached_data.index.tz is not None:
                    cached_data.index = cached_data.index.tz_convert(None)

                # Normalize indices to midnight (00:00:00) for both DataFrames
                new_data_close.index = new_data_close.index.normalize()
                if not cached_data.empty:
                    cached_data.index = cached_data.index.normalize()
                
                # Safely concatenate the DataFrames
                if not cached_data.empty:
                    all_data = pd.concat([cached_data, new_data_close])
                    all_data = all_data[~all_data.index.duplicated(keep='last')].sort_index()
                else:
                    all_data = new_data_close.copy()
                
                # Keep only the Close column (index is Date)
                all_data = all_data[['Close']]
                # Round Close to 4 decimals
                # all_data['Close'] = all_data['Close']
                
                print(f"Fetched {len(new_data)} new rows from Yahoo Finance")
                print(f"Total rows after merge: {len(all_data)}")
                print(all_data.to_string())
                # Recalculate indicators on the combined dataset
                all_data['EMA12'] = all_data['Close'].ewm(span=12, adjust=False).mean()
                all_data['EMA26'] = all_data['Close'].ewm(span=26, adjust=False).mean()
                all_data['MACD'] = (all_data['EMA12'] - all_data['EMA26'])
                all_data['Signal_Line'] = all_data['MACD'].ewm(span=9, adjust=False).mean()
                
                # Filter to only the rows we need to save (the missing dates)
                missing_dates_set = set(missing_dates)
                to_cache = all_data[all_data.index.map(lambda x: x.date() in missing_dates_set)]
                
                if not to_cache.empty:
                    save_bulk_to_cache(symbol, to_cache)
                    # Calculate how many calendar dates don't have market data
                    unavailable_dates = len(missing_dates) - len(to_cache)
                    print(f"Fetched and cached {len(to_cache)} trading days")
                    if unavailable_dates > 0:
                        print(f"Note: {unavailable_dates} calendar days had no market data (weekends/holidays)")
                else:
                    print("No missing data to fetch")
            else:
                print("No data returned from Yahoo Finance")
    else:
        print("All dates present in cache")

# Load all cache_duration_in_days days from cache for use
with get_connection() as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT date, close, ema12, ema26, macd, signal_line
        FROM stock_cache
        WHERE symbol=%s AND date BETWEEN %s AND %s
        ORDER BY date
    """, (symbol, start_date, end_date))
    rows = cur.fetchall()
    for row in rows:
        print(row)  # Print each row to console