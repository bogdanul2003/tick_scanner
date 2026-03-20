import os
import argparse
from datetime import datetime
import matplotlib
# Set the backend to Agg (non-interactive) before importing mplfinance/pyplot
# This prevents the Python icon from popping up in the macOS Dock
matplotlib.use('Agg')
import mplfinance as mpf
import pandas as pd
from db_utils import get_watchlist_symbols, load_cached_data

def generate_charts_for_watchlist(watchlist_name):
    # --- 1. SETUP FOLDERS ---
    today_str = datetime.now().strftime('%Y-%m-%d')
    # Safe-format watchlist name for folder (replace spaces/special chars)
    safe_watchlist_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in watchlist_name)
    
    # Root folder structure: generated_charts/Watchlist_Name/YYYY-MM-DD/
    output_dir = os.path.join('..', 'generated_charts', safe_watchlist_name, today_str)
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. GET SYMBOLS ---
    try:
        symbols = get_watchlist_symbols(watchlist_name)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if not symbols:
        print(f"No symbols found in watchlist '{watchlist_name}'")
        return

    print(f"Generating charts for {len(symbols)} symbols in watchlist '{watchlist_name}'...")

    # --- 3. DEFINE THE PLOT STYLE & SIZE ---
    my_colors = mpf.make_marketcolors(
        up='green', 
        down='red', 
        edge='inherit', 
        wick='black',    
        volume='inherit'
    )

    my_style = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        marketcolors=my_colors, 
        gridstyle=''           
    )

    # Target pixels: 683x768. At 144 DPI: 683/144 = 4.743, 768/144 = 5.333
    fig_size = (4.35, 5.65) 

    # --- 4. PROCESS EACH SYMBOL ---
    intervals = [
        {'months': 6, 'label': '6m'},
        {'months': 3, 'label': '3m'}
    ]

    for symbol in symbols:
        print(f"Processing {symbol}...")
        
        # Get data from DB
        full_data = load_cached_data(symbol)
        
        if full_data.empty:
            print(f"  No data found in DB for {symbol}, skipping.")
            continue

        for interval in intervals:
            # Create interval-specific subfolder
            interval_dir = os.path.join(output_dir, interval['label'])
            os.makedirs(interval_dir, exist_ok=True)

            # Filter data for the specific interval
            start_date = pd.Timestamp.now() - pd.DateOffset(months=interval['months'])
            data = full_data[full_data.index >= start_date].copy()

            if data.empty:
                print(f"  No data found in the last {interval['label']} for {symbol}, skipping.")
                continue

            # Ensure index is DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Map DB column names to what mplfinance expects
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_cols):
                # If High/Low are missing, fill them with Close
                if 'High' not in data.columns: data['High'] = data['Close']
                if 'Low' not in data.columns: data['Low'] = data['Close']

            # File path including the interval label
            filename = f"{symbol}_{interval['label']}_{today_str}.png"
            filepath = os.path.join(interval_dir, filename)

            # --- 5. GENERATE AND SAVE ---
            try:
                mpf.plot(
                    data, 
                    type='candle', 
                    style=my_style, 
                    ylabel='Price',
                    figsize=fig_size,
                    tight_layout=True,
                    show_nontrading=False,
                    datetime_format='%b %d',
                    xrotation=0,
                    savefig=dict(fname=filepath, dpi=144)
                )
                print(f"  Saved {interval['label']} chart to {filepath}")
            except Exception as e:
                print(f"  Failed to generate {interval['label']} chart for {symbol}: {e}")
