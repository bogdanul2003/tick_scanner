import os
import argparse
from datetime import datetime
import matplotlib
# Set the backend to Agg (non-interactive) before importing mplfinance/pyplot
# This prevents the Python icon from popping up in the macOS Dock
matplotlib.use('Agg')
import mplfinance as mpf
import pandas as pd
import cv2
from db_utils import get_watchlist_symbols, load_cached_data

def generate_charts_for_watchlist(watchlist_name, selected_date=None, show_volume=False):
    # --- 1. SETUP FOLDERS ---
    if selected_date:
        today_dt = pd.to_datetime(selected_date)
        today_str = today_dt.strftime('%Y-%m-%d')
    else:
        today_dt = pd.Timestamp.now()
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

    print(f"Generating charts for {len(symbols)} symbols in watchlist '{watchlist_name}' for {today_str} (Volume: {show_volume})...")

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
            start_date = today_dt - pd.DateOffset(months=interval['months'])
            data = full_data[(full_data.index >= start_date) & (full_data.index <= today_dt)].copy()

            if data.empty:
                print(f"  No data found between {start_date.date()} and {today_dt.date()} for {symbol}, skipping.")
                continue

            # Ensure index is DatetimeIndex
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Map DB column names to what mplfinance expects
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                # If High/Low are missing, fill them with Close
                if 'High' not in data.columns: data['High'] = data['Close']
                if 'Low' not in data.columns: data['Low'] = data['Close']
                if 'Volume' not in data.columns: data['Volume'] = 0

            # File path including the interval label
            filename = f"{symbol}_{interval['label']}_{today_str}.png"
            filepath = os.path.join(interval_dir, filename)

            # --- 5. GENERATE AND SAVE ---
            generate_single_chart_file(data, filepath, my_style, fig_size, show_volume=show_volume)

def draw_boxes_on_image(filepath, detections, show_volume=False):
    """
    Draw boxes on the saved image. 
    Each pattern gets a different color.
    """
    if not detections:
        return
        
    img = cv2.imread(filepath)
    if img is None:
        return
    
    # BGR Colors for OpenCV
    COLORS = [
        (0, 255, 0),   # Green
        (255, 165, 0), # Orange (BGR: 0, 165, 255) - wait, OpenCV is BGR
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (255, 0, 0),   # Blue
        (0, 165, 255), # Orange
        (0, 0, 255),   # Red
    ]
    
    h, w, _ = img.shape
    
    # We no longer apply a scale_y factor because we've increased the figure 
    # height to accommodate the volume panel while keeping the price panel 
    # roughly the same pixel height.
    
    for i, d in enumerate(detections):
        box = d["box"] # [x1, y1, x2, y2]
        name = d["name"]
        conf = d["conf"]
        
        x1, y1, x2, y2 = [int(val) for val in box]
        
        # Select color from list
        color = COLORS[i % len(COLORS)]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{name} {conf:.2f}"
        # Draw a small background for the text to make it readable
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Position background inside the box (under y1)
        cv2.rectangle(img, (x1, y1), (x1 + label_w + 5, y1 + label_h + 10), color, -1)
        # Position text inside the box
        cv2.putText(img, label, (x1 + 2, y1 + label_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    cv2.imwrite(filepath, img)

def generate_single_chart_file(data, filepath, style, fig_size, show_volume=False, detections=None):
    try:
        # If show_volume is True, we use panel_ratios to make volume "aggressive" (smaller)
        # and we pass the fig_size which is already taller.
        plot_kwargs = dict(
            type='candle', 
            style=style, 
            ylabel='Price',
            volume=show_volume,
            figsize=fig_size,
            tight_layout=True,
            show_nontrading=False,
            datetime_format='%b %d',
            xrotation=0,
            savefig=dict(fname=filepath, dpi=144)
        )
        
        if show_volume:
            plot_kwargs['panel_ratios'] = (6, 1) # Price panel 6x larger than volume
            
        mpf.plot(data, **plot_kwargs)
        
        if detections:
            draw_boxes_on_image(filepath, detections, show_volume=show_volume)
        print(f"  Saved chart to {filepath}")
    except Exception as e:
        print(f"  Failed to generate chart for {filepath}: {e}")

def generate_symbol_chart(symbol, interval_label, months, end_date_str, output_path, show_volume=True, detections=None):
    """
    Utility to generate a single chart for a symbol with specific parameters.
    Used by API to regenerate filtered charts with volume.
    """
    # Get data from DB
    full_data = load_cached_data(symbol)
    if full_data.empty:
        return False

    today_dt = pd.to_datetime(end_date_str)
    start_date = today_dt - pd.DateOffset(months=months)
    data = full_data[(full_data.index >= start_date) & (full_data.index <= today_dt)].copy()

    if data.empty:
        return False

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # Map DB columns
    if 'High' not in data.columns: data['High'] = data['Close']
    if 'Low' not in data.columns: data['Low'] = data['Close']
    if 'Volume' not in data.columns: data['Volume'] = 0

    # Define style
    my_colors = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='black', volume='inherit')
    my_style = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=my_colors, gridstyle='')
    
    # Increase height for volume charts: 5.65 -> 7.2 (approx 27% increase)
    # This accommodates the volume panel (1/7 of height) without shrinking the price panel.
    fig_size = (4.35, 7.2) if show_volume else (4.35, 5.65)

    generate_single_chart_file(data, output_path, my_style, fig_size, show_volume=show_volume, detections=detections)
    return True
