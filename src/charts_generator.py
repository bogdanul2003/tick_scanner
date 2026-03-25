import os
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import matplotlib
# Set the backend to Agg (non-interactive) before importing mplfinance/pyplot
# This prevents the Python icon from popping up in the macOS Dock
matplotlib.use('Agg')
import mplfinance as mpf
import pandas as pd
import cv2
from db_utils import get_watchlist_symbols, load_cached_data


def _get_chart_style():
    """Create and return the chart style (must be created in each process)."""
    my_colors = mpf.make_marketcolors(
        up='green', 
        down='red', 
        edge='inherit', 
        wick='black',    
        volume='inherit'
    )
    return mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        marketcolors=my_colors, 
        gridstyle=''           
    )


def _process_symbol(symbol, today_dt, today_str, output_dir, intervals, show_volume):
    """
    Process a single symbol - generate charts for all intervals.
    This function runs in a worker process.
    """
    from db_utils import load_cached_data  # Re-import in worker process
    
    results = []
    fig_size = (4.35, 5.65)
    my_style = _get_chart_style()
    
    # Get data from DB
    full_data = load_cached_data(symbol)
    
    if full_data.empty:
        return [(symbol, None, f"No data found in DB for {symbol}")]

    for interval in intervals:
        # Create interval-specific subfolder
        interval_dir = os.path.join(output_dir, interval['label'])
        os.makedirs(interval_dir, exist_ok=True)

        # Filter data for the specific interval
        start_date = today_dt - pd.DateOffset(months=interval['months'])
        data = full_data[(full_data.index >= start_date) & (full_data.index <= today_dt)].copy()

        if data.empty:
            results.append((symbol, interval['label'], f"No data between {start_date.date()} and {today_dt.date()}"))
            continue

        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Map DB column names to what mplfinance expects
        if 'High' not in data.columns: data['High'] = data['Close']
        if 'Low' not in data.columns: data['Low'] = data['Close']
        if 'Volume' not in data.columns: data['Volume'] = 0

        # File path including the interval label
        filename = f"{symbol}_{interval['label']}_{today_str}.png"
        filepath = os.path.join(interval_dir, filename)

        # Generate and save
        try:
            generate_single_chart_file(data, filepath, my_style, fig_size, show_volume=show_volume)
            results.append((symbol, interval['label'], filepath))
        except Exception as e:
            results.append((symbol, interval['label'], f"Error: {e}"))
    
    return results


def generate_charts_for_watchlist(watchlist_name, selected_date=None, show_volume=False, max_workers=4):
    """
    Generate charts for all symbols in a watchlist.
    
    Args:
        watchlist_name: Name of the watchlist
        selected_date: Optional date string (YYYY-MM-DD)
        show_volume: Whether to include volume bars
        max_workers: Number of parallel workers (default 4)
    """
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

    print(f"Generating charts for {len(symbols)} symbols in watchlist '{watchlist_name}' for {today_str} (Volume: {show_volume}, Workers: {max_workers})...")

    # --- 3. DEFINE INTERVALS ---
    intervals = [
        {'months': 6, 'label': '6m'},
        {'months': 3, 'label': '3m'}
    ]

    # --- 4. PARALLEL PROCESSING ---
    import time
    start_time = time.time()
    
    # Create output directories upfront
    for interval in intervals:
        os.makedirs(os.path.join(output_dir, interval['label']), exist_ok=True)
    
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                _process_symbol, 
                symbol, 
                today_dt, 
                today_str, 
                output_dir, 
                intervals, 
                show_volume
            ): symbol for symbol in symbols
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results = future.result()
                for sym, interval, result in results:
                    if result and not result.startswith("No data") and not result.startswith("Error"):
                        print(f"  Saved chart to {result}")
                        completed += 1
                    else:
                        print(f"  {sym} ({interval}): {result}")
                        failed += 1
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
                failed += 1
    
    elapsed = time.time() - start_time
    print(f"Completed: {completed} charts in {elapsed:.2f}s ({completed/elapsed:.1f} charts/sec), Failed: {failed}")

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
        
        # Calculate text size
        label = f"{name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Calculate background rectangle coordinates for right-aligned text inside the box
        # We want the text to end near x2 (right edge of box)
        bg_x2 = x2 - 2
        bg_x1 = bg_x2 - label_w - 4 # 4px padding
        
        bg_y1 = y1 + 2 # Just below the top edge
        bg_y2 = bg_y1 + label_h + 6 # Height + padding
        
        # Draw background rectangle
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        
        # Draw text
        # Text position is bottom-left origin
        text_x = bg_x1 + 2
        text_y = bg_y2 - 4
        
        cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
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


def _upgrade_single_chart(args):
    """
    Worker function for parallel volume chart upgrade.
    Args is a tuple: (symbol, interval_label, months, end_date_str, output_path, boxes)
    """
    symbol, interval_label, months, end_date_str, output_path, boxes = args
    try:
        result = generate_symbol_chart(
            symbol=symbol,
            interval_label=interval_label,
            months=months,
            end_date_str=end_date_str,
            output_path=output_path,
            show_volume=True,
            detections=boxes
        )
        return (symbol, interval_label, output_path, result)
    except Exception as e:
        return (symbol, interval_label, None, str(e))


def upgrade_charts_with_volume_parallel(detections_3m, detections_6m, base_dir, today_str, max_workers=4):
    """
    Upgrade filtered charts with volume bars in parallel.
    
    Args:
        detections_3m: List of detection results for 3-month charts
        detections_6m: List of detection results for 6-month charts
        base_dir: Base directory for output
        today_str: Date string (YYYY-MM-DD)
        max_workers: Number of parallel workers
        
    Returns:
        Number of successfully upgraded charts
    """
    import time
    start_time = time.time()
    
    # Prepare all tasks
    tasks = []
    
    for d in detections_3m:
        fname = d["filename"]
        boxes = d.get("boxes", [])
        symbol = fname.split('_')[0]
        save_path = os.path.join(base_dir, "3m", "filtered", fname)
        tasks.append((symbol, "3m", 3, today_str, save_path, boxes))
    
    for d in detections_6m:
        fname = d["filename"]
        boxes = d.get("boxes", [])
        symbol = fname.split('_')[0]
        save_path = os.path.join(base_dir, "6m", "filtered", fname)
        tasks.append((symbol, "6m", 6, today_str, save_path, boxes))
    
    if not tasks:
        return 0
    
    print(f"Upgrading {len(tasks)} charts with volume (Workers: {max_workers})...")
    
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_upgrade_single_chart, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, interval, path, result = future.result()
                if result is True:
                    print(f"  Upgraded {sym} ({interval}) with volume")
                    completed += 1
                else:
                    print(f"  Failed {sym} ({interval}): {result}")
                    failed += 1
            except Exception as e:
                print(f"  Error upgrading {symbol}: {e}")
                failed += 1
    
    elapsed = time.time() - start_time
    print(f"Volume upgrade completed: {completed} charts in {elapsed:.2f}s ({completed/elapsed:.1f} charts/sec), Failed: {failed}")
    
    return completed
