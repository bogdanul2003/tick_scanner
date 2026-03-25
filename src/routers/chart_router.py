"""Chart generation and neural detection API endpoints."""
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os
import json
import traceback

router = APIRouter(prefix="/charts", tags=["Charts"])


class ChartGenerationRequest(BaseModel):
    selected_date: Optional[str] = None


@router.get("/watchlist/{watchlist_name}/available_dates")
async def api_get_available_dates(watchlist_name: str):
    """Get dates where data is available for a significant number of symbols in the watchlist."""
    try:
        from db_utils import get_available_dates_for_watchlist
        dates = get_available_dates_for_watchlist(watchlist_name)
        return {"dates": dates}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Exception in api_get_available_dates: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/{watchlist_name}/generate")
async def api_generate_watchlist_charts(
    watchlist_name: str, 
    payload: ChartGenerationRequest = None
):
    """
    Generate candle charts and run pattern detection for all companies in a watchlist.
    Returns the relative URLs of the images that passed the pattern filter.
    """
    try:
        from charts_generator import generate_charts_for_watchlist, generate_symbol_chart
        from detector_neural import run_detection
        from db_utils import save_patterns_to_cache
        
        # 0. Define paths
        selected_date = payload.selected_date if payload and payload.selected_date else None
        today_str = selected_date or datetime.now().strftime('%Y-%m-%d')

        safe_watchlist = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in watchlist_name)
        base_dir = os.path.join("..", "generated_charts", safe_watchlist, today_str)
        results_cache_path = os.path.join(base_dir, "results.json")
        
        # 1. Check if we already have results for this date
        if os.path.exists(results_cache_path):
            print(f"Loading cached results for '{watchlist_name}' on {today_str} from {results_cache_path}")
            try:
                with open(results_cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}. Re-generating...")

        # 2. Generate the charts (WITHOUT volume for detection)
        generate_charts_for_watchlist(watchlist_name, selected_date=selected_date, show_volume=False)
        
        # 3. Run Neural Detection for 3m (find_x=550) and 6m (find_x=555)
        detections_3m = run_detection(
            os.path.join(base_dir, "3m"), 
            os.path.join(base_dir, "3m", "filtered"), 
            find_x=550
        )
        detections_6m = run_detection(
            os.path.join(base_dir, "6m"), 
            os.path.join(base_dir, "6m", "filtered"), 
            find_x=565
        )

        # 4. Re-generate filtered charts WITH volume
        def upgrade_to_volume_charts(detections, interval_label, months):
            for d in detections:
                fname = d["filename"]
                boxes = d.get("boxes", [])
                symbol = fname.split('_')[0]
                save_path = os.path.join(base_dir, interval_label, "filtered", fname)
                print(f"Upgrading {fname} to include volume bars and bounding boxes...")
                generate_symbol_chart(
                    symbol=symbol,
                    interval_label=interval_label,
                    months=months,
                    end_date_str=today_str,
                    output_path=save_path,
                    show_volume=True,
                    detections=boxes
                )

        upgrade_to_volume_charts(detections_3m, "3m", 3)
        upgrade_to_volume_charts(detections_6m, "6m", 6)

        # 5. Save the two rightmost patterns to DB for each symbol
        target_date = datetime.strptime(today_str, '%Y-%m-%d').date()
        
        all_symbol_detections = {}
        for d in detections_3m + detections_6m:
            fname = d["filename"]
            symbol = fname.split('_')[0]
            if symbol not in all_symbol_detections:
                all_symbol_detections[symbol] = []
            
            if "boxes" in d:
                for b in d["boxes"]:
                    exists = False
                    for existing in all_symbol_detections[symbol]:
                        if existing["name"] == b["name"] and existing["box"] == b["box"]:
                            exists = True
                            break
                    if not exists:
                        all_symbol_detections[symbol].append(b)

        for symbol, boxes in all_symbol_detections.items():
            if not boxes:
                continue
            sorted_boxes = sorted(boxes, key=lambda x: x["box"][2], reverse=True)
            rightmost_two = sorted_boxes[:2]
            print(f"Saving {len(rightmost_two)} patterns to DB for {symbol} on {today_str}")
            save_patterns_to_cache(symbol, target_date, rightmost_two)
        
        # 6. Build URLs and categorize
        bullish_images = []
        bearish_images = []
        
        def process_detections(detections, interval_label):
            for d in detections:
                fname = d["filename"]
                pattern = d["rightmost_pattern"]
                url = f"/charts/{safe_watchlist}/{today_str}/{interval_label}/filtered/{fname}"
                
                is_bullish = False
                if pattern:
                    p_lower = pattern.lower()
                    if "bottom" in p_lower:
                        is_bullish = True
                
                if is_bullish:
                    bullish_images.append(url)
                else:
                    bearish_images.append(url)

        process_detections(detections_3m, "3m")
        process_detections(detections_6m, "6m")
        
        final_result = {
            "message": f"Charts generated and scanned for '{watchlist_name}' on {today_str}",
            "watchlist": watchlist_name,
            "date": today_str,
            "count": len(bullish_images) + len(bearish_images),
            "bullish": bullish_images,
            "bearish": bearish_images,
            "images": bullish_images + bearish_images,
            "status": "success"
        }
        
        # Save to cache
        os.makedirs(base_dir, exist_ok=True)
        with open(results_cache_path, "w") as f:
            json.dump(final_result, f)
            
        return final_result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Exception in api_generate_watchlist_charts: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
