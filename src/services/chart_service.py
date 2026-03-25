"""Chart generation and pattern detection service."""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ChartService:
    """Service for generating charts and running neural pattern detection."""
    
    def __init__(self, charts_base_dir: str = "../generated_charts"):
        """
        Initialize the chart service.
        
        Args:
            charts_base_dir: Base directory for generated charts
        """
        self.charts_base_dir = charts_base_dir
    
    def generate_and_scan_watchlist(
        self, 
        watchlist_name: str, 
        selected_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate charts for watchlist and run neural detection.
        
        Args:
            watchlist_name: Name of the watchlist
            selected_date: Optional date string (YYYY-MM-DD)
            
        Returns:
            Result dictionary with generated chart URLs and detection results
        """
        from charts_generator import generate_charts_for_watchlist, generate_symbol_chart
        from detector_neural import run_detection
        from db_utils import save_patterns_to_cache
        
        today_str = selected_date or datetime.now().strftime('%Y-%m-%d')
        safe_watchlist = self._sanitize_name(watchlist_name)
        base_dir = os.path.join(self.charts_base_dir, safe_watchlist, today_str)
        
        # Check cache
        cached = self._load_cached_results(base_dir)
        if cached:
            logger.info(f"Using cached results for {watchlist_name} on {today_str}")
            return cached
        
        # Generate charts without volume (for detection)
        logger.info(f"Generating charts for {watchlist_name}")
        generate_charts_for_watchlist(
            watchlist_name, 
            selected_date=selected_date, 
            show_volume=False
        )
        
        # Run pattern detection
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
        
        # Upgrade filtered charts with volume
        self._upgrade_charts_with_volume(
            detections_3m, "3m", 3, base_dir, today_str, generate_symbol_chart
        )
        self._upgrade_charts_with_volume(
            detections_6m, "6m", 6, base_dir, today_str, generate_symbol_chart
        )
        
        # Save patterns to DB
        target_date = datetime.strptime(today_str, '%Y-%m-%d').date()
        self._save_detected_patterns(
            detections_3m + detections_6m, 
            target_date, 
            save_patterns_to_cache
        )
        
        # Build result
        result = self._build_result(
            watchlist_name, 
            today_str, 
            safe_watchlist, 
            detections_3m, 
            detections_6m
        )
        
        # Cache results
        self._cache_results(base_dir, result)
        
        return result
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize watchlist name for filesystem use."""
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    
    def _load_cached_results(self, base_dir: str) -> Optional[Dict[str, Any]]:
        """Load cached results if available."""
        cache_path = os.path.join(base_dir, "results.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None
    
    def _cache_results(self, base_dir: str, result: Dict[str, Any]) -> None:
        """Cache results to filesystem."""
        os.makedirs(base_dir, exist_ok=True)
        cache_path = os.path.join(base_dir, "results.json")
        with open(cache_path, "w") as f:
            json.dump(result, f)
    
    def _upgrade_charts_with_volume(
        self, 
        detections: List[Dict], 
        interval_label: str, 
        months: int,
        base_dir: str,
        today_str: str,
        generate_func
    ) -> None:
        """Re-generate detected charts with volume bars."""
        for d in detections:
            fname = d["filename"]
            boxes = d.get("boxes", [])
            symbol = fname.split('_')[0]
            save_path = os.path.join(base_dir, interval_label, "filtered", fname)
            logger.debug(f"Upgrading {fname} with volume bars")
            generate_func(
                symbol=symbol,
                interval_label=interval_label,
                months=months,
                end_date_str=today_str,
                output_path=save_path,
                show_volume=True,
                detections=boxes
            )
    
    def _save_detected_patterns(
        self, 
        all_detections: List[Dict],
        target_date,
        save_func
    ) -> None:
        """Save the two rightmost patterns for each symbol to DB."""
        symbol_detections = {}
        
        for d in all_detections:
            fname = d["filename"]
            symbol = fname.split('_')[0]
            if symbol not in symbol_detections:
                symbol_detections[symbol] = []
            
            if "boxes" in d:
                for b in d["boxes"]:
                    # Check for duplicates
                    exists = any(
                        existing["name"] == b["name"] and existing["box"] == b["box"]
                        for existing in symbol_detections[symbol]
                    )
                    if not exists:
                        symbol_detections[symbol].append(b)
        
        for symbol, boxes in symbol_detections.items():
            if not boxes:
                continue
            
            # Sort by x2 coordinate (rightmost first)
            sorted_boxes = sorted(boxes, key=lambda x: x["box"][2], reverse=True)
            rightmost_two = sorted_boxes[:2]
            
            logger.debug(f"Saving {len(rightmost_two)} patterns for {symbol}")
            save_func(symbol, target_date, rightmost_two)
    
    def _build_result(
        self,
        watchlist_name: str,
        today_str: str,
        safe_watchlist: str,
        detections_3m: List[Dict],
        detections_6m: List[Dict]
    ) -> Dict[str, Any]:
        """Build the response dictionary."""
        bullish_images = []
        bearish_images = []
        
        def process_detections(detections: List[Dict], interval_label: str):
            for d in detections:
                fname = d["filename"]
                pattern = d.get("rightmost_pattern", "")
                url = f"/charts/{safe_watchlist}/{today_str}/{interval_label}/filtered/{fname}"
                
                is_bullish = pattern and "bottom" in pattern.lower()
                
                if is_bullish:
                    bullish_images.append(url)
                else:
                    bearish_images.append(url)
        
        process_detections(detections_3m, "3m")
        process_detections(detections_6m, "6m")
        
        return {
            "message": f"Charts generated and scanned for '{watchlist_name}' on {today_str}",
            "watchlist": watchlist_name,
            "date": today_str,
            "count": len(bullish_images) + len(bearish_images),
            "bullish": bullish_images,
            "bearish": bearish_images,
            "images": bullish_images + bearish_images,
            "status": "success"
        }


# Default service instance
chart_service = ChartService()
