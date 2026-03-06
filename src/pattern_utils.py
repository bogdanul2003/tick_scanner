"""
Pattern detection utilities for stock chart patterns.
Implements Head and Shoulders and Inverse Head and Shoulders pattern detection.
"""

import numpy as np
from scipy.signal import find_peaks, argrelextrema
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import date


@dataclass
class PatternMatch:
    """Represents a detected chart pattern."""
    pattern_type: str  # 'head_and_shoulders' or 'inverse_head_and_shoulders'
    symbol: str
    start_date: date
    end_date: date
    confidence: float  # 0.0 to 1.0
    # Key points in the pattern
    left_shoulder_idx: int
    left_shoulder_price: float
    left_shoulder_date: date
    head_idx: int
    head_price: float
    head_date: date
    right_shoulder_idx: int
    right_shoulder_price: float
    right_shoulder_date: date
    neckline_price: float  # Average of the two troughs
    # For inverse patterns, troughs become peaks
    left_trough_idx: Optional[int] = None
    left_trough_price: Optional[float] = None
    left_trough_date: Optional[date] = None
    right_trough_idx: Optional[int] = None
    right_trough_price: Optional[float] = None
    right_trough_date: Optional[date] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        def format_date(d):
            if d is None:
                return None
            if isinstance(d, str):
                return d
            return d.isoformat()
        
        return {
            "pattern_type": self.pattern_type,
            "symbol": self.symbol,
            "start_date": format_date(self.start_date),
            "end_date": format_date(self.end_date),
            "confidence": round(self.confidence, 3),
            "left_shoulder": {
                "index": self.left_shoulder_idx,
                "price": round(self.left_shoulder_price, 2),
                "date": format_date(self.left_shoulder_date)
            },
            "head": {
                "index": self.head_idx,
                "price": round(self.head_price, 2),
                "date": format_date(self.head_date)
            },
            "right_shoulder": {
                "index": self.right_shoulder_idx,
                "price": round(self.right_shoulder_price, 2),
                "date": format_date(self.right_shoulder_date)
            },
            "neckline_price": round(self.neckline_price, 2),
            "left_trough": {
                "index": self.left_trough_idx,
                "price": round(self.left_trough_price, 2) if self.left_trough_price else None,
                "date": format_date(self.left_trough_date)
            } if self.left_trough_idx is not None else None,
            "right_trough": {
                "index": self.right_trough_idx,
                "price": round(self.right_trough_price, 2) if self.right_trough_price else None,
                "date": format_date(self.right_trough_date)
            } if self.right_trough_idx is not None else None
        }


class PatternDetector:
    """
    Detects chart patterns in price data.
    """
    
    def __init__(self, 
                 peak_distance: int = 5,
                 shoulder_tolerance: float = 0.03,
                 head_min_prominence: float = 0.02,
                 min_pattern_length: int = 10,
                 max_pattern_length: int = 60):
        """
        Initialize the pattern detector.
        
        Args:
            peak_distance: Minimum distance between peaks/troughs
            shoulder_tolerance: Max percentage difference between shoulders (0.03 = 3%)
            head_min_prominence: Min percentage the head must be above shoulders
            min_pattern_length: Minimum number of days for a valid pattern
            max_pattern_length: Maximum number of days for a valid pattern
        """
        self.peak_distance = peak_distance
        self.shoulder_tolerance = shoulder_tolerance
        self.head_min_prominence = head_min_prominence
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
    
    def find_peaks_and_troughs(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find local peaks (maxima) and troughs (minima) in price data.
        
        Args:
            prices: Array of closing prices
            
        Returns:
            Tuple of (peak_indices, trough_indices)
        """
        # Find peaks
        peaks, _ = find_peaks(prices, distance=self.peak_distance)
        
        # Find troughs (peaks in inverted data)
        troughs, _ = find_peaks(-prices, distance=self.peak_distance)
        
        return peaks, troughs
    
    def _calculate_confidence(self, 
                              left_shoulder: float, 
                              head: float, 
                              right_shoulder: float,
                              left_trough: float,
                              right_trough: float,
                              is_inverse: bool = False) -> float:
        """
        Calculate confidence score for a pattern match.
        
        Factors considered:
        - Symmetry of shoulders
        - Symmetry of troughs (neckline)
        - Head prominence
        """
        # Shoulder symmetry (how close are left and right shoulders)
        avg_shoulder = (left_shoulder + right_shoulder) / 2
        shoulder_diff = abs(left_shoulder - right_shoulder) / avg_shoulder
        shoulder_score = max(0, 1 - shoulder_diff / self.shoulder_tolerance)
        
        # Neckline symmetry (how level is the neckline)
        avg_trough = (left_trough + right_trough) / 2
        trough_diff = abs(left_trough - right_trough) / avg_trough
        neckline_score = max(0, 1 - trough_diff / self.shoulder_tolerance)
        
        # Head prominence
        if is_inverse:
            # For inverse, head should be lower than shoulders
            head_prominence = (avg_shoulder - head) / avg_shoulder
        else:
            # For regular, head should be higher than shoulders
            head_prominence = (head - avg_shoulder) / avg_shoulder
        
        prominence_score = min(1.0, head_prominence / (self.head_min_prominence * 2))
        
        # Combined confidence
        confidence = (shoulder_score * 0.35 + neckline_score * 0.30 + prominence_score * 0.35)
        return max(0.0, min(1.0, confidence))
    
    def detect_head_and_shoulders(self, 
                                   symbol: str,
                                   prices: np.ndarray, 
                                   dates: List[date]) -> List[PatternMatch]:
        """
        Detect Head and Shoulders patterns in price data.
        
        A Head and Shoulders pattern consists of:
        1. Left shoulder (peak)
        2. Trough
        3. Head (higher peak)
        4. Trough
        5. Right shoulder (peak similar to left shoulder)
        
        Args:
            symbol: Stock symbol
            prices: Array of closing prices
            dates: List of corresponding dates
            
        Returns:
            List of detected PatternMatch objects
        """
        if len(prices) < self.min_pattern_length:
            return []
        
        prices = np.array(prices)
        peaks, troughs = self.find_peaks_and_troughs(prices)
        
        patterns = []
        
        # Need at least 3 peaks and 2 troughs for H&S
        if len(peaks) < 3 or len(troughs) < 2:
            return []
        
        # Iterate through possible head positions
        for head_i in range(1, len(peaks) - 1):
            head_idx = peaks[head_i]
            head_price = prices[head_idx]
            
            # Try each combination of left and right shoulders
            for left_i in range(head_i):
                left_shoulder_idx = peaks[left_i]
                left_shoulder_price = prices[left_shoulder_idx]
                
                # Left shoulder must be before head
                if left_shoulder_idx >= head_idx:
                    continue
                
                for right_i in range(head_i + 1, len(peaks)):
                    right_shoulder_idx = peaks[right_i]
                    right_shoulder_price = prices[right_shoulder_idx]
                    
                    # Right shoulder must be after head
                    if right_shoulder_idx <= head_idx:
                        continue
                    
                    # Check pattern length
                    pattern_length = right_shoulder_idx - left_shoulder_idx
                    if pattern_length < self.min_pattern_length or pattern_length > self.max_pattern_length:
                        continue
                    
                    # Head must be higher than both shoulders
                    if head_price <= left_shoulder_price or head_price <= right_shoulder_price:
                        continue
                    
                    # Head must be at least head_min_prominence higher
                    avg_shoulder = (left_shoulder_price + right_shoulder_price) / 2
                    if (head_price - avg_shoulder) / avg_shoulder < self.head_min_prominence:
                        continue
                    
                    # Shoulders should be within tolerance of each other
                    shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / avg_shoulder
                    if shoulder_diff > self.shoulder_tolerance:
                        continue
                    
                    # Find troughs between shoulders
                    left_trough_candidates = troughs[(troughs > left_shoulder_idx) & (troughs < head_idx)]
                    right_trough_candidates = troughs[(troughs > head_idx) & (troughs < right_shoulder_idx)]
                    
                    if len(left_trough_candidates) == 0 or len(right_trough_candidates) == 0:
                        continue
                    
                    # Use the lowest trough on each side
                    left_trough_idx = left_trough_candidates[np.argmin(prices[left_trough_candidates])]
                    right_trough_idx = right_trough_candidates[np.argmin(prices[right_trough_candidates])]
                    
                    left_trough_price = prices[left_trough_idx]
                    right_trough_price = prices[right_trough_idx]
                    
                    # Troughs should form a relatively level neckline
                    avg_trough = (left_trough_price + right_trough_price) / 2
                    trough_diff = abs(left_trough_price - right_trough_price) / avg_trough
                    if trough_diff > self.shoulder_tolerance * 1.5:  # Slightly more tolerance for neckline
                        continue
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        left_shoulder_price, head_price, right_shoulder_price,
                        left_trough_price, right_trough_price, is_inverse=False
                    )
                    
                    if confidence < 0.5:  # Minimum confidence threshold
                        continue
                    
                    pattern = PatternMatch(
                        pattern_type="head_and_shoulders",
                        symbol=symbol,
                        start_date=dates[left_shoulder_idx],
                        end_date=dates[right_shoulder_idx],
                        confidence=confidence,
                        left_shoulder_idx=int(left_shoulder_idx),
                        left_shoulder_price=float(left_shoulder_price),
                        left_shoulder_date=dates[left_shoulder_idx],
                        head_idx=int(head_idx),
                        head_price=float(head_price),
                        head_date=dates[head_idx],
                        right_shoulder_idx=int(right_shoulder_idx),
                        right_shoulder_price=float(right_shoulder_price),
                        right_shoulder_date=dates[right_shoulder_idx],
                        neckline_price=float(avg_trough),
                        left_trough_idx=int(left_trough_idx),
                        left_trough_price=float(left_trough_price),
                        left_trough_date=dates[left_trough_idx],
                        right_trough_idx=int(right_trough_idx),
                        right_trough_price=float(right_trough_price),
                        right_trough_date=dates[right_trough_idx]
                    )
                    patterns.append(pattern)
        
        # Remove overlapping patterns, keep highest confidence
        patterns = self._remove_overlapping_patterns(patterns)
        
        return patterns
    
    def detect_inverse_head_and_shoulders(self,
                                           symbol: str,
                                           prices: np.ndarray,
                                           dates: List[date]) -> List[PatternMatch]:
        """
        Detect Inverse Head and Shoulders patterns in price data.
        
        An Inverse Head and Shoulders pattern is the mirror of H&S:
        1. Left shoulder (trough)
        2. Peak
        3. Head (lower trough)
        4. Peak
        5. Right shoulder (trough similar to left shoulder)
        
        Args:
            symbol: Stock symbol
            prices: Array of closing prices
            dates: List of corresponding dates
            
        Returns:
            List of detected PatternMatch objects
        """
        if len(prices) < self.min_pattern_length:
            return []
        
        prices = np.array(prices)
        peaks, troughs = self.find_peaks_and_troughs(prices)
        
        patterns = []
        
        # Need at least 3 troughs and 2 peaks for inverse H&S
        if len(troughs) < 3 or len(peaks) < 2:
            return []
        
        # Iterate through possible head positions (troughs for inverse)
        for head_i in range(1, len(troughs) - 1):
            head_idx = troughs[head_i]
            head_price = prices[head_idx]
            
            # Try each combination of left and right shoulders (also troughs)
            for left_i in range(head_i):
                left_shoulder_idx = troughs[left_i]
                left_shoulder_price = prices[left_shoulder_idx]
                
                if left_shoulder_idx >= head_idx:
                    continue
                
                for right_i in range(head_i + 1, len(troughs)):
                    right_shoulder_idx = troughs[right_i]
                    right_shoulder_price = prices[right_shoulder_idx]
                    
                    if right_shoulder_idx <= head_idx:
                        continue
                    
                    # Check pattern length
                    pattern_length = right_shoulder_idx - left_shoulder_idx
                    if pattern_length < self.min_pattern_length or pattern_length > self.max_pattern_length:
                        continue
                    
                    # Head must be LOWER than both shoulders (inverse)
                    if head_price >= left_shoulder_price or head_price >= right_shoulder_price:
                        continue
                    
                    # Head must be at least head_min_prominence lower
                    avg_shoulder = (left_shoulder_price + right_shoulder_price) / 2
                    if (avg_shoulder - head_price) / avg_shoulder < self.head_min_prominence:
                        continue
                    
                    # Shoulders should be within tolerance
                    shoulder_diff = abs(left_shoulder_price - right_shoulder_price) / avg_shoulder
                    if shoulder_diff > self.shoulder_tolerance:
                        continue
                    
                    # Find peaks between shoulders (neckline for inverse)
                    left_peak_candidates = peaks[(peaks > left_shoulder_idx) & (peaks < head_idx)]
                    right_peak_candidates = peaks[(peaks > head_idx) & (peaks < right_shoulder_idx)]
                    
                    if len(left_peak_candidates) == 0 or len(right_peak_candidates) == 0:
                        continue
                    
                    # Use the highest peak on each side
                    left_peak_idx = left_peak_candidates[np.argmax(prices[left_peak_candidates])]
                    right_peak_idx = right_peak_candidates[np.argmax(prices[right_peak_candidates])]
                    
                    left_peak_price = prices[left_peak_idx]
                    right_peak_price = prices[right_peak_idx]
                    
                    # Peaks should form a relatively level neckline
                    avg_peak = (left_peak_price + right_peak_price) / 2
                    peak_diff = abs(left_peak_price - right_peak_price) / avg_peak
                    if peak_diff > self.shoulder_tolerance * 1.5:
                        continue
                    
                    # Calculate confidence
                    confidence = self._calculate_confidence(
                        left_shoulder_price, head_price, right_shoulder_price,
                        left_peak_price, right_peak_price, is_inverse=True
                    )
                    
                    if confidence < 0.5:
                        continue
                    
                    pattern = PatternMatch(
                        pattern_type="inverse_head_and_shoulders",
                        symbol=symbol,
                        start_date=dates[left_shoulder_idx],
                        end_date=dates[right_shoulder_idx],
                        confidence=confidence,
                        left_shoulder_idx=int(left_shoulder_idx),
                        left_shoulder_price=float(left_shoulder_price),
                        left_shoulder_date=dates[left_shoulder_idx],
                        head_idx=int(head_idx),
                        head_price=float(head_price),
                        head_date=dates[head_idx],
                        right_shoulder_idx=int(right_shoulder_idx),
                        right_shoulder_price=float(right_shoulder_price),
                        right_shoulder_date=dates[right_shoulder_idx],
                        neckline_price=float(avg_peak),
                        left_trough_idx=int(left_peak_idx),
                        left_trough_price=float(left_peak_price),
                        left_trough_date=dates[left_peak_idx],
                        right_trough_idx=int(right_peak_idx),
                        right_trough_price=float(right_peak_price),
                        right_trough_date=dates[right_peak_idx]
                    )
                    patterns.append(pattern)
        
        patterns = self._remove_overlapping_patterns(patterns)
        
        return patterns
    
    def _remove_overlapping_patterns(self, patterns: List[PatternMatch]) -> List[PatternMatch]:
        """Remove overlapping patterns, keeping the one with highest confidence."""
        if len(patterns) <= 1:
            return patterns
        
        # Sort by confidence descending
        patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)
        
        kept_patterns = []
        for pattern in patterns:
            overlaps = False
            for kept in kept_patterns:
                # Check if patterns overlap
                if not (pattern.right_shoulder_idx < kept.left_shoulder_idx or 
                        pattern.left_shoulder_idx > kept.right_shoulder_idx):
                    overlaps = True
                    break
            if not overlaps:
                kept_patterns.append(pattern)
        
        return kept_patterns
    
    def detect_all_patterns(self,
                            symbol: str,
                            prices: np.ndarray,
                            dates: List[date]) -> List[PatternMatch]:
        """
        Detect all supported patterns in price data.
        
        Args:
            symbol: Stock symbol
            prices: Array of closing prices
            dates: List of corresponding dates
            
        Returns:
            List of all detected PatternMatch objects
        """
        all_patterns = []
        
        all_patterns.extend(self.detect_head_and_shoulders(symbol, prices, dates))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(symbol, prices, dates))
        
        # Sort by end date descending (most recent first)
        all_patterns.sort(key=lambda p: p.end_date, reverse=True)
        
        return all_patterns


def scan_symbol_for_patterns(symbol: str, 
                              days: int = 120,
                              pattern_type: str = "all") -> List[dict]:
    """
    Scan a symbol for chart patterns.
    
    Args:
        symbol: Stock symbol to scan
        days: Number of days of historical data to analyze
        pattern_type: 'all', 'head_and_shoulders', or 'inverse_head_and_shoulders'
        
    Returns:
        List of detected patterns as dictionaries
    """
    from datetime import timedelta
    from macd_utils import get_latest_market_date, get_macd_for_range
    
    end_date = get_latest_market_date()
    start_date = end_date - timedelta(days=days)
    
    # Get price data
    data = get_macd_for_range(symbol, start_date, end_date)
    
    if not data:
        return []
    
    # Extract closing prices and dates
    prices = []
    dates = []
    for d in data:
        if "close" in d and d["close"] is not None:
            prices.append(d["close"])
            dates.append(d["date"])
    
    if len(prices) < 20:  # Need minimum data
        return []
    
    detector = PatternDetector()
    
    if pattern_type == "head_and_shoulders":
        patterns = detector.detect_head_and_shoulders(symbol, np.array(prices), dates)
    elif pattern_type == "inverse_head_and_shoulders":
        patterns = detector.detect_inverse_head_and_shoulders(symbol, np.array(prices), dates)
    else:
        patterns = detector.detect_all_patterns(symbol, np.array(prices), dates)
    
    return [p.to_dict() for p in patterns]


def scan_watchlist_for_patterns(watchlist_name: str,
                                 days: int = 120,
                                 pattern_type: str = "all") -> dict:
    """
    Scan all symbols in a watchlist for chart patterns.
    
    Args:
        watchlist_name: Name of the watchlist to scan
        days: Number of days of historical data to analyze
        pattern_type: 'all', 'head_and_shoulders', or 'inverse_head_and_shoulders'
        
    Returns:
        Dictionary with symbol as key and list of patterns as value
    """
    from db_utils import get_watchlist_symbols
    
    symbols = get_watchlist_symbols(watchlist_name)
    results = {}
    
    for symbol in symbols:
        try:
            patterns = scan_symbol_for_patterns(symbol, days, pattern_type)
            if patterns:  # Only include symbols with detected patterns
                results[symbol] = patterns
        except Exception as e:
            print(f"Error scanning {symbol} for patterns: {e}")
            continue
    
    return results
