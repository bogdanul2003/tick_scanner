import React, { useState } from "react";
import '../App.css'
import { API_BASE } from "../api";

// Add this new component to render MACD history chart
function MacdChart({ data }) {
  // Return early if no data
  if (!data || !data.dates || !data.macd || !data.signal_line) {
    return <div>No data available</div>;
  }

  // Simple canvas-based chart
  const canvasRef = React.useRef(null);
  
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height - 20; // Reserve space for x-axis labels
    const padding = { top: 20, right: 5, bottom: 20, left: 5 };
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height + padding.bottom);
    
    // Find min and max values for scaling
    const allValues = [...data.macd, ...data.signal_line];
    let minValue = Math.min(...allValues);
    let maxValue = Math.max(...allValues);
    
    // Always ensure zero is in the range for proper scaling
    if (minValue > 0) minValue = 0;
    if (maxValue < 0) maxValue = 0;
    
    // Add a small buffer to min/max for better visualization
    const buffer = (maxValue - minValue) * 0.1;
    minValue -= buffer;
    maxValue += buffer;
    
    const range = maxValue - minValue;
    
    // Draw zero line - now it will always be visible
    const zeroY = height - ((0 - minValue) / range) * height;
    ctx.beginPath();
    ctx.strokeStyle = "#888";
    ctx.setLineDash([2, 2]);
    ctx.moveTo(padding.left, zeroY);
    ctx.lineTo(width - padding.right, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw MACD line (blue)
    ctx.beginPath();
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2;
    
    data.macd.forEach((value, i) => {
      const x = padding.left + (i / (data.macd.length - 1)) * (width - padding.left - padding.right);
      const y = height - ((value - minValue) / range) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Draw signal line (red)
    ctx.beginPath();
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    
    data.signal_line.forEach((value, i) => {
      const x = padding.left + (i / (data.signal_line.length - 1)) * (width - padding.left - padding.right);
      const y = height - ((value - minValue) / range) * height;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Removed X-axis date labels as requested
    
    // Add Y-axis min/max labels
    ctx.textAlign = "left";
    ctx.fillStyle = "black";
    ctx.font = "10px Arial";
    ctx.fillText(maxValue.toFixed(3), padding.left, padding.top - 5);
    ctx.fillText(minValue.toFixed(3), padding.left, height - 5);
    ctx.fillText("0", padding.left, zeroY - 5);
    
    // Add legend - move to middle-top to avoid overlap with Y-axis labels
    const legendY = 10; // Keep the same Y position
    const legendX = width / 2 - 60; // Center in the canvas, adjust for legend width
    
    ctx.fillStyle = "black";
    ctx.font = "10px Arial";
    ctx.textAlign = "left";
    ctx.fillText("MACD", legendX, legendY);
    ctx.fillStyle = "blue";
    ctx.fillRect(legendX + 35, legendY - 5, 15, 5);
    
    ctx.fillStyle = "black";
    ctx.fillText("Signal", legendX + 60, legendY);
    ctx.fillStyle = "red";
    ctx.fillRect(legendX + 95, legendY - 5, 15, 5);
    
  }, [data]);
  
  return (
    <div style={{ padding: 5 }}>
      <canvas 
        ref={canvasRef} 
        width={250} 
        height={170} // Kept the same height even though X-axis labels are removed
        style={{ border: '1px solid #ddd' }}
      />
    </div>
  );
}

function MacdLookup() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchMacd = async () => {
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/macd/${symbol}`);
      setResult(await res.json());
    } catch (e) {
      setResult({ error: "Error fetching MACD" });
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>MACD Lookup</h2>
      <input value={symbol} onChange={e => setSymbol(e.target.value)} placeholder="Symbol" />
      <button onClick={fetchMacd}>Get MACD</button>
      {loading && <div>Loading...</div>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

function BulkMacd() {
  const [symbols, setSymbols] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchBulk = async () => {
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/macd/bulk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols: symbols.split(",").map(s => s.trim()) }),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: "Error fetching bulk MACD" });
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>Bulk MACD</h2>
      <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="AAPL,MSFT,GOOG" />
      <button onClick={fetchBulk}>Get Bulk MACD</button>
      {loading && <div>Loading...</div>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

function BullishSignal() {
  const [symbols, setSymbols] = useState("");
  const [days, setDays] = useState(30);
  const [threshold, setThreshold] = useState(0.05);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchSignal = async () => {
    setResult(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/macd/bullish_signal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols: symbols.split(",").map(s => s.trim()), days, threshold }),
      });
      setResult(await res.json());
    } catch (e) {
      setResult({ error: "Error fetching bullish signal" });
    }
    setLoading(false);
  };

  return (
    <div>
      <h2>Bullish MACD Signal</h2>
      <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="AAPL,MSFT,GOOG" />
      <input type="number" value={days} onChange={e => setDays(Number(e.target.value))} />
      <input type="number" value={threshold} step="0.01" onChange={e => setThreshold(Number(e.target.value))} />
      <button onClick={fetchSignal}>Check Signal</button>
      {loading && <div>Loading...</div>}
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

const INTERVAL_OPTIONS = [
  { value: 30, label: "1 month" },
  { value: 90, label: "3 months" },
  { value: 180, label: "6 months" },
  { value: 365, label: "12 months" }
];

function WatchlistBullishSignal({ watchlist, onClose }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [hoveredSymbol, setHoveredSymbol] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [chartPosition, setChartPosition] = useState({ x: 0, y: 0 });
  const [chartLoading, setChartLoading] = useState(false);
  // Add a ref to track if we've already made the API call
  const fetchedRef = React.useRef(false);
  // Add download state
  const [downloading, setDownloading] = useState(false);
  // History chart state
  const [historyDays, setHistoryDays] = useState(180);
  const [historyData, setHistoryData] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [hoveredHistoryIndex, setHoveredHistoryIndex] = useState(null);



  React.useEffect(() => {
    const fetchSignal = async () => {
      // Skip if we've already fetched data in this component instance
      if (fetchedRef.current) return;
      fetchedRef.current = true;
      
      setLoading(true);
      setError("");
      setResult(null);
      try {
        const res = await fetch(`${API_BASE}/watchlist/${encodeURIComponent(watchlist)}/bullish_signal`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ days: 30, threshold: 0.05 }),
        });
        if (!res.ok) {
          const err = await res.json();
          setError(err.detail || "Error fetching signal");
        } else {
          setResult(await res.json());
        }
      } catch (e) {
        setError("Error fetching signal");
      }
      setLoading(false);
    };
    fetchSignal();
  }, [watchlist]);

  // New effect to fetch MACD history when a symbol is hovered
  React.useEffect(() => {
    if (!hoveredSymbol) {
      setChartData(null);
      return;
    }
    
    const fetchMacdHistory = async () => {
      setChartLoading(true);
      try {
        const res = await fetch(`${API_BASE}/macd/${hoveredSymbol}/history?days=30`);
        if (res.ok) {
          const responseData = await res.json();
          
          // Transform the data into the format expected by MacdChart
          const validData = responseData.filter(item => !item.error);
          if (validData.length > 0) {
            const transformedData = {
              dates: validData.map(item => item.date),
              macd: validData.map(item => item.macd),
              signal_line: validData.map(item => item.signal_line)
            };
            setChartData(transformedData);
          } else {
            setChartData(null);
          }
        } else {
          console.error("Failed to fetch MACD history");
          setChartData(null);
        }
      } catch (e) {
        console.error("Error fetching MACD history:", e);
        setChartData(null);
      }
      setChartLoading(false);
    };
    
    fetchMacdHistory();
  }, [hoveredSymbol]);

  // Handle mouse events
  const handleMouseEnter = (symbol, e) => {
    setHoveredSymbol(symbol);
    setChartPosition({
      x: e.clientX,
      y: e.clientY
    });
  };

  const handleMouseLeave = () => {
    setHoveredSymbol(null);
  };

  // Fetch history for the column-count line chart
  React.useEffect(() => {
    const fetchHistory = async () => {
      setHistoryLoading(true);
      try {
        const res = await fetch(
          `${API_BASE}/watchlist/${encodeURIComponent(watchlist)}/bullish_signal_history?days=${historyDays}`
        );
        if (res.ok) {
          const data = await res.json();
          setHistoryData(data.history || []);
        }
      } catch (e) {
        console.error("Error fetching bullish signal history:", e);
      }
      setHistoryLoading(false);
    };
    fetchHistory();
  }, [watchlist, historyDays]);



  // Compute symbols categorized into the 4 columns
  const columns = React.useMemo(() => {
    if (!result || !result.results) {
      return {
        getsPositive: [],
        alreadyCrossed: [],
        underSignalPositive: [],
        underSignalNegative: []
      };
    }

    const getsPositive = [];
    const alreadyCrossed = [];
    const underSignalPositive = [];
    const underSignalNegative = [];

    Object.entries(result.results).forEach(([symbol, signal]) => {
      if (!signal || typeof signal !== "object") return;

      const isMacdPositive = Boolean(signal.macd_is_positive);
      const isAboveSignal = Boolean(signal.bullish_macd_above_signal);
      const justBecamePositive = Boolean(signal.macd_just_became_positive);
      const hasRecentCrossover = Boolean(signal.recent_crossover);

      // 1. "MACD gets positive": both "MACD just became positive" and "MACD is currently above signal line"
      if (justBecamePositive && isAboveSignal) {
        getsPositive.push(symbol);
      }

      // 2. "Already crossed": (recent crossover OR both about to cross & recent crossover) AND MACD is already positive
      if (hasRecentCrossover && isMacdPositive) {
        alreadyCrossed.push(symbol);
      }

      // 3. "MACD under signal positive": MACD below signal line but still positive
      if (!isAboveSignal && isMacdPositive) {
        underSignalPositive.push(symbol);
      }

      // 4. "MACD under signal negative": MACD below signal line and negative
      if (!isAboveSignal && !isMacdPositive) {
        underSignalNegative.push(symbol);
      }
    });

    return {
      getsPositive,
      alreadyCrossed,
      underSignalPositive,
      underSignalNegative
    };
  }, [result]);

  const columnDefs = [
    {
      key: "getsPositive",
      title: "MACD gets positive",
      symbols: columns.getsPositive,
      badgeColor: "#27ae60",
      headerBg: "#eef9f2",
      borderColor: "#a3e0b8"
    },
    {
      key: "alreadyCrossed",
      title: "Already crossed",
      symbols: columns.alreadyCrossed,
      badgeColor: "#8e44ad",
      headerBg: "#f5eefb",
      borderColor: "#d2b4de"
    },
    {
      key: "underSignalPositive",
      title: "MACD under signal positive",
      symbols: columns.underSignalPositive,
      badgeColor: "#2980b9",
      headerBg: "#ebf5fb",
      borderColor: "#aed6f1"
    },
    {
      key: "underSignalNegative",
      title: "MACD under signal negative",
      symbols: columns.underSignalNegative,
      badgeColor: "#7f8c8d",
      headerBg: "#f2f4f4",
      borderColor: "#d5dbdb"
    }
  ];

  // Download CSV handler
  const handleDownloadCSV = async () => {
    setDownloading(true);
    try {
      const res = await fetch(
        `${API_BASE}/watchlist/${encodeURIComponent(watchlist)}/bullish_companies_csv`,
        { method: "POST" }
      );
      if (!res.ok) {
        alert("Failed to download CSV");
        setDownloading(false);
        return;
      }
      const blob = await res.blob();
      // Try to get filename from Content-Disposition header
      let filename = "bullish_companies.csv";
      const disposition = res.headers.get("Content-Disposition");
      if (disposition) {
        // Fix: Use a regex that matches filename= without quotes
        const match = disposition.match(/filename=([^;]+)/i);
        if (match && match[1]) {
          filename = match[1].trim();
          console.log("Downloading file:", filename);
        }
      }
      console.log("Downloading file2:", filename);
      // Create a link and trigger download
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.setAttribute("download", filename);
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }, 100);
    } catch (e) {
      alert("Error downloading CSV");
    }
    setDownloading(false);
  };

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 12, borderRadius: "6px", position: "relative" }}>
      <h4>Bullish MACD Signal for "{watchlist}"</h4>
      <button onClick={onClose} style={{ marginBottom: 10 }}>Close</button>
      <button onClick={handleDownloadCSV} style={{ marginLeft: 10, marginBottom: 10 }} disabled={downloading}>
        {downloading ? "Downloading..." : "Download Bullish Companies CSV"}
      </button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: "red" }}>{error}</div>}
      {result && result.results && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "14px",
            marginTop: "12px",
            alignItems: "start"
          }}
        >
          {columnDefs.map((col) => (
            <div
              key={col.key}
              style={{
                border: `1px solid ${col.borderColor}`,
                borderRadius: "8px",
                background: "#fff",
                overflow: "hidden",
                boxShadow: "0 1px 3px rgba(0,0,0,0.05)"
              }}
            >
              <div
                style={{
                  background: col.headerBg,
                  padding: "9px 12px",
                  borderBottom: `1px solid ${col.borderColor}`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "8px"
                }}
              >
                <span style={{ fontWeight: "bold", fontSize: "0.9em", color: "#2c3e50" }}>
                  {col.title}
                </span>
                <span
                  style={{
                    background: col.badgeColor,
                    color: "#fff",
                    borderRadius: "12px",
                    padding: "2px 8px",
                    fontSize: "0.78em",
                    fontWeight: "bold"
                  }}
                >
                  {col.symbols.length}
                </span>
              </div>
              <div style={{ padding: "10px 12px", minHeight: "70px", maxHeight: "360px", overflowY: "auto" }}>
                {col.symbols.length === 0 ? (
                  <div style={{ color: "#888", fontStyle: "italic", fontSize: "0.85em", padding: "6px 0" }}>
                    No symbols
                  </div>
                ) : (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                    {col.symbols.map((symbol) => (
                      <a
                        key={symbol}
                        href={getChartUrl(symbol)}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          display: "inline-block",
                          padding: "3px 7px",
                          borderRadius: "4px",
                          background: "#f8f9fa",
                          border: "1px solid #e2e6ea",
                          color: "#2c3e50",
                          textDecoration: "underline",
                          fontWeight: "bold",
                          fontSize: "0.92em",
                          cursor: "pointer"
                        }}
                        onMouseEnter={(e) => handleMouseEnter(symbol, e)}
                        onMouseLeave={handleMouseLeave}
                      >
                        {symbol}
                      </a>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Column count history chart */}
      <div style={{ marginTop: "20px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: "8px",
            flexWrap: "wrap",
            gap: "8px"
          }}
        >
          <div style={{ fontWeight: "bold", color: "#2c3e50", fontSize: "0.95em" }}>
            Column counts – last {INTERVAL_OPTIONS.find(o => o.value === historyDays)?.label || `${historyDays} days`}
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <label htmlFor="history-interval-select" style={{ fontSize: "0.85em", color: "#555", fontWeight: "bold" }}>
              Interval:
            </label>
            <select
              id="history-interval-select"
              value={historyDays}
              onChange={(e) => setHistoryDays(Number(e.target.value))}
              style={{
                padding: "3px 8px",
                borderRadius: "4px",
                border: "1px solid #ccc",
                fontSize: "0.85em",
                background: "#fff",
                cursor: "pointer",
                fontWeight: "bold",
                color: "#2c3e50"
              }}
            >
              {INTERVAL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {historyLoading ? (
          <div style={{ color: "#888", fontSize: "0.85em" }}>Loading history chart...</div>
        ) : !historyData || historyData.length === 0 ? (
          <div style={{ color: "#888", fontStyle: "italic", fontSize: "0.85em" }}>
            No historical data yet — data accumulates over time as you run the scanner daily.
          </div>
        ) : (() => {
          // --- SVG line chart (responsive 100% width) ---
          const W = 1000, H = 220, PADL = 45, PADR = 25, PADT = 15, PADB = 35;
          const innerW = W - PADL - PADR;
          const innerH = H - PADT - PADB;

          const cols = [
            { key: "macd_gets_positive",    label: "MACD gets positive",       color: "#27ae60" },
            { key: "already_crossed",        label: "Already crossed",          color: "#8e44ad" },
            { key: "under_signal_positive",  label: "Under signal +",           color: "#2980b9" },
            { key: "under_signal_negative",  label: "Under signal –",           color: "#7f8c8d" },
          ];

          // Interpolate missing days
          const dates = historyData.map(d => d.date);
          const allMax = Math.max(...historyData.flatMap(d =>
            cols.map(c => d.counts[c.key] || 0)
          ), 1);

          const n = historyData.length;
          const xOf = (i) => PADL + (i / Math.max(n - 1, 1)) * innerW;
          const yOf = (v) => PADT + innerH - (v / allMax) * innerH;

          // X-axis ticks: up to 10 evenly-spaced dates
          const tickCount = Math.min(10, n);
          const tickIndices = tickCount <= 1 ? [0]
            : Array.from({ length: tickCount }, (_, k) => Math.round(k * (n - 1) / (tickCount - 1)));

          // Y-axis ticks
          const yTicks = [0, Math.round(allMax / 2), allMax];

          const formatDateLabel = (d) => {
            if (!d) return "";
            if (historyDays >= 365) {
              return d.slice(2); // YY-MM-DD
            }
            return d.slice(5); // MM-DD
          };

          const polyline = (colKey, color) => {
            const points = historyData
              .map((d, i) => `${xOf(i)},${yOf(d.counts[colKey] || 0)}`)
              .join(" ");
            return (
              <polyline
                key={colKey}
                points={points}
                fill="none"
                stroke={color}
                strokeWidth="2.5"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            );
          };

          const handleSvgMouseMove = (e) => {
            if (!historyData || historyData.length === 0) return;
            const rect = e.currentTarget.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const svgX = (mouseX / rect.width) * W;
            const clampedX = Math.max(PADL, Math.min(PADL + innerW, svgX));
            const rawIdx = ((clampedX - PADL) / innerW) * (n - 1);
            const idx = Math.max(0, Math.min(n - 1, Math.round(rawIdx)));
            setHoveredHistoryIndex(idx);
          };

          const handleSvgMouseLeave = () => {
            setHoveredHistoryIndex(null);
          };

          return (
            <div style={{ width: "100%", marginTop: "6px" }}>
              <svg
                width="100%"
                height={H}
                viewBox={`0 0 ${W} ${H}`}
                onMouseMove={handleSvgMouseMove}
                onMouseLeave={handleSvgMouseLeave}
                style={{
                  display: "block",
                  fontFamily: "inherit",
                  width: "100%",
                  height: `${H}px`,
                  cursor: "crosshair"
                }}
              >
                {/* Y-axis grid lines and labels */}
                {yTicks.map(v => (
                  <g key={v}>
                    <line
                      x1={PADL} y1={yOf(v)} x2={W - PADR} y2={yOf(v)}
                      stroke="#e8e8e8" strokeWidth="1"
                    />
                    <text
                      x={PADL - 8} y={yOf(v) + 4}
                      textAnchor="end" fontSize="11" fill="#888"
                    >{v}</text>
                  </g>
                ))}

                {/* Axes */}
                <line x1={PADL} y1={PADT} x2={PADL} y2={PADT + innerH} stroke="#ccc" strokeWidth="1" />
                <line x1={PADL} y1={PADT + innerH} x2={W - PADR} y2={PADT + innerH} stroke="#ccc" strokeWidth="1" />

                {/* X-axis date labels */}
                {tickIndices.map(i => (
                  <text
                    key={i}
                    x={xOf(i)}
                    y={PADT + innerH + 18}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#666"
                  >
                    {formatDateLabel(dates[i])}
                  </text>
                ))}

                {/* Lines */}
                {cols.map(c => polyline(c.key, c.color))}

                {/* Hover crosshair and tooltip */}
                {hoveredHistoryIndex !== null && historyData[hoveredHistoryIndex] && (() => {
                  const hData = historyData[hoveredHistoryIndex];
                  const hX = xOf(hoveredHistoryIndex);
                  const cardW = 185;
                  const cardH = 95;
                  const tooltipX = hX > W - cardW - 30 ? hX - cardW - 12 : hX + 12;
                  const tooltipY = PADT + 5;

                  return (
                    <g pointerEvents="none">
                      {/* Vertical dashed crosshair */}
                      <line
                        x1={hX}
                        y1={PADT}
                        x2={hX}
                        y2={PADT + innerH}
                        stroke="#7f8c8d"
                        strokeWidth="1.2"
                        strokeDasharray="4 3"
                      />

                      {/* Circles on each curve */}
                      {cols.map(c => (
                        <circle
                          key={c.key}
                          cx={hX}
                          cy={yOf(hData.counts[c.key] || 0)}
                          r="4.5"
                          fill={c.color}
                          stroke="#ffffff"
                          strokeWidth="2"
                        />
                      ))}

                      {/* Floating tooltip box */}
                      <rect
                        x={tooltipX}
                        y={tooltipY}
                        width={cardW}
                        height={cardH}
                        rx="6"
                        ry="6"
                        fill="#ffffff"
                        stroke="#dcdde1"
                        strokeWidth="1.5"
                        filter="drop-shadow(0 2px 6px rgba(0,0,0,0.18))"
                      />
                      {/* Date header */}
                      <text
                        x={tooltipX + 10}
                        y={tooltipY + 18}
                        fontSize="11"
                        fontWeight="bold"
                        fill="#2f3640"
                      >
                        {hData.date}
                      </text>
                      <line
                        x1={tooltipX + 8}
                        y1={tooltipY + 24}
                        x2={tooltipX + cardW - 8}
                        y2={tooltipY + 24}
                        stroke="#f1f2f6"
                        strokeWidth="1"
                      />
                      {/* Series Rows */}
                      {cols.map((c, ci) => (
                        <g key={c.key} transform={`translate(${tooltipX + 10}, ${tooltipY + 38 + ci * 14})`}>
                          <circle cx="4" cy="-3" r="3.5" fill={c.color} />
                          <text x="14" y="0" fontSize="10" fill="#57606f">
                            {c.label}:
                          </text>
                          <text x={cardW - 20} y="0" fontSize="10" fontWeight="bold" textAnchor="end" fill="#2f3640">
                            {hData.counts[c.key] || 0}
                          </text>
                        </g>
                      ))}
                    </g>
                  );
                })()}
              </svg>



              {/* Legend */}
              <div style={{ display: "flex", gap: "16px", marginTop: "4px", flexWrap: "wrap" }}>
                {cols.map(c => (
                  <span key={c.key} style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "0.78em", color: "#444" }}>
                    <span style={{ display: "inline-block", width: 20, height: 3, background: c.color, borderRadius: 2 }} />
                    {c.label}
                  </span>
                ))}
              </div>
            </div>
          );
        })()}
      </div>

      {/* Chart tooltip */}
      {hoveredSymbol && (
        <div 
          style={{
            position: "fixed",
            top: chartPosition.y + 20,
            left: chartPosition.x + 20,
            zIndex: 1000,
            background: "white",
            border: "1px solid #ddd",
            borderRadius: "4px",
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
            padding: 5
          }}
        >
          <h5>{hoveredSymbol} MACD History</h5>
          {chartLoading ? (
            <div>Loading chart...</div>
          ) : (
            <MacdChart data={chartData} />
          )}
        </div>
      )}
    </div>
  );
}

function WatchlistBullishForecast({ watchlist, symbols, onClose }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const didRun = React.useRef(false);
  const formatDecimal = (value) => {
    if (value === null || value === undefined) return "N/A";
    const num = typeof value === "number" ? value : Number(value);
    if (Number.isFinite(num)) {
      return num.toFixed(3);
    }
    if (value && typeof value.toFixed === "function") {
      try {
        return value.toFixed(3);
      } catch {
        return String(value);
      }
    }
    return String(value);
  };

  React.useEffect(() => {
    if (didRun.current) return;
    didRun.current = true;
    const fetchForecast = async () => {
      setLoading(true);
      setError("");
      setResult(null);
      try {
        const res = await fetch(`${API_BASE}/forecast/macd/arima_positive`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbols }),
        });
        if (!res.ok) {
          const err = await res.json();
          setError(err.detail || "Error fetching forecast");
        } else {
          setResult(await res.json());
        }
      } catch (e) {
        setError("Error fetching forecast");
      }
      setLoading(false);
    };
    fetchForecast();
    // Only run once per mount
    // eslint-disable-next-line
  }, []);

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10 }}>
      <h4>Bullish MACD Forecast for "{watchlist}"</h4>
      <button onClick={onClose} style={{ marginBottom: 10 }}>Close</button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: "red" }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 10 }}>
          {Object.entries(result).map(([symbol, forecast]) => (
            <div key={symbol} style={{ marginBottom: 8 }}>
              <a
                href={getChartUrl(symbol)}
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontWeight: "bold", textDecoration: "underline", color: "inherit", cursor: "pointer" }}
              >
                {symbol}
              </a>
              {": "}
              {forecast && forecast.will_become_positive !== undefined ? (
                <span style={{ color: forecast.will_become_positive ? "green" : "gray" }}>
                  {forecast.will_become_positive ? "Will become positive" : "Not forecasted positive"}
                </span>
              ) : (
                <span style={{ color: "red" }}>Error</span>
              )}
              {forecast && forecast.forecasted_macd && (
                <span style={{ marginLeft: 10, fontSize: "0.95em" }}>
                  {(() => {
                    const lastMacd = forecast.details && Object.prototype.hasOwnProperty.call(forecast.details, "last_macd")
                      ? forecast.details.last_macd
                      : forecast.last_macd;
                    const valuesArray = Array.isArray(forecast.forecasted_macd)
                      ? forecast.forecasted_macd
                      : Object.values(forecast.forecasted_macd);
                    const formattedForecast = valuesArray.length
                      ? valuesArray.map(val => formatDecimal(val)).join(", ")
                      : "N/A";
                    return `[` +
                      `Last MACD: ${formatDecimal(lastMacd)}` +
                      (valuesArray.length ? ` | Forecast: ${formattedForecast}` : "") +
                      `]`;
                  })()}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WatchlistBullishMAForecast({ watchlist, symbols, onClose }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const didRun = React.useRef(false);

  React.useEffect(() => {
    if (didRun.current) return;
    didRun.current = true;
    const fetchForecast = async () => {
      setLoading(true);
      setError("");
      setResult(null);
      try {
        const res = await fetch(`${API_BASE}/forecast/ma/arima_above_50`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbols }),
        });
        if (!res.ok) {
          const err = await res.json();
          setError(err.detail || "Error fetching MA forecast");
        } else {
          setResult(await res.json());
        }
      } catch (e) {
        setError("Error fetching MA forecast");
      }
      setLoading(false);
    };
    fetchForecast();
    // Only run once per mount
    // eslint-disable-next-line
  }, []);

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10 }}>
      <h4>Bullish MA20 MA50 Forecast for "{watchlist}"</h4>
      <button onClick={onClose} style={{ marginBottom: 10 }}>Close</button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: "red" }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 10 }}>
          {Object.entries(result).map(([symbol, forecast]) => (
            <div key={symbol} style={{ marginBottom: 8 }}>
              <a
                href={getChartUrl(symbol)}
                target="_blank"
                rel="noopener noreferrer"
                style={{ fontWeight: "bold", textDecoration: "underline", color: "inherit", cursor: "pointer" }}
              >
                {symbol}
              </a>
              {": "}
              {forecast && forecast.ma20_will_be_above_ma50 !== undefined ? (
                <span style={{ color: forecast.ma20_will_be_above_ma50 ? "green" : "gray" }}>
                  {forecast.ma20_will_be_above_ma50 ? "MA20 will cross above MA50" : "No bullish MA forecast"}
                </span>
              ) : (
                <span style={{ color: "red" }}>Error</span>
              )}
              {forecast && forecast.forecasted_ma20 && forecast.forecasted_ma50 && (
                <span style={{ marginLeft: 10, fontSize: "0.95em" }}>
                  [MA20: {Object.values(forecast.forecasted_ma20).map(x => x && x.toFixed ? x.toFixed(3) : x).join(", ")}]
                  <br />
                  [MA50: {Object.values(forecast.forecasted_ma50).map(x => x && x.toFixed ? x.toFixed(3) : x).join(", ")}]
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function WatchlistCombinedForecast({ watchlist, onClose }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const didRun = React.useRef(false);

  React.useEffect(() => {
    if (didRun.current) return;
    didRun.current = true;

    const fetchCombined = async () => {
      try {
        const res = await fetch(`${API_BASE}/forecast/combined/${encodeURIComponent(watchlist)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" }
        });
        if (!res.ok) {
          const errorData = await res.json();
          throw new Error(errorData.detail || "Failed to fetch combined forecast");
        }
        const data = await res.json();
        setResult(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchCombined();
  }, []);

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10 }}>
      <h4>Combined Forecast for "{watchlist}"</h4>
      <button onClick={onClose} style={{ marginBottom: 10 }}>Close</button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: "red" }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 10 }}>
          <p>Symbols with both MACD positive forecast AND MA20 above MA50 forecast (as of {result.date}):</p>
          {result.symbols && result.symbols.length > 0 ? (
            <div>
              {result.symbols.map(symbol => (
                <div key={symbol} style={{ marginBottom: 8 }}>
                  <a
                    href={getChartUrl(symbol)}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ fontWeight: "bold", textDecoration: "underline", color: "green", cursor: "pointer" }}
                  >
                    {symbol}
                  </a>
                </div>
              ))}
            </div>
          ) : (
            <p style={{ color: "gray" }}>No symbols match both criteria.</p>
          )}
        </div>
      )}
    </div>
  );
}

// Pattern Chart Component - renders price data with pattern markers
function PatternChart({ symbol, prices, dates, patterns }) {
  const canvasRef = React.useRef(null);
  
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !prices || prices.length === 0) return;
    
    // Filter out null/undefined/NaN values
    const validPrices = prices.filter(p => p != null && !isNaN(p));
    if (validPrices.length === 0) return;
    
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const padding = { top: 30, right: 20, bottom: 30, left: 60 };
    
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Find min and max values from valid prices
    const minPrice = Math.min(...validPrices) * 0.98;
    const maxPrice = Math.max(...validPrices) * 1.02;
    const priceRange = maxPrice - minPrice || 1; // Avoid division by zero
    
    // Helper functions
    const getX = (index) => padding.left + (index / (prices.length - 1)) * chartWidth;
    const getY = (price) => padding.top + chartHeight - ((price - minPrice) / priceRange) * chartHeight;
    
    // Draw grid lines
    ctx.strokeStyle = "#eee";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (i / 5) * chartHeight;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      
      // Price labels
      const price = maxPrice - (i / 5) * priceRange;
      ctx.fillStyle = "#666";
      ctx.font = "10px Arial";
      ctx.textAlign = "right";
      ctx.fillText(price.toFixed(2), padding.left - 5, y + 3);
    }
    
    // Draw price line - skip null values
    ctx.beginPath();
    ctx.strokeStyle = "#2196F3";
    ctx.lineWidth = 2;
    let started = false;
    prices.forEach((price, i) => {
      if (price == null || isNaN(price)) return;
      const x = getX(i);
      const y = getY(price);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Draw patterns
    if (patterns && patterns.length > 0) {
      patterns.forEach(pattern => {
        const isInverse = pattern.pattern_type === "inverse_head_and_shoulders";
        const color = isInverse ? "#4CAF50" : "#f44336"; // Green for bullish, red for bearish
        
        // Draw neckline
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        
        const leftTroughX = getX(pattern.left_trough?.index || pattern.left_shoulder.index);
        const rightTroughX = getX(pattern.right_trough?.index || pattern.right_shoulder.index);
        const necklineY = getY(pattern.neckline_price);
        
        ctx.moveTo(leftTroughX, necklineY);
        ctx.lineTo(rightTroughX, necklineY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw markers for L, H, R
        const markers = [
          { label: "L", idx: pattern.left_shoulder.index, price: pattern.left_shoulder.price },
          { label: "H", idx: pattern.head.index, price: pattern.head.price },
          { label: "R", idx: pattern.right_shoulder.index, price: pattern.right_shoulder.price }
        ];
        
        markers.forEach(marker => {
          const x = getX(marker.idx);
          const y = getY(marker.price);
          
          // Draw circle
          ctx.beginPath();
          ctx.arc(x, y, 12, 0, 2 * Math.PI);
          ctx.fillStyle = color;
          ctx.fill();
          
          // Draw label
          ctx.fillStyle = "white";
          ctx.font = "bold 12px Arial";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(marker.label, x, y);
        });
        
        // Draw pattern type label
        ctx.fillStyle = color;
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        const labelY = isInverse ? padding.top + 15 : padding.top + 15;
        ctx.fillText(
          `${isInverse ? "Inverse H&S" : "H&S"} (${(pattern.confidence * 100).toFixed(0)}%)`,
          padding.left + 5,
          labelY
        );
      });
    }
    
    // Draw symbol name
    ctx.fillStyle = "#333";
    ctx.font = "bold 14px Arial";
    ctx.textAlign = "right";
    ctx.fillText(symbol, width - padding.right, padding.top - 10);
    
  }, [symbol, prices, dates, patterns]);
  
  return (
    <canvas 
      ref={canvasRef} 
      width={700} 
      height={250}
      style={{ border: '1px solid #ddd', borderRadius: 4, margin: 5, display: 'block' }}
    />
  );
}

function WatchlistPatterns({ watchlist, onClose }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [availableDates, setAvailableDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("");
  const [loadingDates, setLoadingDates] = useState(false);
  const [bulkDays, setBulkDays] = useState(5);
  const [bulkLoading, setBulkLoading] = useState(false);
  const [bulkMessage, setBulkMessage] = useState("");

  React.useEffect(() => {
    const fetchDates = async () => {
      setLoadingDates(true);
      try {
        const res = await fetch(`${API_BASE}/charts/watchlist/${encodeURIComponent(watchlist)}/available_dates`);
        if (res.ok) {
          const data = await res.json();
          setAvailableDates(data.dates || []);
          if (data.dates && data.dates.length > 0) {
            setSelectedDate(data.dates[0]);
          }
        }
      } catch (err) {
        console.error("Error fetching available dates:", err);
      } finally {
        setLoadingDates(false);
      }
    };
    fetchDates();
  }, [watchlist]);

  const generateCharts = async () => {
    setLoading(true);
    setError("");
    setMessage("");
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/charts/watchlist/${encodeURIComponent(watchlist)}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_date: selectedDate })
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to generate charts");
      }
      const data = await res.json();
      setResult(data);
      setMessage(`Found ${data.count} charts matching the pattern filter!`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const bulkGenerateCharts = async () => {
    setBulkLoading(true);
    setBulkMessage("");
    setError("");
    try {
      const res = await fetch(`${API_BASE}/charts/watchlist/${encodeURIComponent(watchlist)}/bulk_generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ num_days: bulkDays })
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Failed to bulk generate charts");
      }
      const data = await res.json();
      setBulkMessage(`Bulk generation complete: ${data.processed} days processed (${data.errors} errors)`);
    } catch (err) {
      setError(err.message);
    } finally {
      setBulkLoading(false);
    }
  };

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10 }}>
      <h4>Generate Charts & Scan for "{watchlist}"</h4>
      <div style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: "10px" }}>
        <button onClick={onClose}>Close</button>
        
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <label htmlFor="date-select">Select Date:</label>
          {loadingDates ? (
            <span>Loading dates...</span>
          ) : (
            <select 
              id="date-select" 
              value={selectedDate} 
              onChange={(e) => setSelectedDate(e.target.value)}
              disabled={loading || availableDates.length === 0}
            >
              {availableDates.length === 0 && <option value="">No dates available</option>}
              {availableDates.map(date => (
                <option key={date} value={date}>{date}</option>
              ))}
            </select>
          )}
        </div>

        <button onClick={generateCharts} disabled={loading || !selectedDate}>
          {loading ? "Generating & Scanning..." : "Generate Charts"}
        </button>
      </div>
      
      <div style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: "10px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <label htmlFor="bulk-days">Days to generate:</label>
          <input
            id="bulk-days"
            type="number"
            min="1"
            max="365"
            value={bulkDays}
            onChange={(e) => setBulkDays(Math.max(1, parseInt(e.target.value) || 1))}
            style={{ width: "60px" }}
            disabled={bulkLoading}
          />
        </div>
        <button onClick={bulkGenerateCharts} disabled={bulkLoading || loading}>
          {bulkLoading ? "Bulk Generating..." : "Bulk Generate (No Display)"}
        </button>
        {bulkMessage && <span style={{ color: "blue", marginLeft: "10px" }}>{bulkMessage}</span>}
      </div>
      
      {loading && <div>Generating charts and running neural pattern detection...</div>}
      {bulkLoading && <div>Bulk generating charts for {bulkDays} days. This may take a few minutes...</div>}
      {error && <div style={{ color: "red" }}>Error: {error}</div>}
      {message && <div style={{ color: "green", fontWeight: "bold" }}>{message}</div>}
      
      {result && result.images && result.images.length > 0 && (
        <div style={{ marginTop: 20 }}>
          {result.bullish && result.bullish.length > 0 && (
            <div style={{ marginBottom: 30 }}>
              <h3 style={{ color: "#2c3e50", borderBottom: "2px solid #27ae60", paddingBottom: "5px" }}>
                Bullish Signals ({result.bullish.length})
              </h3>
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))", 
                gap: "15px" 
              }}>
                {result.bullish.map((url, idx) => (
                  <div key={`bullish-${idx}`} style={{ border: "1px solid #ddd", padding: "5px", borderRadius: "4px" }}>
                    <img 
                      src={`${API_BASE}${url}`} 
                      alt={`Bullish Pattern ${idx}`} 
                      style={{ width: "100%", height: "auto", display: "block" }} 
                    />
                    <div style={{ fontSize: "12px", marginTop: "5px", color: "#666" }}>
                      {url.split('/').pop()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.bearish && result.bearish.length > 0 && (
            <div style={{ marginBottom: 30 }}>
              <h3 style={{ color: "#2c3e50", borderBottom: "2px solid #c0392b", paddingBottom: "5px" }}>
                Bearish Signals ({result.bearish.length})
              </h3>
              <div style={{ 
                display: "grid", 
                gridTemplateColumns: "repeat(auto-fill, minmax(350px, 1fr))", 
                gap: "15px" 
              }}>
                {result.bearish.map((url, idx) => (
                  <div key={`bearish-${idx}`} style={{ border: "1px solid #ddd", padding: "5px", borderRadius: "4px" }}>
                    <img 
                      src={`${API_BASE}${url}`} 
                      alt={`Bearish Pattern ${idx}`} 
                      style={{ width: "100%", height: "auto", display: "block" }} 
                    />
                    <div style={{ fontSize: "12px", marginTop: "5px", color: "#666" }}>
                      {url.split('/').pop()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {result && result.images && result.images.length === 0 && !loading && (
        <div style={{ marginTop: 10, color: "#666" }}>
          No patterns matching the filter (x &gt; 550) were found.
        </div>
      )}
    </div>
  );
}

function WatchlistsManager({ onSelectWatchlist }) {
  const [watchlists, setWatchlists] = useState([]);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState(null);
  const [symbols, setSymbols] = useState("");
  const [removeSymbol, setRemoveSymbol] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const fileInputRef = React.useRef(null);

  const fetchWatchlists = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/watchlists`);
      const data = await res.json();
      if (data && data.watchlists) {
        setWatchlists(data.watchlists);
      } else if (Array.isArray(data)) {
        setWatchlists(data);
      } else {
        setWatchlists([]);
      }
    } catch (e) {
      setWatchlists([]);
    }
    setLoading(false);
  };

  const createWatchlist = async () => {
    setMessage("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (res.ok) {
        setMessage("Created!");
        setName("");
        await fetchWatchlists();
      } else {
        const err = await res.json();
        setMessage("Error creating watchlist: " + (err.detail || ""));
      }
    } catch (e) {
      setMessage("Error creating watchlist");
    }
    setLoading(false);
  };

  const deleteWatchlist = async (wlName) => {
    setMessage("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/watchlist/${encodeURIComponent(wlName)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setMessage("Deleted!");
        if (selected === wlName) setSelected(null);
        await fetchWatchlists();
      } else {
        setMessage("Error deleting watchlist");
      }
    } catch (e) {
      setMessage("Error deleting watchlist");
    }
    setLoading(false);
  };

  const addSymbols = async () => {
    setMessage("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/watchlist/${selected}/add_symbol`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols: symbols.split(",").map(s => s.trim()) }),
      });
      if (res.ok) {
        setMessage("Symbols added!");
        setSymbols("");
        await fetchWatchlists();
      } else {
        setMessage("Error adding symbols");
      }
    } catch (e) {
      setMessage("Error adding symbols");
    }
    setLoading(false);
  };

  const removeSymbols = async () => {
    setMessage("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/watchlist/${selected}/remove_symbol`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbols: removeSymbol.split(",").map(s => s.trim()) }),
      });
      if (res.ok) {
        setMessage("Symbols removed!");
        setRemoveSymbol("");
        await fetchWatchlists();
      } else {
        setMessage("Error removing symbols");
      }
    } catch (e) {
      setMessage("Error removing symbols");
    }
    setLoading(false);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setMessage("");
    setLoading(true);

    const formData = new FormData();
    formData.append("watchlist_name", selected);
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/watchlist/upload`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        setMessage(`Uploaded! Added: ${data.symbols_added.length} symbols.`);
        if (data.errors && data.errors.length > 0) {
           setMessage(prev => prev + ` Errors: ${data.errors.length}`);
        }
        await fetchWatchlists();
      } else {
        const err = await res.json();
        setMessage("Error uploading file: " + (err.detail || ""));
      }
    } catch (e) {
      setMessage("Error uploading file");
    }
    setLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  React.useEffect(() => { fetchWatchlists(); }, []);

  // Helper to get symbols for selected watchlist
  const selectedSymbols = React.useMemo(() => {
    const wl = watchlists.find(wl => wl.name === selected);
    return wl && wl.symbols ? wl.symbols : [];
  }, [selected, watchlists]);

  return (
    <div>
      <h2>Watchlists</h2>
      <input value={name} onChange={e => setName(e.target.value)} placeholder="New watchlist name" />
      <button onClick={createWatchlist}>Create</button>
      {loading && <div>Loading...</div>}
      <div>
        <h3>Existing Watchlists</h3>
        <ul>
          {watchlists.length === 0 ? (
            <li>No watchlists found.</li>
          ) : (
            watchlists.map(wl => (
              <li key={wl.name} style={{ marginBottom: 4 }}>
                <span
                  onClick={() => setSelected(wl.name)}
                  style={{ cursor: "pointer", fontWeight: selected === wl.name ? "bold" : "normal" }}
                >
                  {wl.name}: {wl.symbols && wl.symbols.length > 0 ? wl.symbols.join(", ") : "No symbols"}
                </span>
                <button style={{ marginLeft: 8 }} onClick={() => deleteWatchlist(wl.name)}>Delete</button>
                <button style={{ marginLeft: 4 }} onClick={() => onSelectWatchlist(wl.name, wl.symbols || [])}>View Signals</button>
                <button 
                  style={{ marginLeft: 4 }} 
                  onClick={async () => {
                    setMessage(`Refreshing data for ${wl.name}...`);
                    setLoading(true);
                    try {
                      const res = await fetch(`${API_BASE}/watchlist/${encodeURIComponent(wl.name)}/refresh`, {
                        method: "POST"
                      });
                      const data = await res.json();
                      setMessage(data.message);
                    } catch (e) {
                      setMessage("Error refreshing data");
                    }
                    setLoading(false);
                  }}
                  disabled={loading}
                >
                  Refresh Data
                </button>
              </li>
            ))
          )}
        </ul>
      </div>
      {selected && (
        <div>
          <h4>Manage symbols for {selected}</h4>
          <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="AAPL,MSFT" />
          <button onClick={addSymbols}>Add</button>
          <input value={removeSymbol} onChange={e => setRemoveSymbol(e.target.value)} placeholder="AAPL,MSFT" style={{ marginLeft: 10 }} />
          <button onClick={removeSymbols}>Remove</button>
          
          <input 
            type="file" 
            ref={fileInputRef} 
            style={{ display: 'none' }} 
            onChange={handleFileUpload}
            accept=".txt,.csv" 
          />
          <button 
            onClick={() => fileInputRef.current.click()} 
            style={{ marginLeft: 10 }}
          >
            Upload File
          </button>
        </div>
      )}
      {message && <div>{message}</div>}
    </div>
  );
}

function WatchlistSignalsPage({ watchlist, symbols, onBack }) {
  const [showSignal, setShowSignal] = useState(true);
  const [showForecast, setShowForecast] = useState(false);
  const [showMAForecast, setShowMAForecast] = useState(false);
  const [showCombinedForecast, setShowCombinedForecast] = useState(false);
  const [showPatterns, setShowPatterns] = useState(false);

  return (
    <div>
      <button onClick={onBack} style={{ marginBottom: 10 }}>Back to Watchlists</button>
      <h2>Signals for "{watchlist}"</h2>
      <button onClick={() => setShowSignal(true)} disabled={showSignal}>Show Bullish Signal</button>
      <button onClick={() => setShowForecast(true)} style={{ marginLeft: 10 }} disabled={showForecast}>Show Bullish Forecast</button>
      <button onClick={() => setShowMAForecast(true)} style={{ marginLeft: 10 }} disabled={showMAForecast}>Show MA20&gt;MA50 Forecast</button>
      <button onClick={() => setShowCombinedForecast(true)} style={{ marginLeft: 10 }} disabled={showCombinedForecast}>Combined Forecast</button>
      <button onClick={() => setShowPatterns(true)} style={{ marginLeft: 10 }} disabled={showPatterns}>Chart Patterns</button>
      {showSignal && (
        <WatchlistBullishSignal
          watchlist={watchlist}
          onClose={() => setShowSignal(false)}
        />
      )}
      {showForecast && (
        <WatchlistBullishForecast
          watchlist={watchlist}
          symbols={symbols}
          onClose={() => setShowForecast(false)}
        />
      )}
      {showMAForecast && (
        <WatchlistBullishMAForecast
          watchlist={watchlist}
          symbols={symbols}
          onClose={() => setShowMAForecast(false)}
        />
      )}
      {showCombinedForecast && (
        <WatchlistCombinedForecast
          watchlist={watchlist}
          onClose={() => setShowCombinedForecast(false)}
        />
      )}
      {showPatterns && (
        <WatchlistPatterns
          watchlist={watchlist}
          onClose={() => setShowPatterns(false)}
        />
      )}
    </div>
  );
}

// Utility to get the correct chart URL for a symbol
function getChartUrl(symbol) {
  if (symbol.includes(".")) {
    return `https://finance.yahoo.com/chart/${symbol}`;
  }
  return `https://www.tradingview.com/chart/5hYl19L3/?symbol=${symbol}`;
}

export default function MacdDashboard({ onHome }) {
  const [page, setPage] = useState("watchlists");
  const [selectedWatchlist, setSelectedWatchlist] = useState(null);
  const [selectedSymbols, setSelectedSymbols] = useState([]);

  const handleSelectWatchlist = (name, symbols) => {
    setSelectedWatchlist(name);
    setSelectedSymbols(symbols);
    setPage("signals");
  };

  const handleBack = () => {
    setPage("watchlists");
    setSelectedWatchlist(null);
    setSelectedSymbols([]);
  };

  return (
    <div style={{ padding: 20 }}>
      <button onClick={onHome} style={{ marginBottom: 10 }}>← Dashboards</button>
      <h1>Stock MACD Dashboard</h1>
      {page === "watchlists" ? (
        <WatchlistsManager onSelectWatchlist={handleSelectWatchlist} />
      ) : (
        <WatchlistSignalsPage
          watchlist={selectedWatchlist}
          symbols={selectedSymbols}
          onBack={handleBack}
        />
      )}
    </div>
  );
}


