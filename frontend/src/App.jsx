import React, { useState } from "react";
import './App.css'

const API_BASE = "http://localhost:8000";

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

  // Updated color logic and hover explanations
  const getSymbolColor = (signal) => {
    if (!signal || typeof signal !== "object") return undefined;
    if (signal.about_to_cross && signal.recent_crossover) return "purple";
    if (signal.about_to_cross) return "green";
    if (signal.recent_crossover) return "magenta";
    return undefined;
  };

  const getSymbolTitle = (signal) => {
    if (!signal || typeof signal !== "object") return "";
    if (signal.about_to_cross && signal.recent_crossover)
      return "Both bullish signals: about to cross AND recent crossover (purple)";
    if (signal.about_to_cross)
      return "Bullish: MACD is about to cross above the signal line (green)";
    if (signal.recent_crossover)
      return "Bullish: Recent MACD crossover above the signal line (magenta)";
    return "";
  };

  // Blinking state for macd_just_became_positive and ma20_just_became_above_ma50
  const [blinkOn, setBlinkOn] = React.useState(true);
  React.useEffect(() => {
    const interval = setInterval(() => setBlinkOn(b => !b), 500);
    return () => clearInterval(interval);
  }, []);

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
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10, position: "relative" }}>
      <h4>Bullish MACD Signal for "{watchlist}"</h4>
      <button onClick={onClose} style={{ marginBottom: 10 }}>Close</button>
      <button onClick={handleDownloadCSV} style={{ marginLeft: 10, marginBottom: 10 }} disabled={downloading}>
        {downloading ? "Downloading..." : "Download Bullish Companies CSV"}
      </button>
      {loading && <div>Loading...</div>}
      {error && <div style={{ color: "red" }}>{error}</div>}
      {result && result.results && (
        <div style={{ marginTop: 10 }}>
          {Object.entries(result.results).map(([symbol, signal]) => (
            <span
              key={symbol}
              style={{
                color: getSymbolColor(signal),
                fontWeight: "bold",
                marginRight: 18,
                fontSize: "1.1em",
                display: "inline-flex",
                alignItems: "center",
                cursor: "pointer"
              }}
              title={getSymbolTitle(signal)}
              onMouseEnter={(e) => handleMouseEnter(symbol, e)}
              onMouseLeave={handleMouseLeave}
            >
              <a
                href={getChartUrl(symbol)}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: "inherit", textDecoration: "underline", fontWeight: "bold", marginRight: 2 }}
                onClick={e => e.stopPropagation()}
              >
                {symbol}
              </a>
              {signal && signal.bullish_macd_above_signal && (
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: "green",
                    marginLeft: 6
                  }}
                  title="MACD is currently above the signal line"
                />
              )}
              {signal && signal.about_to_become_positive && (
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: "blue",
                    marginLeft: 6
                  }}
                  title="MACD or Signal Line is about to become positive"
                />
              )}
              {signal && signal.about_to_become_negative && (
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: "50%",
                    background: "red",
                    marginLeft: 6
                  }}
                  title="MACD or Signal Line is about to become negative"
                />
              )}
              {signal && signal.macd_just_became_positive && (
                <span
                  style={{
                    display: "inline-block",
                    width: 12,
                    height: 12,
                    borderRadius: "50%",
                    background: blinkOn ? "green" : "yellow",
                    marginLeft: 6,
                    boxShadow: blinkOn ? "0 0 8px 2px yellow" : "0 0 8px 2px green"
                  }}
                  title="MACD just became positive (recently crossed from negative)"
                />
              )}
              {/* Blinking cross for ma20_just_became_above_ma50 */}
              {signal && signal.ma20_just_became_above_ma50 && (
                <span
                  style={{
                    display: "inline-block",
                    width: 14,
                    height: 14,
                    marginLeft: 6,
                    position: "relative"
                  }}
                  title={
                    `MA20 just became above MA50 (bullish cross)` +
                    (signal.ma20_just_became_above_ma50_date
                      ? ` on ${signal.ma20_just_became_above_ma50_date}`
                      : "")
                  }
                >
                  <svg width="14" height="14" style={{ display: "block" }}>
                    <line
                      x1="2" y1="2" x2="12" y2="12"
                      stroke={blinkOn ? "green" : "yellow"}
                      strokeWidth="2"
                      style={{ filter: blinkOn ? "drop-shadow(0 0 4px yellow)" : "drop-shadow(0 0 4px green)" }}
                    />
                    <line
                      x1="12" y1="2" x2="2" y2="12"
                      stroke={blinkOn ? "green" : "yellow"}
                      strokeWidth="2"
                      style={{ filter: blinkOn ? "drop-shadow(0 0 4px yellow)" : "drop-shadow(0 0 4px green)" }}
                    />
                  </svg>
                </span>
              )}
            </span>
          ))}
        </div>
      )}
      
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

export default function App() {
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


