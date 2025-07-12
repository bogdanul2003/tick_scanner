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

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10, position: "relative" }}>
      <h4>Bullish MACD Signal for "{watchlist}"</h4>
      <button onClick={onClose} style={{ marginBottom: 10 }}>Close</button>
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
              {symbol}
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

  React.useEffect(() => {
    if (didRun.current) return;
    didRun.current = true;
    const fetchForecast = async () => {
      setLoading(true);
      setError("");
      setResult(null);
      try {
        const res = await fetch("http://localhost:8000/macd/arima_positive_forecast", {
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
              <span style={{ fontWeight: "bold" }}>{symbol}: </span>
              {forecast && forecast.will_become_positive !== undefined ? (
                <span style={{ color: forecast.will_become_positive ? "green" : "gray" }}>
                  {forecast.will_become_positive ? "Will become positive" : "Not forecasted positive"}
                </span>
              ) : (
                <span style={{ color: "red" }}>Error</span>
              )}
              {forecast && forecast.forecasted_macd && (
                <span style={{ marginLeft: 10, fontSize: "0.95em" }}>
                  [Forecast: {Array.isArray(forecast.forecasted_macd)
                    ? forecast.forecasted_macd.map(x => x && x.toFixed ? x.toFixed(3) : x).join(", ")
                    : Object.values(forecast.forecasted_macd).map(x => x && x.toFixed ? x.toFixed(3) : x).join(", ")}]
                </span>
              )}
            </div>
          ))}
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
        </div>
      )}
      {message && <div>{message}</div>}
    </div>
  );
}

function WatchlistSignalsPage({ watchlist, symbols, onBack }) {
  const [showSignal, setShowSignal] = useState(true);
  const [showForecast, setShowForecast] = useState(false);

  return (
    <div>
      <button onClick={onBack} style={{ marginBottom: 10 }}>Back to Watchlists</button>
      <h2>Signals for "{watchlist}"</h2>
      <button onClick={() => setShowSignal(true)} disabled={showSignal}>Show Bullish Signal</button>
      <button onClick={() => setShowForecast(true)} style={{ marginLeft: 10 }} disabled={showForecast}>Show Bullish Forecast</button>
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
    </div>
  );
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

