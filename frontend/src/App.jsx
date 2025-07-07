import React, { useState } from "react";
import './App.css'

const API_BASE = "http://localhost:8000";

function MacdLookup() {
  const [symbol, setSymbol] = useState("");
  const [result, setResult] = useState(null);

  const fetchMacd = async () => {
    setResult(null);
    const res = await fetch(`${API_BASE}/macd/${symbol}`);
    setResult(await res.json());
  };

  return (
    <div>
      <h2>MACD Lookup</h2>
      <input value={symbol} onChange={e => setSymbol(e.target.value)} placeholder="Symbol" />
      <button onClick={fetchMacd}>Get MACD</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

function BulkMacd() {
  const [symbols, setSymbols] = useState("");
  const [result, setResult] = useState(null);

  const fetchBulk = async () => {
    setResult(null);
    const res = await fetch(`${API_BASE}/macd/bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbols: symbols.split(",").map(s => s.trim()) }),
    });
    setResult(await res.json());
  };

  return (
    <div>
      <h2>Bulk MACD</h2>
      <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="AAPL,MSFT,GOOG" />
      <button onClick={fetchBulk}>Get Bulk MACD</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

function BullishSignal() {
  const [symbols, setSymbols] = useState("");
  const [days, setDays] = useState(30);
  const [threshold, setThreshold] = useState(0.05);
  const [result, setResult] = useState(null);

  const fetchSignal = async () => {
    setResult(null);
    const res = await fetch(`${API_BASE}/macd/bullish_signal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ symbols: symbols.split(",").map(s => s.trim()), days, threshold }),
    });
    setResult(await res.json());
  };

  return (
    <div>
      <h2>Bullish MACD Signal</h2>
      <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="AAPL,MSFT,GOOG" />
      <input type="number" value={days} onChange={e => setDays(Number(e.target.value))} />
      <input type="number" value={threshold} step="0.01" onChange={e => setThreshold(Number(e.target.value))} />
      <button onClick={fetchSignal}>Check Signal</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}

function WatchlistBullishSignal({ watchlist, onClose }) {
  // Remove days/threshold state and UI, use defaults
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  React.useEffect(() => {
    const fetchSignal = async () => {
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

  // Updated color logic and hover explanations
  const getSymbolColor = (signal) => {
    if (!signal || typeof signal !== "object") return undefined;
    if (signal.about_to_cross && signal.recent_crossover) return "green";
    if (signal.about_to_cross || signal.recent_crossover) return "magenta";
    return undefined;
  };

  const getSymbolTitle = (signal) => {
    if (!signal || typeof signal !== "object") return "";
    if (signal.about_to_cross && signal.recent_crossover)
      return "Both bullish signals: about to cross AND recent crossover";
    if (signal.about_to_cross)
      return "Bullish: MACD is about to cross above the signal line";
    if (signal.recent_crossover)
      return "Bullish: Recent MACD crossover above the signal line";
    return "";
  };

  return (
    <div style={{ border: "1px solid #ccc", margin: "10px 0", padding: 10 }}>
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
                alignItems: "center"
              }}
              title={getSymbolTitle(signal)}
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
    </div>
  );
}

function WatchlistBullishForecast({ watchlist, symbols, onClose }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  React.useEffect(() => {
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
  }, [symbols]);

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
                  [Forecast: {forecast.forecasted_macd.map(x => x && x.toFixed ? x.toFixed(3) : x).join(", ")}]
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Watchlists() {
  const [watchlists, setWatchlists] = useState([]);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState(null);
  const [symbols, setSymbols] = useState("");
  const [message, setMessage] = useState("");
  const [showSignal, setShowSignal] = useState(false);
  const [showForecast, setShowForecast] = useState(false);

  const fetchWatchlists = async () => {
    try {
      const res = await fetch(`${API_BASE}/watchlists`);
      const data = await res.json();
      console.log("Watchlists API response:", data);
      
      // Handle the response more carefully
      if (data && data.watchlists) {
        console.log("Setting watchlists to:", data.watchlists);
        setWatchlists(data.watchlists);
      } else if (Array.isArray(data)) {
        console.log("Setting watchlists to array:", data);
        setWatchlists(data);
      } else {
        console.log("No valid watchlists data found");
        setWatchlists([]);
      }
    } catch (e) {
      console.error("Error fetching watchlists:", e);
      setWatchlists([]);
    }
  };

  const createWatchlist = async () => {
    setMessage("");
    try {
      const res = await fetch(`${API_BASE}/watchlist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (res.ok) {
        setMessage("Created!");
        setName("");
        await fetchWatchlists(); // Refresh after creation
      } else {
        const err = await res.json();
        setMessage("Error creating watchlist: " + (err.detail || ""));
      }
    } catch (e) {
      setMessage("Error creating watchlist");
    }
  };

  const addSymbols = async () => {
    setMessage("");
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
      <div>
        <h3>Existing Watchlists</h3>
        <ul>
          {watchlists.length === 0 ? (
            <li>No watchlists found.</li>
          ) : (
            watchlists.map(wl => (
              <li key={wl.name} onClick={() => setSelected(wl.name)} style={{ cursor: "pointer", fontWeight: selected === wl.name ? "bold" : "normal" }}>
                {wl.name}: {wl.symbols && wl.symbols.length > 0 ? wl.symbols.join(", ") : "No symbols"}
              </li>
            ))
          )}
        </ul>
      </div>
      {selected && (
        <div>
          <h4>Add symbols to {selected}</h4>
          <input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="AAPL,MSFT" />
          <button onClick={addSymbols}>Add</button>
          <button onClick={() => setShowSignal(true)} style={{ marginLeft: 10 }}>
            Check Bullish Signal
          </button>
          <button onClick={() => setShowForecast(true)} style={{ marginLeft: 10 }}>
            Check bullish forecast
          </button>
        </div>
      )}
      {showSignal && selected && (
        <WatchlistBullishSignal
          watchlist={selected}
          onClose={() => setShowSignal(false)}
        />
      )}
      {showForecast && selected && (
        <WatchlistBullishForecast
          watchlist={selected}
          symbols={selectedSymbols}
          onClose={() => setShowForecast(false)}
        />
      )}
      {message && <div>{message}</div>}
    </div>
  );
}

export default function App() {
  return (
    <div style={{ padding: 20 }}>
      <h1>Stock MACD Dashboard</h1>
      <MacdLookup />
      <BulkMacd />
      <BullishSignal />
      <Watchlists />
    </div>
  );
}
