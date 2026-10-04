import React from "react";

/**
 * The dashboards available from the start page. Adding a new dashboard is one
 * entry here plus a branch in App.jsx.
 */
export const DASHBOARDS = [
  {
    key: "macd",
    icon: "📈",
    title: "MACD Dashboard",
    blurb: "Watchlists, bullish MACD crossovers, MA20/MA50 forecasts and chart pattern detection.",
  },
  {
    key: "notes",
    icon: "📝",
    title: "Notes Dashboard",
    blurb: "Track your thinking per symbol: dated notes with markdown text and screenshots.",
  },
];

export default function HomePage({ onOpen }) {
  return (
    <div style={styles.page}>
      <h1 style={styles.heading}>Tick Scanner</h1>
      <p style={styles.subheading}>Pick a dashboard.</p>
      <div style={styles.grid}>
        {DASHBOARDS.map(dashboard => (
          <button
            key={dashboard.key}
            onClick={() => onOpen(dashboard.key)}
            style={styles.card}
            onMouseEnter={e => { e.currentTarget.style.borderColor = "#2c7"; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = "#ccc"; }}
          >
            <div style={styles.icon}>{dashboard.icon}</div>
            <div style={styles.cardTitle}>{dashboard.title}</div>
            <div style={styles.cardBlurb}>{dashboard.blurb}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

const styles = {
  page: { padding: 40, maxWidth: 900, margin: "0 auto" },
  heading: { marginBottom: 4 },
  subheading: { color: "#666", marginTop: 0, marginBottom: 28 },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
    gap: 16,
  },
  card: {
    display: "block",
    textAlign: "left",
    padding: 20,
    border: "1px solid #ccc",
    borderRadius: 8,
    background: "#fff",
    cursor: "pointer",
    font: "inherit",
    transition: "border-color 150ms",
  },
  icon: { fontSize: 30, marginBottom: 8 },
  cardTitle: { fontSize: "1.15em", fontWeight: "bold", marginBottom: 6 },
  cardBlurb: { color: "#555", fontSize: "0.92em", lineHeight: 1.45 },
};
