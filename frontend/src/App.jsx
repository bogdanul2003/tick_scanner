import React, { useEffect, useState } from "react";
import './App.css'
import HomePage, { DASHBOARDS } from "./pages/HomePage";
import MacdDashboard from "./dashboards/MacdDashboard";
import NotesDashboard from "./dashboards/notes/NotesDashboard";

const HOME = "home";

/**
 * Read the current dashboard from the URL hash (#/macd, #/notes, #/).
 * Keeping the view in the hash means refresh and browser back/forward work
 * without pulling in a router dependency.
 */
function viewFromHash() {
  const key = window.location.hash.replace(/^#\/?/, "");
  return DASHBOARDS.some(d => d.key === key) ? key : HOME;
}

export default function App() {
  const [view, setView] = useState(viewFromHash);

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = (next) => {
    window.location.hash = next === HOME ? "#/" : `#/${next}`;
    setView(next);
  };

  const goHome = () => navigate(HOME);

  if (view === "macd") return <MacdDashboard onHome={goHome} />;
  if (view === "notes") return <NotesDashboard onHome={goHome} />;
  return <HomePage onOpen={navigate} />;
}
