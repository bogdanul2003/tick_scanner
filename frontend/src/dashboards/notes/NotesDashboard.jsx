import React, { useCallback, useEffect, useState } from "react";
import './notes.css';
import SymbolNotesPanel from "./SymbolNotesPanel";
import {
  listWatchlists,
  getWatchlist,
  createWatchlist,
  updateWatchlist,
  deleteWatchlist,
  addSymbols,
  removeSymbol,
} from "./notesApi";

/**
 * Notes Dashboard: notes-only watchlists, their symbols, and the note timeline
 * for the selected symbol.
 */
export default function NotesDashboard({ onHome }) {
  const [watchlists, setWatchlists] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const [newName, setNewName] = useState("");
  const [symbolInput, setSymbolInput] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameName, setRenameName] = useState("");
  const [renameDescription, setRenameDescription] = useState("");

  const run = async (action) => {
    setError("");
    setBusy(true);
    try {
      await action();
    } catch (e) {
      setError(e.message);
    }
    setBusy(false);
  };

  const refreshWatchlists = useCallback(async () => {
    const lists = await listWatchlists();
    setWatchlists(lists);
    return lists;
  }, []);

  const refreshDetail = useCallback(async (id) => {
    if (id == null) {
      setDetail(null);
      return;
    }
    const data = await getWatchlist(id);
    setDetail(data);
    setRenameName(data.name);
    setRenameDescription(data.description || "");
  }, []);

  useEffect(() => { run(refreshWatchlists); }, [refreshWatchlists]);
  useEffect(() => { run(() => refreshDetail(selectedId)); }, [selectedId, refreshDetail]);

  const selectWatchlist = (id) => {
    setSelectedId(id);
    setSelectedSymbol(null);
    setRenaming(false);
  };

  const handleCreate = () => run(async () => {
    const name = newName.trim();
    if (!name) return;
    const created = await createWatchlist(name);
    setNewName("");
    await refreshWatchlists();
    selectWatchlist(created.id);
  });

  const handleRename = () => run(async () => {
    await updateWatchlist(selectedId, {
      name: renameName.trim(),
      description: renameDescription,
    });
    setRenaming(false);
    await refreshWatchlists();
    await refreshDetail(selectedId);
  });

  const handleDeleteWatchlist = (watchlist) => run(async () => {
    const label = `"${watchlist.name}"`;
    const counts = `${watchlist.symbol_count} symbol(s) and ${watchlist.note_count} note(s)`;
    if (!window.confirm(`Delete ${label}? This removes ${counts}, including images.`)) return;
    await deleteWatchlist(watchlist.id);
    if (selectedId === watchlist.id) {
      setSelectedId(null);
      setSelectedSymbol(null);
      setDetail(null);
    }
    await refreshWatchlists();
  });

  const handleAddSymbols = () => run(async () => {
    const symbols = symbolInput.split(",").map(s => s.trim()).filter(Boolean);
    if (!symbols.length) return;
    await addSymbols(selectedId, symbols);
    setSymbolInput("");
    await refreshDetail(selectedId);
    await refreshWatchlists();
  });

  const handleRemoveSymbol = (entry) => run(async () => {
    const warning = entry.note_count > 0
      ? `Remove ${entry.symbol}? Its ${entry.note_count} note(s) and images are deleted too.`
      : `Remove ${entry.symbol}?`;
    if (!window.confirm(warning)) return;
    await removeSymbol(selectedId, entry.symbol);
    if (selectedSymbol === entry.symbol) setSelectedSymbol(null);
    await refreshDetail(selectedId);
    await refreshWatchlists();
  });

  const handleNotesChanged = () => run(async () => {
    await refreshDetail(selectedId);
    await refreshWatchlists();
  });

  return (
    <div className="notes-dashboard">
      <div className="notes-topbar">
        <button className="notes-btn" onClick={onHome}>← Dashboards</button>
        <h1>Notes Dashboard</h1>
        {busy && <span className="notes-hint">Working…</span>}
      </div>

      {error && <div className="notes-error notes-error-top">{error}</div>}

      <div className="notes-layout">
        <aside className="notes-column notes-column-lists">
          <h2>Watchlists</h2>
          <div className="notes-row">
            <input
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter") handleCreate(); }}
              placeholder="New watchlist name"
            />
            <button className="notes-btn notes-btn-primary" onClick={handleCreate}>Create</button>
          </div>

          {watchlists.length === 0 ? (
            <div className="notes-empty">No notes watchlists yet.</div>
          ) : (
            <ul className="notes-list">
              {watchlists.map(watchlist => (
                <li
                  key={watchlist.id}
                  className={watchlist.id === selectedId ? "notes-list-item notes-selected" : "notes-list-item"}
                >
                  <button className="notes-list-main" onClick={() => selectWatchlist(watchlist.id)}>
                    <span className="notes-list-title">{watchlist.name}</span>
                    <span className="notes-hint">
                      {watchlist.symbol_count} symbol{watchlist.symbol_count === 1 ? "" : "s"}
                      {" · "}
                      {watchlist.note_count} note{watchlist.note_count === 1 ? "" : "s"}
                    </span>
                    {watchlist.description && (
                      <span className="notes-list-desc">{watchlist.description}</span>
                    )}
                  </button>
                  <button className="notes-btn" onClick={() => handleDeleteWatchlist(watchlist)}>
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        {detail && (
          <aside className="notes-column notes-column-symbols">
            <div className="notes-detail-head">
              {renaming ? (
                <div className="notes-rename">
                  <input
                    value={renameName}
                    onChange={e => setRenameName(e.target.value)}
                    placeholder="Name"
                  />
                  <input
                    value={renameDescription}
                    onChange={e => setRenameDescription(e.target.value)}
                    placeholder="Description"
                  />
                  <button className="notes-btn notes-btn-primary" onClick={handleRename}>Save</button>
                  <button className="notes-btn" onClick={() => setRenaming(false)}>Cancel</button>
                </div>
              ) : (
                <>
                  <h2>{detail.name}</h2>
                  <button className="notes-btn" onClick={() => setRenaming(true)}>Edit</button>
                  {detail.description && <p className="notes-list-desc">{detail.description}</p>}
                </>
              )}
            </div>

            <div className="notes-row">
              <input
                value={symbolInput}
                onChange={e => setSymbolInput(e.target.value)}
                onKeyDown={e => { if (e.key === "Enter") handleAddSymbols(); }}
                placeholder="AAPL, MSFT"
              />
              <button className="notes-btn notes-btn-primary" onClick={handleAddSymbols}>Add</button>
            </div>

            {detail.symbols.length === 0 ? (
              <div className="notes-empty">No symbols yet.</div>
            ) : (
              <ul className="notes-list">
                {detail.symbols.map(entry => (
                  <li
                    key={entry.symbol}
                    className={entry.symbol === selectedSymbol ? "notes-list-item notes-selected" : "notes-list-item"}
                  >
                    <button className="notes-list-main" onClick={() => setSelectedSymbol(entry.symbol)}>
                      <span className="notes-list-title">{entry.symbol}</span>
                      <span className="notes-hint">
                        {entry.note_count} note{entry.note_count === 1 ? "" : "s"}
                        {entry.last_note_date ? ` · last ${entry.last_note_date}` : ""}
                      </span>
                    </button>
                    <button className="notes-btn" onClick={() => handleRemoveSymbol(entry)}>✕</button>
                  </li>
                ))}
              </ul>
            )}
          </aside>
        )}

        <main className="notes-column notes-column-notes">
          {!detail && <div className="notes-empty">Pick a watchlist to start.</div>}
          {detail && !selectedSymbol && (
            <div className="notes-empty">Pick a symbol to read or add notes.</div>
          )}
          {detail && selectedSymbol && (
            <SymbolNotesPanel
              key={`${detail.id}-${selectedSymbol}`}
              watchlistId={detail.id}
              symbol={selectedSymbol}
              onNotesChanged={handleNotesChanged}
            />
          )}
        </main>
      </div>
    </div>
  );
}
