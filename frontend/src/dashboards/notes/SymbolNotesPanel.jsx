import React, { useCallback, useEffect, useState } from "react";
import NoteCard from "./NoteCard";
import NoteEditor from "./NoteEditor";
import ImageLightbox from "./ImageLightbox";
import { listNotes, updateNote, deleteNote } from "./notesApi";

const NEW_NOTE = "new";

/**
 * The note timeline for one symbol, newest recorded date first.
 */
export default function SymbolNotesPanel({ watchlistId, symbol, onNotesChanged }) {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(null);      // NEW_NOTE | note object | null
  const [lightboxImage, setLightboxImage] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setNotes(await listNotes(watchlistId, symbol));
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  }, [watchlistId, symbol]);

  useEffect(() => { setEditing(null); load(); }, [load]);

  const handleSaved = async () => {
    setEditing(null);
    await load();
    onNotesChanged?.();
  };

  const handleStatusChange = async (note, newStatus) => {
    // Optimistically update status in UI immediately
    setNotes(prev => prev.map(n => (n.id === note.id ? { ...n, status: newStatus } : n)));
    try {
      await updateNote(note.id, { status: newStatus });
      onNotesChanged?.();
    } catch (e) {
      setError(e.message);
      await load();
    }
  };

  const handleDelete = async (note) => {
    if (!window.confirm(`Delete the note from ${note.note_date}? Its images go too.`)) return;
    try {
      await deleteNote(note.id);
      await load();
      onNotesChanged?.();
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <section className="notes-panel">
      <header className="notes-panel-head">
        <h3>{symbol}</h3>
        <button
          className="notes-btn notes-btn-primary"
          onClick={() => setEditing(NEW_NOTE)}
          disabled={editing === NEW_NOTE}
        >
          ＋ New note
        </button>
        <span className="notes-hint">
          {notes.length} note{notes.length === 1 ? "" : "s"}
        </span>
      </header>

      {error && <div className="notes-error">{error}</div>}

      {editing === NEW_NOTE && (
        <NoteEditor
          watchlistId={watchlistId}
          symbol={symbol}
          onSaved={handleSaved}
          onCancel={() => setEditing(null)}
        />
      )}

      {loading && <div className="notes-hint">Loading notes…</div>}

      {!loading && notes.length === 0 && editing !== NEW_NOTE && (
        <div className="notes-empty">
          No notes for {symbol} yet. Start with what you are watching and why.
        </div>
      )}

      {notes.map(note => (
        editing && editing !== NEW_NOTE && editing.id === note.id ? (
          <NoteEditor
            key={note.id}
            watchlistId={watchlistId}
            symbol={symbol}
            note={note}
            onSaved={handleSaved}
            onCancel={() => setEditing(null)}
          />
        ) : (
          <NoteCard
            key={note.id}
            note={note}
            onEdit={setEditing}
            onDelete={handleDelete}
            onOpenImage={setLightboxImage}
            onStatusChange={handleStatusChange}
          />
        )
      ))}


      <ImageLightbox image={lightboxImage} onClose={() => setLightboxImage(null)} />
    </section>
  );
}
