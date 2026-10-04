import React, { useEffect, useRef, useState } from "react";
import Markdown from "./Markdown";
import { createNote, updateNote, uploadImages, deleteImage, imageUrl } from "./notesApi";

/** Today in the browser's timezone, as YYYY-MM-DD. */
function today() {
  const now = new Date();
  const pad = n => String(n).padStart(2, "0");
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

const isImage = file => file && file.type.startsWith("image/");

/**
 * Create or edit one note: a recorded date, markdown text, and images that can
 * be pasted, dropped or picked from disk.
 */
export default function NoteEditor({ watchlistId, symbol, note, onSaved, onCancel }) {
  const editing = Boolean(note);
  const [noteDate, setNoteDate] = useState(note?.note_date || today());
  const [status, setStatus] = useState(note?.status || "INITIAL");
  const [body, setBody] = useState(note?.body || "");
  const [pending, setPending] = useState([]);          // [{file, url}] not yet uploaded
  const [existing, setExisting] = useState(note?.images || []);
  const [showPreview, setShowPreview] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);
  const pendingRef = useRef(pending);

  pendingRef.current = pending;
  // Release object URLs when the editor closes.
  useEffect(() => () => pendingRef.current.forEach(p => URL.revokeObjectURL(p.url)), []);

  const addFiles = (files) => {
    const images = Array.from(files || []).filter(isImage);
    if (!images.length) return;
    setPending(prev => [
      ...prev,
      ...images.map(file => ({ file, url: URL.createObjectURL(file) })),
    ]);
  };

  const handlePaste = (e) => {
    const files = Array.from(e.clipboardData?.items || [])
      .filter(item => item.kind === "file")
      .map(item => item.getAsFile())
      .filter(isImage);
    if (files.length) {
      e.preventDefault();   // keep the image out of the text body
      addFiles(files);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer?.files);
  };

  const removePending = (index) => {
    setPending(prev => {
      const next = [...prev];
      URL.revokeObjectURL(next[index].url);
      next.splice(index, 1);
      return next;
    });
  };

  const removeExisting = async (image) => {
    if (!window.confirm(`Delete image "${image.filename || image.id}"?`)) return;
    try {
      await deleteImage(image.id);
      setExisting(prev => prev.filter(i => i.id !== image.id));
    } catch (e) {
      setError(e.message);
    }
  };

  const handleSave = async () => {
    setError("");
    setSaving(true);
    try {
      const saved = editing
        ? await updateNote(note.id, { body, note_date: noteDate, status })
        : await createNote(watchlistId, symbol, { body, noteDate, status });
      if (pending.length) {
        // Images are a separate call so they can also be added to older notes.
        await uploadImages(saved.id, pending.map(p => p.file));
      }
      pending.forEach(p => URL.revokeObjectURL(p.url));
      setPending([]);
      onSaved();
    } catch (e) {
      setError(e.message);
    }
    setSaving(false);
  };

  return (
    <div
      className={`notes-editor${dragging ? " notes-dragging" : ""}`}
      onPaste={handlePaste}
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
    >
      <div className="notes-editor-head">
        <strong>{editing ? `Edit note · ${symbol}` : `New note · ${symbol}`}</strong>
        <label className="notes-date">
          Date recorded{" "}
          <input type="date" value={noteDate} onChange={e => setNoteDate(e.target.value)} />
        </label>
        <label className="notes-date">
          Status{" "}
          <select
            className={`notes-status-select notes-status-select-${status.toLowerCase()}`}
            value={status}
            onChange={e => setStatus(e.target.value)}
          >
            <option value="INITIAL">INITIAL</option>
            <option value="CONFIRMED">CONFIRMED</option>
            <option value="WRONG">WRONG</option>
          </select>
        </label>
        <button className="notes-btn" onClick={() => setShowPreview(p => !p)}>
          {showPreview ? "Write" : "Preview"}
        </button>
      </div>


      {showPreview ? (
        <div className="notes-preview"><Markdown text={body} /></div>
      ) : (
        <textarea
          className="notes-textarea"
          value={body}
          onChange={e => setBody(e.target.value)}
          placeholder="What are you thinking about this symbol? Markdown works. Paste a screenshot with Cmd+V."
          rows={10}
        />
      )}

      <div className="notes-dropzone">
        <span>
          Paste, drop, or{" "}
          <button className="notes-link" onClick={() => fileInputRef.current?.click()}>
            choose images
          </button>
        </span>
        <input
          type="file"
          ref={fileInputRef}
          multiple
          accept="image/*"
          style={{ display: "none" }}
          onChange={e => { addFiles(e.target.files); e.target.value = ""; }}
        />
      </div>

      {(existing.length > 0 || pending.length > 0) && (
        <div className="notes-thumbs">
          {existing.map(image => (
            <figure key={`saved-${image.id}`} className="notes-thumb">
              <img src={imageUrl(image.thumb_url)} alt={image.filename || "note image"} />
              <figcaption>
                <button className="notes-link" onClick={() => removeExisting(image)}>remove</button>
              </figcaption>
            </figure>
          ))}
          {pending.map((item, index) => (
            <figure key={`pending-${index}`} className="notes-thumb notes-thumb-pending">
              <img src={item.url} alt={item.file.name} />
              <figcaption>
                <button className="notes-link" onClick={() => removePending(index)}>remove</button>
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      {error && <div className="notes-error">{error}</div>}

      <div className="notes-editor-actions">
        <button className="notes-btn notes-btn-primary" onClick={handleSave} disabled={saving}>
          {saving ? "Saving…" : "Save note"}
        </button>
        <button className="notes-btn" onClick={onCancel} disabled={saving}>Cancel</button>
        {pending.length > 0 && (
          <span className="notes-hint">{pending.length} image(s) will be uploaded</span>
        )}
      </div>
    </div>
  );
}
