import React from "react";
import Markdown from "./Markdown";
import { imageUrl } from "./notesApi";

/** Render a timestamp as a readable local date-time. */
function formatStamp(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

/**
 * One saved note: its recorded date, markdown body and image thumbnails.
 */
export default function NoteCard({ note, onEdit, onDelete, onOpenImage }) {
  const edited = note.updated_at > note.created_at;

  return (
    <article className="notes-card">
      <header className="notes-card-head">
        <span className="notes-date-badge">{note.note_date}</span>
        <span className="notes-card-meta" title={`Created ${formatStamp(note.created_at)}`}>
          {edited ? `edited ${formatStamp(note.updated_at)}` : `added ${formatStamp(note.created_at)}`}
        </span>
        <span className="notes-card-actions">
          <button className="notes-btn" onClick={() => onEdit(note)}>Edit</button>
          <button className="notes-btn" onClick={() => onDelete(note)}>Delete</button>
        </span>
      </header>

      <Markdown text={note.body} />

      {note.images.length > 0 && (
        <div className="notes-thumbs">
          {note.images.map(image => (
            <figure key={image.id} className="notes-thumb">
              <img
                src={imageUrl(image.thumb_url)}
                alt={image.filename || "note image"}
                title={image.filename || ""}
                onClick={() => onOpenImage(image)}
              />
            </figure>
          ))}
        </div>
      )}
    </article>
  );
}
