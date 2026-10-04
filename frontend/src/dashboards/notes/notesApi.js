// Client for the /notes endpoints.
import { apiJson, apiUrl } from "../../api";

const json = (method, body) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const listWatchlists = () =>
  apiJson("/notes/watchlists");

export const getWatchlist = (id) =>
  apiJson(`/notes/watchlists/${id}`);

export const createWatchlist = (name, description) =>
  apiJson("/notes/watchlists", json("POST", { name, description: description || null }));

export const updateWatchlist = (id, fields) =>
  apiJson(`/notes/watchlists/${id}`, json("PATCH", fields));

export const deleteWatchlist = (id) =>
  apiJson(`/notes/watchlists/${id}`, { method: "DELETE" });

export const addSymbols = (id, symbols) =>
  apiJson(`/notes/watchlists/${id}/symbols`, json("POST", { symbols }));

export const removeSymbol = (id, symbol) =>
  apiJson(`/notes/watchlists/${id}/symbols/${encodeURIComponent(symbol)}`, { method: "DELETE" });

export const listNotes = (id, symbol) =>
  apiJson(`/notes/watchlists/${id}/symbols/${encodeURIComponent(symbol)}/notes`);

export const createNote = (id, symbol, { body, noteDate, status }) =>
  apiJson(
    `/notes/watchlists/${id}/symbols/${encodeURIComponent(symbol)}/notes`,
    json("POST", { body, note_date: noteDate || null, status: status || "INITIAL" }),
  );


export const updateNote = (noteId, fields) =>
  apiJson(`/notes/entries/${noteId}`, json("PATCH", fields));

export const deleteNote = (noteId) =>
  apiJson(`/notes/entries/${noteId}`, { method: "DELETE" });

export const deleteImage = (imageId) =>
  apiJson(`/notes/images/${imageId}`, { method: "DELETE" });

/**
 * Upload images to an existing note. The backend validates every file before
 * storing any of them, so a rejected batch leaves the note untouched.
 */
export const uploadImages = (noteId, files) => {
  const form = new FormData();
  files.forEach(file => form.append("files", file, file.name));
  return apiJson(`/notes/entries/${noteId}/images`, { method: "POST", body: form });
};

// Image urls come back from the API as relative paths.
export const imageUrl = (path) => apiUrl(path);
