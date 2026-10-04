// Shared API configuration and fetch helpers.
export const API_BASE = "http://localhost:8000";

/**
 * Build an absolute URL for a path returned by the API (e.g. an image url).
 */
export function apiUrl(path) {
  return `${API_BASE}${path}`;
}

/**
 * Fetch JSON from the API, throwing an Error carrying the server's message.
 *
 * FastAPI reports errors either as {detail: ...} (HTTPException) or as
 * {message, error_code} (the centralized TickScanner handlers), so both are
 * unwrapped into a single readable message.
 */
export async function apiJson(path, options = {}) {
  const res = await fetch(apiUrl(path), options);
  if (!res.ok) {
    throw new Error(await readError(res));
  }
  if (res.status === 204) return null;
  return res.json();
}

async function readError(res) {
  let payload;
  try {
    payload = await res.json();
  } catch {
    return `Request failed (${res.status})`;
  }
  const { detail, message } = payload || {};
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // Pydantic validation errors: [{loc: [...], msg: "..."}]
    return detail.map(d => d.msg || JSON.stringify(d)).join("; ");
  }
  if (message) return message;
  return `Request failed (${res.status})`;
}
