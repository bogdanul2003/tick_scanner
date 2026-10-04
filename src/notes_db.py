"""
Database access for the Notes Dashboard.

Notes watchlists are intentionally separate from the trading `watchlists` tables:
they hold only what is needed to track a thinking process per symbol
(free-form markdown notes plus images), not OHLCV/indicator data.

Tables (all created by `create_note_tables`):
    note_watchlists          - a named collection of symbols to write notes about
    note_watchlist_symbols   - symbols belonging to a notes watchlist
    symbol_notes             - one dated note (markdown body) for a symbol
    note_images              - images attached to a note, stored as BYTEA
"""
import psycopg2

from db_utils import get_connection, put_connection


def create_note_tables():
    """Create the Notes Dashboard tables if they do not exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS note_watchlists (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS note_watchlist_symbols (
                watchlist_id INTEGER NOT NULL REFERENCES note_watchlists(id) ON DELETE CASCADE,
                symbol TEXT NOT NULL,
                added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (watchlist_id, symbol)
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS symbol_notes (
                id SERIAL PRIMARY KEY,
                watchlist_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                note_date DATE NOT NULL DEFAULT CURRENT_DATE,
                body TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'INITIAL',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                FOREIGN KEY (watchlist_id, symbol)
                    REFERENCES note_watchlist_symbols (watchlist_id, symbol)
                    ON DELETE CASCADE
            );
            """)
            cur.execute("""
            ALTER TABLE symbol_notes ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'INITIAL';
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_notes_wl_symbol
            ON symbol_notes (watchlist_id, symbol, note_date DESC, id DESC);
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS note_images (
                id SERIAL PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES symbol_notes(id) ON DELETE CASCADE,
                filename TEXT,
                mime_type TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                data BYTEA NOT NULL,
                thumbnail BYTEA,
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_note_images_note
            ON note_images (note_id, sort_order);
            """)
            conn.commit()
    finally:
        put_connection(conn)


# --- serialization helpers -------------------------------------------------

def _iso(value):
    """Format a datetime/date as an ISO string, tolerating None."""
    return value.isoformat() if value is not None else None


def _image_meta(row):
    """Build image metadata (no bytes) from a (id, filename, mime, size, w, h, order) row."""
    image_id = row[0]
    return {
        "id": image_id,
        "filename": row[1],
        "mime_type": row[2],
        "byte_size": row[3],
        "width": row[4],
        "height": row[5],
        "sort_order": row[6],
        "url": f"/notes/images/{image_id}",
        "thumb_url": f"/notes/images/{image_id}/thumb",
    }


def _note_dict(row, images=None):
    """Build a note dict from a (id, watchlist_id, symbol, note_date, body, status, created, updated) row."""
    return {
        "id": row[0],
        "watchlist_id": row[1],
        "symbol": row[2],
        "note_date": _iso(row[3]),
        "body": row[4],
        "status": row[5] or "INITIAL",
        "created_at": _iso(row[6]),
        "updated_at": _iso(row[7]),
        "images": images if images is not None else [],
    }


NOTE_COLUMNS = "id, watchlist_id, symbol, note_date, body, status, created_at, updated_at"



def _touch_watchlist(cur, watchlist_id):
    """Bump a watchlist's updated_at so the dashboard can sort by recent activity."""
    cur.execute(
        "UPDATE note_watchlists SET updated_at = now() WHERE id = %s",
        (watchlist_id,),
    )


# --- watchlists ------------------------------------------------------------

def list_note_watchlists():
    """List all notes watchlists with symbol and note counts, most recently touched first."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT w.id, w.name, w.description, w.created_at, w.updated_at,
                       (SELECT count(*) FROM note_watchlist_symbols s WHERE s.watchlist_id = w.id),
                       (SELECT count(*) FROM symbol_notes n WHERE n.watchlist_id = w.id)
                FROM note_watchlists w
                ORDER BY w.updated_at DESC, w.name
            """)
            return [
                {
                    "id": r[0],
                    "name": r[1],
                    "description": r[2],
                    "created_at": _iso(r[3]),
                    "updated_at": _iso(r[4]),
                    "symbol_count": r[5],
                    "note_count": r[6],
                }
                for r in cur.fetchall()
            ]
    finally:
        put_connection(conn)


def get_note_watchlist(watchlist_id):
    """Get one notes watchlist with a per-symbol summary, or None if it does not exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, description, created_at, updated_at
                FROM note_watchlists WHERE id = %s
            """, (watchlist_id,))
            row = cur.fetchone()
            if not row:
                return None
            cur.execute("""
                SELECT s.symbol,
                       count(n.id),
                       max(n.note_date)
                FROM note_watchlist_symbols s
                LEFT JOIN symbol_notes n
                       ON n.watchlist_id = s.watchlist_id AND n.symbol = s.symbol
                WHERE s.watchlist_id = %s
                GROUP BY s.symbol
                ORDER BY s.symbol
            """, (watchlist_id,))
            symbols = [
                {
                    "symbol": r[0],
                    "note_count": r[1],
                    "last_note_date": _iso(r[2]),
                }
                for r in cur.fetchall()
            ]
            return {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "created_at": _iso(row[3]),
                "updated_at": _iso(row[4]),
                "symbols": symbols,
            }
    finally:
        put_connection(conn)


def create_note_watchlist(name, description=None):
    """Create a notes watchlist. Returns the new id, or None if the name is taken."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO note_watchlists (name, description)
                VALUES (%s, %s)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
            """, (name, description))
            result = cur.fetchone()
            conn.commit()
            return result[0] if result else None
    finally:
        put_connection(conn)


def update_note_watchlist(watchlist_id, name=None, description=None):
    """
    Rename a notes watchlist and/or change its description.

    Only non-None fields are written. Returns True if the row existed.
    Raises ValueError if the new name is already used by another watchlist.
    """
    updates = []
    params = []
    if name is not None:
        updates.append("name = %s")
        params.append(name)
    if description is not None:
        updates.append("description = %s")
        params.append(description)
    if not updates:
        return get_note_watchlist(watchlist_id) is not None

    updates.append("updated_at = now()")
    params.append(watchlist_id)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    f"UPDATE note_watchlists SET {', '.join(updates)} WHERE id = %s",
                    tuple(params),
                )
            except psycopg2.IntegrityError:
                conn.rollback()
                raise ValueError(f"A notes watchlist named '{name}' already exists.")
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        put_connection(conn)


def delete_note_watchlist(watchlist_id):
    """Delete a notes watchlist and everything under it. Returns True if it existed."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM note_watchlists WHERE id = %s", (watchlist_id,))
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    finally:
        put_connection(conn)


# --- symbols ---------------------------------------------------------------

def add_symbols_to_note_watchlist(watchlist_id, symbols):
    """
    Add symbols to a notes watchlist (idempotent).

    Returns the list of symbols that were newly inserted.
    Raises ValueError if the watchlist does not exist.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM note_watchlists WHERE id = %s", (watchlist_id,))
            if not cur.fetchone():
                raise ValueError(f"Notes watchlist {watchlist_id} does not exist.")
            added = []
            for symbol in symbols:
                cur.execute("""
                    INSERT INTO note_watchlist_symbols (watchlist_id, symbol)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING symbol
                """, (watchlist_id, symbol))
                row = cur.fetchone()
                if row:
                    added.append(row[0])
            if added:
                _touch_watchlist(cur, watchlist_id)
            conn.commit()
            return added
    finally:
        put_connection(conn)


def remove_symbol_from_note_watchlist(watchlist_id, symbol):
    """
    Remove a symbol from a notes watchlist, cascading to its notes and images.

    Returns the number of notes deleted, or None if the symbol was not in the watchlist.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM symbol_notes WHERE watchlist_id = %s AND symbol = %s",
                (watchlist_id, symbol),
            )
            note_count = cur.fetchone()[0]
            cur.execute(
                "DELETE FROM note_watchlist_symbols WHERE watchlist_id = %s AND symbol = %s",
                (watchlist_id, symbol),
            )
            if cur.rowcount == 0:
                conn.rollback()
                return None
            _touch_watchlist(cur, watchlist_id)
            conn.commit()
            return note_count
    finally:
        put_connection(conn)


# --- notes -----------------------------------------------------------------

def list_notes(watchlist_id, symbol):
    """List a symbol's notes (newest first) with image metadata attached."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT {NOTE_COLUMNS} FROM symbol_notes
                WHERE watchlist_id = %s AND symbol = %s
                ORDER BY note_date DESC, id DESC
            """, (watchlist_id, symbol))
            rows = cur.fetchall()
            if not rows:
                return []
            note_ids = [r[0] for r in rows]
            cur.execute("""
                SELECT id, filename, mime_type, byte_size, width, height, sort_order, note_id
                FROM note_images
                WHERE note_id = ANY(%s)
                ORDER BY note_id, sort_order, id
            """, (note_ids,))
            images_by_note = {}
            for img in cur.fetchall():
                images_by_note.setdefault(img[7], []).append(_image_meta(img))
            return [_note_dict(r, images_by_note.get(r[0], [])) for r in rows]
    finally:
        put_connection(conn)


def get_note(note_id):
    """Get a single note with its image metadata, or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {NOTE_COLUMNS} FROM symbol_notes WHERE id = %s",
                (note_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute("""
                SELECT id, filename, mime_type, byte_size, width, height, sort_order
                FROM note_images WHERE note_id = %s
                ORDER BY sort_order, id
            """, (note_id,))
            images = [_image_meta(img) for img in cur.fetchall()]
            return _note_dict(row, images)
    finally:
        put_connection(conn)


def create_note(watchlist_id, symbol, body, note_date=None, status="INITIAL"):
    """
    Create a note for a symbol. The symbol is added to the watchlist if missing.

    `note_date` is a 'YYYY-MM-DD' string or None (defaults to today).
    `status` is 'INITIAL', 'CONFIRMED', or 'WRONG' (defaults to 'INITIAL').
    Raises ValueError if the watchlist does not exist.
    """
    status_val = (status or "INITIAL").upper()
    if status_val not in ("INITIAL", "CONFIRMED", "WRONG"):
        status_val = "INITIAL"

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM note_watchlists WHERE id = %s", (watchlist_id,))
            if not cur.fetchone():
                raise ValueError(f"Notes watchlist {watchlist_id} does not exist.")
            cur.execute("""
                INSERT INTO note_watchlist_symbols (watchlist_id, symbol)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
            """, (watchlist_id, symbol))
            if note_date:
                cur.execute(f"""
                    INSERT INTO symbol_notes (watchlist_id, symbol, note_date, body, status)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING {NOTE_COLUMNS}
                """, (watchlist_id, symbol, note_date, body, status_val))
            else:
                cur.execute(f"""
                    INSERT INTO symbol_notes (watchlist_id, symbol, body, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING {NOTE_COLUMNS}
                """, (watchlist_id, symbol, body, status_val))
            note = _note_dict(cur.fetchone())
            _touch_watchlist(cur, watchlist_id)
            conn.commit()
            return note
    finally:
        put_connection(conn)


def update_note(note_id, body=None, note_date=None, status=None):
    """Update a note's body, recorded date, and/or status. Returns the updated note, or None."""
    updates = []
    params = []
    if body is not None:
        updates.append("body = %s")
        params.append(body)
    if note_date is not None:
        updates.append("note_date = %s")
        params.append(note_date)
    if status is not None:
        status_val = status.upper()
        if status_val not in ("INITIAL", "CONFIRMED", "WRONG"):
            raise ValueError(f"Invalid status '{status}'. Must be INITIAL, CONFIRMED, or WRONG.")
        updates.append("status = %s")
        params.append(status_val)
    if not updates:
        return get_note(note_id)

    updates.append("updated_at = now()")
    params.append(note_id)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE symbol_notes SET {', '.join(updates)} WHERE id = %s"
                f" RETURNING {NOTE_COLUMNS}",
                tuple(params),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return None
            _touch_watchlist(cur, row[1])
            conn.commit()
            return get_note(note_id)
    finally:
        put_connection(conn)



def delete_note(note_id):
    """Delete a note and its images. Returns True if it existed."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM symbol_notes WHERE id = %s RETURNING watchlist_id",
                (note_id,),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return False
            _touch_watchlist(cur, row[0])
            conn.commit()
            return True
    finally:
        put_connection(conn)


# --- images ----------------------------------------------------------------

def add_note_image(note_id, filename, mime_type, data, thumbnail=None, width=None, height=None):
    """
    Attach an image to a note. Returns its metadata, or None if the note is gone.

    Image bytes live in Postgres (BYTEA); `thumbnail` is a small re-encoded copy
    used by the notes list so it never has to transfer full-size screenshots.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT watchlist_id FROM symbol_notes WHERE id = %s", (note_id,))
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                "SELECT COALESCE(max(sort_order), -1) + 1 FROM note_images WHERE note_id = %s",
                (note_id,),
            )
            sort_order = cur.fetchone()[0]
            cur.execute("""
                INSERT INTO note_images
                    (note_id, filename, mime_type, byte_size, width, height,
                     data, thumbnail, sort_order)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id, filename, mime_type, byte_size, width, height, sort_order
            """, (
                note_id, filename, mime_type, len(data), width, height,
                psycopg2.Binary(data),
                psycopg2.Binary(thumbnail) if thumbnail else None,
                sort_order,
            ))
            meta = _image_meta(cur.fetchone())
            cur.execute(
                "UPDATE symbol_notes SET updated_at = now() WHERE id = %s",
                (note_id,),
            )
            _touch_watchlist(cur, row[0])
            conn.commit()
            return meta
    finally:
        put_connection(conn)


def get_note_image(image_id, thumb=False):
    """
    Read image bytes for serving.

    Returns (mime_type, bytes) or None. When `thumb` is True the stored
    thumbnail is returned (as image/webp), falling back to the full image.
    """
    column = "thumbnail" if thumb else "data"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT mime_type, {column}, data FROM note_images WHERE id = %s",
                (image_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            if thumb:
                if row[1] is not None:
                    return "image/webp", bytes(row[1])
                return row[0], bytes(row[2])
            return row[0], bytes(row[1])
    finally:
        put_connection(conn)


def delete_note_image(image_id):
    """Delete one image. Returns True if it existed."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM note_images WHERE id = %s RETURNING note_id",
                (image_id,),
            )
            row = cur.fetchone()
            if not row:
                conn.rollback()
                return False
            cur.execute(
                "UPDATE symbol_notes SET updated_at = now() WHERE id = %s",
                (row[0],),
            )
            conn.commit()
            return True
    finally:
        put_connection(conn)
