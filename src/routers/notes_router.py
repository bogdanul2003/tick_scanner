"""Notes Dashboard API endpoints.

Notes watchlists are addressed by id (not by name like the trading watchlists)
so renaming one does not break links held by the frontend.
"""
from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from typing import List

import notes_db
from models.requests import (
    NoteCreateRequest,
    NoteSymbolsRequest,
    NoteUpdateRequest,
    NoteWatchlistCreateRequest,
    NoteWatchlistUpdateRequest,
)
from models.responses import (
    NoteImageResponse,
    NoteResponse,
    NoteWatchlistDetailResponse,
    NoteWatchlistSummary,
)
from utils.image_utils import process_note_image

router = APIRouter(prefix="/notes", tags=["Notes"])

# Images are immutable once uploaded, so browsers may cache them indefinitely.
IMAGE_CACHE_CONTROL = "public, max-age=31536000, immutable"


def _get_watchlist_or_404(watchlist_id: int):
    """Load a notes watchlist, raising 404 if it does not exist."""
    watchlist = notes_db.get_note_watchlist(watchlist_id)
    if watchlist is None:
        raise HTTPException(status_code=404, detail=f"Notes watchlist {watchlist_id} not found")
    return watchlist


# --- watchlists ------------------------------------------------------------

@router.get("/watchlists", response_model=List[NoteWatchlistSummary])
async def list_note_watchlists():
    """List all notes watchlists with symbol and note counts."""
    return notes_db.list_note_watchlists()


@router.post("/watchlists", response_model=NoteWatchlistDetailResponse, status_code=201)
async def create_note_watchlist(payload: NoteWatchlistCreateRequest):
    """Create a notes watchlist."""
    watchlist_id = notes_db.create_note_watchlist(payload.name, payload.description)
    if watchlist_id is None:
        raise HTTPException(
            status_code=409,
            detail=f"A notes watchlist named '{payload.name}' already exists",
        )
    return notes_db.get_note_watchlist(watchlist_id)


@router.get("/watchlists/{watchlist_id}", response_model=NoteWatchlistDetailResponse)
async def get_note_watchlist(watchlist_id: int):
    """Get a notes watchlist with its per-symbol note summary."""
    return _get_watchlist_or_404(watchlist_id)


@router.patch("/watchlists/{watchlist_id}", response_model=NoteWatchlistDetailResponse)
async def update_note_watchlist(watchlist_id: int, payload: NoteWatchlistUpdateRequest):
    """Rename a notes watchlist and/or edit its description."""
    try:
        updated = notes_db.update_note_watchlist(
            watchlist_id, name=payload.name, description=payload.description
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not updated:
        raise HTTPException(status_code=404, detail=f"Notes watchlist {watchlist_id} not found")
    return notes_db.get_note_watchlist(watchlist_id)


@router.delete("/watchlists/{watchlist_id}")
async def delete_note_watchlist(watchlist_id: int):
    """Delete a notes watchlist along with all of its symbols, notes and images."""
    if not notes_db.delete_note_watchlist(watchlist_id):
        raise HTTPException(status_code=404, detail=f"Notes watchlist {watchlist_id} not found")
    return {"deleted": watchlist_id}


# --- symbols ---------------------------------------------------------------

@router.post("/watchlists/{watchlist_id}/symbols")
async def add_note_watchlist_symbols(watchlist_id: int, payload: NoteSymbolsRequest):
    """Add one or more symbols to a notes watchlist."""
    try:
        added = notes_db.add_symbols_to_note_watchlist(watchlist_id, payload.symbols)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"watchlist_id": watchlist_id, "symbols_added": added}


@router.delete("/watchlists/{watchlist_id}/symbols/{symbol}")
async def remove_note_watchlist_symbol(watchlist_id: int, symbol: str):
    """Remove a symbol from a notes watchlist, deleting its notes and images."""
    notes_deleted = notes_db.remove_symbol_from_note_watchlist(watchlist_id, symbol.upper())
    if notes_deleted is None:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{symbol.upper()}' is not in notes watchlist {watchlist_id}",
        )
    return {"symbol": symbol.upper(), "notes_deleted": notes_deleted}


# --- notes -----------------------------------------------------------------

@router.get(
    "/watchlists/{watchlist_id}/symbols/{symbol}/notes",
    response_model=List[NoteResponse],
)
async def list_symbol_notes(watchlist_id: int, symbol: str):
    """List a symbol's notes, newest recorded date first."""
    _get_watchlist_or_404(watchlist_id)
    return notes_db.list_notes(watchlist_id, symbol.upper())


@router.post(
    "/watchlists/{watchlist_id}/symbols/{symbol}/notes",
    response_model=NoteResponse,
    status_code=201,
)
async def create_symbol_note(watchlist_id: int, symbol: str, payload: NoteCreateRequest):
    """Create a note for a symbol, adding the symbol to the watchlist if needed."""
    try:
        return notes_db.create_note(
            watchlist_id, symbol.upper(), payload.body, payload.note_date, payload.status
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.patch("/entries/{note_id}", response_model=NoteResponse)
async def update_note(note_id: int, payload: NoteUpdateRequest):
    """Edit a note's text, recorded date, and/or status."""
    try:
        note = notes_db.update_note(
            note_id, body=payload.body, note_date=payload.note_date, status=payload.status
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if note is None:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
    return note



@router.delete("/entries/{note_id}")
async def delete_note(note_id: int):
    """Delete a note and its images."""
    if not notes_db.delete_note(note_id):
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
    return {"deleted": note_id}


# --- images ----------------------------------------------------------------

@router.post("/entries/{note_id}/images", response_model=List[NoteImageResponse], status_code=201)
async def upload_note_images(note_id: int, files: List[UploadFile] = File(...)):
    """
    Attach one or more images to an existing note.

    Every file is validated and thumbnailed before anything is written, so a bad
    file in the batch fails the whole request instead of leaving partial uploads.
    """
    if notes_db.get_note(note_id) is None:
        raise HTTPException(status_code=404, detail=f"Note {note_id} not found")

    processed = []
    for upload in files:
        raw = await upload.read()
        image = process_note_image(raw, upload.filename or "image")
        processed.append((upload.filename, image))

    stored = []
    for filename, image in processed:
        meta = notes_db.add_note_image(
            note_id,
            filename=filename,
            mime_type=image["mime_type"],
            data=image["data"],
            thumbnail=image["thumbnail"],
            width=image["width"],
            height=image["height"],
        )
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
        stored.append(meta)
    return stored


@router.get("/images/{image_id}")
async def get_note_image(image_id: int):
    """Serve a note image's full-size bytes."""
    image = notes_db.get_note_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    mime_type, data = image
    return Response(
        content=data,
        media_type=mime_type,
        headers={"Cache-Control": IMAGE_CACHE_CONTROL},
    )


@router.get("/images/{image_id}/thumb")
async def get_note_image_thumbnail(image_id: int):
    """Serve a note image's thumbnail (falls back to the full image if absent)."""
    image = notes_db.get_note_image(image_id, thumb=True)
    if image is None:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    mime_type, data = image
    return Response(
        content=data,
        media_type=mime_type,
        headers={"Cache-Control": IMAGE_CACHE_CONTROL},
    )


@router.delete("/images/{image_id}")
async def delete_note_image(image_id: int):
    """Delete a single image from a note."""
    if not notes_db.delete_note_image(image_id):
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    return {"deleted": image_id}
