"""
Validation and thumbnail generation for images attached to notes.

Note images are stored in Postgres as BYTEA, so every upload is verified with
Pillow before it reaches the database, and a small thumbnail is generated so
listing notes never has to transfer full-size screenshots.
"""
import io

from PIL import Image, UnidentifiedImageError

from core.config import settings
from utils.exceptions import InvalidImageError, ImageTooLargeError

# Pillow format -> mime type. The mime is derived from the decoded image rather
# than the client-supplied content type, so a renamed file cannot lie about it.
SUPPORTED_FORMATS = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}

THUMBNAIL_QUALITY = 80
JPEG_REENCODE_QUALITY = 92


def process_note_image(data: bytes, filename: str = "image"):
    """
    Validate an uploaded image and build its stored representation.

    Returns a dict with `mime_type`, `data`, `thumbnail`, `width`, `height`.
    Original bytes are kept as-is (lossless) except for JPEGs, which are
    re-encoded to drop EXIF metadata such as GPS coordinates.

    Raises:
        ImageTooLargeError: if the upload exceeds `note_image_max_bytes`.
        InvalidImageError: if the bytes are empty or not a supported image.
    """
    if not data:
        raise InvalidImageError(filename, "File is empty")
    if len(data) > settings.note_image_max_bytes:
        raise ImageTooLargeError(filename, len(data), settings.note_image_max_bytes)

    try:
        with Image.open(io.BytesIO(data)) as img:
            image_format = img.format
            if image_format not in SUPPORTED_FORMATS:
                raise InvalidImageError(
                    filename, f"Unsupported image format: {image_format}"
                )
            # Fully decode to reject files that only have a valid header.
            img.load()
            width, height = img.size
            thumbnail = _build_thumbnail(img)
            stored_data = _strip_jpeg_metadata(img, data, image_format)
    except InvalidImageError:
        raise
    except UnidentifiedImageError:
        # Pillow's message embeds a BytesIO repr, which is noise for the user.
        raise InvalidImageError(filename, "Not a recognizable image file")
    except Image.DecompressionBombError:
        raise InvalidImageError(filename, "Image dimensions are unreasonably large")
    except (OSError, ValueError) as exc:
        raise InvalidImageError(filename, str(exc) or "Could not decode image")

    return {
        "mime_type": SUPPORTED_FORMATS[image_format],
        "data": stored_data,
        "thumbnail": thumbnail,
        "width": width,
        "height": height,
    }


def _build_thumbnail(img: Image.Image):
    """Return WEBP thumbnail bytes bounded by `note_image_thumb_px`, or None on failure."""
    max_px = settings.note_image_thumb_px
    try:
        thumb = img.copy()
        # GIF/palette frames need a mode WEBP can encode.
        if thumb.mode not in ("RGB", "RGBA"):
            thumb = thumb.convert("RGBA" if "A" in thumb.getbands() else "RGB")
        thumb.thumbnail((max_px, max_px))
        buffer = io.BytesIO()
        thumb.save(buffer, format="WEBP", quality=THUMBNAIL_QUALITY)
        return buffer.getvalue()
    except (OSError, ValueError):
        # A missing thumbnail is not fatal - the full image is served instead.
        return None


def _strip_jpeg_metadata(img: Image.Image, data: bytes, image_format: str) -> bytes:
    """Re-encode JPEGs without EXIF; other formats keep their original bytes."""
    if image_format != "JPEG":
        return data
    try:
        clean = Image.new(img.mode, img.size)
        clean.putdata(list(img.getdata()))
        buffer = io.BytesIO()
        clean.save(buffer, format="JPEG", quality=JPEG_REENCODE_QUALITY)
        return buffer.getvalue()
    except (OSError, ValueError):
        return data
