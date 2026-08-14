import React, { useEffect } from "react";
import { imageUrl } from "./notesApi";

/**
 * Full-size overlay for a note image, closed with Escape or a click outside.
 */
export default function ImageLightbox({ image, onClose }) {
  useEffect(() => {
    const onKeyDown = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  if (!image) return null;

  return (
    <div className="notes-lightbox" onClick={onClose}>
      <div className="notes-lightbox-inner" onClick={e => e.stopPropagation()}>
        <div className="notes-lightbox-bar">
          <span>{image.filename || "image"}</span>
          <span className="notes-lightbox-meta">
            {image.width && image.height ? `${image.width}×${image.height}` : ""}
            {" "}
            <button className="notes-btn" onClick={onClose}>Close</button>
          </span>
        </div>
        <img src={imageUrl(image.url)} alt={image.filename || "note image"} />
      </div>
    </div>
  );
}
