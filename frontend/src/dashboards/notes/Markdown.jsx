import React, { useMemo } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";

marked.setOptions({ breaks: true, gfm: true });

/**
 * Render a note's markdown body.
 *
 * Note text is user-authored and stored verbatim, so the generated HTML is
 * always sanitized before it is injected.
 */
export default function Markdown({ text }) {
  const html = useMemo(() => {
    if (!text) return "";
    return DOMPurify.sanitize(marked.parse(text));
  }, [text]);

  if (!text) {
    return <div className="notes-markdown notes-empty">(no text)</div>;
  }
  return (
    <div className="notes-markdown" dangerouslySetInnerHTML={{ __html: html }} />
  );
}
