"""Polish outbound assistant text for readable chat UI (markdown-friendly)."""

from __future__ import annotations

import re

_SECTION_START = re.compile(
    r"^(\d{1,2})[\.\)]\s*(.*)$",
    re.UNICODE,
)
_BULLET_START = re.compile(r"^[•●▪◦]\s*(.*)$", re.UNICODE)
_SENTENCE_END = re.compile(r'[。\.!??:][\s"\'\)\]]*$')


def collapse_pdf_line_breaks(text: str) -> str:
    """Merge lines broken by PDF/OCR while keeping real paragraphs."""
    if not text:
        return ""
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    buf: list[str] = []

    def flush_buf() -> None:
        if buf:
            out.append(" ".join(buf))
            buf.clear()

    for line in raw_lines:
        s = line.strip()
        if not s:
            flush_buf()
            if out and out[-1] != "":
                out.append("")
            continue

        if _SECTION_START.match(s) or _BULLET_START.match(s) or s.startswith(("#", "**", "- ", "* ")):
            flush_buf()
            out.append(s)
            continue

        words = s.split()
        is_fragment = (
            len(s) < 80
            and len(words) <= 6
            and not _SENTENCE_END.search(s)
        )
        if is_fragment:
            buf.append(s)
        else:
            if buf:
                out.append(" ".join(buf + [s]))
                buf.clear()
            else:
                out.append(s)

    flush_buf()
    joined = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def normalize_markdown_bullets(text: str) -> str:
    """Convert PDF bullets to markdown list markers."""
    lines: list[str] = []
    pending_bullet = False
    for line in text.split("\n"):
        s = line.strip()
        if s in ("•", "●", "▪", "◦"):
            pending_bullet = True
            continue
        m = _BULLET_START.match(s)
        if m:
            body = (m.group(1) or "").strip()
            lines.append(f"- {body}" if body else "-")
            pending_bullet = False
            continue
        sm = _SECTION_START.match(s)
        if sm and not sm.group(2).strip():
            lines.append(f"**{sm.group(1)}.**")
            pending_bullet = False
            continue
        if sm and len(sm.group(2)) < 60:
            lines.append(f"**{sm.group(1)}. {sm.group(2).strip()}**")
            pending_bullet = False
            continue
        if pending_bullet and s:
            lines.append(f"- {s}")
            pending_bullet = False
        else:
            lines.append(line)
            pending_bullet = False
    return "\n".join(lines)


def polish_policy_answer(text: str) -> str:
    """Readable policy/RAG answer: fix broken lines, bullets, spacing."""
    t = collapse_pdf_line_breaks(text or "")
    t = normalize_markdown_bullets(t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"([^\n])\n(- )", r"\1\n\n\2", t)
    return t.strip()
