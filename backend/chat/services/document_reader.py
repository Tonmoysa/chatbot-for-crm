from __future__ import annotations

import io
import logging
import re
import os
import shutil
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentExtractResult:
    text: str
    warnings: list[str]
    source: str  # e.g. "pdf_text", "ocr", "none"


def _norm_text(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").replace("\x00", " ")).strip()


def extract_text_from_upload(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    max_chars: int = 60_000,
) -> DocumentExtractResult:
    """
    Best-effort text extraction for receipts/invoices.
    - PDF: try embedded text via pypdf
    - Images: optional OCR via pytesseract (requires system tesseract)

    Returns extracted text (possibly empty) and warnings.
    """
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    warnings: list[str] = []

    is_pdf = name.endswith(".pdf") or "pdf" in ctype
    if is_pdf:
        txt = _pdf_text(data, warnings)
        txt = _norm_text(txt)
        # Keep current fast-path: embedded PDF text is the best quality and cheapest.
        if txt:
            return DocumentExtractResult(
                text=txt[:max_chars], warnings=warnings, source="pdf_text"
            )

        # Scanned PDFs are image-based and will typically produce empty (or tiny) embedded text.
        # In production, it's common to accept receipts/invoices as scanned PDFs, so we attempt
        # OCR as a fallback when text extraction yields nothing useful.
        if not txt or len(txt) < 30:
            warnings.append(
                "No embedded text detected in PDF (may be scanned). Attempting OCR."
            )
            ocr_txt = _pdf_ocr(data, warnings, max_chars=max_chars)
            ocr_txt = _norm_text(ocr_txt)
            if ocr_txt:
                return DocumentExtractResult(
                    text=ocr_txt[:max_chars], warnings=warnings, source="ocr"
                )

        return DocumentExtractResult(text="", warnings=warnings, source="none")

    is_image = any(
        name.endswith(ext)
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    ) or (ctype.startswith("image/"))
    if is_image:
        txt = _image_ocr(data, warnings)
        txt = _norm_text(txt)
        return DocumentExtractResult(
            text=txt[:max_chars], warnings=warnings, source="ocr" if txt else "none"
        )

    warnings.append("Unsupported file type. Upload a PDF or an image (png/jpg).")
    return DocumentExtractResult(text="", warnings=warnings, source="none")


def _pdf_text(data: bytes, warnings: list[str]) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        warnings.append(
            "Missing dependency: pypdf. Install it to enable PDF text extraction."
        )
        return ""

    try:
        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)
    except Exception:
        warnings.append("Failed to parse PDF.")
        return ""


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def _configure_tesseract(warnings: list[str] | None = None) -> bool:
    """
    Configure pytesseract executable location safely.

    WHY: In production (and especially on Windows), tesseract may not be on PATH.
    We allow explicit configuration via `TESSERACT_CMD` while keeping a PATH fallback.
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        if warnings is not None:
            warnings.append(
                "Missing dependency: pytesseract (and system tesseract). Install/configure to enable OCR."
            )
        return False

    cmd = (os.getenv("TESSERACT_CMD") or "").strip()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return True

    which = shutil.which("tesseract") or shutil.which("tesseract.exe")
    if which:
        pytesseract.pytesseract.tesseract_cmd = which
        return True

    if warnings is not None:
        warnings.append(
            "OCR engine (tesseract) not found. Install Tesseract and ensure it's in PATH, "
            "or set TESSERACT_CMD to the full path of tesseract.exe."
        )
    return False


def _prepare_image_for_ocr(img, warnings: list[str]):
    """
    Lightweight OCR preprocessing.

    WHY: Receipts/invoices often suffer from low contrast or odd orientation. We keep this
    minimal (Pillow-only) to avoid heavy CV dependencies while improving accuracy.
    """
    try:
        from PIL import ImageOps  # type: ignore
    except Exception:
        # If pillow is missing, callers will already have warned; keep defensive.
        return img

    img = ImageOps.exif_transpose(img)
    if getattr(img, "mode", None) not in ("RGB", "L"):
        img = img.convert("RGB")

    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)

    # Upscale tiny images a bit; OCR quality drops sharply below ~800px shortest edge.
    # Limit overall pixels to avoid abusive requests.
    max_pixels = _env_int(
        "OCR_MAX_IMAGE_PIXELS", 30_000_000
    )  # ~30MP default safety cap
    w, h = gray.size
    pixels = int(w) * int(h)
    if pixels > max_pixels:
        warnings.append("Image too large for OCR. Please upload a smaller image/PDF.")
        raise ValueError("image too large")

    min_edge = min(w, h)
    if min_edge and min_edge < 800:
        scale = min(2.0, 800 / float(min_edge))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        # Re-check after resize (scale might still exceed pixel cap if image is weirdly shaped).
        if new_w * new_h <= max_pixels:
            try:
                gray = gray.resize((new_w, new_h))
            except Exception:
                # Resizing failure shouldn't break extraction; proceed with original.
                pass

    return gray


def _tesseract_image_to_string(img, warnings: list[str]) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception:
        warnings.append(
            "Missing dependency: pytesseract (and system tesseract). Install/configure to enable OCR."
        )
        return ""

    # lang = (os.getenv("TESSERACT_LANG") or "eng").strip() or "eng"
    lang = (os.getenv("TESSERACT_LANG") or "ben+eng").strip() or "ben+eng"
    # config = "--psm 6"
    config = "--oem 3 --psm 6"
    try:
        return pytesseract.image_to_string(img, lang=lang, config=config) or ""
    except Exception:
        # Retry without lang/config in case of missing traineddata or unsupported params.
        txt = pytesseract.image_to_string(img) or ""
        if lang != "eng":
            warnings.append(
                f"OCR language '{lang}' may be unavailable; fell back to default."
            )
        return txt


def _image_ocr(data: bytes, warnings: list[str]) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        warnings.append("Missing dependency: pillow. Install it to enable image OCR.")
        return ""

    if not _configure_tesseract(warnings):
        return ""

    try:
        img = Image.open(io.BytesIO(data))
        prepped = _prepare_image_for_ocr(img, warnings)
        txt = _tesseract_image_to_string(prepped, warnings)
        return txt
    except Exception as exc:
        # Never leak raw exception details into warnings (API response); log for ops.
        logger.exception("Image OCR failed: %s", exc)
        warnings.append("OCR failed on the uploaded image.")
        return ""


def _pdf_ocr(data: bytes, warnings: list[str], *, max_chars: int) -> str:
    """
    OCR a scanned PDF by rasterizing pages then running tesseract.

    WHY: scanned PDFs contain image-based pages with no embedded text. We rasterize pages
    with pdf2image (poppler) and OCR each page with safety limits.
    """
    try:
        from pdf2image import convert_from_bytes  # type: ignore
    except Exception:
        warnings.append(
            "Missing dependency: pdf2image. Install it to enable scanned PDF OCR."
        )
        return ""

    try:
        from PIL import Image  # type: ignore
    except Exception:
        warnings.append(
            "Missing dependency: pillow. Install it to enable scanned PDF OCR."
        )
        return ""

    if not _configure_tesseract(warnings):
        return ""

    page_limit = _env_int("OCR_PAGE_LIMIT", 10)
    hard_char_limit = min(
        max_chars * 3, 200_000
    )  # prevent runaway OCR output/memory use

    try:
        # Moderate DPI balances accuracy and performance/memory.
        # We cap pages at conversion-time to avoid rasterizing the entire PDF.
        images: list[Image.Image] = convert_from_bytes(
            data,
            dpi=200,
            first_page=1,
            last_page=page_limit,
            fmt="png",
        )
    except Exception as exc:
        logger.exception("PDF rasterization failed: %s", exc)
        warnings.append("Failed to rasterize PDF for OCR.")
        return ""

    if not images:
        warnings.append("No pages found in PDF for OCR.")
        return ""

    # If the PDF has more pages than we processed, be explicit.
    # (pdf2image doesn't reliably tell total pages without extra work; we keep it simple.)
    if len(images) >= page_limit:
        warnings.append(f"PDF OCR processed up to {page_limit} page(s).")

    parts: list[str] = []
    total = 0
    for idx, img in enumerate(images, start=1):
        if total >= hard_char_limit:
            warnings.append("OCR output truncated due to size limits.")
            break
        try:
            prepped = _prepare_image_for_ocr(img, warnings)
            page_txt = _tesseract_image_to_string(prepped, warnings)
            page_txt = (page_txt or "").strip()
            if page_txt:
                parts.append(f"[Page {idx}]\n{page_txt}")
                total += len(page_txt)
        except ValueError:
            # _prepare_image_for_ocr raises ValueError when image is too large; keep going.
            warnings.append(f"Skipped page {idx} due to size limits.")
            continue
        except Exception as exc:
            logger.exception("PDF OCR failed on page %s: %s", idx, exc)
            warnings.append(f"OCR failed on PDF page {idx}.")
            continue

    return "\n\n".join(parts)
