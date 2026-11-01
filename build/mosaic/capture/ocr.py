"""OCR fallback with recorded availability (audit N-60).

Tesseract is resolved from PATH or the standard Windows install path; its
version goes into the recipe digest. If neither Tesseract nor RapidOCR is
available the page disposition says `ocr_unavailable`; an engine that runs
and returns nothing usable is `ocr_attempted_failed`. OCR output is a
separate `ocr_recognized` text artifact; native text is never overwritten.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from mosaic.recipe import rapidocr_info, tesseract_info


@dataclass
class OcrResult:
    status: str  # ocr_unavailable | ocr_attempted_failed | ocr_succeeded
    engine: str | None
    engine_version: str | None
    text: str
    detail: str = ""


def ocr_png(png: bytes, prefer: str = "tesseract") -> OcrResult:
    tess = tesseract_info()
    if prefer == "tesseract" and tess["available"]:
        try:
            proc = subprocess.run([tess["path"], "stdin", "stdout", "--psm", "3"], input=png,
                                  capture_output=True, timeout=180)
            # Tesseract on Windows ends lines with CRLF; the artifact is LF on every platform.
            text = proc.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
            if proc.returncode == 0 and text.strip():
                return OcrResult("ocr_succeeded", "tesseract", tess["version"], text)
            return OcrResult("ocr_attempted_failed", "tesseract", tess["version"], "",
                             f"exit {proc.returncode}; {len(text.strip())} chars")
        except (OSError, subprocess.SubprocessError) as exc:
            return OcrResult("ocr_attempted_failed", "tesseract", tess["version"], "", type(exc).__name__)
    rapid = rapidocr_info()
    if rapid["available"]:
        try:
            import numpy as np  # noqa: WPS433
            import cv2  # noqa: WPS433
            from rapidocr_onnxruntime import RapidOCR  # noqa: WPS433

            image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
            result, _ = RapidOCR()(image)
            lines = [r[1] for r in (result or [])]
            if lines:
                return OcrResult("ocr_succeeded", "rapidocr", rapid["version"], "\n".join(lines) + "\n")
            return OcrResult("ocr_attempted_failed", "rapidocr", rapid["version"], "", "no text lines")
        except Exception as exc:  # engine failure is recorded, not raised
            return OcrResult("ocr_attempted_failed", "rapidocr", rapid["version"], "", type(exc).__name__)
    return OcrResult("ocr_unavailable", None, None, "", "neither tesseract nor rapidocr available")
