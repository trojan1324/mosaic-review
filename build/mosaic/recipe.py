"""Public OCR runtime discovery, with private named recipes unavailable.

The public workflow builds its recipe from explicit source/model/runtime inputs.
Shared legacy extraction helpers may import ``resolve_recipe``; invoking their
private experiment entry points refuses rather than discovering private roots.
OCR discovery is copied unchanged from MOSAIC's recipe module.
"""
from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

TESSERACT_CANDIDATES = [Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")]


def _pkg_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def tesseract_info() -> dict[str, Any]:
    path = shutil.which("tesseract")
    if not path:
        for cand in TESSERACT_CANDIDATES:
            if cand.exists():
                path = str(cand)
                break
    if not path:
        return {"available": False, "path": None, "version": None}
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20)
        first = (out.stdout or out.stderr).strip().splitlines()[0]
        version = first.split()[-1] if first else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return {"available": False, "path": path, "version": None}
    return {"available": bool(version), "path": path.replace("\\", "/"), "version": version}


def rapidocr_info() -> dict[str, Any]:
    version = _pkg_version("rapidocr-onnxruntime")
    return {"available": version is not None, "version": version}


def resolve_recipe(name: str):
    raise ValueError("public_private_recipe_unavailable: use bootstrap.py prepare-local or corpus with explicit runtime inputs")
