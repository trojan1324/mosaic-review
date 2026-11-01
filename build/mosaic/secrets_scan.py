"""Secret scan used by the pre-commit hook and the history-wide scan.

Hard stop (6) of HANDOFF §6.6. Rules: Vercel gateway keys (``vck_``), OpenAI-
style keys (``sk-``), an ``AI_GATEWAY_API_KEY=`` assignment with a non-empty
value, and any 32+ character high-entropy token. Findings never include the
matched value: only location, rule name and a short hash of the match so two
reports of the same finding can be correlated without revealing it.
"""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

RULES: list[tuple[str, re.Pattern[str]]] = [
    ("vercel_gateway_key", re.compile(r"(?<![A-Za-z0-9])vck_[A-Za-z0-9_\-]{8,}")),
    ("sk_api_key", re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_\-]{16,}")),
    (
        "gateway_key_assignment",
        # '=' followed (optionally quoted) by an actual token of 4+ key characters.
        re.compile(r"AI_GATEWAY_API_KEY[ \t]*=[ \t]*['\"]?[A-Za-z0-9_\-]{4,}"),
    ),
]
_ENTROPY_TOKEN = re.compile(r"(?<![A-Za-z0-9_\-+=])[A-Za-z0-9_\-+=]{32,}(?![A-Za-z0-9_\-+=])")
ENTROPY_THRESHOLD_BITS = 4.5
# A token hitting only two of the three character classes (e.g. an all-lowercase or all-uppercase
# key) needs a higher bar: real identifiers such as FABLE_5_1_SANITIZED_PACKET_BOUNDARY_2026-09-03
# (upper+digit, no lower) sit just above ENTROPY_THRESHOLD_BITS (D-005).
ENTROPY_THRESHOLD_BITS_TWO_CLASS = 4.7
_HEX = re.compile(r"^[0-9a-fA-F\-]+$")

# Paths never scanned: binary renders, the ignored secret file itself and the
# virtual environment. docs/inputs IS scanned.
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".gif", ".ico", ".sqlite", ".db", ".pyc", ".onnx"}


@dataclass(frozen=True)
class Finding:
    location: str
    line: int
    rule: str
    fingerprint: str

    def render(self) -> str:
        return f"{self.location}:{self.line}: {self.rule} (match sha256 prefix {self.fingerprint})"


def shannon_entropy(token: str) -> float:
    counts = Counter(token)
    n = len(token)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_high_entropy(token: str) -> bool:
    if _HEX.match(token):
        return False  # hex digests/UUIDs top out at 4 bits/char
    # Random key material mixes at least two of lower/upper/digit; an identifier such as
    # snake_case_words or ALL_CAPS_2026-09-03 hits at most one or two classes too, so the
    # three-class-only gate missed genuine two-class secrets (lowercase- or uppercase-only
    # keys). A two-class token needs a higher entropy bar to keep those identifiers out.
    classes = sum((any(c.islower() for c in token), any(c.isupper() for c in token), any(c.isdigit() for c in token)))
    if classes < 2:
        return False
    threshold = ENTROPY_THRESHOLD_BITS if classes == 3 else ENTROPY_THRESHOLD_BITS_TWO_CLASS
    return shannon_entropy(token) >= threshold


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:10]


def scan_text(text: str, location: str, allowlist: set[str] | None = None) -> list[Finding]:
    allowlist = allowlist or set()
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in RULES:
            for m in pattern.finditer(line):
                fp = _fingerprint(m.group(0))
                if fp not in allowlist:
                    findings.append(Finding(location, lineno, rule, fp))
        for m in _ENTROPY_TOKEN.finditer(line):
            token = m.group(0)
            if _is_high_entropy(token):
                fp = _fingerprint(token)
                if fp not in allowlist:
                    findings.append(Finding(location, lineno, "high_entropy_token", fp))
    return findings


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:4096]:
        return None
    return data.decode("utf-8", errors="replace")


def git_worktree_files(repo: Path) -> list[Path]:
    """Tracked plus untracked-but-not-ignored files (never the ignored .env)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
    ).stdout
    names = [n for n in out.decode("utf-8").split("\x00") if n]
    return [repo / n for n in names if (repo / n).is_file()]


def scan_paths(paths: Iterable[Path], repo: Path, allowlist: set[str] | None = None) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        text = _read_text(path)
        if text is None:
            continue
        try:
            rel = path.relative_to(repo).as_posix()
        except ValueError:
            rel = str(path)
        findings.extend(scan_text(text, rel, allowlist))
    return findings


def scan_staged_diff(repo: Path, allowlist: set[str] | None = None) -> list[Finding]:
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--no-color", "--unified=0"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8", errors="replace")
    added = "\n".join(l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    return scan_text(added, "<staged diff>", allowlist)


def iter_history_lines(repo: Path) -> Iterator[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "log", "-p", "--all", "--no-color", "--full-history"],
        check=True,
        capture_output=True,
    )
    yield from proc.stdout.decode("utf-8", errors="replace").splitlines()


def scan_history(repo: Path, allowlist: set[str] | None = None) -> tuple[list[Finding], int]:
    lines = list(iter_history_lines(repo))
    return scan_text("\n".join(lines), "<git log -p --all>", allowlist), len(lines)


def load_allowlist(repo: Path) -> set[str]:
    path = repo / "config" / "secret_scan_allowlist.txt"
    if not path.exists():
        return set()
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries
