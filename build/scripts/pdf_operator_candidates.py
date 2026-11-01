"""Propose directly evidenced PDF relation operators without rewriting source text.

This is an isolated diagnostic utility.  It reads a PDF and its already-admitted
body units, and emits only candidates whose glyph identity follows either from
the font Encoding Differences or from a deliberately narrow CFF outline test.
It never changes the supplied source text and never uses statistical context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz

from mosaic.workspace import Workspace


_RELATION_GLYPHS = {"equals": "=", "less": "<", "greater": ">"}
_WIN_ANSI_NAMES = {
    43: "plus", 44: "comma", 45: "hyphen", 50: "two", 53: "five",
    60: "less", 61: "equals", 62: "greater",
}
_SID_NAMES = {
    0: ".notdef", 1: "space", 12: "plus", 13: "comma", 14: "hyphen",
    19: "two", 22: "five", 29: "less", 30: "equal", 31: "greater",
}


@dataclass(frozen=True)
class RawChar:
    char: str
    font: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ShownCode:
    font_resource: str
    code: int


@dataclass(frozen=True)
class TraceChar:
    font: str
    glyph_id: int
    unicode: int
    bbox: tuple[float, float, float, float]


def _normal_font_name(name: str) -> str:
    return name.split("+", 1)[-1]


def _literal_bytes(token: bytes) -> bytes:
    """Decode one PDF literal string, retaining only its bytes."""
    assert token.startswith(b"(") and token.endswith(b")")
    body, out, i = token[1:-1], bytearray(), 0
    while i < len(body):
        value = body[i]
        if value != 92:  # backslash
            out.append(value)
            i += 1
            continue
        i += 1
        if i >= len(body):
            break
        escaped = body[i]
        simple = {110: 10, 114: 13, 116: 9, 98: 8, 102: 12}
        if escaped in simple:
            out.append(simple[escaped]); i += 1; continue
        if 48 <= escaped <= 55:
            digits = bytes([escaped]); i += 1
            while i < len(body) and len(digits) < 3 and 48 <= body[i] <= 55:
                digits += bytes([body[i]]); i += 1
            out.append(int(digits, 8)); continue
        if escaped in (10, 13):
            if escaped == 13 and i + 1 < len(body) and body[i + 1] == 10:
                i += 1
            i += 1; continue
        out.append(escaped); i += 1
    return bytes(out)


def _tokens(data: bytes) -> Iterable[tuple[str, Any]]:
    """Small PDF-content lexer sufficient for Tf/Tj/TJ text showing."""
    i = 0
    while i < len(data):
        c = data[i]
        if c in b"\x00\x09\x0a\x0c\x0d ":
            i += 1; continue
        if c == 37:  # comment
            i = data.find(b"\n", i)
            if i < 0: return
            continue
        if c in (91, 93):
            yield ("array", chr(c)); i += 1; continue
        if c == 47:  # name
            j = i + 1
            while j < len(data) and data[j] not in b"\x00\x09\x0a\x0c\x0d /[]<>()": j += 1
            yield ("name", data[i + 1:j].decode("latin1")); i = j; continue
        if c == 40:  # literal string, balance nested parentheses
            j, depth = i + 1, 1
            while j < len(data) and depth:
                if data[j] == 92: j += 2; continue
                if data[j] == 40: depth += 1
                elif data[j] == 41: depth -= 1
                j += 1
            if depth: raise ValueError("unclosed PDF literal string")
            yield ("string", _literal_bytes(data[i:j])); i = j; continue
        if c == 60 and i + 1 < len(data) and data[i + 1] != 60:
            j = data.find(b">", i + 1)
            if j < 0: raise ValueError("unclosed PDF hexadecimal string")
            hexed = re.sub(rb"\s+", b"", data[i + 1:j])
            if len(hexed) % 2: hexed += b"0"
            yield ("string", bytes.fromhex(hexed.decode("ascii"))); i = j + 1; continue
        if c in (60, 62, 41):
            raise ValueError("unsupported PDF content delimiter")
        j = i
        while j < len(data) and data[j] not in b"\x00\x09\x0a\x0c\x0d /[]<>()": j += 1
        yield ("word", data[i:j].decode("latin1")); i = j


def shown_codes(page: fitz.Page) -> list[ShownCode]:
    """Return PDF text-show bytes with their active font resource names."""
    current_font: str | None = None
    previous: list[tuple[str, Any]] = []
    arrays: list[list[tuple[str, Any]]] = []
    out: list[ShownCode] = []
    try:
        streams = [(kind, value) for stream_xref in page.get_contents() for kind, value in _tokens(page.parent.xref_stream(stream_xref))]
    except ValueError:
        return []
    for kind, value in streams:
            if kind == "array":
                if value == "[": arrays.append([])
                elif arrays:
                    previous.append(("array_value", arrays.pop()))
                continue
            if arrays:
                arrays[-1].append((kind, value))
                continue
            if kind == "word" and value == "Tf":
                names = [v for k, v in previous[-4:] if k == "name"]
                if names: current_font = names[-1]
            elif kind == "word" and value == "Tj" and current_font:
                strings = [v for k, v in previous[-2:] if k == "string"]
                if strings:
                    out.extend(ShownCode(current_font, code) for code in strings[-1])
            elif kind == "word" and value == "TJ" and current_font:
                values = [v for k, v in previous[-2:] if k == "array_value"]
                if values:
                    for item_kind, item in values[-1]:
                        if item_kind == "string":
                            out.extend(ShownCode(current_font, code) for code in item)
            previous.append((kind, value))
            if len(previous) > 8: previous.pop(0)
    return out


def raw_chars(page: fitz.Page) -> list[RawChar]:
    out: list[RawChar] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for char in span["chars"]:
                    out.append(RawChar(char["c"], _normal_font_name(span["font"]), tuple(char["bbox"])))
    return out


def trace_chars(page: fitz.Page) -> list[TraceChar]:
    out: list[TraceChar] = []
    for span in page.get_texttrace():
        font = _normal_font_name(span["font"])
        for unicode_value, glyph_id, _origin, bbox in span["chars"]:
            out.append(TraceChar(font, glyph_id, unicode_value, tuple(bbox)))
    return out


def unique_compact_range(page_text: str, unit_text: str) -> tuple[int, int] | None:
    """Align only one admitted unit after whitespace-only normalization."""
    compact_page = "".join(char for char in page_text if not char.isspace())
    compact_unit = "".join(char for char in unit_text if not char.isspace())
    if not compact_unit:return None
    starts = [match.start() for match in re.finditer('(?='+re.escape(compact_unit)+')', compact_page)]
    if len(starts) != 1: return None
    compact_to_raw = [index for index, char in enumerate(page_text) if not char.isspace()]
    start, end = starts[0], starts[0] + len(compact_unit)
    return compact_to_raw[start], compact_to_raw[end - 1] + 1


def exact_font_pairs(codes: list[int], chars: list[RawChar]) -> list[tuple[RawChar, int]] | None:
    """Pair a font only when every non-space glyph has one stream byte."""
    visible = [item for item in chars if item.char != " "]
    return list(zip(visible, codes)) if len(visible) == len(codes) else None


def _resource_fonts(page: fitz.Page) -> dict[str, tuple[int, str]]:
    return {name: (xref, _normal_font_name(base)) for xref, _, _, base, name, *_ in page.get_fonts(full=True)}


def _differences(doc: fitz.Document, font_xref: int) -> dict[int, str]:
    obj = doc.xref_object(font_xref, compressed=False)
    ref = re.search(r"/Encoding\s+(\d+)\s+0\s+R", obj)
    if not ref: return {}
    encoding = doc.xref_object(int(ref.group(1)), compressed=False)
    match = re.search(r"/Differences\s*\[([^]]*)\]", encoding, re.S)
    if not match: return {}
    code: int | None = None; out: dict[int, str] = {}
    for token in re.findall(r"/[^\s\[\]]+|-?\d+", match.group(1)):
        if token.startswith("/"):
            if code is not None: out[code] = token[1:]; code += 1
        else: code = int(token)
    return out


def _index(data: bytes, offset: int) -> tuple[list[bytes], int]:
    count = int.from_bytes(data[offset:offset + 2], "big"); offset += 2
    if not count: return [], offset
    size = data[offset]; offset += 1
    offsets = [int.from_bytes(data[offset + size * i:offset + size * (i + 1)], "big") for i in range(count + 1)]
    start = offset + size * (count + 1)
    return [data[start + offsets[i] - 1:start + offsets[i + 1] - 1] for i in range(count)], start + offsets[-1] - 1


def _dict_numbers(data: bytes) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}; values: list[float] = []; i = 0
    while i < len(data):
        b = data[i]; i += 1
        if b <= 21:
            op = 1200 + data[i] if b == 12 else b
            if b == 12: i += 1
            out[op] = values; values = []; continue
        if 32 <= b <= 246: values.append(float(b - 139)); continue
        if 247 <= b <= 250: values.append(float((b - 247) * 256 + data[i] + 108)); i += 1; continue
        if 251 <= b <= 254: values.append(float(-(b - 251) * 256 - data[i] - 108)); i += 1; continue
        if b == 28: values.append(float(int.from_bytes(data[i:i + 2], "big", signed=True))); i += 2; continue
        if b == 29: values.append(float(int.from_bytes(data[i:i + 4], "big", signed=True))); i += 4; continue
        raise ValueError("unsupported CFF dict operand")
    return out


def _cff_charstrings(font_bytes: bytes) -> tuple[dict[int, bytes], dict[int, str]]:
    """Return charstrings and glyph names for compact Type1C fonts."""
    hdr = font_bytes[2]
    _, pos = _index(font_bytes, hdr)  # names
    top, pos = _index(font_bytes, pos)
    custom_strings, pos = _index(font_bytes, pos)
    _, _ = _index(font_bytes, pos)    # global subroutines
    fields = _dict_numbers(top[0])
    charstrings_offset = int(fields[17][0])
    charset_offset = int(fields.get(15, [0])[0])
    charstrings, _ = _index(font_bytes, charstrings_offset)
    sids = {0: 0}
    if charset_offset > 2 and charset_offset < len(font_bytes):
        fmt = font_bytes[charset_offset]; pos = charset_offset + 1
        if fmt == 0:
            for gid in range(1, len(charstrings)):
                sids[gid] = int.from_bytes(font_bytes[pos:pos + 2], "big"); pos += 2
        else:
            return {}, {}
    names = {gid: _SID_NAMES.get(sid, custom_strings[sid - 391].decode("latin1") if 391 <= sid < 391 + len(custom_strings) else "") for gid, sid in sids.items()}
    return {gid: value for gid, value in enumerate(charstrings)}, names


def _outline(charstring: bytes) -> list[list[tuple[float, float]]] | None:
    """Decode only move/line Type2 programs; reject curves, subroutines, and hints."""
    points: list[list[tuple[float, float]]] = []; args: list[float] = []; x = y = 0.0; i = 0
    def number(b: int) -> tuple[float, int]:
        nonlocal i
        if 32 <= b <= 246: return float(b - 139), 0
        if 247 <= b <= 250: value = float((b - 247) * 256 + charstring[i] + 108); i += 1; return value, 1
        if 251 <= b <= 254: value = float(-(b - 251) * 256 - charstring[i] - 108); i += 1; return value, 1
        if b == 28: value = float(int.from_bytes(charstring[i:i + 2], "big", signed=True)); i += 2; return value, 2
        return 0.0, -1
    while i < len(charstring):
        b = charstring[i]; i += 1
        if b >= 28:
            value, consumed = number(b)
            if consumed < 0: return None
            args.append(value); continue
        if b in (1, 3, 18, 19, 20, 23, 10, 29, 30, 31, 8, 24, 25, 26, 27): return None
        if b == 14: args = []; continue
        if b == 21 and len(args) >= 2:
            x += args[-2]; y += args[-1]; points.append([(x, y)]); args = []; continue
        if b == 22 and args:
            x += args[-1]; points.append([(x, y)]); args = []; continue
        if b == 4 and args:
            y += args[-1]; points.append([(x, y)]); args = []; continue
        if b == 5 and points and len(args) % 2 == 0:
            for dx, dy in zip(args[::2], args[1::2]): x += dx; y += dy; points[-1].append((x, y))
            args = []; continue
        if b == 6 and points:
            horizontal = True
            for value in args:
                if horizontal: x += value
                else: y += value
                points[-1].append((x, y)); horizontal = not horizontal
            args = []; continue
        if b == 7 and points:
            vertical = True
            for value in args:
                if vertical: y += value
                else: x += value
                points[-1].append((x, y)); vertical = not vertical
            args = []; continue
        return None
    return points


def _geometry_operator(charstring: bytes) -> str | None:
    paths = _outline(charstring)
    if not paths: return None
    rects: list[tuple[float, float, float, float]] = []
    for path in paths:
        if len(path) != 4: break
        xs, ys = {p[0] for p in path}, {p[1] for p in path}
        if len(xs) != 2 or len(ys) != 2: break
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    if len(paths) == len(rects) == 2:
        a, b = sorted(rects, key=lambda r: r[1])
        same_x = abs(a[0] - b[0]) < 0.01 and abs(a[2] - b[2]) < 0.01
        similar_height = abs((a[3] - a[1]) - (b[3] - b[1])) < 0.01
        if same_x and similar_height and (a[2] - a[0]) > 5 * (a[3] - a[1]) and a[3] < b[1]: return "="
    if len(paths) == 1 and len(paths[0]) in (6, 7):
        p = paths[0]; xs = [point[0] for point in p]; ys = [point[1] for point in p]
        left, right = min(xs), max(xs)
        # A comparison wedge has two thin vertical ends on one side and two
        # interior points near the other apex.  This excludes ordinary commas
        # and diagonal strokes even when their font metadata is misleading.
        if not p[0][1]<p[1][1]<p[2][1]<p[3][1]<p[4][1]:return None
        if len(p)==7 and not (p[5][0]==p[6][0] and p[6][1]<p[2][1]<p[5][1]):return None
        if len(p)==6 and not p[1][1]<p[5][1]<p[3][1]:return None
        end_left = p[5][0] < p[2][0] < right
        end_right = p[5][0] > p[2][0] > left
        if p[0][0] == p[1][0] == p[3][0] == p[4][0] == right and p[2][0] < right and end_left and (max(ys) - min(ys)) > 2 * (right - left) / 5:
            return "<"
        if p[0][0] == p[1][0] == p[3][0] == p[4][0] == left and p[2][0] > left and end_right and (max(ys) - min(ys)) > 2 * (right - left) / 5:
            return ">"
    return None


def _geometry_minus(charstring: bytes) -> str | None:
    """Recognize one positive-area horizontal bar, and nothing broader.

    This is intentionally separate from the relation-operator geometry.  A
    bar by itself is only projected as a numeric sign after the caller proves
    it occupies the first character of a numeric token.
    """
    paths = _outline(charstring)
    if not paths or len(paths) != 1 or len(paths[0]) != 4:
        return None
    path = paths[0]
    xs, ys = {point[0] for point in path}, {point[1] for point in path}
    if len(xs) != 2 or len(ys) != 2 or len(set(path)) != 4:
        return None
    if any((a[0] == b[0]) == (a[1] == b[1]) for a,b in zip(path,path[1:]+path[:1])):
        return None
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    if width <= 0 or height <= 0 or width < 5 * height:
        return None
    return "-"


def _numeric_sign_position(text: str, offset: int) -> bool:
    """Accept only a bar occupying the first character of a numeric token."""
    if not 0 <= offset < len(text) or not text[offset].isdigit():
        return False
    if offset and (text[offset - 1].isalnum() or text[offset - 1] in "_.+-−–"):
        return False
    suffix = text[offset + 1:]
    return bool(re.match(r"(?:\d|\.\d)", suffix))


def _operator_from_glyph(doc: fitz.Document, font_xref: int, glyph_id: int) -> tuple[str, str] | None:
    data = doc.extract_font(font_xref)[3]
    try:
        charstrings, names = _cff_charstrings(data)
    except (IndexError, ValueError):
        return None
    name = names.get(glyph_id, "")
    if name in _RELATION_GLYPHS and name in set(_differences(doc, font_xref).values()):
        return _RELATION_GLYPHS[name], "encoding_differences"
    if glyph_id in charstrings:
        operator = _geometry_operator(charstrings[glyph_id])
        if operator: return operator, "cff_geometry"
        minus = _geometry_minus(charstrings[glyph_id])
        if minus: return minus, "cff_geometry"
    return None


def candidates(pdf_path: Path, units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return source-aligned relation-operator candidates, never corrected text."""
    doc = fitz.open(pdf_path); output: list[dict[str, Any]] = []
    try:
        by_page: dict[int, list[dict[str, Any]]] = {}
        for unit in units: by_page.setdefault(int(unit["page"]), []).append(unit)
        for page_no, page_units in by_page.items():
            page = doc[page_no - 1]; chars = raw_chars(page); resources = _resource_fonts(page)
            by_font: dict[str, list[tuple[int, str]]] = {}
            for _resource, (xref, raw_name) in resources.items(): by_font.setdefault(raw_name, []).append((xref, _resource))
            aliased_fonts = {name for name, values in by_font.items() if len(values) != 1}
            raw_by_signature: dict[tuple[str, tuple[float, float, float, float]], list[RawChar]] = {}
            for item in chars: raw_by_signature.setdefault((item.font, tuple(round(v, 3) for v in item.bbox)), []).append(item)
            traced_matches: dict[int, list[TraceChar]] = {}
            for trace in trace_chars(page):
                matched = raw_by_signature.get((trace.font, tuple(round(v, 3) for v in trace.bbox)), [])
                if len(matched) == 1:traced_matches.setdefault(id(matched[0]),[]).append(trace)
            traced_raw={key:values[0] for key,values in traced_matches.items() if len(values)==1}
            page_text = "".join(item.char for item in chars)
            index_by_object = {id(item): index for index, item in enumerate(chars)}
            for unit in page_units:
                # The admitted unit and MuPDF character stream can differ only
                # in inferred line/word spacing.  Preserve every non-space
                # character and require a unique compact match; anything else
                # is an alignment refusal, not an opportunity to guess.
                alignment = unique_compact_range(page_text, unit["text"])
                if alignment is None: continue
                raw_start, raw_end = alignment
                compact_start = sum(not value.isspace() for value in page_text[:raw_start])
                unit_compact_offsets = [offset for offset, value in enumerate(unit["text"]) if not value.isspace()]
                for char in chars[raw_start:raw_end]:
                    index = index_by_object[id(char)]; trace = traced_raw.get(id(char))
                    if trace is None or trace.font in aliased_fonts or trace.font not in by_font: continue
                    font_xref = by_font[trace.font][0][0]
                    proposal = _operator_from_glyph(doc, font_xref, trace.glyph_id)
                    if not proposal: continue
                    operator, basis = proposal
                    if char.char == operator: continue
                    compact_offset = sum(not value.isspace() for value in page_text[:index]) - compact_start
                    if not (0 <= compact_offset < len(unit_compact_offsets)): continue
                    unit_offset = unit_compact_offsets[compact_offset]
                    if unit["text"][unit_offset] != char.char: continue
                    if operator == "-" and not _numeric_sign_position(unit["text"], unit_offset):
                        continue
                    neighbors = []
                    for step in (-1, 1):
                        probe = index + step
                        while raw_start <= probe < raw_end and chars[probe].char.isspace(): probe += step
                        neighbor = traced_raw.get(id(chars[probe])) if raw_start <= probe < raw_end else None
                        neighbors.append(neighbor is not None and neighbor.unicode != 0xFFFD and chr(neighbor.unicode) == chars[probe].char)
                    if not all(neighbors): continue
                    output.append({
                            "unit_id": unit["unit_id"], "source_locator": unit["source_locator"], "page": page_no,
                            "raw_character": char.char, "proposed_operator": operator, "basis": basis,
                            "font_xref": font_xref, "font_sha256": hashlib.sha256(doc.extract_font(font_xref)[3]).hexdigest(),
                            "glyph_id": trace.glyph_id, "bbox": [round(value, 4) for value in char.bbox],
                            "unit_char_offset": unit_offset, "unit_char_end": unit_offset + 1,
                            "raw_text_preserved": True,
                        })
    finally:
        doc.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unit", action="append")
    parser.add_argument("--case", action="append", help="source;pdf;comma-separated unit ids (repeatable)")
    args = parser.parse_args()
    cases: list[tuple[Path, Path, set[str]]] = []
    if args.case:
        for value in args.case:
            source_text, pdf_text, units_text = value.split(";", 2)
            cases.append((Path(source_text), Path(pdf_text), set(units_text.split(","))))
    elif args.pdf and args.source and args.unit:
        cases.append((args.source, args.pdf, set(args.unit)))
    else:
        raise SystemExit("supply --pdf/--source/--unit or one or more --case values")
    summaries, all_candidates = [], []
    for source_path, pdf_path, wanted in cases:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        selected = [unit for unit in source["units"] if unit["unit_id"] in wanted]
        if {unit["unit_id"] for unit in selected} != wanted: raise SystemExit("requested unit is absent")
        found = candidates(pdf_path, selected); all_candidates.extend(found)
        summaries.append({"pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "units": sorted(wanted), "candidate_count": len(found)})
    report = {"cases": summaries, "candidates": all_candidates}
    Workspace().write_canonical_json(args.out, report)


if __name__ == "__main__":
    main()
