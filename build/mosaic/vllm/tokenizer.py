"""Optional, pinned local sizing; never downloads assets or calls a provider."""
from __future__ import annotations

import hashlib
from importlib import metadata
from typing import Callable

from mosaic.workspace import Workspace


class TokenizerRefused(ValueError):
    """The explicitly selected local tokenizer cannot be verified."""


def local_tokenizer(spec: dict) -> Callable[[str], int] | None:
    """Load verified bytes once; return None only when no local mode is selected.

    Model/revision, tokenizer bytes, runtime and special-token policy are recipe
    inputs. A bad or missing local asset never falls back to a different method.
    The backend still performs its document-scope check before calling this.
    """
    selection = spec.get("local_tokenizer")
    if selection is None:
        return None
    if (selection.get("model") != spec.get("model") or selection.get("revision") != spec.get("revision")
            or type(selection.get("add_special_tokens")) is not bool):
        raise TokenizerRefused("local_tokenizer_identity_mismatch")
    try:
        installed = metadata.version("tokenizers")
    except metadata.PackageNotFoundError as exc:
        raise TokenizerRefused("local_tokenizer_runtime_missing") from exc
    if installed != selection["runtime_version"]:
        raise TokenizerRefused("local_tokenizer_runtime_mismatch")
    from tokenizers import Tokenizer
    path = Workspace().check(selection["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != selection["sha256"]:
        raise TokenizerRefused("local_tokenizer_asset_mismatch")
    tokenizer = Tokenizer.from_str(data.decode("utf-8"))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise TokenizerRefused("local_tokenizer_truncation_or_padding")
    add_special_tokens = selection["add_special_tokens"]

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)

    return count
