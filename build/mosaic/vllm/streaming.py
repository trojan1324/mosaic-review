"""Strict assembly of OpenAI-compatible chat-completion server-sent events.

The wire bytes remain the primary record.  This module accepts a completed
SSE response only when it has a terminal ``[DONE]`` marker, a final choice-0
finish reason, and one valid usage record.  Anything else is deliberately an
unknown outcome: the request may have run, but its response is not safe to
adapt or retry.
"""

from __future__ import annotations

import json
from typing import Any, Iterator


class StreamDecodeError(ValueError):
    """A received SSE body cannot safely become a completed chat response."""


def _events(raw: bytes) -> Iterator[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StreamDecodeError("stream was not UTF-8") from exc
    # CRLF and LF are both valid SSE line endings.  Split only on blank lines;
    # data lines in one event are joined as SSE specifies.
    for block in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        lines = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                lines.append(line[5:].lstrip(" "))
            elif line.startswith(("event:", "id:", "retry:")):
                continue
            else:
                raise StreamDecodeError("malformed SSE field")
        if lines:
            yield "\n".join(lines)


def _valid_usage(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    prompt = value.get("prompt_tokens", value.get("input_tokens"))
    completion = value.get("completion_tokens", value.get("output_tokens"))
    return (isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0
            and isinstance(completion, int) and not isinstance(completion, bool) and completion >= 0)


def decode_chat_stream(raw: bytes) -> dict[str, Any]:
    """Decode one complete SSE chat response into the normal choice-0 shape.

    The returned fields are intentionally limited to the downstream contract:
    model, choice-0 content/reasoning content, finish reason, and exact final
    usage.  A server error event, a missing terminal marker, or incomplete
    choice/usage data raises :class:`StreamDecodeError`.
    """
    content: list[str] = []
    reasoning: list[str] = []
    model: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    done = False
    saw_event = False
    for payload in _events(raw):
        if done:
            raise StreamDecodeError("data after DONE")
        if payload == "[DONE]":
            done = True
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise StreamDecodeError("malformed JSON data event") from exc
        if not isinstance(event, dict) or "error" in event:
            raise StreamDecodeError("stream error event")
        saw_event = True
        event_model = event.get("model")
        if event_model is not None:
            if not isinstance(event_model, str):
                raise StreamDecodeError("invalid model")
            if model is not None and model != event_model:
                raise StreamDecodeError("model changed during stream")
            model = event_model
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            raise StreamDecodeError("invalid choices")
        if choices and finish_reason is not None:
            raise StreamDecodeError("choice after final finish reason")
        if choices and usage is not None:
            raise StreamDecodeError("choice after final usage")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index") != 0:
                raise StreamDecodeError("only choice zero is supported")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise StreamDecodeError("invalid delta")
            # Newer servers use `reasoning`; retain the established internal
            # field without silently discarding it or double-counting aliases.
            if delta.get("reasoning") is not None:
                if delta.get("reasoning_content") is not None and delta["reasoning_content"] != delta["reasoning"]:
                    raise StreamDecodeError("conflicting reasoning aliases")
                delta = {**delta, "reasoning_content": delta["reasoning"]}
            for field, target in (("content", content), ("reasoning_content", reasoning)):
                value = delta.get(field)
                if value is not None:
                    if not isinstance(value, str):
                        raise StreamDecodeError(f"invalid {field}")
                    target.append(value)
            current_finish = choice.get("finish_reason")
            if current_finish is not None:
                if not isinstance(current_finish, str) or finish_reason is not None:
                    raise StreamDecodeError("invalid final finish reason")
                finish_reason = current_finish
        if "usage" in event and event["usage"] is not None:
            if usage is not None:
                raise StreamDecodeError("multiple usage records")
            # Usage may share the finish event, but cannot precede it.
            if finish_reason is None or not _valid_usage(event["usage"]):
                raise StreamDecodeError("invalid final usage")
            usage = event["usage"]
    if not saw_event or not done or model is None or finish_reason is None or usage is None:
        raise StreamDecodeError("incomplete stream")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    return {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage}
