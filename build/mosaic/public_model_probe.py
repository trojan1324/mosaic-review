"""One bounded fictional loopback check before public corpus execution.

The probe isolates model readiness from the extraction route.  It never starts a
server, downloads a model, reads a PDF, or retries a send.  A successful probe
only establishes an observed response from the configured loopback endpoint; it
does not establish scientific accuracy or complete MOSAIC execution.
"""
from __future__ import annotations

from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from mosaic.workspace import Workspace


VERSION = "mosaic-public-model-probe/v1"
ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
SERVED_MODEL = "screen-candidate"
MAX_OUTPUT_TOKENS = 32
MAX_SECONDS = 60
MAX_RESPONSE_BYTES = 1_048_576
PROMPT = "This is a fictional operational readiness check. Reply with READY only."


class PublicModelProbeRefusal(ValueError):
    """The probe cannot safely make its one permitted send."""


def _need(value: bool, code: str) -> None:
    if not value:
        raise PublicModelProbeRefusal(code)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PublicModelProbeRefusal("public_model_probe_redirect_refused")


def _payload() -> dict[str, Any]:
    return {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _new_output(output: Path, workspace: Workspace) -> Path:
    output = workspace.check(output)
    _need(not output.exists(), "public_model_probe_output_exists")
    return workspace.mkdir(output)


def _write_outcome(workspace: Workspace, output: Path, **values: Any) -> dict[str, Any]:
    outcome = {"version": VERSION, **values}
    workspace.write_canonical_json(output / "outcome.json", outcome)
    return outcome


def _validate_response(value: object) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "public_model_probe_response_invalid")
    _need(value.get("model") == SERVED_MODEL, "public_model_probe_response_model_mismatch")
    choices = value.get("choices")
    _need(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], Mapping),
          "public_model_probe_response_choices_invalid")
    choice = choices[0]
    message = choice.get("message")
    _need(isinstance(message, Mapping) and message.get("role") == "assistant"
          and isinstance(message.get("content"), str) and bool(message["content"].strip()),
          "public_model_probe_response_message_invalid")
    finish = choice.get("finish_reason")
    _need(finish == "stop",
          "public_model_probe_response_finish_invalid")
    usage = value.get("usage")
    _need(isinstance(usage, Mapping), "public_model_probe_response_usage_missing")
    completion = usage.get("completion_tokens")
    prompt = usage.get("prompt_tokens")
    _need(type(completion) is int and 0 <= completion <= MAX_OUTPUT_TOKENS
          and type(prompt) is int and prompt >= 0,
          "public_model_probe_response_usage_invalid")
    return {
        "finish_reason": finish,
        "completion_tokens": completion,
        "prompt_tokens": prompt,
        "assistant_content_sha256": sha256(message["content"].encode("utf-8")).hexdigest(),
    }


def probe_loopback_model(
    output: Path,
    *,
    workspace: Workspace,
    deadline_epoch: float,
    test_transport: bool = False,
    opener: Callable[..., Any] | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Send exactly one fictional readiness request and retain a terminal record.

    ``opener`` exists only for tests.  Passing it requires ``test_transport``;
    that outcome is explicitly not a model-inference observation.
    """
    _need(type(deadline_epoch) in (int, float) and math.isfinite(float(deadline_epoch)),
          "public_model_probe_deadline_invalid")
    remaining = float(deadline_epoch) - now()
    _need(remaining > 0, "public_model_probe_deadline_expired")
    _need(opener is None or test_transport, "public_model_probe_test_transport_required")
    output = _new_output(output, workspace)
    request_body = json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    evidence = "mocked_test_transport" if test_transport else "observed_local_loopback"
    created_epoch = now()
    workspace.write_canonical_json(output / "intent.json", {
        "version": VERSION,
        "endpoint": ENDPOINT,
        "model": SERVED_MODEL,
        "request_cap": 1,
        "output_token_cap": MAX_OUTPUT_TOKENS,
        "deadline_epoch": deadline_epoch,
        "created_epoch": created_epoch,
        "request_sha256": sha256(request_body).hexdigest(),
        "fictional_operational_prompt": True,
        "transport_evidence": evidence,
        "scientifically_accepted": False,
    })
    workspace.write_once(output / "request.json", request_body)
    request = Request(ENDPOINT, data=request_body, method="POST", headers={"Content-Type": "application/json"})
    timeout = min(MAX_SECONDS, remaining)
    active_opener = opener or build_opener(ProxyHandler({}), _NoRedirect()).open
    sent_epoch = now()
    workspace.write_canonical_json(output / "send-intent.json", {
        "version": VERSION, "sent_epoch": sent_epoch,
        "request_sha256": sha256(request_body).hexdigest(), "endpoint": ENDPOINT,
        "transport_evidence": evidence,
    })
    def terminal(**values: Any) -> dict[str, Any]:
        return _write_outcome(workspace, output, finished_epoch=now(), **values)
    try:
        with active_opener(request, timeout=timeout) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            _need(len(body) <= MAX_RESPONSE_BYTES, "public_model_probe_response_too_large")
            _need(getattr(response, "status", None) == 200, "public_model_probe_http_status_invalid")
    except PublicModelProbeRefusal as exc:
        return {"outcome_path": str(output / "outcome.json"), "outcome": terminal(
            status="failed", send_attempted=True, transport_evidence=evidence,
            real_model_inference_run=False, reason=str(exc), scientifically_accepted=False)}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)[:512]}
        workspace.write_canonical_json(output / "send-error.json", error)
        return {"outcome_path": str(output / "outcome.json"), "outcome": terminal(
            status="failed", send_attempted=True, transport_evidence=evidence,
            real_model_inference_run=False, reason="public_model_probe_send_failed",
            error_sha256=sha256(json.dumps(error, sort_keys=True).encode("utf-8")).hexdigest(),
            scientifically_accepted=False)}
    workspace.write_once(output / "response.json", body)
    if now() > deadline_epoch:
        return {"outcome_path": str(output / "outcome.json"), "outcome": terminal(
            status="failed", send_attempted=True, transport_evidence=evidence,
            real_model_inference_run=False, reason="public_model_probe_deadline_elapsed",
            response_sha256=sha256(body).hexdigest(), scientifically_accepted=False)}
    try:
        details = _validate_response(json.loads(body))
    except (PublicModelProbeRefusal, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"outcome_path": str(output / "outcome.json"), "outcome": terminal(
            status="failed", send_attempted=True, transport_evidence=evidence,
            real_model_inference_run=False, reason=str(exc), response_sha256=sha256(body).hexdigest(),
            scientifically_accepted=False)}
    outcome = terminal(
        status="completed", send_attempted=True, transport_evidence=evidence,
        real_model_inference_run=not test_transport, response_sha256=sha256(body).hexdigest(),
        response=details, scientifically_accepted=False,
        limitation="One fictional readiness response; not extraction, scientific qualification, or full route success.")
    return {"outcome_path": str(output / "outcome.json"), "outcome": outcome}


def verify_retained_model_probe(root: Path, *, require_observed: bool = True) -> dict[str, Any]:
    """Recompute a completed probe from its immutable request/response evidence."""
    def obj(name: str) -> dict[str, Any]:
        try:
            value = json.loads((root / name).read_bytes())
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            raise PublicModelProbeRefusal("public_model_probe_retained_" + name.replace(".json", "") + "_invalid") from exc
        _need(isinstance(value, dict), "public_model_probe_retained_object_invalid")
        return value
    intent, sent, outcome = obj("intent.json"), obj("send-intent.json"), obj("outcome.json")
    request = (root / "request.json").read_bytes(); response = (root / "response.json").read_bytes()
    expected = json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    _need(request == expected, "public_model_probe_retained_request_invalid")
    request_sha, response_sha = sha256(request).hexdigest(), sha256(response).hexdigest()
    for value in (intent, sent, outcome):
        _need(value.get("version") == VERSION, "public_model_probe_retained_version_invalid")
    _need(intent.get("endpoint") == ENDPOINT and intent.get("model") == SERVED_MODEL
          and intent.get("request_cap") == 1 and intent.get("output_token_cap") == MAX_OUTPUT_TOKENS
          and intent.get("fictional_operational_prompt") is True and intent.get("scientifically_accepted") is False
          and intent.get("request_sha256") == request_sha, "public_model_probe_retained_intent_invalid")
    deadline = intent.get("deadline_epoch"); created = intent.get("created_epoch"); sent_epoch = sent.get("sent_epoch"); finished = outcome.get("finished_epoch")
    _need(all(type(value) in (int, float) and math.isfinite(float(value)) for value in (deadline, created, sent_epoch, finished)),
          "public_model_probe_retained_timestamps_invalid")
    _need(created <= sent_epoch <= finished <= deadline and finished - sent_epoch <= MAX_SECONDS,
          "public_model_probe_retained_deadline_invalid")
    _need(sent.get("request_sha256") == request_sha and sent.get("endpoint") == ENDPOINT,
          "public_model_probe_retained_send_intent_invalid")
    details = _validate_response(json.loads(response))
    _need(outcome.get("status") == "completed" and outcome.get("send_attempted") is True
          and outcome.get("response_sha256") == response_sha and outcome.get("response") == details
          and outcome.get("scientifically_accepted") is False, "public_model_probe_retained_outcome_invalid")
    evidence = intent.get("transport_evidence")
    _need(sent.get("transport_evidence") == evidence and outcome.get("transport_evidence") == evidence,
          "public_model_probe_retained_transport_invalid")
    if require_observed:
        _need(evidence == "observed_local_loopback" and outcome.get("real_model_inference_run") is True,
              "public_model_probe_retained_observation_required")
    else:
        _need(evidence in {"observed_local_loopback", "mocked_test_transport"},
              "public_model_probe_retained_transport_invalid")
    return {"version": VERSION, "request_sha256": request_sha, "response_sha256": response_sha,
            "completion_tokens": details["completion_tokens"], "finished_epoch": finished,
            "transport_evidence": evidence, "real_model_inference_run": outcome.get("real_model_inference_run") is True,
            "scientifically_accepted": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    args = parser.parse_args(argv)
    _need(args.run == "run", "public_model_probe_command_invalid")
    result = probe_loopback_model(Path(args.output).absolute(), workspace=Workspace(repo_root=Path.cwd()),
                                  deadline_epoch=args.deadline_epoch)
    print(json.dumps(result["outcome"], sort_keys=True))
    return 0 if result["outcome"].get("status") == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PublicModelProbeRefusal, OSError, ValueError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}))
        raise SystemExit(2)
