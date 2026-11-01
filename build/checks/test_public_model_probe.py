from __future__ import annotations

import json
import time
from urllib.error import URLError

import pytest

from mosaic.public_model_probe import (ENDPOINT, MAX_OUTPUT_TOKENS, PublicModelProbeRefusal,
    SERVED_MODEL, probe_loopback_model, verify_retained_model_probe)
from mosaic.workspace import Workspace


class _Response:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def read(self, limit: int) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _body(**changes):
    result = {
        "model": SERVED_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "READY"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 1},
    }
    result.update(changes)
    return json.dumps(result).encode()


def test_probe_retains_one_explicit_mocked_transport_outcome(tmp_path):
    received = []

    def fake(request, *, timeout):
        received.append((request.full_url, json.loads(request.data), timeout))
        return _Response(_body())

    result = probe_loopback_model(tmp_path / "probe", workspace=Workspace(repo_root=tmp_path),
                                  deadline_epoch=time.time() + 90, test_transport=True, opener=fake)
    assert received[0][0] == ENDPOINT
    assert received[0][1]["model"] == SERVED_MODEL
    assert received[0][1]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert received[0][1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["outcome"]["status"] == "completed"
    assert result["outcome"]["transport_evidence"] == "mocked_test_transport"
    assert result["outcome"]["real_model_inference_run"] is False
    output = tmp_path / "probe"
    assert {p.name for p in output.iterdir()} == {"intent.json", "request.json", "send-intent.json", "response.json", "outcome.json"}
    proof = verify_retained_model_probe(output, require_observed=False)
    assert proof["transport_evidence"] == "mocked_test_transport"
    with pytest.raises(PublicModelProbeRefusal, match="observation_required"):
        verify_retained_model_probe(output)
    with pytest.raises(PublicModelProbeRefusal, match="output_exists"):
        probe_loopback_model(output, workspace=Workspace(repo_root=tmp_path), deadline_epoch=time.time() + 90,
                             test_transport=True, opener=fake)


@pytest.mark.parametrize("payload, reason", [
    (_body(model="wrong"), "response_model_mismatch"),
    (_body(usage={"prompt_tokens": 8, "completion_tokens": 33}), "response_usage_invalid"),
    (_body(choices=[{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]), "response_message_invalid"),
])
def test_probe_retains_invalid_response_without_model_claim(tmp_path, payload, reason):
    result = probe_loopback_model(tmp_path / reason, workspace=Workspace(repo_root=tmp_path),
                                  deadline_epoch=time.time() + 90, test_transport=True,
                                  opener=lambda request, *, timeout: _Response(payload))
    assert result["outcome"]["status"] == "failed"
    assert result["outcome"]["reason"] == "public_model_probe_" + reason
    assert result["outcome"]["real_model_inference_run"] is False
    assert (tmp_path / reason / "response.json").is_file()


def test_probe_retains_send_failure_and_rejects_test_opener_without_label(tmp_path):
    def unavailable(request, *, timeout):
        raise URLError("fictional unavailable")

    with pytest.raises(PublicModelProbeRefusal, match="test_transport_required"):
        probe_loopback_model(tmp_path / "unlabeled", workspace=Workspace(repo_root=tmp_path),
                             deadline_epoch=time.time() + 90, opener=unavailable)
    result = probe_loopback_model(tmp_path / "failed", workspace=Workspace(repo_root=tmp_path),
                                  deadline_epoch=time.time() + 90, test_transport=True, opener=unavailable)
    assert result["outcome"]["reason"] == "public_model_probe_send_failed"
    assert (tmp_path / "failed" / "send-error.json").is_file()


def test_probe_refuses_expired_deadline_without_writes(tmp_path):
    with pytest.raises(PublicModelProbeRefusal, match="deadline_expired"):
        probe_loopback_model(tmp_path / "expired", workspace=Workspace(repo_root=tmp_path),
                             deadline_epoch=time.time() - 1, test_transport=True,
                             opener=lambda request, *, timeout: _Response(_body()))
    assert not (tmp_path / "expired").exists()


def test_probe_stops_if_one_send_returns_after_deadline(tmp_path):
    ticks = iter((100.0, 100.0, 100.0, 102.0, 102.0))
    result = probe_loopback_model(tmp_path / "late", workspace=Workspace(repo_root=tmp_path),
                                  deadline_epoch=101.0, test_transport=True,
                                  opener=lambda request, *, timeout: _Response(_body()), now=lambda: next(ticks))
    assert result["outcome"]["reason"] == "public_model_probe_deadline_elapsed"
    assert result["outcome"]["real_model_inference_run"] is False


@pytest.mark.parametrize("tamper", ["request", "response", "timestamp", "finish", "boolean_usage"])
def test_retained_probe_rejects_unbound_or_nonobserved_evidence(tmp_path, tamper):
    output = tmp_path / tamper
    probe_loopback_model(output, workspace=Workspace(repo_root=tmp_path), deadline_epoch=time.time() + 90,
                         test_transport=True, opener=lambda request, *, timeout: _Response(_body()))
    if tamper == "request":
        (output / "request.json").write_bytes(b"{}")
    elif tamper == "response":
        (output / "response.json").write_bytes(_body(model="wrong"))
    elif tamper == "timestamp":
        value = json.loads((output / "outcome.json").read_bytes()); value["finished_epoch"] = float("inf")
        (output / "outcome.json").write_text(json.dumps(value))
    elif tamper == "finish":
        value = json.loads((output / "response.json").read_bytes()); value["choices"][0]["finish_reason"] = "length"
        (output / "response.json").write_text(json.dumps(value))
    else:
        value = json.loads((output / "response.json").read_bytes()); value["usage"]["completion_tokens"] = True
        (output / "response.json").write_text(json.dumps(value))
    with pytest.raises(PublicModelProbeRefusal):
        verify_retained_model_probe(output, require_observed=False)
