from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time

import pymupdf
import pytest

from mosaic.public_capture import capture_public
from mosaic.public_execution import PublicExecutionRefusal, run_local_program
from mosaic.public_native_runtime import load_capture_envelope, prepare_local_program
from mosaic.public_runtime_manifest import VERSION as MANIFEST_VERSION
from mosaic.public_workflow import source_handoff
from mosaic.workspace import Workspace

pytestmark = pytest.mark.evidence_mock

def _pdf():
    doc = pymupdf.open(); page = doc.new_page()
    for y, text in ((72, "Study 1"), (110, "Abstract"), (145, "Fictional source prose."),
                    (200, "Methods"), (235, "Fictional participants completed a scale."),
                    (290, "Results"), (325, "A fictional result is reported.")):
        page.insert_text((72, y), text)
    raw = doc.tobytes(); doc.close(); return raw


def _prepared(tmp_path):
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    raw = _pdf(); sha = sha256(raw).hexdigest(); sources = ws.mkdir(tmp_path / "sources")
    ws.write_once(sources / "fictional.pdf", raw)
    row = {"paper_id": "public-" + sha, "source_pdf_sha256": sha, "byte_size": len(raw), "locators": ["fictional.pdf"],
           "source_status": "public_unreviewed_source", "permission": "user_declared_permitted", "scientifically_accepted": False}
    capture = tmp_path / "capture"; capture_public(row, "fictional.pdf", sources, capture, title="Unreviewed fictional source", workspace=ws)
    weights = ws.mkdir(tmp_path / "weights")
    tokenizer = (b'{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,'
                 b'"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}')
    model = b"fictional weights"; ws.write_once(weights / "tokenizer.json", tokenizer); ws.write_once(weights / "weights/model.safetensors", model)
    def f(path, data): return {"path": path, "size": len(data), "sha256": sha256(data).hexdigest()}
    t = f("tokenizer.json", tokenizer)
    manifest = {"version": MANIFEST_VERSION, "model": {"id":"fictional/local","revision":"r1","files":[t, f("weights/model.safetensors", model)],"tokenizer":{"path":t["path"],"sha256":t["sha256"]},"tokenizer_runtime_version":metadata.version("tokenizers"),"add_special_tokens":True}, "runtime":{"platform":"linux","container_digest":"a"*64,"wheels":[{"name":"xgrammar","sha256":"b"*64}]}, "hardware":{"declared":{"accelerator":"fictional"},"observed_linux_gpu_smoke_receipt":None}}
    profile = {"version":"mosaic-public-native-driver-profile/v1","model":{"id":"fictional/local","revision":"r1","served_model_name":"fictional-local"},"source":{"permission_declaration":"user_declared_permitted"},"driver":{"candidate":"fictional-execution","run_id":"fictional-execution-v1","max_measure_stages_per_paper":1,"max_request_seconds":5,"request_cap":8,"output_token_cap":49152,"wall_clock_cap_seconds":40,"article_request_cap":8}}
    prepared = tmp_path / "prepared"
    prepare_local_program(load_capture_envelope(capture, workspace=ws), manifest, weights, profile, "http://127.0.0.1:8000/v1", prepared, workspace=ws)
    return ws, capture, prepared


def _mock_observation():
    return {"kind":"mocked_loopback_protocol", "endpoint":"http://127.0.0.1:8000/v1", "model_execution_observed":False,
            "inventory":{"transport":"pytest fictional loopback", "network":"local only"}}


class _Loopback(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        body = b'{"data":[{"id":"fictional-local"}]}' if self.path == "/v1/models" else b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"])); wire = json.loads(raw)
        schema = json.loads(wire["messages"][0]["content"].split("\nRESPONSE_SCHEMA:\n", 1)[1]); fields = schema["properties"]
        if "findings" in fields: answer = {"article_kind":"original_empirical","findings":[],"qualifications":[]}
        elif "studies" in fields: answer = {"article_kind":"original_empirical","studies":[{"identifier_sentence_ids":["c001"],"design":"uncertain","design_sentence_ids":[],"population_sentence_ids":[],"sample_size_sentence_ids":[],"measures":[]}],"analyses":[]}
        else: answer = {"facts":[]}
        event = {"model":wire["model"],"choices":[{"index":0,"delta":{"content":json.dumps(answer)},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":5}}
        wire_out = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers(); self.wfile.write(wire_out)


class _WrongModelLoopback(_Loopback):
    def do_GET(self):
        body = b'{"data":[{"id":"wrong-model"}]}' if self.path == "/v1/models" else b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)


class _IncompleteLoopback(_Loopback):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"])); wire = json.loads(raw)
        partial = {"model":wire["model"],"choices":[{"index":0,"delta":{"content":"{"},"finish_reason":None}]}
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(("data: " + json.dumps(partial) + "\n\n").encode())


class _RedirectLoopback(_Loopback):
    def do_GET(self):
        self.send_response(302); self.send_header("Location", "http://example.invalid/"); self.end_headers()


def _server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 8000), handler); thread = Thread(target=server.serve_forever); thread.start()
    return server, thread


def test_native_driver_runs_against_mocked_loopback_protocol(tmp_path):
    ws, capture, prepared = _prepared(tmp_path)
    server, thread = _server(_Loopback)
    try:
        result = run_local_program(prepared, capture, tmp_path / "executed", workspace=ws, deadline_epoch=time.time()+100,
                                   runtime_observation=_mock_observation())
    finally:
        server.shutdown(); thread.join(); server.server_close()
    assert result["receipt"]["status"] == "terminal"
    assert result["receipt"]["model_execution"] == "mocked_loopback_protocol"
    assert result["receipt"]["real_model_inference_run"] is False
    assert Path(result["snapshot_path"]).is_file()
    assert Path(result["report_path"]).is_file()
    handoff = source_handoff(capture, Path(result["report_path"]), Path(result["snapshot_path"]), workspace=ws)
    assert handoff["snapshot"]["report_path"] == Path(result["report_path"])
    reused = run_local_program(prepared, capture, tmp_path / "executed", workspace=ws, deadline_epoch=time.time()+100,
                               runtime_observation=_mock_observation())
    assert reused["receipt_path"] == result["receipt_path"]
    # The immutable preparation binding forbids a second send under a new output.
    with pytest.raises(PublicExecutionRefusal, match="already_bound"):
        run_local_program(prepared, capture, tmp_path / "other", workspace=ws, deadline_epoch=time.time()+100,
                          runtime_observation=_mock_observation())


def test_execution_refuses_changed_capture_and_unverified_output(tmp_path):
    ws, capture, prepared = _prepared(tmp_path)
    with pytest.raises(PublicExecutionRefusal, match="endpoint_changed"):
        run_local_program(prepared, capture, tmp_path / "out", workspace=ws, deadline_epoch=time.time()+70,
                          runtime_observation={**_mock_observation(), "endpoint":"http://127.0.0.1:9/v1"})
    (capture / "source.pdf").write_bytes(b"changed")
    with pytest.raises(Exception, match="pdf_hash_mismatch"):
        run_local_program(prepared, capture, tmp_path / "out", workspace=ws, deadline_epoch=time.time()+70,
                          runtime_observation=_mock_observation())


def test_execution_requires_new_empty_output_and_no_mock_model_claim(tmp_path):
    ws, capture, prepared = _prepared(tmp_path)
    occupied = tmp_path / "occupied"; occupied.mkdir(); (occupied / "x").write_text("x")
    with pytest.raises(PublicExecutionRefusal, match="output_not_empty"):
        run_local_program(prepared, capture, occupied, workspace=ws, deadline_epoch=time.time()+70,
                          runtime_observation=_mock_observation())
    bad = _mock_observation(); bad["model_execution_observed"] = True
    with pytest.raises(PublicExecutionRefusal, match="mock_claim_refused"):
        run_local_program(prepared, capture, tmp_path / "out", workspace=ws, deadline_epoch=time.time()+70,
                          runtime_observation=bad)


def test_refuses_served_model_mismatch_before_driver_requests(tmp_path):
    ws, capture, prepared = _prepared(tmp_path); server, thread = _server(_WrongModelLoopback)
    try:
        with pytest.raises(PublicExecutionRefusal, match="served_model_refused"):
            run_local_program(prepared, capture, tmp_path / "out", workspace=ws, deadline_epoch=time.time()+40,
                              runtime_observation=_mock_observation())
    finally:
        server.shutdown(); thread.join(); server.server_close()
    assert not (tmp_path / "out" / "ledger").exists()


def test_refuses_loopback_redirect_without_following_it(tmp_path):
    ws, capture, prepared = _prepared(tmp_path); server, thread = _server(_RedirectLoopback)
    try:
        with pytest.raises(PublicExecutionRefusal, match="protocol_redirect_refused"):
            run_local_program(prepared, capture, tmp_path / "out", workspace=ws, deadline_epoch=time.time()+40,
                              runtime_observation=_mock_observation())
    finally:
        server.shutdown(); thread.join(); server.server_close()


def test_incomplete_stream_stops_without_claiming_native_completion(tmp_path):
    ws, capture, prepared = _prepared(tmp_path); server, thread = _server(_IncompleteLoopback)
    try:
        result = run_local_program(prepared, capture, tmp_path / "out", workspace=ws, deadline_epoch=time.time()+40,
                                   runtime_observation=_mock_observation())
    finally:
        server.shutdown(); thread.join(); server.server_close()
    assert result["receipt"]["native_execution_completed"] is False
    assert result["receipt"]["stop_reason"] is not None
    with pytest.raises(PublicExecutionRefusal, match="already_bound"):
        run_local_program(prepared, capture, tmp_path / "other", workspace=ws, deadline_epoch=time.time()+40,
                          runtime_observation=_mock_observation())
