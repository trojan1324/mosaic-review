import json

import pytest

from mosaic.public_literature import (MAX_PROVIDER_REQUEST_BYTES, PublicLiteratureAssistant,
                                      PublicLiteratureConfig, PublicLiteratureRefusal, SERVICE_TIER)
from mosaic.workspace import Workspace


def assistant(tmp_path, *, enabled=False, transport=None, config=None):
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    bibliography = {"paper_items": [{"paper_id": "p1", "title_verbatim": "Trust at Work", "authors_verbatim": ["Smith", "Jones"], "journal_verbatim": "Journal", "publication_year": 2025}],
                    "items": [{"paper_id": "p1", "citation_apa": "Smith, A., & Jones, B. (2025). Trust at work. Journal."}]}
    passages = [{"id": "passage:p1:u1", "paper_id": "p1", "page": 2, "lane": "body", "quote": "Trust predicted collaboration in this source-bound study."},
                {"id": "passage:p1:r1", "paper_id": "p1", "page": 8, "lane": "references", "quote": "Reference material must not become evidence."},
                {"id": "passage:p2:u1", "paper_id": "p2", "page": 1, "lane": "body", "quote": "Incomplete identity is withheld."}]
    return PublicLiteratureAssistant(passages=passages, bibliography=bibliography, snapshot_id="snapshot-1", workspace=ws,
        config=config or PublicLiteratureConfig(enabled=enabled), transport=transport)


def completed():
    draft = {"title": "Trust", "sections": [{"heading": "Evidence", "paragraphs": [{"text": "The source reports a relationship.", "kind": "source_summary", "citations": ["S1"], "support": [{"source_id": "passage:p1:u1", "quote": "Trust predicted collaboration in this source-bound study."}]}]}], "gaps": ["Not exhaustive."]}
    return {"id": "response", "status": "completed", "model": "gpt-6-sol", "usage": {"input_tokens": 100, "output_tokens": 50}, "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(draft)}]}]}


def test_preview_is_offline_and_withholds_incomplete_and_references(tmp_path):
    calls = []
    service = assistant(tmp_path, transport=lambda request: calls.append(request))
    result = service.preview({"question": "What does trust predict?"})
    assert result["status"] == "ok" and len(result["sources"]) == 1
    source = result["sources"][0]
    assert source["citation_complete_apa"] and source["article_url"] == "studies.html?paper=p1"
    assert source["source_id"] == "passage:p1:u1" and not calls
    assert service.status()["generation_configured"] is False


def test_generation_requires_two_explicit_opt_ins(tmp_path):
    service = assistant(tmp_path, enabled=False, transport=lambda _: completed())
    with pytest.raises(PublicLiteratureRefusal, match="generation_not_enabled"):
        service.draft({"question": "trust", "enable_model": True})
    service = assistant(tmp_path, enabled=True, transport=lambda _: completed())
    with pytest.raises(PublicLiteratureRefusal, match="generation_not_enabled"):
        service.draft({"question": "trust"})


def test_generated_draft_requires_exact_source_support_and_records_budget(tmp_path):
    service = assistant(tmp_path, enabled=True, transport=lambda _: completed())
    result = service.draft({"question": "trust", "enable_model": True, "output_type": "literature_review"})
    assert result["status"] == "ok" and result["span_binding_verified"]
    ledger = tmp_path / "public-ledgers" / "literature_assistant_calls.jsonl"
    assert [json.loads(line)["event"] for line in ledger.read_text().splitlines()] == ["reserved", "completed"]


def test_provider_request_fixes_standard_tier_and_conservative_pricing_floor(tmp_path):
    service = assistant(tmp_path, enabled=True, transport=lambda _: completed())
    preview = service.preview({"question": "trust"})
    request = service._provider_request({"question": "trust", "output_type": "answer", "history": []}, preview["sources"])
    assert request["service_tier"] == SERVICE_TIER == "default"
    assert PublicLiteratureConfig().input_upper_usd_per_million == 6.0
    with pytest.raises(PublicLiteratureRefusal, match="rate_bound_invalid"):
        PublicLiteratureConfig(input_upper_usd_per_million=5.99)


def test_provider_payload_cap_refuses_before_reservation_or_transport(tmp_path):
    ws = Workspace(repo_root=tmp_path, temp_root=tmp_path)
    papers, apa, passages = [], [], []
    for number in range(12):
        paper_id = f"p{number}"
        papers.append({"paper_id": paper_id, "title_verbatim": f"Trust {number}", "authors_verbatim": ["Smith"], "journal_verbatim": "Journal", "publication_year": 2025})
        apa.append({"paper_id": paper_id, "citation_apa": f"Smith, A. (2025). Trust {number}. Journal."})
        passages.append({"id": f"passage:{paper_id}", "paper_id": paper_id, "page": 1, "lane": "body", "quote": "trust " + "x" * 19_994})
    calls = []
    service = PublicLiteratureAssistant(passages=passages, bibliography={"paper_items": papers, "items": apa},
        snapshot_id="fictional-large", workspace=ws, config=PublicLiteratureConfig(enabled=True), transport=lambda request: calls.append(request))
    preview = service.preview({"question": "trust", "limit": 12})
    assert len(preview["sources"]) == 12
    with pytest.raises(PublicLiteratureRefusal, match="request_too_large"):
        service.draft({"question": "trust", "limit": 12, "enable_model": True})
    assert not calls and not (tmp_path / "public-ledgers" / "literature_assistant_calls.jsonl").exists()
    assert MAX_PROVIDER_REQUEST_BYTES < 272_000


def test_nonliteral_model_support_is_withheld(tmp_path):
    response = completed()
    body = json.loads(response["output"][0]["content"][0]["text"])
    body["sections"][0]["paragraphs"][0]["support"][0]["quote"] = "invented source text that is not present"
    response["output"][0]["content"][0]["text"] = json.dumps(body)
    service = assistant(tmp_path, enabled=True, transport=lambda _: response)
    with pytest.raises(PublicLiteratureRefusal, match="support_not_exact"):
        service.draft({"question": "trust", "enable_model": True})


def test_missing_or_cross_citation_support_and_freeform_citation_are_withheld(tmp_path):
    response = completed()
    body = json.loads(response["output"][0]["content"][0]["text"])
    body["sections"][0]["paragraphs"][0]["support"] = []
    response["output"][0]["content"][0]["text"] = json.dumps(body)
    with pytest.raises(PublicLiteratureRefusal, match="uncited_summary"):
        assistant(tmp_path, enabled=True, transport=lambda _: response).draft({"question": "trust", "enable_model": True})
    response = completed()
    body = json.loads(response["output"][0]["content"][0]["text"])
    body["sections"][0]["paragraphs"][0]["text"] = "Prior research (Fake, 2023) established this."
    response["output"][0]["content"][0]["text"] = json.dumps(body)
    with pytest.raises(PublicLiteratureRefusal, match="freeform_citation"):
        assistant(tmp_path, enabled=True, transport=lambda _: response).draft({"question": "trust", "enable_model": True})


def test_damaged_statistic_and_generic_query_are_withheld_from_preview(tmp_path):
    service = assistant(tmp_path)
    service.passages.append({"id": "damaged", "paper_id": "p1", "page": 3, "lane": "body", "quote": "A damaged result reports p 5 .03."})
    # A damaged row arriving after construction remains outside ordinary query
    # results only when admission happened through the constructor; this checks
    # the public guard at that admission point separately below.
    assert service.preview({"question": "please summarize the literature"})["status"] == "no_sources"
    passages = [{"id": "damaged", "paper_id": "p1", "page": 3, "lane": "body", "quote": "A damaged result reports p 5 .03."}]
    bibliography = {"paper_items": [{"paper_id": "p1", "title": "T", "authors": "Smith", "journal": "J", "year": 2025}],
                    "items": [{"paper_id": "p1", "citation_apa": "Smith, A. (2025). T. J."}]}
    clean = PublicLiteratureAssistant(passages=passages, bibliography=bibliography, snapshot_id="s", workspace=Workspace(tmp_path, tmp_path))
    assert clean.preview({"question": "damaged result"})["status"] == "no_sources"


def test_unknown_provider_outcome_retains_full_reservation(tmp_path):
    def failed(request):
        raise RuntimeError('unknown outcome')
    service = assistant(tmp_path, enabled=True, transport=failed)
    with pytest.raises(RuntimeError, match='unknown outcome'):
        service.draft({'question':'trust','enable_model':True})
    events=[json.loads(line) for line in (tmp_path/'public-ledgers/literature_assistant_calls.jsonl').read_text().splitlines()]
    assert [row['event'] for row in events] == ['reserved','failed']
    assert service.status()['budget']['accounted_upper_usd'] == events[0]['reserved_usd']


def test_duplicate_completion_cannot_reduce_cumulative_cost(tmp_path):
    service=assistant(tmp_path,enabled=True,transport=lambda _:completed())
    service.draft({'question':'trust','enable_model':True})
    ledger=tmp_path/'public-ledgers/literature_assistant_calls.jsonl'
    event=json.loads(ledger.read_text().splitlines()[1]); event['cost_upper_usd']=0
    service.workspace.append_line(ledger,json.dumps(event))
    with pytest.raises(PublicLiteratureRefusal,match='ledger_invalid'):
        service.status()


def test_canonical_completed_then_postprocessing_failure_keeps_completed_cost(tmp_path):
    ledger = tmp_path / "ledgers" / "research_assistant_calls.jsonl"
    ledger.parent.mkdir()
    rows = [
        {"attempt_id": "fictional", "event": "reserved", "reserved_usd": 1.0},
        {"attempt_id": "fictional", "event": "completed", "cost_upper_usd": 0.25},
        {"attempt_id": "fictional", "event": "failed", "error_class": "ValueError"},
    ]
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    service = assistant(tmp_path, config=PublicLiteratureConfig(ledger_path="ledgers/research_assistant_calls.jsonl"))
    budget = service.status()["budget"]
    assert budget["attempts"] == 1 and budget["accounted_upper_usd"] == 0.25


@pytest.mark.parametrize("rows", [
    [{"attempt_id": "fictional", "event": "completed", "cost_upper_usd": 0.1}],
    [{"attempt_id": "fictional", "event": "reserved", "reserved_usd": 1.0}, {"attempt_id": "fictional", "event": "failed"}, {"attempt_id": "fictional", "event": "completed", "cost_upper_usd": 0.1}],
    [{"attempt_id": "fictional", "event": "reserved", "reserved_usd": 1.0}, {"attempt_id": "fictional", "event": "completed", "cost_upper_usd": 0.1}, {"attempt_id": "fictional", "event": "failed"}, {"attempt_id": "fictional", "event": "failed"}],
    [{"attempt_id": "fictional", "event": "reserved", "reserved_usd": float("nan")}],
])
def test_canonical_ledger_rejects_invalid_terminal_sequences_and_amounts(tmp_path, rows):
    ledger = tmp_path / "ledgers" / "research_assistant_calls.jsonl"
    ledger.parent.mkdir()
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    service = assistant(tmp_path, config=PublicLiteratureConfig(ledger_path="ledgers/research_assistant_calls.jsonl"))
    with pytest.raises(PublicLiteratureRefusal, match="ledger_invalid"):
        service.status()


def test_research_suggestion_cannot_attach_an_unsupported_citation(tmp_path):
    response=completed(); body=json.loads(response['output'][0]['content'][0]['text'])
    paragraph=body['sections'][0]['paragraphs'][0]
    paragraph['kind']='research_suggestion'; paragraph['support']=[]
    response['output'][0]['content'][0]['text']=json.dumps(body)
    with pytest.raises(PublicLiteratureRefusal,match='citation_unsupported'):
        assistant(tmp_path,enabled=True,transport=lambda _:response).draft({'question':'trust','enable_model':True})
