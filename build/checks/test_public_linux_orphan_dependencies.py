"""Contracts for the narrow one-package Linux repair."""
from hashlib import sha256
import base64
import json
from pathlib import Path

import pytest

from mosaic.public_linux_orphan_dependencies import (NCCL, OTEL, TARGET, PDF_WHEEL, PDF_WHEEL_SHA, PDF_WHEEL_URL, OrphanRepairRefusal, load_plan, parse_freeze, replay_graph, run_orphan_dependency_gate, validate_baseline, validate_file_manifests, validate_orphan_pdf_stage, validate_record_ownership, verify_retained_orphan_dependency_gate)
from mosaic.workspace import Workspace


def _freeze(values): return ("\n".join(f"{k}=={v}" for k, v in sorted(values.items())) + "\n").encode()
def _metadata(values, requires=None):
    requires = requires or {}; return json.dumps({"environment":{"implementation_name":"cpython","implementation_version":"3.12.3","os_name":"posix","platform_machine":"x86_64","platform_release":"x","platform_system":"Linux","platform_version":"x","python_full_version":"3.12.3","platform_python_implementation":"CPython","python_version":"3.12","sys_platform":"linux"},"installed":[{"metadata":{"name":name,"version":version,"requires_dist":requires.get(name,[])}} for name,version in values.items()]}).encode()
def _record(paths):
    digest=base64.urlsafe_b64encode(b"x"*32).decode().rstrip("=")
    return ("\n".join(f"{p},," if p.endswith(".dist-info/RECORD") else f"{p},sha256={digest},1" for p in paths)+"\n").encode()

def _baseline(): return {TARGET:"0.66b1","opentelemetry-sdk":"1.40.0","opentelemetry-api":"1.40.0","opentelemetry-exporter-prometheus":"0.61b0","torch":"2.13.0+cu130","nvidia-nccl-cu13":"2.30.7","lmcache":"0.5.6"}
def _requires(): return {TARGET:["opentelemetry-sdk~=1.45.1"],"torch":["nvidia-nccl-cu13==2.29.7; platform_system == 'Linux'"],"lmcache":["opentelemetry-api<=1.40.0,>=1.20.0","opentelemetry-exporter-prometheus<=0.61b0,>=0.50b0"]}

def test_baseline_and_projection_are_exact():
    values=_baseline(); plan=load_plan(); result=validate_baseline(freeze=values,metadata=_metadata(values,_requires()),check={"returncode":1,"stdout":f"{OTEL}\n{NCCL}\n".encode()},plan=plan)
    assert len(result["baseline_graph"]["conflicts"])==2
    assert result["projected_graph"]["conflicts"][0]["child"]=="nvidia-nccl-cu13"

def test_reverse_edge_refuses_removal_even_if_marker_inactive():
    values=_baseline(); req=_requires(); req["lmcache"] += [f"{TARGET}==0.66b1; platform_system == 'Windows'"]
    with pytest.raises(OrphanRepairRefusal,match="not_orphan"):
        validate_baseline(freeze=values,metadata=_metadata(values,req),check={"returncode":1,"stdout":f"{OTEL}\n{NCCL}\n".encode()},plan=load_plan())

def test_lmcache_regression_is_seen_after_bad_upgrade():
    values={**_baseline(),"opentelemetry-api":"1.45.1","opentelemetry-exporter-prometheus":"0.66b1"}; graph=replay_graph(_metadata(values,_requires()),values)
    assert {"lmcache","torch"} <= {x["parent"] for x in graph["conflicts"]}

def test_unknown_conflict_and_bad_direct_url_are_visible():
    values={**_baseline(),"bad":"1"}; req=_requires(); req["lmcache"] += ["missing-package==1", "bad @ https://example.invalid/a.whl"]
    graph=replay_graph(_metadata(values,req),values); assert {x["reason"] for x in graph["conflicts"]}|{x["reason"] for x in graph["invalid_or_unverifiable"]} >= {"missing_distribution","unverifiable_direct_url"}

def test_record_rejects_path_escape_and_collision():
    with pytest.raises(OrphanRepairRefusal,match="path_unsafe"): validate_record_ownership(_record(["../outside"]))
    record=validate_record_ownership(_record(["pkg/a.py","pkg-0.dist-info/RECORD"]))
    before={TARGET:{"paths":list(record),"record_sha256":"a"*64,"metadata_sha256":"b"*64},"other":{"paths":["pkg/a.py"],"record_sha256":"a"*64,"metadata_sha256":"b"*64}}; after={"other":before["other"]}
    with pytest.raises(OrphanRepairRefusal,match="ownership_collision"): validate_file_manifests(before,after,target_record=record,target_file_hashes={path:(b"x"*32).hex() if digest else "d"*64 for path,digest in record.items()})

def test_file_manifest_tamper_refuses():
    record=validate_record_ownership(_record(["pkg/a.py","pkg-0.dist-info/RECORD"])); before={TARGET:{"paths":list(record),"record_sha256":"a"*64,"metadata_sha256":"b"*64},"other":{"paths":["other.py"],"record_sha256":"a"*64,"metadata_sha256":"b"*64}}; after={"other":{**before["other"],"metadata_sha256":"c"*64}}
    with pytest.raises(OrphanRepairRefusal,match="remaining_file_drift"): validate_file_manifests(before,after,target_record=record,target_file_hashes={path:(b"x"*32).hex() if digest else "d"*64 for path,digest in record.items()})

def test_freeze_malformed_refuses():
    with pytest.raises(OrphanRepairRefusal,match="freeze_invalid"): parse_freeze(b"broken\n")


def create_retained_orphan_fixture(root):
    """Create a fictional archive-shaped v4 gate tree without invoking pip."""
    from hashlib import sha256
    output=root/"run"/"gpu-smoke"; output.mkdir(parents=True); ws=Workspace(repo_root=root,temp_root=root)
    plan=load_plan(); base=_baseline(); post={k:v for k,v in base.items() if k != TARGET}; pdf={**post,"pymupdf":PDF_WHEEL_URL+"#sha256="+PDF_WHEEL_SHA}; record=_record(["pkg/a.py","pkg-0.dist-info/RECORD"]).decode()
    target_files={"pkg/a.py":(b"x"*32).hex(),"pkg-0.dist-info/RECORD":"d"*64}
    def manifest(values, include_target):
        rows={name:{"paths":[],"record_sha256":"a"*64,"metadata_sha256":"b"*64} for name in values}
        if include_target: rows[TARGET]={"paths":sorted(target_files),"record_sha256":"a"*64,"metadata_sha256":"b"*64}
        return {"distributions":rows,"shared_opentelemetry_files":({TARGET:target_files} if include_target else {})}
    commands=[]
    def run(command,label,timeout,require_success=True,**_):
        if label=="baseline-freeze": out=_freeze(base)
        elif label=="post-repair-freeze": out=_freeze(post)
        elif label in {"post-pdf-freeze","installed"}: out=_freeze(pdf)
        elif label.endswith("metadata"):
            values=base if label.startswith("baseline") else post if label.startswith("post-repair") else pdf; payload=json.loads(_metadata(({k:v for k,v in values.items() if k!="pymupdf"} if label.startswith("post-pdf") else values),_requires()));
            if label.startswith("post-pdf"): payload["installed"].append({"metadata":{"name":"pymupdf","version":"1.28.2","requires_dist":[]},"direct_url":{"url":PDF_WHEEL_URL,"archive_info":{"hash":"sha256="+PDF_WHEEL_SHA}}})
            out=json.dumps(payload).encode()
        elif label=="baseline-check": out=f"{OTEL}\n{NCCL}\n".encode()
        elif label.endswith("check"): out=(NCCL+"\n").encode()
        elif label=="orphan-target-record": out=json.dumps({"record":record,"files":target_files,"root":"/tmp/fake-site"}).encode()
        elif label=="orphan-before-files": out=json.dumps(manifest(base,True)).encode()
        elif label=="orphan-after-files": out=json.dumps(manifest(post,False)).encode()
        elif label=="orphan-post-remove-absence": out=json.dumps({"absent_paths":sorted(target_files)}).encode()
        elif label=="orphan-post-imports": out=b"orphan-repair-imports-ok\n"
        else: out=b""
        rc=1 if label.endswith("check") else 0; (output/f"{label}.stdout.log").write_bytes(out); (output/f"{label}.stderr.log").write_bytes(b"")
        receipt={"command":list(command),"returncode":rc,"timed_out":False,"stdout_sha256":sha256(out).hexdigest(),"stderr_sha256":sha256(b"").hexdigest()}; ws.write_canonical_json(output/f"{label}.json",receipt); commands.append(receipt); return receipt
    gate=run_orphan_dependency_gate(run=run,python="python",output=output,workspace=ws,image=plan.image,plan=plan)
    lock=output/"requirements-linux-smoke.lock"; ws.write_text(lock,f"pymupdf @ {PDF_WHEEL_URL} --hash=sha256:{PDF_WHEEL_SHA}\n")
    run(["python","-m","pip","install","--no-deps","--require-hashes","-r",str(lock)],"install",30)
    run(["python","-m","pip","freeze","--all"],"installed",30)
    pdf_stage=validate_orphan_pdf_stage(run=run,python="python",output=output,workspace=ws,before=post,plan=plan)
    runtime=root/"run"/"smoke-runtime"; runtime.mkdir(); ws.write_canonical_json(runtime/"declaration.json",{"runtime":{"wheels":[{"name":PDF_WHEEL,"sha256":PDF_WHEEL_SHA}],"installed_inventory_sha256":sha256((output/"installed.stdout.log").read_bytes()).hexdigest(),"dependency_repair":{"action":"remove_orphan_distribution","removed_distribution":TARGET,"plan_sha256":plan.config_sha256}}})
    ws.write_canonical_json(output/"outcome.json",{"commands":commands,"dependency_gate_sha256":sha256((output/"dependency-gate.json").read_bytes()).hexdigest(),"dependency_config_sha256":plan.config_sha256,"dependency_pdf_stage_sha256":sha256((output/"dependency-pdf-stage.json").read_bytes()).hexdigest(),"dependency_gate":gate,"dependency_pdf":pdf_stage})
    return output


def test_retained_fixture_runs_the_real_orphan_gate(tmp_path):
    output=create_retained_orphan_fixture(tmp_path)
    assert (output/"orphan-post-remove-absence.json").is_file()
    assert json.loads((output/"dependency-gate.json").read_text())["action"]=="remove_orphan_distribution"
    assert verify_retained_orphan_dependency_gate(tmp_path,expected_config_sha256=load_plan().config_sha256)["scientifically_accepted"] is False


def test_retained_verifier_refuses_rehashed_semantic_tampering(tmp_path):
    output=create_retained_orphan_fixture(tmp_path); outcome=json.loads((output/"outcome.json").read_text()); receipt=json.loads((output/"orphan-uninstall.json").read_text())
    receipt["command"][-1]="different-package"; ws=Workspace(repo_root=tmp_path,temp_root=tmp_path); ws.write_canonical_json(output/"orphan-uninstall.json",receipt,once=False)
    outcome["commands"]=[receipt if row.get("command",[])[1:]==["-m","pip","uninstall","--yes",TARGET] else row for row in outcome["commands"]]; ws.write_canonical_json(output/"outcome.json",outcome,once=False)
    with pytest.raises(OrphanRepairRefusal,match="command_semantics"): verify_retained_orphan_dependency_gate(tmp_path,expected_config_sha256=load_plan().config_sha256)
