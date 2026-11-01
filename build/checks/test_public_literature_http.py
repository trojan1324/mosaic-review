"""Offline HTTP integration, including actual reviewed fictional capture and refusals."""
from pathlib import Path
import json
import threading
import urllib.request
import urllib.error
import pytest
from mosaic.public_literature_demo import build_literature_demo
from mosaic.public_serving import serve_public
from mosaic.workspace import Workspace

pytestmark = pytest.mark.evidence_mock

def test_public_literature_http_uses_clean_projection_and_explicit_generation(tmp_path: Path):
    ws=Workspace(repo_root=tmp_path,temp_root=tmp_path/'temp')
    result=build_literature_demo(tmp_path,workspace=ws)
    server=serve_public(result['bundle_path'],workspace=ws)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def get(path):
        with urllib.request.urlopen(base+path,timeout=10) as r: return json.load(r)
    def post(path,body,headers=None):
        h={'Content-Type':'application/json','X-Mosaic-Request':status['csrf_token']}
        h.update(headers or {})
        req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers=h)
        with urllib.request.urlopen(req,timeout=10) as r: return json.load(r)
    try:
        status=get('/api/literature/status')
        assert status['generation_configured'] is False
        assert status['budget']['attempts']==0
        body={'question':'Compare these fictional articles','search_terms':'fictional','output_type':'literature_review'}
        preview=post('/api/literature/preview',body)
        assert preview['status']=='ok' and preview['retrieved_papers']==2
        papers=get('/api/display/v1/bibliography')['items']
        assert all(all(isinstance(x,str) for x in p['authors_verbatim']) for p in papers)
        admitted={p['paper_id'] for p in papers}
        passages=get('/api/display/v1/passages?page_size=100')['passages']
        evidence={p['id']:p for p in passages if p['lane']!='references'}
        for source in preview['sources']:
            assert source['paper_id'] in admitted
            assert source['source_id'] in evidence
            assert source['quote']==evidence[source['source_id']]['quote']
            assert source['parenthetical'] and '(' not in source['parenthetical']
        assert get('/api/literature/status')['budget']['attempts']==0
        with pytest.raises(urllib.error.HTTPError) as exc:
            post('/api/literature/draft',{**body,'enable_model':True})
        assert exc.value.code==400
        for headers in ({'X-Mosaic-Request':''},{'Origin':'https://unrelated.example'}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post('/api/literature/preview',body,headers)
            assert exc.value.code==403
        for malformed in ([],{'question':12},{**body,'unsupported_key':True}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post('/api/literature/preview',malformed)
            assert exc.value.code==400
        none=post('/api/literature/preview',{'question':'zzzz-no-matching-evidence'})
        assert none['status']=='no_sources'
        assert not (tmp_path/'public-ledgers/literature_assistant_calls.jsonl').exists()
    finally:
        server.shutdown();server.server_close();thread.join(timeout=5)
