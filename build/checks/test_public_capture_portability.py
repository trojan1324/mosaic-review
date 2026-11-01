"""Same fictional PDF yields stable capture identities after moving the install."""
from hashlib import sha256
import json
from pathlib import Path
import pymupdf
import pytest
from mosaic.public_capture import capture_public
from mosaic.workspace import Workspace

pytestmark = pytest.mark.evidence_mock

def test_capture_identity_does_not_include_installation_directory(tmp_path: Path):
    with pymupdf.open() as doc:
        page=doc.new_page()
        for y,text in [(72,'Fictional source'),(95,'Abstract'),(120,'This fictional abstract discusses workplace trust.'),(160,'Study 1'),(190,'Methods'),(215,'Participants described trust in a fictional exercise.')]:
            page.insert_text((72,y),text)
        table=doc.new_page()
        for y,text in [(72,'Appendix'),(120,'Table 1'),(145,'Fictional table caption'),(175,'Trust     1.00')]:
            table.insert_text((72,y),text)
        raw=doc.tobytes(no_new_id=True)
    digest=sha256(raw).hexdigest()
    row={'paper_id':'public-'+digest,'source_pdf_sha256':digest,'byte_size':len(raw),'locators':['example.pdf'],'source_status':'public_unreviewed_source','permission':'fictional_fixture','scientifically_accepted':False}
    artifacts=[]
    for name in ('first-install','relocated-install'):
        root=tmp_path/name; ws=Workspace(repo_root=root,temp_root=tmp_path)
        ws.write_once(root/'pdfs/example.pdf',raw)
        result=capture_public(row,'example.pdf',root/'pdfs',root/'capture',title='Fictional source',workspace=ws)
        source=json.loads((root/'capture'/'source.json').read_bytes())
        binding=json.loads((root/'capture'/'source-binding.json').read_bytes())
        reader=json.loads((root/'capture'/'native-reader.json').read_bytes())
        receipt=json.loads((root/'capture'/'receipt.json').read_bytes())
        object_payloads={path.name:path.read_bytes() for path in (root/'capture/packages').rglob('objects/*.json')}
        assert any(name.startswith('table-') for name in object_payloads)
        source_payload = next(data for name, data in object_payloads.items() if name.startswith('source-'))
        assert b'"acquisition_path":"source.pdf"' in source_payload
        artifacts.append({'source':source, 'binding':binding, 'reader':reader, 'receipt':receipt,
                          'object_payloads':object_payloads})
        assert result['status']=='source_packet_ready'
        for path in (root/'capture/packages').rglob('*.json'):
            data=path.read_bytes()
            assert str(root).replace('\\','/').encode() not in data
            assert str(root).encode() not in data
    # Object identities and their content are stable across installations. The
    # package manifest carries its real creation timestamp, so manifest IDs and
    # the three public projections that cite it are intentionally not claimed
    # byte-identical.
    assert artifacts[0]['object_payloads'] == artifacts[1]['object_payloads']
    assert artifacts[0]['source'] == artifacts[1]['source']
    for key in ('work_id','status','source_pdf_sha256','source_packet_sha256','source_units','model_calls','network_calls'):
        assert artifacts[0]['receipt'][key] == artifacts[1]['receipt'][key]
    assert artifacts[0]['binding']['unit_bindings'] == artifacts[1]['binding']['unit_bindings']
    assert artifacts[0]['reader']['units'] == artifacts[1]['reader']['units']
    assert artifacts[0]['reader']['source_pdf_sha256'] == artifacts[1]['reader']['source_pdf_sha256']
    assert all(item['receipt']['package_manifest_id'].startswith('manifest-') for item in artifacts)
