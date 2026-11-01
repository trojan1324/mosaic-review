"""Locate line-hyphenated names within existing citations; retain raw spelling."""
from copy import deepcopy
from hashlib import sha256
import re
from mosaic.canonical import canonical_bytes
from scripts.core_evidence_trial import validate
from scripts.core_name_witnesses import recover_cited_names

VERSION='core-name-witnesses-v3'
_BREAK=r'(?:[-\u00ad][^\S\r\n]*\r?\n[^\S\r\n]*)?'

def linebreak_mapping(name, text):
    """Allow a printed line-end hyphen only inside an alphabetic token."""
    tokens=re.findall(r'\S+',name)
    if not tokens:return None
    patterns=[]
    for token in tokens:
        parts=[]
        for index,char in enumerate(token):
            if index and token[index-1].isalpha() and char.isalpha():parts.append(_BREAK)
            parts.append(re.escape(char))
        patterns.append(''.join(parts))
    pattern=(r'(?<!\w)' if tokens[0][0].isalnum() or tokens[0][0]=='_' else '')+r'\s+'.join(patterns)+(r'(?!\w)' if tokens[-1][-1].isalnum() or tokens[-1][-1]=='_' else '')
    matches=list(re.finditer(pattern,text,re.IGNORECASE))
    if len(matches)!=1:return None
    m=matches[0]
    if not re.search(r'[-\u00ad][^\S\r\n]*\r?\n',m.group()):return None
    return {'proposal_name':name,'source_name':m.group(),'match_type':'unique_intra_word_line_hyphen',
        'source_span':{'start':m.start(),'end':m.end(),'text':m.group()},
        'source_text_unchanged':True,'semantic_identity_verified':False}

def recover_linebreak_names(proposal,task,*,eligible_units,source_policy_sha256):
    args={'eligible_units':eligible_units,'source_policy_sha256':source_policy_sha256,'allow_local_layout_match':True}
    baseline=recover_cited_names(proposal,task,**args)
    table=validate(task);changed=deepcopy(proposal);repairs=[]
    for q in baseline['quarantined_fields']:
        if q['reason']!='name_not_exact_or_uniquely_normalisable_in_cited_sentence':continue
        keys=q['path'].split('/')[1:];field=changed
        for key in keys[:-1]:field=field[int(key)] if isinstance(field,list) else field[key]
        name_key=keys[-1]
        assert name_key in ('name','variable_name')
        evidence_key='description_sentence_ids' if name_key=='name' else 'role_sentence_ids'
        citations=list(dict.fromkeys([field['name_sentence_id']]+field[evidence_key]))
        alternatives=[]
        for sid in citations:
            mapping=linebreak_mapping(field[name_key],table[sid]['text'])
            if mapping is not None:alternatives.append({'sentence_id':sid,'mapping':mapping})
        if not alternatives:continue
        selected=alternatives[0]
        repairs.append({'field_path':'/'+('/'.join(keys[:-1])),'original_name':field[name_key],
            'original_name_sentence_id':field['name_sentence_id'],'replacement_name_sentence_id':selected['sentence_id'],
            'mapping':selected['mapping'],'alternatives':alternatives})
        field[name_key]=selected['mapping']['source_name'];field['name_sentence_id']=selected['sentence_id']
    result=recover_cited_names(changed,task,**args)
    # The fallback adds fields only; earlier accepted records are immutable.
    for kind,children in [('studies','measures'),('analyses','roles')]:
        for old,new in zip(baseline[kind],result[kind],strict=True):
            lookup={r['id']:r for r in new[children]}
            assert all(lookup[r['id']]==r for r in old[children])
            assert {k:v for k,v in old.items() if k!=children}=={k:v for k,v in new.items() if k!=children}
    result.update(version=VERSION,original_proposal_sha256=sha256(canonical_bytes(proposal)).hexdigest(),
        original_quarantined_fields=deepcopy(baseline['original_quarantined_fields']),
        name_witness_repairs=deepcopy(baseline['name_witness_repairs']),
        linebreak_base_record_sha256=sha256(canonical_bytes(baseline)).hexdigest(),
        linebreak_name_repairs=repairs,all_prior_fields_unchanged=True,
        repair_status='raw_linebreak_name_locator_only_not_semantic_acceptance')
    return result
