"""Recover only source-declared initialisms inside a field's existing citations."""
from copy import deepcopy
from hashlib import sha256
import re
from mosaic.canonical import canonical_bytes
from scripts.core_linebreak_names import recover_linebreak_names
from scripts.core_evidence_trial import validate
from scripts.core_evidence_recovery import source_name_mapping

VERSION='core-name-witnesses-v4'
_SHORT=re.compile(r'\(([A-Za-z][A-Za-z-]{1,14})\)')
_WORD=re.compile(r'[A-Za-z]+(?:-[A-Za-z]+)*')
digest=lambda value:sha256(canonical_bytes(value)).hexdigest()
normal=lambda text:' '.join(text.lower().split())


def declared_initialisms(units):
    """Require printed long form followed by matching first-letter abbreviation.

    The abbreviation may represent a prefix (partner-specific -> p-s). No
    dictionary, synonym model, acronym inference or role inference is used.
    """
    declarations=[]
    for unit in units:
        text=unit['text']
        for match in _SHORT.finditer(text):
            short=match[1];letters=short.replace('-','').lower()
            if len(letters)<2:continue
            # Ordinary lower-case parenthetical words are not abbreviations.
            # Accept conventional capitals or explicitly separated initials.
            if not (short.isupper() or re.fullmatch(r'[A-Za-z](?:-[A-Za-z])+',short)):continue
            # Quotation punctuation may enclose the long form; other punctuation
            # ends the definition. Only directly adjacent word sequences count.
            prefix=text[:match.start()].rstrip().rstrip('\"”’').rstrip()
            words=list(_WORD.finditer(prefix))
            if not words or words[-1].end()!=len(prefix):continue
            for count in range(1,min(8,len(words))+1):
                chosen=words[-count:];long=prefix[chosen[0].start():]
                if any(not re.fullmatch(r'\s+',prefix[a.end():b.start()]) for a,b in zip(chosen,chosen[1:])):break
                initials=''.join(part[0] for word in chosen for part in word.group().split('-')).lower()
                if initials!=letters:continue
                declarations.append({'long':long,'short':short,'unit_id':unit['unit_id'],
                    'start':chosen[0].start(),'end':match.end(),'text':text[chosen[0].start():match.end()],
                    'unit_text_sha256':sha256(text.encode()).hexdigest()})
    by_short={}
    for row in declarations:by_short.setdefault(normal(row['short']),set()).add(normal(row['long']))
    return [r for r in declarations if len(by_short[normal(r['short'])])==1]


def recover_declared_names(proposal,task,*,eligible_units,source_policy_sha256):
    args={'eligible_units':eligible_units,'source_policy_sha256':source_policy_sha256}
    baseline=recover_linebreak_names(proposal,task,**args)
    table=validate(task);aliases=declared_initialisms(eligible_units);changed=deepcopy(proposal);repairs=[]
    for q in baseline['quarantined_fields']:
        if q['reason']!='name_not_exact_or_uniquely_normalisable_in_cited_sentence':continue
        keys=q['path'].split('/')[1:];field=changed
        for key in keys[:-1]:field=field[int(key)] if isinstance(field,list) else field[key]
        key=keys[-1];name=field[key]
        evidence='description_sentence_ids' if key=='name' else 'role_sentence_ids'
        citations=list(dict.fromkeys([field['name_sentence_id']]+field[evidence]))
        candidates=[]
        for alias in aliases:
            # Match a complete leading phrase. No within-word replacement.
            parts=alias['long'].split()
            prefix=re.match(r'^'+r'\s+'.join(re.escape(s) for s in parts)+r'(?=\s|$)',name,re.I)
            if prefix is None:continue
            alternate=alias['short']+name[prefix.end():]
            for sid in citations:
                mapping=source_name_mapping(alternate,table[sid]['text'])
                if mapping is not None:candidates.append({'sentence_id':sid,'mapping':mapping,'declaration':alias})
        names={normal(c['mapping']['source_name']) for c in candidates}
        if len(names)!=1:continue
        selected=candidates[0]
        repairs.append({'field_path':'/'+('/'.join(keys[:-1])),'original_name':name,
            'original_name_sentence_id':field['name_sentence_id'],'replacement_name_sentence_id':selected['sentence_id'],
            'source_name':selected['mapping']['source_name'],'alternatives':candidates,'semantic_identity_verified':False})
        field[key]=selected['mapping']['source_name'];field['name_sentence_id']=selected['sentence_id']
    result=recover_linebreak_names(changed,task,**args)
    for kind,children in [('studies','measures'),('analyses','roles')]:
        for old,new in zip(baseline[kind],result[kind],strict=True):
            byid={r['id']:r for r in new[children]}
            assert all(byid[r['id']]==r for r in old[children])
            assert {k:v for k,v in old.items() if k!=children}=={k:v for k,v in new.items() if k!=children}
    result.update(version=VERSION,original_proposal_sha256=digest(proposal),
        declared_name_base_record_sha256=digest(baseline),declared_name_repairs=repairs,
        original_quarantined_fields=deepcopy(baseline['original_quarantined_fields']),
        all_prior_fields_unchanged=True,repair_status='explicit_source_initialism_only_not_semantic_acceptance')
    return result
