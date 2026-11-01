"""Experimental source-bound analysis scopes over the unchanged Results input.

No model call is made here. The caller must freeze and authorize its comparison.
Scopes remain proposals; exact quotations do not prove their scientific role.
"""
from copy import deepcopy
import re
from jsonschema import Draft202012Validator
from scripts import typed_test_decomposition as baseline
from scripts.integrated_extraction_trial import compile_section_task
from scripts.source_sentence_choices import sentence_choices,original_segments,cue_text
from scripts.source_quote_locator import locate_quote
from mosaic.adapted.e.headings import study_label,split_section_number


SCOPE_GUIDE = '''ANALYSIS IDENTITY: Preserve the separately reported sample, experimental condition, outcome entity, time point and analysis specification for each test. A measure such as resource allocation can refer to different divisions; name the affected division separately. A direct-effect sample and an indirect-effect sample remain separate even when their outcome and predictor have the same names. Do not combine several estimates into an indistinguishable generic test or rely on array order to distinguish them.
Each test has analysis_scope, an array of {kind,quote,evidence_selection}. kind is sample, condition, outcome_entity, time_point or specification. quote is a SHORT contiguous phrase from THIS Results paragraph, or from a contextual sentence explicitly included in this test's support; evidence_selection identifies that original sentence. Use context only when the source explicitly connects it to this analysis, never because it is nearby. Copy the authors' identifying phrase, not an invented group name or generated summary. Do not calculate a sample size. Empty means not established locally. Never substitute a neighboring test's sample or specification. A kind can occur more than once when the test explicitly compares several named arms.
A within-entity before/after comparison is simple_contrast with the two time conditions and the entity held fixed; do not invent a numeric reference-value test. A qualitative statement that another specification has a similar effect does not establish that it has exactly the same estimate, percentage or p-value. Preserve that qualitative result without borrowing a neighboring specification's number.
A simple slope of X predicting Y at a high or low level of Z is conditional_coefficient: X is the term, Z is the moderator and high/low is its condition. It is not a high-versus-low contrast unless the authors explicitly compare those levels as the test itself.
MEDIATION COMPLETENESS: The indirect effect and each separately reported component path are distinct results. If a paragraph reports X to M, M to Y and X through M to Y, preserve all three; the indirect-path object cannot replace the two coefficients. Likewise preserve a separately stated direct effect. Repeat this distinction separately for every named sample or condition. A later paragraph's similar or replicated result never licenses copying the earlier sample's coefficient. Separate outcome operationalizations (for example self-report and observer ratings) require separate objects even when the authors use the same construct name.
RESULT ADMISSION: A sentence that only says where a table or figure can be found is context_only, not a descriptive scientific result. Keep an actual body-text finding even when it cites a table or figure. Do not infer a complete claim from a broken sentence; use its supplied continuation if present, otherwise mark comparison_unresolved. Reported sample characteristics or the planned procedure alone are Methods context, not findings about the hypothesized relationship.
NAMED CONTROLS: Copy each individually named control as a contiguous phrase from its cited sentence. Generic phrases such as covariates, control variables or village-level controls are NOT individual variable names: use unspecified_controls instead. A pointer to a table containing an adjusted analysis does not establish that the estimates reported here used that adjustment. Preserve that distinction; do not transfer the table's controls to the body-text model.'''


def bounded_study_payload(prepared):
    """Use only already assigned main-body units of an explicitly named study.

    This does not restore withheld text, infer a study, or resolve an inventory
    ambiguity. Joint-study headings cannot silently belong to a single study.
    """
    study=prepared['study'];sid=study['study_id']
    if not re.fullmatch(r'study:[SE]\d+[a-z]?',sid,re.I):
        raise ValueError('explicit_named_study_required')
    units=[u for u in study['units'] if u['source_locator'].startswith('main_')]
    if not units:raise ValueError('empty_study_body')
    for unit in units:
        nodes=[n for path in unit.get('section_paths',[]) for n in path]
        if any(re.search(r'\b(?:studies|experiments)\s+\d+\s*(?:[-–]|and)\s*\d+',n.get('title',''),re.I) for n in nodes):
            raise ValueError('joint_study_heading_requires_separate_routing')
        labels=[study_label(split_section_number(n.get('title',''))[1]) for n in nodes if n.get('role')=='study']
        if not labels or any(label is None or 'study:'+label[1]!=sid for label in labels):
            raise ValueError('unit_not_exclusively_in_named_study')
    return deepcopy({'physical_study_id':sid,'source':units,'study_method_context':{},
        'source_scope':'already_assigned_main_body_of_named_study',
        'inventory_status':study.get('inventory_status'),'routing_uncertainty':study.get('ambiguity'),
        'coverage_claim':'admitted_units_only'})


def prepare_task(recipe,payload,*,bounded_study=False):
    task=baseline.prepare_task(recipe,payload,evidence_mode='coherent')
    task['baseline_schema']=deepcopy(task['schema'])
    scopes={'type':'array','maxItems':8,'items':{'type':'object','properties':{
        'kind':{'enum':['sample','condition','outcome_entity','time_point','specification']},
        'quote':{'type':'string','minLength':1,'maxLength':180},
        'evidence_selection':{'$ref':'#/$defs/selection'}},
        'required':['kind','quote','evidence_selection'],'additionalProperties':False}}
    variants=task['schema']['$defs']['paragraph']['anyOf'][0]['properties']['tests']['items']['anyOf']
    task['schema']['$defs']['analysis_scope']=scopes
    for variant in variants:
        variant['properties']['analysis_scope']={'$ref':'#/$defs/analysis_scope'}
        variant['required'].append('analysis_scope')
    request=deepcopy(task['body'])
    request['messages'][0]['content']=request['messages'][0]['content'].split('\nRESPONSE_SCHEMA:\n',1)[0]+'\n'+SCOPE_GUIDE
    if bounded_study:
        request['messages'][0]['content']+='\nBOUNDED STUDY INPUT: This source is the admitted body of one explicitly bounded study whose Results heading was not recognized. It includes Methods and possibly Discussion. Identify only findings explicitly reported as observed in the supplied body; classify Methods, procedure, predictions and bare headings as context_only/heading_only. A missing Results header is not permission to invent findings. Some paper text may have been withheld before this request; never claim whole-paper completeness.'
    compiled=compile_section_task(request,task['payload'],task['schema'])
    task.update(compiled);task['analysis_scope_version']='source-bound-v1'
    task['bounded_study_input']=bounded_study
    task['source_scope_metadata']={k:deepcopy(payload[k]) for k in
        ('source_scope','inventory_status','routing_uncertainty','coverage_claim') if k in payload}
    return task


def reference_only(text):
    normalized=' '.join(cue_text(text).split())
    if re.search(r'\b(?:significant|positive|negative|higher|lower|increase[ds]?|decrease[ds]?|no difference|support(?:s|ed)?)\b',normalized,re.I):
        return False  # A genuine qualitative assertion may also cite a table.
    return bool(re.fullmatch(r'(?:The\s+)?[^.!?;]{1,180}\s+(?:are|is)\s+(?:shown|reported|presented|provided|listed)\s+in\s+(?:Tables?|Figures?)\s+[A-Za-z0-9., –-]+\.',normalized,re.I))


def availability_only(text):
    """Recognize a complete access notice, never a mixed empirical statement.

    This intentionally narrow syntax is not a general semantic classifier.
    Unrecognized language stays a candidate rather than being silently dropped.
    """
    normalized=' '.join(cue_text(text).split())
    # Guard the optional discourse prefix against carrying an actual finding.
    if re.search(r'\d|\b(?:effect|association|significant|positive|negative|higher|lower|difference|support|predicted|found|unchanged)\w*\b',normalized,re.I):
        return False
    return bool(re.fullmatch(
        r'(?:(?:Also(?: in this case)?|In (?:addition|this case)|Additionally|Furthermore|Again|For completeness),\s*)?'
        r'(?:(?:the|our|these)\s+)?(?:(?:detailed|additional|supplementary|full|further)\s+)*'
        r'(?:analyses|analysis|results|details|data|materials|code)\s+'
        r'(?:are|is)\s+available\s+(?:on|upon)\s+request'
        r'(?:\s+from\s+the\s+(?:corresponding\s+)?authors?)?\.?',normalized,re.I))


def materialize(proposal,task):
    Draft202012Validator(task['schema']).validate(proposal)
    stripped=deepcopy(proposal)
    for row in stripped.values():
        for test in row.get('tests',[]):test.pop('analysis_scope')
    baseline_task={**task,'schema':task['baseline_schema']}
    result=baseline.materialize(stripped,baseline_task)
    choices={c['choice_id']:c for c in sentence_choices(task['payload']['source']+task['payload'].get('source_context',[]))}
    groups={g['unit_id']:g for g in task['source_groups']}
    for record in result['records']:
        uid=record['source']['unit_id']
        record['proposal']=deepcopy(proposal[uid])
        for test,raw in zip(record['tests'],proposal[uid].get('tests',[])):
            test['proposal']['analysis_scope']=deepcopy(raw['analysis_scope'])
            test['scope_copies']=[]
            supported={s['choice_id'] for s in raw['support']}
            for scope in raw['analysis_scope']:
                choice=choices[scope['evidence_selection']['choice_id']]
                span=locate_quote(scope['quote'],choice['text'],allow_alphabetic_ligatures=True)
                if (choice['unit_id']!=uid and choice['choice_id'] not in supported) or span is None:
                    test['copy_errors'].append('analysis_scope_not_in_own_result_evidence')
                    continue
                start,end=choice['start']+span['start'],choice['start']+span['end']
                sid=choice['unit_id']
                test['scope_copies'].append({'proposed_kind':scope['kind'],'unit_id':sid,'start':start,'end':end,
                    'text':groups[sid]['text'][start:end],'source_segments':original_segments(groups[sid],start,end),
                    'contextual_link_proposed':sid!=uid})
            for control in test['control_copies']:
                span=control['original_span']
                if span is not None and locate_quote(control['variable'],span['text'],allow_alphabetic_ligatures=True) is None:
                    test['copy_errors'].append('control_name_not_in_cited_evidence')
            test['assertion_review']={'reference_only':reference_only(test['result_copy']['text']) if test['result_copy'] else None,
                'scientific_assertion_validated':False}
    result['analysis_scope_version']=task['analysis_scope_version']
    result['bounded_study_input']=task['bounded_study_input']
    result['source_scope_metadata']=deepcopy(task['source_scope_metadata'])
    return result


def binding_eligible_tests(upstream):
    """Isolate invalid proposals without dropping their original audit record."""
    selected=deepcopy(upstream);excluded=[]
    for record in selected['records']:
        valid=[]
        for test in record['tests']:
            if test['copy_errors']:
                excluded.append({'test_id':test['test_id'],'copy_errors':deepcopy(test['copy_errors'])})
            else:valid.append(test)
        record['tests']=valid
    return selected,excluded
