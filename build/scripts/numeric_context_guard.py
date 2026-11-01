"""Separate explicitly labelled confidence levels from result percentages."""
from copy import deepcopy
import re
import unicodedata


def context_reason(statistic, groups):
    expression=statistic['expression']
    if expression['statistic_type'] not in ('percentage','reported_quantity'):return None
    if expression['statistic_type']=='reported_quantity' and expression.get('printed_label','').lower()!='percent':return None
    group=groups[expression['unit_id']];text=group['text']
    if text[expression['start']:expression['end']]!=expression['text']:raise ValueError('numeric_context_source_changed')
    after=unicodedata.normalize('NFKC',text[expression['end']:])
    after=re.sub(r'(?<=\w)-\s*\n\s*(?=\w)','',after)
    if re.match(r'\s*(?:(?:bias[-\s]+corrected|bootstrap(?:ped)?|two[-\s]+sided|one[-\s]+sided)\s+)*(?:confidence\s+(?:interval|level|limit)|CI\b)',after,re.I):
        return 'explicit_confidence_level_not_result_percentage'
    return None


def filter_delta(delta):
    """Preserve the original delta and quarantine only the explicit context case."""
    result=deepcopy(delta);groups={g['unit_id']:g for g in delta['baseline']['upstream']['source_group_provenance']}
    kept=[];quarantined=[]
    for binding in result['new_bindings']:
        reason=context_reason(binding['statistic'],groups)
        if reason:quarantined.append({'binding':binding,'reason':reason})
        else:kept.append(binding)
    result['new_bindings']=kept
    result['numeric_context_quarantine']=quarantined
    result['original_unfiltered_delta']=deepcopy(delta)
    return result


def filter_attachments(bound):
    """Apply the same rule to initial attachments, preserving their raw record."""
    result=deepcopy(bound);groups={g['unit_id']:g for g in bound['upstream']['source_group_provenance']}
    quarantined=[]
    for attachment in result['attachments']:
        kept=[]
        for statistic in attachment['statistics']:
            reason=context_reason(statistic,groups)
            if reason:quarantined.append({'test_id':attachment['test_id'],'statistic':statistic,'reason':reason})
            else:kept.append(statistic)
        attachment['statistics']=kept
    result['numeric_context_quarantine']=quarantined;result['original_unfiltered_attachments']=deepcopy(bound)
    return result
