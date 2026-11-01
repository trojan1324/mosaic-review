"""Source-witnessed heading repairs for experimental packets, never capture edits."""
from collections import Counter
from copy import deepcopy
import re
import pymupdf
from mosaic.capture.pdf import _block_text, BOLD_NAME, ITALIC_NAME

INLINE = re.compile(r'^(Results|Discussion|Methods?)\.\s+')


def wrapped_study_parents(tree, objects):
    """Join contiguous, tightly spaced equal-style lines in recorded reading order."""
    found={}
    for left,right in zip(tree.nodes,tree.nodes[1:]):
        a,b=left.heading,right.heading
        if (left.role!='study' or not left.study or right.role!='unclassified'
                or left.parent_id!=right.parent_id or left.level!=right.level
                or right.numbered or not a or not b or a.text_region_id!=b.text_region_id
                or not left.title.isupper() or not right.title.isupper()):continue
        region=objects[a.text_region_id]
        blocks=[block for block in region['blocks'] if block['start']<b.end and a.start<block['end']]
        if not blocks or region['text'][a.end:b.start].strip():continue
        if len(blocks)>1:
            order=region.get('reading_order',[])
            indices=[order.index(block['block_index']) for block in blocks if block.get('block_index') in order]
            if len(indices)!=len(blocks) or indices!=list(range(indices[0],indices[0]+len(indices))):continue
        lines=[ln for block in blocks for ln in block.get('lines',[])]
        al=[ln for ln in lines if a.start<=ln['start']<a.end]
        bl=[ln for ln in lines if b.start<=ln['start']<b.end]
        if len(al)!=1 or len(bl)!=1:continue
        x,y=al[0],bl[0];xb,yb=x.get('bbox'),y.get('bbox')
        if (not xb or not yb or not x.get('bold') or
                any(x.get(k)!=y.get(k) for k in ('size','bold','italic','fonts'))):continue
        size=x.get('size',0)
        aligned=abs(xb[0]-yb[0])<=2 or abs((xb[0]+xb[2])-(yb[0]+yb[2]))<=4
        if not size or not aligned or not -1<=yb[1]-xb[3]<=0.6*size:continue
        found[right.node_id]={'parent_node_id':left.node_id,'study_id':'study:'+left.study[1],
            'basis':'contiguous_equal_style_lines_in_reading_order',
            'witnesses':[{'text_region_id':h.text_region_id,'start':h.start,'end':h.end} for h in (a,b)]}
    return found


def inline_heading_witness(region, start, end, page):
    """Verify a bold/italic run-in label against exact native page text and spans."""
    text=region['text'][start:end];match=INLINE.match(text)
    if not match:return None
    pd=page.get_text('dict',sort=True,flags=pymupdf.TEXTFLAGS_BLOCKS)
    blocks=[b for b in pd['blocks'] if b.get('type')==0]
    if ''.join(_block_text(b)[0] for b in blocks)!=region['text']:return None
    spans=[];pos=0
    for block in blocks:
        for line in block.get('lines',[]):
            for span in line.get('spans',[]):
                word=span.get('text','');spans.append((pos,pos+len(word),span));pos+=len(word)
            pos+=1
    stop=start+len(match.group(1))+1
    prefix=[s for a,b,s in spans if a<stop and start<b and s.get('text','').strip()]
    first_body=next((i for i in range(stop,end) if not region['text'][i].isspace()),end)
    following=[s for a,b,s in spans if a<=first_body<b]
    def style(s):
        both=bool(re.search(r'[.\-,]BI$',s.get('font',''),re.I))
        return (both or bool(s.get('flags',0)&16) or bool(BOLD_NAME.search(s.get('font',''))),
                both or bool(s.get('flags',0)&2) or bool(ITALIC_NAME.search(s.get('font',''))))
    if not prefix or not following:return None
    styles={style(s) for s in prefix}
    if len(styles)!=1 or styles=={(False,False)} or style(following[0]) in styles:return None
    return {'title':match.group(1),'role':{'Results':'results','Discussion':'discussion',
        'Method':'methods','Methods':'methods'}[match.group(1)],'start':start,'end':stop,
        'basis':'native_styled_run_in_label'}


def project_wrapped_units(packet, links, unit_order):
    """Move only unassigned units with one witnessed study path, preserving bytes."""
    result=deepcopy(packet);studies={s['study_id']:s for s in result['studies']};moved=[];remaining=[]
    for unit in result['unassigned_source_units']:
        sids={links[n['node_id']]['study_id'] for path in unit.get('section_paths',[]) for n in path
              if n['node_id'] in links}
        if len(sids)!=1 or not unit['source_locator'].startswith('main_') or unit.get('routing_status')!='unassigned':
            remaining.append(unit);continue
        sid=next(iter(sids))
        if sid not in studies:remaining.append(unit);continue
        studies[sid]['units'].append(unit);moved.append({'unit_id':unit['unit_id'],'study_id':sid})
    result['unassigned_source_units']=remaining
    def ids(p):return Counter(u['unit_id'] for s in p['studies'] for u in s['units'])+Counter(u['unit_id'] for u in p['unassigned_source_units'])
    if ids(result)!=ids(packet):raise ValueError('routing_source_loss')
    # Source binding offsets determine order, not order of newly appended units.
    for study in result['studies']:
        study['units'].sort(key=lambda u:unit_order[u['unit_id']])
    result['wrapped_heading_routing']={'witnesses':links,'links':moved,'semantic_ownership_unassessed':True}
    return result
