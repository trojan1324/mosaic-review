(function(root,factory){const api=factory();if(typeof module==='object')module.exports=api;else root.CitationMapCore=api})(this,function(){
  const query=x=>new URLSearchParams(x).toString(),items=x=>Array.isArray(x)?x:(x?.items||[]);
  function normalizeWork(r){const a=r.references||{},b=r.incoming||{};return {work:r.work||r,references:items(a),references_total:a.total??items(a).length,references_cursor:a.next_cursor??null,incoming:items(b),incoming_total:b.total??items(b).length,incoming_cursor:b.next_cursor??null,coupling:r.bibliographic_coupling||[],co_citation:r.co_citation||[],bridge_candidates:r.bridge_candidates||[],co_citation_scan_truncated:!!r.co_citation_scan_truncated}}
  function appendPage(state,raw,kind){if(!['references','incoming'].includes(kind))throw Error('Unknown citation page kind');const next=normalizeWork(raw);if(next.work.work_id!==state.work.work_id)throw Error('Citation page identity changed');const seen=new Set(state[kind].map(x=>x.reference_id));state[kind].push(...next[kind].filter(x=>!seen.has(x.reference_id)));state[kind+'_cursor']=next[kind+'_cursor'];state[kind+'_total']=next[kind+'_total'];return state}
  function syntheticResponse(data,url){
    if(data?.manifest?.synthetic!==true)throw Error('Synthetic fixture manifest required');
    const u=new URL(url,'http://localhost'),q=u.searchParams;
    if(u.pathname.endsWith('/health'))return {status:'ready',coverage_scope:'fictional demonstration only',...data.health};
    if(u.pathname.endsWith('/search')){const text=(q.get('q')||'').toLowerCase(),start=Number(q.get('cursor')||0),limit=20;const rows=data.works.filter(w=>((w.title||'')+' '+(w.doi||'')).toLowerCase().includes(text));return {items:rows.slice(start,start+limit),total:rows.length,next_cursor:start+limit<rows.length?start+limit:null}}
    if(u.pathname.endsWith('/references')){const text=(q.get('q')||'').toLowerCase(),start=Number(q.get('cursor')||0),limit=20;const rows=Object.values(data.details).flatMap(d=>(d.references?.items||[]).map(r=>({...r,citing_work:d.work}))).filter(r=>r.literal_reference.toLowerCase().includes(text));return {items:rows.slice(start,start+limit),total:rows.length,next_cursor:start+limit<rows.length?start+limit:null}}
    const work=data.details[q.get('work_id')];if(!work)throw Error('Unknown synthetic work');return work;
  }
  return {normalizeWork,appendPage,syntheticResponse,searchUrl:(s,c,mode='works')=>'/api/citations/'+(mode==='references'?'references':'search')+'?'+query({q:s,...(c!==undefined&&c!==null?{cursor:c}:{})}),workUrl:(id,c)=>'/api/citations/work?'+query({work_id:id,...(c!==undefined&&c!==null?{cursor:c}:{})})};
});
