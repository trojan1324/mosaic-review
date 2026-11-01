/* Conservative research display. Raw evidence and snapshot-scoped selections stay intact. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s);
  const M = JSON.parse($('#manifest').textContent), PAGE = 25;
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const SK = 'mosaic-api-saved/' + M.snapshot_id, CK = 'mosaic-api-compare/' + M.snapshot_id;
  const S = {screen:'papers', q:'', paper:'', page:1, raw:false, saved:new Set(), compare:new Set(), cache:new Map(), bibliography:new Map(), request:0};
  for (const [key, field] of [[SK,'saved'],[CK,'compare']]) {
    try { const ids = JSON.parse(localStorage.getItem(key) || '[]');
      S[field] = new Set(Array.isArray(ids) ? ids.filter(x => typeof x === 'string').slice(0, field === 'compare' ? 2 : undefined) : []);
    } catch (_) { /* A damaged local selection never becomes source evidence. */ }
  }
  async function get(path) {
    const r = await fetch(path); if (!r.ok) throw Error('Local index request failed ('+r.status+').');
    const d = await r.json();
    if ((d.snapshot_id || d.snapshot?.snapshot_id) !== M.snapshot_id) throw Error('Collection identity changed. Reload a matching preview.');
    return d;
  }
  function save() {
    try { localStorage.setItem(SK, JSON.stringify([...S.saved])); localStorage.setItem(CK, JSON.stringify([...S.compare])); }
    catch (_) { $('#status').textContent = 'Browser storage unavailable. Export your selections before closing.'; }
  }
  const displayField = (row, field) => typeof row?.[field+'_display'] === 'string' && row[field+'_display'].trim() ? row[field+'_display'] : row?.[field+'_verbatim'] || '';
  const paperTitle = row => displayField(row, 'title');
  const journal = row => displayField(row, 'journal');
  const title = r => paperTitle(S.bibliography.get(r.paper_id)) || 'Article awaiting bibliographic review';
  function byline(row) {
    const display = row.authors_display;
    const names = Array.isArray(display) && display.length && display.every(x => typeof x === 'string' && x.trim()) ? display : typeof display === 'string' && display.trim() ? display : row.authors_verbatim;
    return Array.isArray(names) ? names.join('; ') : String(names || '');
  }
  function bibLine(row) { return row ? byline(row)+' · '+journal(row)+' · '+row.publication_year : 'Bibliographic identity has not passed source review.'; }
  function sourceLink(r) { return r.source_pdf_sha256 ? `<a href="sources/${encodeURIComponent(r.paper_id)}.pdf${r.page>0?'#page='+r.page:''}" target="_blank" rel="noopener">Open source PDF${r.page>0?' · page '+r.page:''}</a>` : ''; }
  function card(r) {
    S.cache.set(r.id, r);
    if (r.opening_image && S.screen==='papers' && !S.raw) {
      const image=r.opening_image;
      const parts=image.parts.map(part=>{
        S.cache.set(part.id,{...part,display_quote:part.quote,display_start:0,display_end:part.quote.length});
        return `<p class="quiet">${esc(part.id)}</p><blockquote>${esc(part.quote)}</blockquote><div class="actions"><label><input data-save="${esc(part.id)}" type="checkbox" ${S.saved.has(part.id)?'checked':''}> Keep this capture in Saved</label><button data-compare="${esc(part.id)}" type="button">${S.compare.has(part.id)?'Remove from comparison':'Compare this capture'}</button></div>`;
      }).join('');
      return `<article class="record" data-passage-id="${esc(r.id)}"><h3>${esc(title(r))}</h3><p class="quiet">${esc(bibLine(S.bibliography.get(r.paper_id)))} · page ${esc(r.page)}</p><figure><img src="${esc(image.url)}" alt="Original source PDF article opening, page ${esc(image.page)}" style="display:block;max-width:100%;height:auto"><figcaption class="quiet">Original PDF opening · wording shown directly from the source.</figcaption></figure><details><summary>Inspect original text captures (${image.parts.length})</summary>${parts}</details><div class="actions">${sourceLink(r)}</div></article>`;
    }
    const shown = typeof r.display_quote === 'string' ? r.display_quote : r.quote;
    return `<article class="record" data-passage-id="${esc(r.id)}"><h3>${esc(title(r))}</h3><p class="quiet">${esc(bibLine(S.bibliography.get(r.paper_id)))} · page ${esc(r.page)}</p><blockquote>${esc(typeof sourceText === 'function' ? sourceText(shown) : shown)}</blockquote><p class="quiet">${S.screen==='papers'&&!S.raw?'Captured article text · source wording preserved.':'Preserved source passage · may include material excluded from the default reader.'}</p><div class="actions">${sourceLink(r)}<label><input data-save="${esc(r.id)}" type="checkbox" ${S.saved.has(r.id)?'checked':''}> Keep in Saved</label><button data-compare="${esc(r.id)}" type="button">${S.compare.has(r.id)?'Remove from comparison':'Compare this passage'}</button></div></article>`;
  }
  function passageCards(rows) {
    const seen=new Set();
    return rows.filter(row=>{
      if (!row.opening_image || S.raw) return true;
      const key=row.paper_id+'/'+row.opening_image.sha256;
      if(seen.has(key)) return false;
      seen.add(key); return true;
    }).map(card).join('');
  }
  function paperRows(rows) {
    rows.forEach(r => S.bibliography.set(r.paper_id, r));
    return rows.map(r => `<button type="button" class="paper" data-title-paper="${esc(r.paper_id)}"><strong>${esc(paperTitle(r))}</strong><span class="bibliography-authors">${esc(byline(r))}</span><span class="bibliography-journal">${esc(journal(r))} · ${esc(r.publication_year)}</span><small>Browse article passages →</small></button>`).join('');
  }
  function pagination(total) {
    const n = Math.max(1, Math.ceil(total/PAGE));
    return n > 1 ? `<div class="actions"><button data-page="prev" ${S.page<=1?'disabled':''}>Previous</button><span>Page ${S.page} of ${n}</span><button data-page="next" ${S.page>=n?'disabled':''}>Next</button></div>` : '';
  }
  function reviewNote(b, p) {
    const count = b.unresolved_count ?? b.unresolved_total ?? 0;
    const exclusions = p?.excluded || {};
    const reasons = Object.entries(exclusions).map(([k,v]) => `${String(k).replace(/_/g,' ')}: ${v}`).join('; ');
    return `<details class="display-review"><summary>Display review and exclusions</summary><p>${esc(count)} article identities are awaiting bibliographic review. Missing values are never inferred from a filename or DOI.</p>${reasons?`<p>Page blocks withheld: ${esc(reasons)}.</p>`:''}<p>Raw page blocks, original extraction outputs and reference strings are preserved. A clean display is not scientific validation.</p>${S.paper?'<label><input id="raw-blocks" type="checkbox" '+(S.raw?'checked':'')+'> Inspect all raw page blocks for this article</label>':''}</details>`;
  }
  async function renderPapers(persist = true) {
    if (persist) writeLocation();
    const token = ++S.request;
    $('#result-status').textContent = 'Loading reviewed display…';
    try {
      const params = new URLSearchParams({q:S.q, paper:S.paper, page:S.page, page_size:PAGE});
      const [b, p] = await Promise.all([
        get('/api/display/v1/bibliography?'+params),
        S.q || S.paper ? get((S.raw?'/api/search?':'/api/display/v1/passages?')+params) : Promise.resolve(null)
      ]);
      if (token !== S.request) return;
      const rows = b.items || [], passages = p?.passages || p?.passage_hits || [], total = p?.total_passage_hits ?? p?.total ?? 0;
      const bibliography = paperRows(rows);
      const sourceActions=S.paper?`<div class="actions"><a href="sources/${encodeURIComponent(S.paper)}.pdf" target="_blank" rel="noopener">Open source PDF</a><a href="citation-map.html?paper=${encodeURIComponent(S.paper)}">Source-checked citations</a></div>`:'';
      $('#result-status').textContent = !p ? `${b.total} source-checked article identities. Titles, authors, journal and year are shown in full.` : `${b.total} matching article ${b.total===1?'identity':'identities'} · ${total} ${S.raw?'raw page blocks':'article-content passages'}${S.paper?' within this article':''}.`;
      $('#results').innerHTML = (bibliography?'<h2>Articles</h2>'+bibliography:!p?'<p>No article identities have passed the current source review. The review count below records the outstanding work.</p>':'')+sourceActions+
        (!p?pagination(b.total):'<h2>Source passages</h2>'+pagination(total)+passageCards(passages)+(!total?'<p>No passages meet the current display boundary checks. Open the source PDF or inspect raw blocks to review this article.</p>':'')+pagination(total))+reviewNote(b,p);
      bind();
    } catch(e) {
      if (token !== S.request) return;
      $('#result-status').textContent = e.message;
      $('#results').innerHTML = '<p class="error">The conservative display is unavailable. No raw-data fallback is substituted.</p>';
    }
  }
  async function hydrate(ids) {
    await Promise.allSettled([...ids].filter(id => !S.cache.has(id)).map(async id => {
      const r = await get('/api/passage?id='+encodeURIComponent(id));
      if (r.id !== id) throw Error('Source identity mismatch.');
      S.cache.set(id,r);
    }));
    return [...ids].filter(id => !S.cache.has(id));
  }
  async function renderSaved() {
    const missing = await hydrate(S.saved);
    $('#saved-count').textContent = `${S.saved.size} saved passages.${missing.length?' '+missing.length+' unavailable; selections retained.':''}`;
    $('#retry-saved').hidden = !missing.length;
    $('#export-json').disabled = $('#export-csv').disabled = !S.saved.size || !!missing.length;
    $('#clear-saved').hidden = !S.saved.size;
    $('#saved-list').innerHTML = [...S.saved].map(x => S.cache.get(x)).filter(Boolean).map(card).join(''); bind();
  }
  async function renderCompare() {
    const missing = await hydrate(S.compare);
    $('#compare-status').textContent = missing.length ? `${missing.length} passages unavailable; selections retained.` : '';
    $('#retry-compare').hidden = !missing.length;
    $('#compare-list').innerHTML = [...S.compare].map(x => S.cache.get(x)).filter(Boolean).map(card).join('') || '<p>Select two passages from Papers.</p>'; bind();
  }
  function bind() {
    document.querySelectorAll('[data-title-paper]').forEach(x => x.onclick = () => {
      S.paper=x.dataset.titlePaper; S.q=''; S.page=1; S.raw=false; $('#paper-query').value=''; selectedFilter(); renderPapers();
    });
    document.querySelectorAll('[data-page]').forEach(x => x.onclick = () => { S.page += x.dataset.page==='next'?1:-1; renderPapers(); });
    const raw = $('#raw-blocks'); if (raw) raw.onchange = () => { S.raw=raw.checked; S.page=1; renderPapers(); };
    document.querySelectorAll('[data-save]').forEach(x => x.onchange = () => {
      x.checked ? S.saved.add(x.dataset.save) : S.saved.delete(x.dataset.save); save(); renderSaved();
      document.querySelectorAll('[data-save]').forEach(y => { y.checked=S.saved.has(y.dataset.save); });
    });
    document.querySelectorAll('[data-compare]').forEach(x => x.onclick = () => {
      const id=x.dataset.compare;
      if(S.compare.has(id)) S.compare.delete(id);
      else if(S.compare.size<2) S.compare.add(id);
      else { $('#status').textContent='Remove a passage from Compare before adding another.'; return; }
      save(); renderCompare(); document.querySelectorAll('[data-compare]').forEach(y => { y.textContent=S.compare.has(y.dataset.compare)?'Remove from comparison':'Compare this passage'; });
    });
  }
  function selectedFilter() {
    const filter=$('#paper-filter');
    if(S.paper && ![...filter.options].some(x => x.value===S.paper)) {
      const option=document.createElement('option'); option.value=S.paper; option.textContent=paperTitle(S.bibliography.get(S.paper)) || 'Selected article · identity awaiting review'; filter.append(option);
    }
    filter.value=S.paper; $('#clear-filter').hidden=!S.paper;
  }
  let lastLocation='';
  function writeLocation() {
    const u=new URL(location.href); u.search='';
    if(S.q) u.searchParams.set('q',S.q); if(S.paper) u.searchParams.set('paper',S.paper);
    if(S.page>1) u.searchParams.set('page',String(S.page)); if(S.raw)u.searchParams.set('raw','1');
    u.hash=S.screen==='start'?'':S.screen;
    if(u.href!==location.href)history.pushState(null,'',u.href); lastLocation=location.href;
  }
  function screen(n,push=true) {
    S.screen=['start','papers','saved','compare','coverage'].includes(n)?n:'papers';
    document.querySelectorAll('[data-view]').forEach(x => { x.hidden=x.dataset.view!==S.screen; });
    document.querySelectorAll('[data-screen]').forEach(x => x.setAttribute('aria-current',x.dataset.screen===S.screen?'page':'false'));
    if(push)writeLocation();
    if(S.screen==='papers')renderPapers(false); if(S.screen==='saved')renderSaved(); if(S.screen==='compare')renderCompare();
  }
  function readLocation() {
    if(lastLocation===location.href)return; lastLocation=location.href;
    const u=new URL(location.href); S.q=u.searchParams.get('q')||''; S.paper=u.searchParams.get('paper')||'';
    const n=Number(u.searchParams.get('page')); S.page=Number.isSafeInteger(n)&&n>0?n:1; S.raw=u.searchParams.get('raw')==='1';
    $('#paper-query').value=S.q; $('#query').value=S.q; selectedFilter(); screen(u.hash.slice(1)||'papers',false);
  }
  function csv(v) { v=String(v??''); if(/^[=+@-]/.test(v.trimStart()))v="'"+v; return '"'+v.replace(/"/g,'""')+'"'; }
  function download(name,text,type) { const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([text],{type})); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),500); }
  async function exports(kind) {
    if((await hydrate(S.saved)).length){renderSaved();return;}
    const rows=[...S.saved].map(x=>S.cache.get(x));
    if(kind==='json')download('mosaic-saved-evidence.json',JSON.stringify({schema_version:'mosaic-saved-evidence/v1',snapshot_id:M.snapshot_id,scientifically_accepted:false,records:rows},null,2),'application/json');
    else download('mosaic-saved-source-list.csv',[['Article title','Bibliographic identity','Page','Exact source quote','Source PDF link','Status'],...rows.map(r=>[title(r),bibLine(S.bibliography.get(r.paper_id)),r.page,r.quote,r.source_pdf_sha256?new URL('sources/'+encodeURIComponent(r.paper_id)+'.pdf'+(r.page>0?'#page='+r.page:''),location.href).href:'','Unqualified source passage'])].map(row=>row.map(csv).join(',')).join('\r\n'),'text/csv;charset=utf-8');
  }
  document.querySelectorAll('[data-screen]').forEach(x=>x.onclick=()=>screen(x.dataset.screen));
  $('#search-form').onsubmit=e=>{e.preventDefault();S.q=$('#paper-query').value.trim();S.page=1;renderPapers();};
  $('#start-form').onsubmit=e=>{e.preventDefault();S.q=$('#query').value.trim();S.paper='';S.page=1;S.raw=false;$('#paper-query').value=S.q;selectedFilter();screen('papers');};
  $('#paper-filter').onchange=()=>{S.paper=$('#paper-filter').value;S.page=1;S.raw=false;selectedFilter();renderPapers();};
  $('#clear-filter').onclick=()=>{S.paper='';S.page=1;S.raw=false;selectedFilter();renderPapers();};
  $('#retry-saved').onclick=renderSaved; $('#retry-compare').onclick=renderCompare;
  $('#clear-saved').onclick=()=>{S.saved.clear();save();renderSaved();bind();};
  $('#export-json').onclick=()=>exports('json'); $('#export-csv').onclick=()=>exports('csv');
  window.addEventListener('hashchange',readLocation); window.addEventListener('popstate',readLocation);
  (async()=>{
    try {
      const h=await get('/api/health'); $('#scope').textContent='Source-checked bibliography and article-content display candidates.';
      $('#coverage').textContent=`${h.snapshot.paper_count} stored identities; ${h.snapshot.source_passage_count} immutable raw page blocks. Display exclusions do not erase these records.`;
      $('#status').textContent='Conservative display · source checks do not establish scientific validity';
      let page=1,done=false;
      while(!done){
        const b=await get('/api/display/v1/bibliography?page_size=100&page='+page);
        for(const r of b.items){
          S.bibliography.set(r.paper_id,r);
          const filter=$('#paper-filter'),existing=[...filter.options].find(o=>o.value===r.paper_id);
          const o=existing||document.createElement('option');o.value=r.paper_id;o.textContent=paperTitle(r);
          if(!existing)filter.append(o);
        }
        done=page*100>=b.total;page++;
      }
      selectedFilter();
      if(S.screen==='saved')renderSaved();if(S.screen==='compare')renderCompare();
    } catch(e){$('#status').textContent=e.message;}
  })();
  readLocation();
})();
