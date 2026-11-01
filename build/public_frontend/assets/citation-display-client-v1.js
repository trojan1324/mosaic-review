/* Default citation view: no fallback to raw graph labels. */
(() => {
  'use strict';
  let page=1, request=0;
  const $=s=>document.querySelector(s);
  const selected=new URL(location.href).searchParams.get('paper')||new URL(location.href).searchParams.get('work')||'';
  const rawLink=document.querySelector('a[href="reference-network-review.html"]');
  if(selected&&rawLink)rawLink.href='reference-network-review.html?work='+encodeURIComponent(selected);
  async function render(){
    const id=++request;$('#citation-status').textContent='Loading complete citations…';
    try{
      const q=$('#citation-q').value.trim(),r=await fetch('/api/display/v1/citations?'+new URLSearchParams({q,paper:selected,page,page_size:25}));
      if(!r.ok)throw Error('The formatted citation display is unavailable.');
      const d=await r.json();if(id!==request)return;
      MosaicDiscoveryDisplayV1.renderCitations($('#formatted-citations'),d);
      $('#citation-status').textContent=d.total+' complete source-checked journal citations'+(selected?' for the selected source article':'')+'.';
      $('#citation-review-count').textContent=(d.raw_reference_count||0)+' original reference strings are preserved. Current review does not establish full citation coverage.';
      const count=Math.max(1,Math.ceil(d.total/25)),box=$('#citation-pager');box.replaceChildren();
      if(count>1){for(const [label,delta] of [['Previous',-1],['Next',1]]){const b=document.createElement('button');b.textContent=label;b.disabled=delta<0?page===1:page>=count;b.onclick=()=>{page+=delta;render();};box.append(b);}}
    }catch(e){if(id!==request)return;$('#citation-status').textContent=e.message;$('#formatted-citations').replaceChildren();}
  }
  $('#formatted-citation-search').onsubmit=e=>{e.preventDefault();page=1;render();};render();
})();
