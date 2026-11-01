/* Versioned, dependency-free renderer for discovery_display_v1 projections. */
(function(root){
  'use strict';
  const text=value=>String(value??'');
  const clear=node=>{while(node.firstChild)node.removeChild(node.firstChild);};
  const add=(node,tag,value,className)=>{const el=document.createElement(tag);if(className)el.className=className;el.textContent=text(value);node.appendChild(el);return el;};
  function summary(node, message){clear(node);add(node,'p',message,'rp-caption');}
  function renderCitations(node,page){
    clear(node); const fragment=document.createDocumentFragment();
    const items=Array.isArray(page?.items)?page.items:[];
    if(!items.length)add(fragment,'p','No APA-ready citations are available in this local snapshot. Incomplete or unverified bibliography stays in the raw graph and is not shown here.','rp-missing');
    for(const item of items){const article=document.createElement('article'),p=document.createElement('p'),parts=item.render_parts;article.className='cm-row';if(parts){p.append(text(parts.authors_year)+' '+text(parts.title)+text(parts.title_stop)+' ');const journal=document.createElement('i');journal.textContent=text(parts.journal);p.append(journal,', ');const volume=document.createElement('i');volume.textContent=text(parts.volume);p.append(volume);if(parts.issue)p.append('('+text(parts.issue)+')');p.append(', '+text(parts.locator)+'.');if(parts.doi_url){p.append(' ');const a=document.createElement('a'),doi=text(parts.doi_url);a.href=doi.startsWith('http')?doi:'https://doi.org/'+doi;a.rel='noopener';a.textContent=a.href;p.append(a);}}else p.textContent=text(item.citation_apa);article.appendChild(p);fragment.appendChild(article);}
    const deficits=page?.deficits||{}, total=Object.values(deficits).reduce((n,v)=>n+(Number(v)||0),0);
    if(total)add(fragment,'p',total.toLocaleString()+' raw reference records are excluded because required verified bibliographic fields are missing or unsupported.','rp-caption');
    node.appendChild(fragment);
  }
  function renderMeasures(node,page){
    clear(node); const fragment=document.createDocumentFragment();
    const items=Array.isArray(page?.items)?page.items:[];
    if(!items.length)add(fragment,'p','No displayable measure or construct labels are available.','rp-missing');
    for(const item of items){const row=document.createElement('article');row.className='rp-record';add(row,'h2',item.label);add(row,'p',String(item.distinct_article_count)+' distinct article'+(item.distinct_article_count===1?'':'s'),'rp-caption');fragment.appendChild(row);}
    const methods=Number(page?.review?.suppressed_generic_method_label||0),artifacts=Number(page?.review?.suppressed_nonmeasure_label||0);if(methods)add(fragment,'p',methods.toLocaleString()+' exact generic method labels are excluded from measure discovery and retained for review.','rp-caption');if(artifacts)add(fragment,'p',artifacts.toLocaleString()+' obvious symbols, units, placeholders, or file labels are excluded from measure discovery and retained for review.','rp-caption');
    node.appendChild(fragment);
  }
  root.MosaicDiscoveryDisplayV1={renderCitations,renderMeasures,summary};
})(this);
