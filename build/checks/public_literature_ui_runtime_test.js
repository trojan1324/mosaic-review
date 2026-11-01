const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const offline=process.argv[3]==='offline';
const live=process.argv[4]?JSON.parse(fs.readFileSync(process.argv[4],'utf8')):null;

class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.listeners={};this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.style={};this.parentElement=null;}
  append(...items){for(const item of items){if(typeof item==='string')continue;item.parentElement=this;this.children.push(item);}return this;}
  prepend(item){item.parentElement=this;this.children.unshift(item);}
  replaceChildren(...items){this.children=[];this.append(...items);}
  addEventListener(event,handler){this.listeners[event]=handler;}
  trigger(event){this.listeners[event]?.({preventDefault(){}});}
  setAttribute(){}
  scrollIntoView(){}
  click(){this.onclick?.();}
}
const ids=['literature-status','literature-error','literature-question','literature-keywords','literature-paper','literature-model','literature-generate','literature-preview','literature-results','literature-form','literature-output'];
const nodes=new Map(ids.map(id=>[id,new Element(id)]));
nodes.get('literature-output').value='literature_review';
const requests=[];
const response={status:'ok',retrieved_papers:1,sources:[{citation_id:'S1',source_id:'u1',paper_id:'p1',title:'<img src=x>',citation:'A. Author (2025). Fiction.',parenthetical:'Author, 2025',citation_complete_apa:true,page:2,quote:'<img src=x onerror=alert(1)>',article_url:'javascript:alert(1)'}]};
async function fetch(url,options){
  requests.push({url,options});
  if(url==='/api/literature/status')return {ok:true,json:async()=>({enabled:true,generation_configured:!offline,csrf_token:'token',models:[{id:'gpt-6-sol',label:'GPT-6 Sol',available:!offline}]})};
  if(url.startsWith('/api/display/v1/bibliography'))return {ok:true,json:async()=>({items:[{paper_id:'p1',title_verbatim:'Fictional article'}]})};
  if(url==='/api/literature/preview')return {ok:true,json:async()=>response};
  if(url==='/api/literature/draft'&&live)return {ok:true,json:async()=>live};
  throw Error('unexpected '+url);
}
const document={getElementById:id=>nodes.get(id),createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text})};
const context=vm.createContext({document,fetch,URL,Blob,setTimeout,location:{href:'http://127.0.0.1:8780/assistant.html',origin:'http://127.0.0.1:8780',protocol:'http:'},localStorage:{getItem(){return null},setItem(){}},navigator:{clipboard:{writeText:async()=>{}}},window:{}});
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
(async()=>{await new Promise(resolve=>setImmediate(resolve));await new Promise(resolve=>setImmediate(resolve));assert.equal(nodes.get('literature-generate').disabled,true,'generation requires a preview');nodes.get('literature-question').value='trust';await nodes.get('literature-preview').onclick();const previewRequest=JSON.parse(requests.find(row=>row.url==='/api/literature/preview').options.body);assert.equal(previewRequest.enable_model,false);assert.equal(previewRequest.model,'gpt-6-sol','offline preview has an allowlisted model value');assert.equal(nodes.get('literature-generate').disabled,offline,'only a configured service enables generation after preview');const preview=nodes.get('literature-results').children[0];assert.ok(preview);const find=(node,predicate)=>predicate(node)||node.children.some(child=>find(child,predicate));assert.ok(find(preview,node=>node.textContent==='<img src=x onerror=alert(1)>'),'returned text remains text');assert.ok(!find(preview,node=>node.tagName==='img'),'response did not create markup');if(live){nodes.get('literature-keywords').value='';nodes.get('literature-question').value='compare';await nodes.get('literature-preview').onclick();nodes.get('literature-form').onsubmit({preventDefault(){}});await new Promise(resolve=>setImmediate(resolve));await new Promise(resolve=>setImmediate(resolve));const draft=nodes.get('literature-results').children[0];const all=(node,predicate,out=[])=>{if(predicate(node))out.push(node);for(const child of node.children||[])all(child,predicate,out);return out;};assert.equal(all(draft,node=>node.tagName==='a'&&String(node.href||'').startsWith('#literature-source-')).length,2);assert.equal(all(draft,node=>node.className==='literature-source').length,6);assert.equal(all(draft,node=>node.tagName==='blockquote').length,10);const copied=context.window.MosaicPublicLiterature.draftText(live);assert.equal((copied.match(/Price, 2024/g)||[]).length,1);assert.equal((copied.match(/Singh, 2025/g)||[]).length,1);const same=context.window.MosaicPublicLiterature.citationSources(['x','y'],new Map([['x',{paper_id:'x',parenthetical:'Same, 2024'}],['y',{paper_id:'y',parenthetical:'Same, 2024'}]]));assert.equal(same.length,2);}
nodes.get('literature-keywords').value='changed';nodes.get('literature-keywords').trigger('input');assert.equal(nodes.get('literature-generate').disabled,true,'filter edits invalidate the preview');console.log(JSON.stringify({passed:true,preview_first:true,unsafe_text_not_markup:true,query_change_invalidates:true,offline}));})().catch(error=>{console.error(error.stack||error);process.exitCode=1;});
