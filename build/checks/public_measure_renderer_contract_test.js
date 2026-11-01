const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

const html=fs.readFileSync(process.argv[2],'utf8');
const payloadHtml=process.argv[3]?fs.readFileSync(process.argv[3],'utf8'):html;
const start=html.indexOf('function sourceText'), end=html.indexOf('function bind',start);
assert.ok(start>=0&&end>start,'measure renderer seam exists');
const marker='<script id="measure-data" type="application/json">', dataStart=payloadHtml.indexOf(marker)+marker.length, dataEnd=payloadHtml.indexOf('</script>',dataStart), data=payloadHtml.slice(dataStart,dataEnd);
const document={querySelector(selector){
  if(selector==='#measure-data')return {textContent:data};
  if(selector==='#measure-display-keys-v1'||selector==='#measure-bibliography-v1')return {textContent:'{}'};
  if(selector==='#measure-method-exclusions-v1')return {textContent:'[]'};
  return {textContent:'',hidden:false};
}};
const context=vm.createContext({document,URL,localStorage:{getItem(){return null},setItem(){}},navigator:{clipboard:{writeText:async()=>{}}},console});
vm.runInContext(html.slice(start,end)+';globalThis.contract={DATA,itemRows,evidence,versionText};',context);
const version=context.contract.DATA.versions.find(v=>v.id==='public-measure-cbf6f11ca724103822b24d6bb59fd98e7c32e8adafd0b93e41786f04884e0d73');
assert.ok(version,'fictional Team Coordination version present');
const rows=context.contract.itemRows(version);
assert.deepEqual(JSON.parse(JSON.stringify(rows)),[{quote:'Team members coordinate their timing.',pages:[1]},{quote:'Team members clarify shared responsibilities.',pages:[1]}]);
const cards=context.contract.evidence(version,rows,'Captured item wording');
for(const wording of version.items)assert.ok(cards.includes(wording),'item appears in rendered card');
assert.equal((cards.match(/Source · page 1/g)||[]).length,2,'exact evidence pages retained');
const copied=context.contract.versionText(version);
for(const wording of version.items)assert.ok(copied.includes(wording),'item appears in copy/download text');
assert.deepEqual(JSON.parse(JSON.stringify(context.contract.itemRows({items:[{quote:'Legacy object item.',pages:[7]}],item_evidence:[]}))),[{quote:'Legacy object item.',pages:[7]}],'legacy object rows retained');
assert.deepEqual(JSON.parse(JSON.stringify(context.contract.itemRows({items:['Exact item.'],item_evidence:[{quote:'Wrong item.',pages:[99]}]}))),[{quote:'Exact item.'}],'mismatched evidence cannot supply an unrelated locator');
console.log(JSON.stringify({passed:true,fictional_items_visible:true,locators_retained:true,copy_download_text:true,legacy_object_rows:true,mismatched_evidence_unbound:true}));
