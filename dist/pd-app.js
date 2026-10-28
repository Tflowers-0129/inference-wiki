(function(){
  'use strict';
  const $=id=>document.getElementById(id);
  let scenario,index=0,timer=null;
  function stop(){clearInterval(timer);timer=null;$('pdPlay').innerHTML='▶ <span>播放</span>';$('pdPlay').setAttribute('aria-label','自动播放 PD 交接');}
  const sourceText={none:'尚未分配',building:'构建中',held:'交接期间保留',released:'请求引用已归还'};
  const targetText={none:'尚未分配',reserved:'目标块已预留',receiving:'正在写入',ready:'KV 就绪',running:'Decode 使用中',released:'请求引用已归还'};
  function chip(id,text,kind=''){return `<span class="pd-request ${kind}"><b>${id}</b><small>${text}</small></span>`;}
  function sourceBlocks(s){const a=[];for(const [id,key]of[['R1','r1Source'],['R2','r2Source']])if(['building','held'].includes(s[key]))a.push(chip(id,sourceText[s[key]],s[key]==='building'?'building':'hold'));return a.join('')||'<span class="pd-empty">没有请求持有源引用</span>';}
  function targetBlocks(s){const a=[];for(const [id,key]of[['R1','r1Target'],['R2','r2Target']])if(!['none','released'].includes(s[key]))a.push(chip(id,targetText[s[key]],s[key]==='ready'?'ready':'target'));return a.join('')||'<span class="pd-empty">尚无目标 KV 引用</span>';}
  function render(){
    const s=scenario.frames[index];
    $('pdStepValue').textContent=String(index+1).padStart(2,'0');$('pdStepTotal').textContent=`/ ${scenario.frames.length}`;
    $('pdEventTitle').textContent=s.title;$('pdEventNote').textContent=s.note;$('pdScope').textContent=scenario.scope;
    $('pdPQueue').innerHTML=s.pQueue.map(id=>chip(id,'待 Prefill')).join('')||'<span class="pd-empty">队列为空</span>';
    $('pdPQueueCount').textContent=s.pQueue.length;
    $('pdPRun').innerHTML=s.pRun?`<strong>${s.pRun}</strong><span>Prefill</span><i class="pd-work-dots"><b></b><b></b><b></b></i>`:'<span class="pd-run-idle">本事件没有 P 计算</span>';
    $('pdPSource').innerHTML=sourceBlocks(s);
    $('pdDQueue').innerHTML=s.dQueue.map(q=>chip(q.id,q.reason,q.reason.includes('不足')||q.reason==='等目标 KV 分配'?'hold':q.reason.includes('就绪')?'ready':'')).join('')||'<span class="pd-empty">队列为空</span>';
    $('pdDQueueCount').textContent=s.dQueue.length;
    $('pdDRun').innerHTML=s.dRun?`<strong>${s.dRun}</strong><span>Decode</span><i class="pd-work-dots"><b></b><b></b><b></b></i>`:'<span class="pd-run-idle">本事件没有 D 计算</span>';
    $('pdDTarget').innerHTML=targetBlocks(s);
    $('pdKVStatus').textContent=s.kvFull?'R2 目标块不足':'按场景条件可分配';$('pdKVStatus').classList.toggle('is-full',s.kvFull);
    const wires={idle:'本事件没有拉取',pulling:`正在拉取 ${s.wireRequest}`,complete:`${s.wireRequest} 目标 KV 就绪`,blocked:'尚未发起拉取'};
    $('pdWireLabel').textContent=wires[s.wire];$('pdWireFill').style.width=s.progress+'%';$('pdWireProgress').textContent=s.wire==='blocked'?'等目的空间':s.wire==='idle'?'—':`${s.progress}% · 示意`;
    $('pdWire').dataset.state=s.wire;$('pdRoute').textContent=s.route;
    $('pdRequestLocation').textContent=s.focus;
    $('pdR1SourceState').textContent=sourceText[s.r1Source];$('pdR1TargetState').textContent=targetText[s.r1Target];$('pdDoneState').textContent=s.r1Done&&s.r2Done?'2 / 2':s.r1Done?'1 / 2':'0 / 2';
    $('pdRange').value=index+1;$('pdRangeLabel').textContent=`${index+1} / ${scenario.frames.length}`;
    $('pdPrev').disabled=index===0;$('pdNext').disabled=index===scenario.frames.length-1;
    $('pdNormalContrast').hidden=scenario.name!=='kv';
  }
  function load(){stop();scenario=WikiPDPlayback.scenario($('pdScenario').value);index=0;$('pdRange').max=scenario.frames.length;render();}
  function go(n){stop();index=Math.max(0,Math.min(scenario.frames.length-1,n));render();}
  $('pdPrev').addEventListener('click',()=>go(index-1));$('pdNext').addEventListener('click',()=>go(index+1));$('pdReset').addEventListener('click',()=>go(0));
  $('pdScenario').addEventListener('change',load);$('pdRange').addEventListener('input',e=>go(+e.target.value-1));
  document.querySelectorAll('[data-pd-jump]').forEach(b=>b.addEventListener('click',()=>{const key=b.dataset.pdJump;go(scenario.frames.findIndex(s=>s.key===key));}));
  $('pdPlay').addEventListener('click',()=>{if(timer){stop();return;}if(index===scenario.frames.length-1)index=0;$('pdPlay').innerHTML='Ⅱ <span>暂停</span>';$('pdPlay').setAttribute('aria-label','暂停 PD 交接');timer=setInterval(()=>{if(index<scenario.frames.length-1){index++;render();}else stop();},1800);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  if('IntersectionObserver'in window)new IntersectionObserver(e=>{if(!e[0].isIntersecting)stop();}).observe($('pdPlayer'));
  load();
})();
