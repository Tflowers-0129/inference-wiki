(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const fmt = n => n.toLocaleString('en-US');
  let model, current = 0, playTimer = null;
  const outputSelect = $('outputSelect');
  const stop = () => {clearInterval(playTimer); playTimer=null; $('playSteps').innerHTML='▶ <span>播放</span>'; $('playSteps').setAttribute('aria-label','自动播放调度');};
  function insight(s) {
    if (s.number===1) return '还没有 Decode。两条完整 Prefill 各占 4,096 tokens，刚好用满预算；本轮完成后，各请求已可得到首个输出 token。';
    if (s.number===2) return '先安排 2 条 Decode，仅用 2 tokens。剩余 8,190：R3 完整处理 4,096，R4 处理 4,094，还差 2 tokens，留到下一轮。';
    if (s.number===3) return '3 条 Decode 之后，先补完 R4 的 2-token 尾块，再放入完整 R5 和 R6 的分块。同一轮会接触 3 条 P，但总 token 预算不变。';
    if (s.complete===model.total) return '这批请求已全部完成。它是单批请求的排空终点；若持续补入请求，调度又会面对新的 P/D 混合，实际稳态需要另行验证。';
    if (!s.prefill && !s.waiting) return `所有输入已处理完，本轮只安排 ${s.decode} 条 Decode，使用 ${fmt(s.used)} / ${fmt(model.budget)} tokens。预算是上限，无需填满；token 用量也不能直接代表耗时或算力利用率。`;
    if (s.number===model.peakStep && +outputSelect.value===64) return `短输出时，早到请求已经开始完成。这个模型的运行序列数峰值为 ${model.peak}，没有达到 200；ms 只是上限，并不保证填满。`;
    if (s.active===model.maxSeqs) return '轮末运行序列达到 200。本批请求的 Prefill 将很快处理完，随后主要是 Decode；长输出下，这个纯 D 窗口可能持续很多轮。';
    if (s.finishedNow && s.prefill) return `本轮 ${s.finishedNow} 条请求结束，同时还有 Prefill 在推进。短输出可能让解码生命周期与混合填充期大量重叠；长轮次会影响 token 间隔，但这里没有模拟实际耗时。`;
    if (s.chunks.some(c=>c.continued)) return '优先给已有 Decode 分配 token，再续接上轮未完成的 Prefill，最后接纳新请求。分块的尾部可能很小，因此一轮可能出现多条 P。';
    return '每轮都重新核算 Decode、序列数和剩余 token 预算。当前有新的 Prefill 入场，完成 Prefill 的请求从下一轮参与 Decode。';
  }
  function render() {
    const s=model.steps[current];
    $('stepValue').textContent=String(s.number).padStart(2,'0');
    $('decodeValue').innerHTML=`${s.decode} <small>条</small>`;
    $('prefillValue').innerHTML=`${fmt(s.prefill)} <small>tokens</small>`;
    $('activeValue').innerHTML=`${s.active} <small>/ ${model.maxSeqs}</small>`;
    $('usedValue').textContent=`${fmt(s.used)} / ${fmt(model.budget)}`;
    const segs=[];
    if (s.decode) segs.push({tokens:s.decode,type:'d',label:`D × ${s.decode}`,title:`${s.decode} 条 Decode：${s.decode} tokens`});
    for (const c of s.chunks) segs.push({tokens:c.tokens,type:c.continued?'remainder':'p',label:`R${c.id} · ${fmt(c.tokens)}`,title:`R${c.id} Prefill：${fmt(c.tokens)} tokens${c.left?`，剩余 ${c.left}`:'，完成'}`});
    if(s.free)segs.push({tokens:s.free,type:'empty',label:'未用预算',title:`未用预算：${fmt(s.free)} tokens`});
    $('budgetTrack').innerHTML=segs.map(g=>`<span class="budget-segment ${g.type}" style="width:${100*g.tokens/model.budget}%" title="${g.title}">${g.tokens/model.budget>.12?g.label:''}</span>`).join('');
    $('budgetTrack').setAttribute('aria-label',segs.map(g=>g.title).join('；'));
    const rows=[];
    if(s.decode)rows.push(`<div class="detail-row"><span class="detail-kind d">D</span><strong>${s.decode} 条序列各推进 1 token</strong><code>${s.decode} tokens</code></div>`);
    for(const c of s.chunks)rows.push(`<div class="detail-row"><span class="detail-kind">P</span><strong>R${c.id} ${c.continued?'续接输入':'处理输入'}</strong><span>${c.complete?'→ Prefill 完成':'分块进行中'}</span>${c.left?`<span class="remain-note">还差 ${fmt(c.left)}</span>`:''}<code>${fmt(c.tokens)} tokens</code></div>`);
    if(s.free)rows.push(`<div class="detail-row"><span class="detail-kind empty">—</span><span>未使用预算</span><code>${fmt(s.free)} tokens</code></div>`);
    rows.push(`<div class="detail-row"><span class="detail-kind empty">Σ</span><span>尚未入场 ${s.waiting} · 正在 Prefill ${s.prefilling} · 已完成 ${s.complete}</span></div>`);
    $('scheduleDetail').innerHTML=rows.join('');
    $('stepInsight').textContent=insight(s);
    $('stepRange').value=s.number;
    $('rangeLabel').textContent=`${s.number} / ${model.steps.length}`;
    $('prevStep').disabled=current===0;
    $('nextStep').disabled=current===model.steps.length-1;
    $('jumpPeak').textContent=+outputSelect.value===64?'运行数峰值':'首次满载附近';
    $('jumpPeak').title=`跳至第 ${model.peakStep} 轮`;
  }
  function loadModel() {
    stop();model=WikiScheduler.simulate({output:+outputSelect.value});current=0;
    $('stepRange').max=model.steps.length;
    render();
  }
  function gotoStep(number) {stop();current=Math.max(0,Math.min(model.steps.length-1,number-1));render();}
  $('prevStep').addEventListener('click',()=>gotoStep(current));
  $('nextStep').addEventListener('click',()=>gotoStep(current+2));
  $('resetStep').addEventListener('click',()=>gotoStep(1));
  $('stepRange').addEventListener('input',e=>gotoStep(+e.target.value));
  outputSelect.addEventListener('change',loadModel);
  document.querySelectorAll('[data-jump]').forEach(b=>b.addEventListener('click',()=>gotoStep(+b.dataset.jump)));
  $('jumpPeak').addEventListener('click',()=>gotoStep(model.peakStep));
  $('playSteps').addEventListener('click',()=>{
    if(playTimer){stop();return;}
    if(current===model.steps.length-1)current=0;
    $('playSteps').innerHTML='Ⅱ <span>暂停</span>';$('playSteps').setAttribute('aria-label','暂停自动播放');
    playTimer=setInterval(()=>{if(current<model.steps.length-1){current++;render();}else stop();},1500);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  if('IntersectionObserver' in window){new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();},{threshold:0}).observe($('simulator'));}
  loadModel();

  const sections=Array.from(document.querySelectorAll('.observed'));
  const navLinks=Array.from(document.querySelectorAll('.main-nav a,.page-toc a'));
  let scrollQueued=false;
  function updateScroll(){
    const denom=document.documentElement.scrollHeight-window.innerHeight;
    $('readingProgress').style.width=`${denom>0?window.scrollY/denom*100:0}%`;
    let active=sections[0].id;
    for(const section of sections){if(section.getBoundingClientRect().top<160)active=section.id;}
    navLinks.forEach(a=>{const on=a.hash===`#${active}`;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});
    scrollQueued=false;
  }
  window.addEventListener('scroll',()=>{if(!scrollQueued){scrollQueued=true;requestAnimationFrame(updateScroll);}},{passive:true});
  window.addEventListener('resize',updateScroll);updateScroll();
  $('menuToggle').addEventListener('click',()=>{const on=$('sidebar').classList.toggle('open');$('menuToggle').setAttribute('aria-expanded',String(on));$('menuToggle').setAttribute('aria-label',on?'收起目录':'展开目录');});
  document.addEventListener('click',e=>{if(!$('sidebar').contains(e.target)&&!$('menuToggle').contains(e.target)){$('sidebar').classList.remove('open');$('menuToggle').setAttribute('aria-expanded','false');$('menuToggle').setAttribute('aria-label','展开目录');}});
  document.querySelectorAll('.main-nav a').forEach(a=>a.addEventListener('click',()=>{$('sidebar').classList.remove('open');$('menuToggle').setAttribute('aria-expanded','false');$('menuToggle').setAttribute('aria-label','展开目录');}));

  const searchDialog=$('searchDialog');
  const searchIndex=sections.map((section,i)=>({id:section.id,title:section.dataset.title,body:section.textContent.replace(/\s+/g,' ').trim(),keywords:section.dataset.searchText||'',index:i}));
  function snippet(item,q){
    if(!q)return item.keywords.slice(0,60);
    const position=item.body.toLocaleLowerCase().indexOf(q.toLocaleLowerCase());
    return position<0?item.keywords:`${position>25?'…':''}${item.body.slice(Math.max(0,position-25),position+95)}…`;
  }
  function renderSearch(){
    const q=$('searchInput').value.trim();
    const terms=q.toLocaleLowerCase().split(/\s+/).filter(Boolean);
    const results=searchIndex.filter(item=>terms.every(t=>(item.title+' '+item.keywords+' '+item.body).toLocaleLowerCase().includes(t)));
    $('searchResults').replaceChildren();
    if(!results.length){const p=document.createElement('p');p.className='no-results';p.textContent='没有匹配的知识点，试试「TPOT」「稳态」或「KV」。';$('searchResults').append(p);return;}
    for(const item of results){
      const a=document.createElement('a');a.href=`#${item.id}`;
      const index=document.createElement('span');index.className='search-result-index';index.textContent=String(item.index).padStart(2,'0');
      const text=document.createElement('span');const title=document.createElement('strong');title.textContent=item.title;const sub=document.createElement('small');sub.textContent=snippet(item,q);text.append(title,sub);a.append(index,text);
      a.addEventListener('click',()=>searchDialog.close());$('searchResults').append(a);
    }
  }
  function openSearch(){stop();searchDialog.showModal();$('searchInput').value='';renderSearch();$('searchInput').focus();}
  document.querySelectorAll('[data-search]').forEach(b=>b.addEventListener('click',openSearch));
  $('closeSearch').addEventListener('click',()=>searchDialog.close());
  searchDialog.addEventListener('click',e=>{if(e.target===searchDialog){const r=searchDialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)searchDialog.close();}});
  $('searchInput').addEventListener('input',renderSearch);
  $('searchInput').addEventListener('keydown',e=>{if(e.key==='Enter'){const first=$('searchResults').querySelector('a');if(first){e.preventDefault();first.click();}}});
  document.addEventListener('keydown',e=>{const tag=document.activeElement?.tagName;const typing=['INPUT','TEXTAREA','SELECT'].includes(tag);if((e.key.toLowerCase()==='k'&&(e.metaKey||e.ctrlKey))||(e.key==='/'&&!typing)){e.preventDefault();if(!searchDialog.open)openSearch();}});

  document.querySelectorAll('.quiz-card').forEach(card=>{
    card.querySelectorAll('[data-choice]').forEach(button=>button.addEventListener('click',()=>{
      card.querySelectorAll('[data-choice]').forEach(b=>{b.classList.remove('selected-correct','selected-wrong');b.setAttribute('aria-pressed','false');});
      const right=button.dataset.choice===card.dataset.answer;
      button.classList.add(right?'selected-correct':'selected-wrong');button.setAttribute('aria-pressed','true');
      const feedback=card.querySelector('.quiz-feedback');feedback.hidden=false;feedback.setAttribute('role','status');
      feedback.dataset.result=right?'correct':'incorrect';
      if(!feedback.dataset.original)feedback.dataset.original=feedback.textContent;
      feedback.textContent=(right?'判断正确。':'再检查一下预算和前提。')+feedback.dataset.original;
    }));
  });
})();
