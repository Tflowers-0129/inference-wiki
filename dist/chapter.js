(function(){
  'use strict';
  const $=id=>document.getElementById(id), sections=[...document.querySelectorAll('article section[id]')];
  const menu=$('menuToggle'), sidebar=$('sidebar'), dialog=$('searchDialog');
  const closeMenu=()=>{sidebar.classList.remove('open');menu.setAttribute('aria-expanded','false');menu.setAttribute('aria-label','展开目录');};
  menu.addEventListener('click',()=>{const open=sidebar.classList.toggle('open');menu.setAttribute('aria-expanded',String(open));menu.setAttribute('aria-label',open?'收起目录':'展开目录');});
  document.querySelectorAll('.main-nav a').forEach(a=>a.addEventListener('click',closeMenu));
  document.addEventListener('click',e=>{if(!sidebar.contains(e.target)&&!menu.contains(e.target))closeMenu();});
  const index=sections.map(s=>({id:s.id,title:s.dataset.title||s.querySelector('h2,h3')?.textContent||s.id,body:s.textContent.replace(/\s+/g,' ')}));
  function renderSearch(){
    const words=$('searchInput').value.trim().toLowerCase().split(/\s+/).filter(Boolean), out=$('searchResults');out.replaceChildren();
    index.filter(s=>words.every(w=>(s.title+' '+s.body).toLowerCase().includes(w))).forEach(s=>{const a=document.createElement('a');a.href='#'+s.id;const text=document.createElement('strong');text.textContent=s.title;a.append(text);a.addEventListener('click',()=>dialog.close());out.append(a);});
    if(!out.children.length){const p=document.createElement('p');p.className='no-results';p.textContent='本章没有匹配内容。';out.append(p);}
  }
  function openSearch(){closeMenu();renderSearch();dialog.showModal();$('searchInput').focus();}
  document.querySelectorAll('[data-search]').forEach(b=>b.addEventListener('click',openSearch));
  $('closeSearch').addEventListener('click',()=>dialog.close());$('searchInput').addEventListener('input',renderSearch);
  $('searchInput').addEventListener('keydown',e=>{if(e.key==='Enter'){const a=$('searchResults').querySelector('a');if(a){e.preventDefault();a.click();}}});
  document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='k'&&(e.metaKey||e.ctrlKey)){e.preventDefault();openSearch();}});
  let queued=false;
  function update(){const total=document.documentElement.scrollHeight-innerHeight;$('readingProgress').style.width=(total>0?scrollY/total*100:0)+'%';let active=sections[0]?.id;sections.forEach(s=>{if(s.getBoundingClientRect().top<160)active=s.id;});document.querySelectorAll('.chapter-nav a,.page-toc a').forEach(a=>{const on=a.hash==='#'+active;a.classList.toggle('active',on);if(on)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});queued=false;}
  addEventListener('scroll',()=>{if(!queued){queued=true;requestAnimationFrame(update);}},{passive:true});addEventListener('resize',update);update();
})();
