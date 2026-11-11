(function () {
  'use strict';

  const player = document.getElementById('kv-decode-player');
  const { MAX_STEP, stateAt } = window.KVDecodeModel;
  const delay = 2200;
  const steps = [
    ['已有两页历史', 'a8 是本轮输入。位置 0～7 的 K/V 已在块 7 和块 2。'],
    ['预留物理块 9', '块表为逻辑块 2 补上块号 9。位置 8 对应新页第 0 槽，扁平槽号为 36。'],
    ['写入 a8 的 K/V', '第 0 槽写入后，有效长度为 9。块 9 的其余三槽还没有这条请求的 KV。'],
    ['按块表读取位置 0～3', '逻辑块 0 指向物理块 7。页中的四行都有效。'],
    ['按块表读取位置 4～7', '逻辑块 1 指向物理块 2。读取范围推进到位置 7。'],
    ['按块表读取位置 8', '逻辑块 2 指向物理块 9。有效长度为 9，尾页只用第 0 槽。'],
    ['生成 Attention 输出', 'Q(a8) 与九行 K 的分数经 softmax 得到权重，再用权重汇总对应的 V。']
  ];

  const field = name => player.querySelector(`[data-field="${name}"]`);
  const region = name => player.querySelector(`[data-region="${name}"]`);
  const control = name => player.querySelector(`[data-control="${name}"]`);
  const previous = control('prev');
  const next = control('next');
  const play = control('play');
  const seek = control('seek');
  const finalState = stateAt(MAX_STEP);
  const physicalOrder = finalState.pages.map(page => page.physicalBlock).sort((a, b) => a - b);

  region('table').innerHTML = finalState.blockTable.map((_, index) =>
    `<div class="kv-decode-entry" role="listitem" data-logical="${index}">
       <span>逻辑块 ${index}</span><b></b>
     </div>`
  ).join('');

  region('routes').innerHTML = finalState.blockTable.map((block, index) => {
    const x = 50 + index * 100;
    const target = 50 + physicalOrder.indexOf(block) * 100;
    return `<g data-route="${index}">
      <path d="M${x} 0 C${x} 19 ${target} 22 ${target} 38"/>
      <path d="M${target - 2.2} 36 L${target} 42 L${target + 2.2} 36"/>
    </g>`;
  }).join('');

  region('pages').innerHTML = physicalOrder.map(block =>
    `<div class="kv-decode-page" role="listitem" data-page="${block}">
       <div class="kv-decode-page-label">物理块 <b>${block}</b></div>
       <ol class="kv-decode-slots" aria-label="块 ${block} 的槽位">
         ${Array.from({ length: finalState.blockSize }, () => '<li data-slot><span></span></li>').join('')}
       </ol>
       <div class="kv-decode-page-state"></div>
     </div>`
  ).join('');

  const entries = [...region('table').children];
  const routes = [...region('routes').children];
  const pageViews = new Map([...region('pages').children].map(view => [
    Number(view.dataset.page),
    {
      view,
      slots: [...view.querySelectorAll('[data-slot]')],
      status: view.querySelector('.kv-decode-page-state')
    }
  ]));

  let step = 0;
  let timer = null;

  function draw() {
    const state = stateAt(step);
    const writing = step === 1 || step === 2;
    const readPositions = new Set(state.readPositions);
    const writeLogical = Math.floor(state.position / state.blockSize);
    const lastRead = state.readPositions.at(-1);
    const activeLogical = writing ? writeLogical
      : state.outputReady || lastRead === undefined ? -1 : Math.floor(lastRead / state.blockSize);
    const activeBlock = state.blockTable[activeLogical];

    player.dataset.step = String(step);
    player.dataset.outputReady = String(state.outputReady);
    field('step-title').textContent = steps[step][0];
    field('step-note').textContent = steps[step][1];
    field('progress').textContent = `${step + 1} / ${MAX_STEP + 1}`;
    field('write-slot').textContent = state.writeSlot ?? '待定位';
    field('kv-length').textContent = state.pages.reduce(
      (count, page) => count + page.slots.filter(position => position !== null).length, 0
    );
    field('read-count').textContent = readPositions.size;

    entries.forEach((entry, index) => {
      const block = state.blockTable[index];
      entry.querySelector('b').textContent = block === undefined ? '空' : '→ ' + block;
      entry.dataset.state = block === undefined ? 'empty' : 'mapped';
      entry.dataset.active = index === activeLogical ? (writing ? 'write' : 'read') : '';
      routes[index].dataset.state = entry.dataset.state;
      routes[index].dataset.active = entry.dataset.active;
    });

    for (const page of state.pages) {
      const { view, slots, status } = pageViews.get(page.physicalBlock);
      const valid = page.slots.filter(position => position !== null).length;
      const read = page.slots.filter(position => readPositions.has(position)).length;
      view.dataset.allocated = String(page.allocated);
      view.dataset.active = page.physicalBlock === activeBlock ? (writing ? 'write' : 'read') : '';
      status.textContent = read > 0 ? `已读 ${read} 行`
        : valid > 0 ? `${valid} 行有效` : page.allocated ? '已分配' : '未分配';
      slots.forEach((slot, offset) => {
        const position = page.slots[offset];
        const tone = position === null ? 'empty'
          : readPositions.has(position) ? 'read' : position === state.position ? 'new' : 'valid';
        slot.dataset.state = tone;
        slot.firstElementChild.textContent = position === null ? '·' : `a${position}`;
        if (position === null) delete slot.dataset.position;
        else slot.dataset.position = String(position);
        const label = position === null ? '本请求无有效 KV' : `a${position}${tone === 'read' ? '，已读' : ''}`;
        slot.setAttribute('aria-label', `槽 ${offset}，${label}`);
      });
    }

    seek.value = String(step);
    seek.setAttribute('aria-valuetext', `第 ${step + 1} 步，${steps[step][0]}`);
    previous.disabled = step === 0;
    next.disabled = step === MAX_STEP;
    play.textContent = timer !== null ? '暂停' : step === MAX_STEP ? '重播' : '播放';
  }

  function pause() {
    window.clearTimeout(timer);
    timer = null;
  }

  function goTo(target) {
    pause();
    step = target;
    draw();
  }

  function advance() {
    step += 1;
    timer = step === MAX_STEP ? null : window.setTimeout(advance, delay);
    draw();
  }

  previous.addEventListener('click', () => goTo(step - 1));
  next.addEventListener('click', () => goTo(step + 1));
  seek.addEventListener('input', () => goTo(Number(seek.value)));
  play.addEventListener('click', () => {
    if (timer !== null) {
      pause();
    } else {
      if (step === MAX_STEP) step = 0;
      timer = window.setTimeout(advance, delay);
    }
    draw();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && timer !== null) goTo(step);
  });

  draw();
  region('interactive').hidden = false;
})();
