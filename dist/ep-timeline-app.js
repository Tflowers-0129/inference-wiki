(function () {
  'use strict';
  const M = window.EPTimelineModel;
  const lab = document.getElementById('ep-timeline');
  if (!M || !lab) return;
  const byId = id => document.getElementById(id);
  const escape = value => String(value).replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
  const number = value => Number(value.toFixed(2)).toString();
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const X = 178, W = 1060, RIGHT = 25, AXIS = 32, ROW = 39, RANK = 230, HEIGHT = 526;
  const roles = ['host', 'boundary', 'aiv', 'aic', 'consumer'];
  const labels = ['Host 提交', 'Device 算子边界', 'AIV 通信 / 向量', 'AIC 矩阵 / 局部算子', '后继消费'];
  let comparison, time = 0, playing = false, frame = null, last = null;
  let selection = {method: 'fused', id: 'fused-r0-gmm1-0'};
  const y = (rank, role) => AXIS + rank * RANK + roles.indexOf(role) * ROW + 43;
  const eventY = event => y(event.rank, event.lane.split('-').slice(1).join('-'));
  const x = value => X + value / comparison.duration * (W - X - RIGHT);
  const colorClass = kind => `ep-time-${kind}`;

  function svg(schedule) {
    const uid = `ep-time-${schedule.method}`;
    const chosen = selection.method === schedule.method && schedule.byId[selection.id];
    const selectedBoundary = selection.method === schedule.method && schedule.boundaries.find(boundary => boundary.id === selection.id);
    const selectedIds = new Set(chosen ? [chosen.id] : selectedBoundary ? selectedBoundary.events : []);
    const step = Math.ceil(comparison.duration / 8);
    let html = `<svg class="ep-time-svg" viewBox="0 0 ${W} ${HEIGHT}" width="${W}" height="${HEIGHT}" role="img" aria-labelledby="${uid}-title ${uid}-desc"><title id="${uid}-title">${escape(schedule.label)}：教学依赖泳道</title><desc id="${uid}-desc">两逻辑 rank；横轴为模拟推进格，非实测时间。点击矩形可查看算子、来源和前驱。AIV 与 AIC 行是角色聚合，不是真实 stream 号。</desc><defs><pattern id="${uid}-wait" width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#fff7ed"/><path d="M-2 2L2-2M0 7L7 0M5 9L9 5" stroke="#c48b48" stroke-width=".7"/></pattern><marker id="${uid}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M1 1L6 3.5L1 6" fill="none" stroke="#9a4d10" stroke-width="1.3"/></marker></defs><rect width="${W}" height="${HEIGHT}" fill="#fff"/>`;
    for (let rank = 0; rank < M.RANKS; rank++) {
      const top = AXIS + rank * RANK + 14;
      html += `<rect x="4" y="${top}" width="${W - 8}" height="${RANK - 10}" rx="8" fill="${rank ? '#f5f9fc' : '#f8fafc'}" stroke="#d8e3eb"/><text x="16" y="${top + 20}" class="ep-time-rank">逻辑 rank ${rank}</text>`;
      roles.forEach((role, index) => {
        const yy = y(rank, role);
        html += `<text x="16" y="${yy + 5}" class="ep-time-lane">${labels[index]}</text><line x1="${X}" x2="${W - RIGHT}" y1="${yy}" y2="${yy}" class="ep-time-track"/>`;
      });
    }
    for (let tick = 0; tick <= comparison.duration; tick += step) {
      html += `<line x1="${x(tick)}" x2="${x(tick)}" y1="${AXIS + 7}" y2="${HEIGHT - 18}" class="ep-time-grid"/><text x="${x(tick)}" y="24" text-anchor="middle" class="ep-time-tick">${number(tick)}</text>`;
    }
    html += `<text x="16" y="24" class="ep-time-axis">模拟推进格 →</text>`;
    // Wait areas only refer to this role's next event. They do not assert that the device is idle.
    for (const event of schedule.events) {
      const opaque = schedule.events.find(candidate => candidate.rank === event.rank && candidate.opaque);
      const waitStart = opaque && event.kind === 'combine' ? Math.max(event.laneReady, opaque.end) : event.laneReady;
      const gap = event.start - waitStart;
      if (event.kind === 'host' || event.kind === 'consume' || gap < .7) continue;
      const xx = x(waitStart), width = x(event.start) - xx;
      html += `<rect x="${xx}" y="${eventY(event) - 12}" width="${width}" height="24" fill="url(#${uid}-wait)" stroke="#c49050" stroke-width=".7"><title>该角色的下一事件 ${escape(event.label)} 等待 ${escape(event.waitsFor.map(id => schedule.byId[id].label).join('、') || '输入到达')}；不是设备全局空闲。</title></rect>`;
      if (width > 82) html += `<text x="${xx + width / 2}" y="${eventY(event) + 4}" text-anchor="middle" class="ep-time-wait-label">等前驱就绪</text>`;
    }
    for (const event of schedule.events.filter(candidate => candidate.opaque)) {
      const xx = x(event.start), width = x(event.end) - xx;
      html += `<g class="ep-time-opaque"><rect x="${xx}" y="${y(event.rank, 'aiv') - 12}" width="${width}" height="24" rx="3"/><text x="${xx + width / 2}" y="${y(event.rank, 'aiv') + 4}" text-anchor="middle">局部融合内部随 tiling 变化 · 不展开</text><title>这不是 AIV 忙闲轨迹；该 API 的内部时序未建模。</title></g>`;
    }
    for (const boundary of schedule.boundaries) {
      const xx = x(boundary.start), width = x(boundary.end) - xx;
      const selected = selection.method === schedule.method && selection.id === boundary.id;
      html += `<g class="ep-time-boundary${selected ? ' is-selected' : ''}" tabindex="0" role="button" data-method="${schedule.method}" data-event="${boundary.id}" aria-label="${escape(boundary.operator)}；点击查看外部算子边界"><rect x="${xx}" y="${y(boundary.rank, 'boundary') - 12}" width="${Math.max(width, 2)}" height="24" rx="3"/><text x="${xx + width / 2}" y="${y(boundary.rank, 'boundary') + 4}" text-anchor="middle">${escape(boundary.label)}</text><title>${escape(boundary.label)}；边界不等于 Host 同步。</title></g>`;
    }
    if (chosen) {
      const dependencies = chosen.deps.map(id => [schedule.byId[id], chosen]);
      const consumers = schedule.events.filter(event => event.deps.includes(chosen.id)).map(event => [chosen, event]);
      for (const [from, to] of [...dependencies, ...consumers]) {
        if (from.kind === 'host' && to.kind === 'host') continue;
        const fromX = x(from.end), toX = x(to.start), fromY = eventY(from), toY = eventY(to);
        const middle = (fromX + toX) / 2;
        html += `<path d="M${fromX} ${fromY}C${middle} ${fromY},${middle} ${toY},${toX} ${toY}" class="ep-time-dependency" marker-end="url(#${uid}-arrow)"><title>${escape(from.label)} 完成 → ${escape(to.label)} 可消费</title></path>`;
      }
    }
    for (const event of schedule.events) {
      const xx = x(event.start), width = x(event.end) - xx;
      const status = event.end <= time ? 'is-complete' : event.start <= time && event.end > time ? 'is-active' : 'is-pending';
      const selected = selectedIds.has(event.id);
      const shortLabel = event.kind === 'host' ? '↑' : event.kind === 'sync' ? '↔' : event.kind === 'unpermute' ? 'U' : event.kind === 'consume' ? '下游' : event.kind === 'init' ? '准备' : schedule.method === 'fused' && event.kind === 'activation' ? event.id.endsWith('-0') ? 'S0' : 'S1' : event.label;
      const drawLabel = width >= (shortLabel.length * 6.8 + 5);
      html += `<g class="ep-time-event ${colorClass(event.kind)} ${status}${selected ? ' is-selected' : ''}" tabindex="0" role="button" data-method="${schedule.method}" data-event="${event.id}" aria-label="逻辑 rank ${event.rank}，${escape(event.label)}，${number(event.start)} 至 ${number(event.end)} 模拟推进格，点击查看依赖"><rect x="${xx}" y="${eventY(event) - 12}" width="${Math.max(width, 2)}" height="24" rx="3"/>${drawLabel ? `<text x="${xx + width / 2}" y="${eventY(event) + 4}" text-anchor="middle">${escape(shortLabel)}</text>` : ''}<title>${escape(event.operator || event.label)} · ${number(event.start)}–${number(event.end)} 教学格</title></g>`;
    }
    const cursor = x(time);
    html += `<line x1="${cursor}" x2="${cursor}" y1="${AXIS + 4}" y2="${HEIGHT - 12}" class="ep-time-cursor"/><path d="M${cursor - 4} ${AXIS + 4}L${cursor + 4} ${AXIS + 4}L${cursor} ${AXIS + 10}Z" fill="#163854"/></svg>`;
    return html;
  }

  function detail() {
    const schedule = comparison[selection.method];
    const event = schedule.byId[selection.id] || schedule.boundaries.find(boundary => boundary.id === selection.id);
    if (!event) return;
    const deps = event.deps || [];
    const successors = schedule.events.filter(next => next.deps.includes(event.id));
    const source = event.source;
    const dependencyText = deps.length ? deps.map(id => {
      const predecessor = schedule.byId[id];
      return `<button type="button" data-jump="${escape(id)}" data-method="${selection.method}">R${predecessor.rank} · ${escape(predecessor.label)}</button>`;
    }).join('') : event.events ? '选择下方内部角色事件可查看其前驱。' : '本模型中的提交起点。';
    const active = event.start <= time && event.end > time;
    const state = event.end <= time ? '已完成' : active ? '当前推进中' : '尚未推进到';
    byId('ep-time-detail').innerHTML = `<div class="ep-time-detail-heading"><div><span>${escape(schedule.label)} · 逻辑 rank ${event.rank}</span><h5>${escape(event.operator || event.label)}</h5></div><strong>${state}</strong></div><dl><div><dt>源码顺序 / 边界</dt><dd>${escape(event.evidence)}</dd></div><div><dt>教学时长</dt><dd>${number(event.start)} → ${number(event.end)} 格${event.duration ? `；块宽 ${number(event.duration)} 格` : ''}。可切换假设，不能换算为设备耗时。</dd></div><div><dt>前驱依赖</dt><dd class="ep-time-dep-buttons">${dependencyText}</dd></div>${successors.length ? `<div><dt>下游消费者</dt><dd>${successors.map(next => `R${next.rank} · ${escape(next.label)}`).join('；')}</dd></div>` : ''}<div><dt>源码定位</dt><dd><a href="${escape(source.url)}" target="_blank" rel="noopener">${escape(source.path)}</a><br><code>${escape(source.symbol)}</code></dd></div></dl>`;
  }

  function render() {
    const focused = document.activeElement && document.activeElement.dataset ? {id: document.activeElement.dataset.event, method: document.activeElement.dataset.method} : {};
    const selectedSchedule = comparison[selection.method];
    if (!selectedSchedule.byId[selection.id] && !selectedSchedule.boundaries.some(boundary => boundary.id === selection.id)) {
      selection = {method: selection.method, id: selectedSchedule.boundaries[0].id};
    }
    for (const method of ['mc2', 'fused']) {
      byId(`ep-time-${method}`).innerHTML = svg(comparison[method]);
      byId(`ep-time-${method}-state`).textContent = `${comparison[method].boundaries.length / M.RANKS} 个外部边界 / rank${method === 'mc2' ? comparison.localFusion ? ' · 默认局部融合开启，内部时序不展开' : ' · 受控教学配置：局部融合关闭' : ' · AIV / AIC 以 flag 交接，末端同步后归并'}`;
    }
    byId('ep-time-position').value = time;
    byId('ep-time-position').max = comparison.duration;
    byId('ep-time-clock').textContent = `${number(time)} / ${number(comparison.duration)} 格`;
    byId('ep-time-play').textContent = playing ? '暂停' : time >= comparison.duration ? '重播' : '播放';
    byId('ep-time-play').setAttribute('aria-pressed', String(playing));
    byId('ep-time-step').disabled = time >= comparison.duration;
    byId('ep-time-config-note').textContent = comparison.localFusion ? '当前普通 MC2 使用默认局部融合：GMM1+SwiGLU+Quant 是一个外部调用。其内部随 tiling 改变，图中仅给任意总时长；AIV 位置的灰框表示内部未展开，不是忙闲结论。两图的终点不用于比较性能。' : '当前是受控教学配置：关闭普通 MC2 的 GMM1+SwiGLU+Quant 局部融合，才能逐调用展开 GMM1→SwiGLU/Quant→GMM2。它不代表默认配置；两图的终点不用于比较性能。';
    detail();
    if (focused.id) {
      const replacement = lab.querySelector(`[data-method="${focused.method}"][data-event="${focused.id}"]`);
      if (replacement) replacement.focus({preventScroll: true});
    }
  }
  function pause() {
    playing = false;
    last = null;
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
  }
  function tick(now) {
    if (!playing) return;
    if (last !== null) time = Math.min(comparison.duration, time + Math.min(now - last, 100) / 1000 * 2.5);
    last = now;
    if (time >= comparison.duration) pause();
    render();
    if (playing) frame = window.requestAnimationFrame(tick);
  }
  function choose(method, id) { selection = {method, id}; render(); }
  byId('ep-time-profile').addEventListener('change', event => {
    pause(); time = 0; comparison = M.buildComparison(event.target.value, {localFusion: byId('ep-time-local-fusion').checked}); render();
  });
  byId('ep-time-local-fusion').addEventListener('change', event => {
    pause(); time = 0; comparison = M.buildComparison(byId('ep-time-profile').value, {localFusion: event.target.checked}); render();
  });
  byId('ep-time-layout').addEventListener('change', event => {
    byId('ep-time-panels').classList.toggle('ep-time-side-by-side', event.target.value === 'side');
  });
  byId('ep-time-play').addEventListener('click', () => {
    if (playing) pause();
    else { if (time >= comparison.duration) time = 0; playing = true; frame = window.requestAnimationFrame(tick); }
    render();
  });
  byId('ep-time-step').addEventListener('click', () => {
    pause(); time = comparison.steps.find(step => step > time + 1e-7) ?? comparison.duration; render();
  });
  byId('ep-time-reset').addEventListener('click', () => { pause(); time = 0; render(); });
  byId('ep-time-position').addEventListener('input', event => { pause(); time = Number(event.target.value); render(); });
  lab.addEventListener('click', event => {
    const target = event.target.closest('[data-event], [data-jump]');
    if (target) choose(target.dataset.method, target.dataset.event || target.dataset.jump);
  });
  lab.addEventListener('keydown', event => {
    const target = event.target.closest('[data-event]');
    if (target && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); choose(target.dataset.method, target.dataset.event); }
  });
  reduced.addEventListener('change', () => { pause(); render(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { pause(); render(); } });
  comparison = M.buildComparison();
  render();
})();
