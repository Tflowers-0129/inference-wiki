/* Explanatory controls. Expert widths represent expected task counts, never measured time. */
(() => {
  'use strict';
  const M = window.LoadBalanceModel;
  const $ = id => document.getElementById(id);
  if (!M || !$('dplb-lab') || !$('eplb-lab')) return;
  const fmt = value => Number.isInteger(value) ? String(value) : value.toFixed(1);
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let state = M.initialDplbState(), policy = 'kv';
  function renderDplb() {
    const scores = M.dplbScores(state, policy);
    const next = M.routeNext(state, policy).lastTarget;
    $('lb-count').setAttribute('aria-pressed', String(policy === 'count'));
    $('lb-kv').setAttribute('aria-pressed', String(policy === 'kv'));
    $('lb-route-result').textContent = `下一条目标：DP ${next}，当前最低分 ${fmt(scores[next])}。已送入 ${state.assigned.length} 条新请求。`;
    $('lb-ranks').innerHTML = scores.map((score, rank) => `<div class="lb-rank${rank === next ? ' is-target' : ''}"><h5>DP rank ${rank}<span>${rank === next ? '下一条目标' : '队列留在原 rank'}</span></h5><dl><div><dt>waiting＋信用</dt><dd>${state.waiting[rank]}</dd></div><div><dt>running</dt><dd>${state.running[rank]}</dd></div><div><dt>KV 使用率</dt><dd>${Math.round(state.kvUsage[rank] * 100)}%</dd></div><div><dt>本 client 在途</dt><dd>${state.ownInflight[rank]}</dd></div></dl><div class="lb-score"><span>当前评分</span><strong>${fmt(score)}</strong></div></div>`).join('');
    $('lb-route-history').textContent = state.assigned.length ? state.assigned.map((rank, i) => `R${i + 1} → DP ${rank}`).join('　·　') : '新请求尚未到达，表中已有请求不被搬走。';
    $('lb-send').disabled = state.assigned.length >= 8;
  }
  for (const choice of ['count', 'kv']) $('lb-' + choice).addEventListener('click', () => {policy = choice; state = M.initialDplbState(); renderDplb();});
  $('lb-send').addEventListener('click', () => {if (state.assigned.length < 8) state = M.routeNext(state, policy); renderDplb();});
  $('lb-reset').addEventListener('click', () => {state = M.initialDplbState(); renderDplb();});

  let placementKey = 'old', heatKey = 'original', phase = 4, timer = null;
  const phaseLabels = ['0 · 布局与输入', '1 · dispatch', '2 · 本 rank 专家计算', '3 · combine / 结果依赖', '4 · 后续消费'];
  const text = (x, y, label, size = 16, fill = '#334155', weight = 400) => `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}">${escape(label)}</text>`;
  const rect = (x,y,w,h,fill,stroke='#cbd5e1',extra='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}" ${extra}/>`;
  function scene(placement, result) {
    const start = 278, scale = 8, end = start + result.max * scale, consumer = end + 84;
    const out = [`<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="596" viewBox="0 0 1020 596" role="img" aria-labelledby="lb-svg-title lb-svg-desc" font-family="system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif"><title id="lb-svg-title">四个 EP rank 的专家工作与结果依赖</title><desc id="lb-svg-desc">教学示意：${phaseLabels[phase]}。各 rank 期望专家任务为 ${result.rankLoads.map(fmt).join('、')}。同一层共八十次专家任务，十二个物理槽。副本均分模型，非实测时长。虚线区是教学同步路径中的结果依赖，不表示整设备空闲。</desc><defs><pattern id="lb-wait" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#fff7e7"/><path d="M-2 8L8-2M6 10L10 6" stroke="#c59a50" stroke-width="1"/></pattern></defs>`];
    out.push(rect(1,1,1018,594,'#fff','#d9e1eb'));
    out.push(text(24,32,'单层 MoE · rank × stream',21,'#172b4d',700));
    out.push(text(24,59,'教学示意：计算条按期望任务数缩放；通信区段只表达依赖，不是毫秒。',16));
    out.push(text(195,91,'dispatch',15,'#1d4ed8',600),text(start,91,'专家计算',15,'#0f766e',600),text(760,91,'输入/权重与真实 Trace 须另核对',14,'#475569'));
    placement.forEach((slots, rank) => {
      const y = 110 + rank * 107, width = result.rankLoads[rank] * scale;
      out.push(rect(16,y,988,96,'#f8fafc','#e2e8f0'));
      out.push(text(30,y+25,`EP rank ${rank}`,18,'#172b4d',700));
      out.push(text(30,y+49,slots.map(e => 'E'+e).join(' / '),16,'#475569'));
      out.push(text(30,y+76,`任务 ${fmt(result.rankLoads[rank])}`,16,'#0f766e',600));
      out.push(text(170,y+29,'计算',14),text(170,y+70,'通信',14));
      out.push(`<g opacity="${phase >= 1 ? 1 : .25}">`+rect(204,y+48,62,27,'#dbeafe','#93b4ed')+text(210,y+67,'发送',14,'#1d4ed8')+'</g>');
      out.push(`<g opacity="${phase >= 2 ? 1 : .25}">`+rect(start,y+8,Math.max(width,1),29,'#d8eee7','#7fb4a3')+(width >= 46 ? text(start+8,y+29,fmt(result.rankLoads[rank]),16,'#155e4c',600) : '')+'</g>');
      if (end - start - width > 1) out.push(`<g opacity="${phase >= 3 ? 1 : .18}">`+rect(start+width,y+48,end-start-width,27,'url(#lb-wait)','#d3ac65')+'</g>');
      out.push(`<g opacity="${phase >= 3 ? 1 : .25}">`+rect(end,y+48,70,27,'#eee8f7','#b69acb')+text(end+6,y+67,'归还',14,'#6b478a')+'</g>');
      out.push(`<path d="M${consumer} ${y+7}V${y+80}" stroke="${phase >= 4 ? '#0f766e' : '#94a3b8'}" stroke-width="2" stroke-dasharray="4 4"/>`);
    });
    out.push(text(24,566,phase >= 3 ? '斜线：结果依赖窗口，可体现为通信等待，也可能被其他工作覆盖。' : '步骤展示依赖顺序；条宽不是播放器停留时间。',16,'#805c1d'));
    out.push(text(24,589,'按整组同步消费简化；真实后端可分块/重叠，不代表源码存在额外全局 barrier。',14,'#475569'),'</svg>');
    return out.join('');
  }
  function renderEplb() {
    const placement = placementKey === 'old' ? M.OLD_PLACEMENT : M.NEW_PLACEMENT;
    const heat = heatKey === 'original' ? M.HEAT : M.SHIFTED_HEAT;
    const result = M.expertLoads(placement, heat);
    $('lb-old').setAttribute('aria-pressed', String(placementKey === 'old'));
    $('lb-new').setAttribute('aria-pressed', String(placementKey === 'new'));
    $('lb-heat-values').innerHTML = heat.map((h, e) => `<div class="lb-heat-cell${h >= 16 ? ' is-hot' : ''}">E${e}<strong>${h}</strong></div>`).join('');
    $('lb-expert-result').textContent = `各 rank 期望任务：[${result.rankLoads.map(fmt).join(', ')}]；总计 ${result.total}；计数均衡度（均值 / 最大值）${(result.balance*100).toFixed(1)}%。`;
    $('lb-phase').value = String(phase);
    $('lb-phase-label').value = phaseLabels[phase];
    $('lb-phase-label').textContent = phaseLabels[phase];
    $('lb-expert-scene').innerHTML = scene(placement, result);
    $('lb-expert-note').textContent = (heatKey === 'shifted' ? '热度已经改变，原先适配 E0/E1 的放置可能更差；需要重新统计与安排，不能瞬时知道未来。' : '旧、新放置使用相同的 12 个槽；逻辑选择仍是同一组 80 次专家任务。') + ' 期望值假设副本均分；计算任务计数不等于真实耗时，也不表示加速比。';
  }
  function stop() {if (timer) clearInterval(timer); timer = null; $('lb-play').textContent = '播放步骤'; $('lb-play').classList.remove('is-playing');}
  for (const key of ['old','new']) $('lb-'+key).addEventListener('click', () => {stop(); placementKey=key; phase=4; renderEplb();});
  $('lb-heat').addEventListener('change', event => {stop(); heatKey=event.target.value; phase=4; renderEplb();});
  $('lb-phase').addEventListener('input', event => {stop(); phase=Number(event.target.value); renderEplb();});
  $('lb-play').addEventListener('click', () => {
    if (timer) {stop(); return;}
    phase=0; renderEplb(); $('lb-play').textContent='暂停'; $('lb-play').classList.add('is-playing');
    timer=setInterval(() => {phase++; renderEplb(); if (phase >= 4) stop();}, 1200);
  });
  document.addEventListener('visibilitychange', () => {if (document.hidden) stop();});
  renderDplb(); renderEplb();
})();
