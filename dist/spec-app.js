(function () {
  'use strict';
  const model = globalThis.WikiSpecPlayback;
  const byId = id => document.getElementById(id);
  if (!model || !byId('spec-player') || !byId('spec-stream')) return;

  let accepted = Number(byId('spec-accepted').value);
  let index = 0;
  let events = model.frames(accepted);
  let timer = null;
  const scene = globalThis.WikiSpecScene;
  let fullScene = false;

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function token(label, position, status, description) {
    const chip = node('span', `spec-token spec-token-${status}`);
    chip.append(node('strong', '', label), node('span', 'spec-token-position', `pos ${position}`));
    if (description) chip.append(node('span', 'spec-token-status', description));
    return chip;
  }
  function stat(label, value, suffix = '') {
    const box = node('div', 'spec-stat');
    box.append(node('span', 'spec-stat-label', label), node('strong', '', String(value)));
    if (suffix) box.append(node('span', 'spec-stat-suffix', suffix));
    return box;
  }
  function renderContinuation() {
    const next = model.continuation(accepted, Number(byId('spec-next-k').value));
    const previous = next.previous;
    const list = node('dl', 'spec-meta-list');
    const array = values => `[${values.join(', ')}]`;
    const pairs = [
      ['上一轮 K / 接受 a', `${previous.draftCount} / ${previous.accepted}`],
      ['下一轮候选 K', String(next.nextK)],
      ['query tokens', array(next.queryTokens)],
      ['positions', array(next.queryPositions)],
      ['query_start_loc', array(next.queryOffsets)],
      ['seq_len', String(next.seqLength)],
      ['slot_mapping', array(next.slotMapping)]
    ];
    pairs.forEach(([term, value]) => list.append(node('dt', '', term), node('dd', '', value)));
    byId('spec-continuation-metadata').replaceChildren(list);
    byId('spec-continuation-correction').textContent =
      `上一轮拒绝 ${previous.draftCount} − ${previous.accepted} = ${previous.invalidTailLength} 个候选，起点从 12 纠正到 ${previous.validKvLength}。下一轮从 x@${previous.validKvLength} 开始，${next.nextK === 0 ? '没有新候选，也要计算 x 的 KV。' : `再加 ${next.nextK} 个新候选，共 ${next.queryTokens.length} 行 Query。`}`;
    byId('spec-continuation-address').textContent = next.queryPositions.map((position, i) => {
      const logicalBlock = Math.floor(position / next.blockSize);
      return `${next.queryTokens[i]}@${position} → 块 ${logicalBlock} → 物理块 ${next.blockTable[logicalBlock]}，slot ${next.slotMapping[i]}`;
    }).join('；');
  }
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    byId('spec-play').textContent = '播放';
    byId('spec-play').setAttribute('aria-pressed', 'false');
  }
  function renderScene() {
    if (!scene) return;
    const host = byId('spec-scene');
    const useFull = fullScene || host.clientWidth >= 1050;
    const draw = !useFull && scene.renderCompact ? scene.renderCompact : scene.render;
    host.innerHTML = draw(events[index], Number(byId('spec-next-k').value));
    host.dataset.layout = useFull ? 'full' : 'compact';
    byId('spec-toggle-view').textContent = fullScene ? '返回自适应图示' : '展开完整泳道';
    byId('spec-toggle-view').setAttribute('aria-pressed', String(fullScene));
  }
  function render() {
    const frame = events[index];
    const result = frame.result;
    const stages = byId('spec-stages');
    stages.replaceChildren(...events.map((event, i) => {
      const item = node('li', i === index ? 'is-current' : i < index ? 'is-complete' : '');
      const jump = node('button', '', `${i + 1} ${event.shortTitle}`);
      jump.type = 'button';
      jump.setAttribute('aria-label', `跳到第 ${i + 1} 步：${event.shortTitle}`);
      if (i === index) jump.setAttribute('aria-current', 'step');
      jump.addEventListener('click', () => {stop(); index = i; render();});
      item.append(jump);
      return item;
    }));
    byId('spec-progress').max = events.length - 1;
    byId('spec-progress').value = index;
    byId('spec-progress').setAttribute('aria-valuetext', events[index].shortTitle);
    byId('spec-progress-label').textContent = `${index + 1} / ${events.length}`;
    renderScene();
    byId('spec-event-title').textContent = frame.title;
    byId('spec-event-note').textContent = frame.note;
    byId('spec-prev').disabled = index === 0;
    byId('spec-next').disabled = index === events.length - 1;

    byId('spec-prefix').replaceChildren(...Array.from({length: 8}, (_, i) => token(`h${i}`, i, 'valid', '有效 KV')));
    const query = [token('t0', 8, frame.hasVerification ? 'valid' : 'anchor', frame.hasVerification ? '有效 KV' : '待计算 KV')];
    for (let i = 0; i < frame.proposedCount; i++) {
      const acceptedHere = i < accepted;
      const status = frame.hasVerification ? acceptedHere ? 'valid' : frame.committed ? 'invalid' : 'reject' : 'draft';
      const label = frame.hasVerification ? acceptedHere ? '接受 · KV 有效' : frame.committed ? '逻辑作废' : '不提交' : 'Draft 候选';
      query.push(token(`d${i + 1}`, i + 9, status, label));
    }
    byId('spec-query').replaceChildren(...query);
    if (!frame.proposedCount) byId('spec-query').append(node('span', 'spec-empty', '尚未提出候选'));
    const output = byId('spec-output');
    if (frame.committed) {
      output.replaceChildren(...result.acceptedTokens.map((label, i) => token(label, i + 9, 'valid', '已提交 · 有 KV')),
        token('x', result.extraToken.position, 'next', result.extraToken.kind === 'bonus' ? 'bonus · 无 KV' : '纠正 · 无 KV'));
    } else {
      output.replaceChildren(node('span', 'spec-empty', frame.hasVerification ? '验证结果已确定，下一事件提交。' : '尚未提交本轮输出。'));
    }

    const metadata = byId('spec-metadata');
    metadata.replaceChildren();
    if (frame.hasMetadata) {
      const list = node('dl', 'spec-meta-list');
      const pairs = [
        ['Query tokens', '[t0, d1, d2, d3]'], ['positions', '[8, 9, 10, 11]'],
        ['query offsets', '[0, 4]'], ['provisional seq_len', '12（含本轮 Query）']
      ];
      pairs.forEach(([term, value]) => list.append(node('dt', '', term), node('dd', '', value)));
      metadata.append(list);
    } else {
      metadata.append(node('p', 'spec-empty', '候选生成完后再打包 Target 输入；当前仍只有前缀的 Target KV。'));
    }
    byId('spec-stats').replaceChildren(
      stat('本轮 Target Query', frame.hasMetadata ? 4 : '待组装', frame.hasMetadata ? '行；不是 4 个有效新输出' : ''),
      stat('有效 Target KV 长度', frame.committed ? result.validKvLength : frame.hasVerification ? '待提交' : 8, frame.committed ? `位置 0–${result.validKvLength - 1}` : '提交前保留原有效边界'),
      stat('本轮提交输出', frame.committed ? result.emittedCount : 0, frame.committed ? 'a + 1；不含已确认的 t0' : '候选仍未提交'),
      stat('无效候选 KV 尾部', frame.committed ? result.invalidTailLength : '待验证', frame.committed ? '个位置；不承诺物理清零' : '')
    );
    byId('spec-kv-boundary').textContent = frame.committed
      ? `提交后下一位置是 ${result.extraToken.position}：这里的 x 已确认输出，但没有 Target KV。${result.invalidTailLength ? '它将替换同一逻辑位置上被拒绝的候选，后续无效位置也不能作为历史读取。' : '它在最后一个已接受候选之后，留待下一轮计算。'}`
      : '本例始终区分三件事：候选已提出、Target 已计算 KV、输出已确认提交。';
  }
  byId('spec-toggle-view').addEventListener('click', () => {fullScene = !fullScene; renderScene();});
  window.addEventListener('resize', renderScene);
  byId('spec-prev').addEventListener('click', () => {stop(); index = Math.max(0, index - 1); render();});
  byId('spec-next').addEventListener('click', () => {stop(); index = Math.min(events.length - 1, index + 1); render();});
  byId('spec-reset').addEventListener('click', () => {stop(); index = 0; render();});
  byId('spec-accepted').addEventListener('change', event => {
    stop(); accepted = Number(event.target.value); events = model.frames(accepted); index = 0; render(); renderContinuation();
  });
  byId('spec-next-k').addEventListener('change', () => {renderContinuation(); render();});
  byId('spec-progress').addEventListener('input', event => {stop(); index = Number(event.target.value); render();});
  byId('spec-play').addEventListener('click', () => {
    if (timer !== null) {stop(); return;}
    if (index === events.length - 1) {index = 0; render();}
    byId('spec-play').textContent = '暂停';
    byId('spec-play').setAttribute('aria-pressed', 'true');
    timer = setInterval(() => {
      index = Math.min(events.length - 1, index + 1); render();
      if (index === events.length - 1) stop();
    }, 1800);
  });
  document.addEventListener('visibilitychange', () => {if (document.hidden) stop();});

  function number(value) {return value === null ? '无样本' : Number.isInteger(value) ? String(value) : value.toFixed(2);}
  function renderStream() {
    const scenario = model.streamScenario(byId('spec-stream-scenario').value);
    byId('spec-stream-note').textContent = scenario.note;
    let ordinal = 0;
    byId('spec-stream-events').replaceChildren(...scenario.events.map((event, i) => {
      const item = node('li', 'spec-stream-event');
      item.append(node('strong', 'spec-stream-time', `${event.time} ms`), node('span', 'spec-stream-count', `收到 ${event.count} token`));
      const chips = node('div', 'spec-stream-tokens');
      for (let j = 0; j < event.count; j++) chips.append(node('span', '', `y${++ordinal}`));
      item.append(chips);
      item.append(node('span', 'spec-stream-gap', i ? `距上次 ${scenario.chunkGaps[i - 1]} ms` : '首个输出事件'));
      return item;
    }));
    const internal = byId('spec-stream-internal');
    internal.hidden = scenario.internalGenerationTimes === null;
    internal.textContent = scenario.internalGenerationTimes ? `本场景另行给定的内部生成时间：${scenario.internalGenerationTimes.join('、')} ms。它是构造条件，不能从上面的客户端事件反推。` : '';
    byId('spec-stream-stats').replaceChildren(
      stat('TTFT', number(scenario.ttft), 'ms；请求从 0 ms 发出'),
      stat('最后输出事件', number(scenario.lastTime), 'ms'),
      stat('总输出 O', scenario.outputTokens, 'token'),
      stat('请求 TPOT', number(scenario.tpot), 'ms/token'),
      stat('客户端输出事件 ITL 均值', number(scenario.meanChunkGap), 'ms；每个事件间隔一条样本')
    );
    const formula = node('p', '', `TPOT = (${scenario.lastTime} − ${scenario.firstTime}) ÷ (${scenario.outputTokens} − 1) = ${number(scenario.tpot)} ms/token`);
    const gaps = node('p', '', `输出事件间隔 = [${scenario.chunkGaps.join('、')}] ms；算术均值 = ${number(scenario.meanChunkGap)} ms`);
    byId('spec-stream-formulas').replaceChildren(formula, gaps);
  }
  byId('spec-stream-scenario').addEventListener('change', renderStream);
  render();
  renderContinuation();
  renderStream();
})();
