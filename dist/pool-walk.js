/* Fixed teaching addresses; snapshots stop tracking a request after its handoff. */
(function (root, factory) {
  'use strict';
  const model = factory();
  if (typeof module === 'object' && module.exports) module.exports = model;
  if (typeof document === 'undefined') return;
  const panel = document.getElementById('pool-walk');
  if (!panel) return;

  const find = name => panel.querySelector(`[data-pool-${name}]`);
  const stepButtons = [...panel.querySelectorAll('[data-pool-step]')];
  const placementButtons = [...panel.querySelectorAll('[data-pool-placement]')];
  let selection = model.selectStep({ step: 0, placement: 'A' }, 0);
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function pageCard(page) {
    const node = element('div', `pool-walk-page ${page.kind === 'prefix' ? 'is-prefix' : 'is-tail'}`);
    node.append(element('b', '', `页 ${page.number}`));
    node.append(element('span', '', page.kind === 'prefix' ? '公共前缀 · 128 token' : page.token));
    return node;
  }

  function instanceCard(name, instance, active) {
    const node = element('section', `pool-walk-instance${active ? ' is-active' : ''}`);
    const header = element('div', 'pool-walk-instance-head');
    header.append(element('h5', '', `${name} HBM`));
    header.append(element('span', '', model.statusLabels[instance.status]));
    node.append(header);
    if (instance.pages.length) {
      const pages = element('div', 'pool-walk-pages');
      instance.pages.forEach(page => pages.append(pageCard(page)));
      node.append(pages);
    } else {
      node.append(element('p', 'pool-walk-untracked', instance.status === 'past' ? '已完成交接，图中不再跟踪这些页' : '尚无本例的 KV 页'));
    }
    return node;
  }

  function machine(name, role, className) {
    const node = element('section', `pool-walk-machine ${className}`);
    const header = element('header', 'pool-walk-machine-head');
    header.append(element('b', '', name));
    header.append(element('span', '', role));
    node.append(header);
    return node;
  }

  function storeSegment(host, state) {
    const segment = element('section', `pool-walk-segment${state.active.includes(`segment-${host}`) ? ' is-active' : ''}`);
    segment.dataset.poolSegment = host;
    segment.append(element('h5', '', `共享 ${host} 段 · Host DRAM`));
    const prefixBlock = state.store.prefixBlock;
    if (prefixBlock && prefixBlock.location === host) {
      const card = element('div', 'pool-walk-object');
      card.dataset.poolObject = 'prefixBlock';
      card.append(element('b', '', '前缀块'));
      card.append(element('span', '', '公共前缀的 KV · 128 token'));
      card.append(element('small', '', state.step === 4 ? '读取后，副本仍在本段' : '共享池中的唯一副本'));
      segment.append(card);
    } else {
      segment.append(element('p', 'pool-walk-segment-empty', prefixBlock ? '此段未保存前缀块' : '前缀块尚未写入'));
    }
    return segment;
  }

  function renderMachines(state) {
    const hostA = machine('P 机 A', 'P0 + 本机共享 DRAM 段', 'pool-walk-host-a');
    hostA.append(instanceCard('P0', state.instances.P0, state.active.includes('P0')), storeSegment('A', state));
    const hostB = machine('P 机 B', 'P1 + 本机共享 DRAM 段', 'pool-walk-host-b');
    hostB.append(instanceCard('P1', state.instances.P1, state.active.includes('P1')), storeSegment('B', state));
    const decode = machine('Decode 机', 'D0、D1 各用一张卡做 Decode', 'pool-walk-host-decode');
    ['D0', 'D1'].forEach(name => decode.append(instanceCard(name, state.instances[name], state.active.includes(name))));
    find('machines').replaceChildren(hostA, hostB, decode);
  }

  function renderRoute(state) {
    const route = find('route');
    route.className = `pool-walk-route is-${state.flow.kind}`;
    if (state.flow.kind === 'idle') {
      route.replaceChildren(element('p', '', 'P0 的显存和共享池中都没有 R1 所需的前缀块。'));
      return;
    }
    const source = element('div', 'pool-walk-endpoint');
    source.append(element('b', '', state.flow.from), element('span', '', state.flow.fromDetail));
    const action = element('div', 'pool-walk-action');
    action.append(element('b', '', state.flow.label), element('span', 'pool-walk-arrow', '→'), element('small', '', state.flow.initiator));
    const target = element('div', 'pool-walk-endpoint');
    target.append(element('b', '', state.flow.to), element('span', '', state.flow.toDetail));
    route.replaceChildren(source, action, target);
  }

  function render() {
    const state = model.getSnapshot(selection.step, selection.placement);
    stepButtons.forEach((button, step) => button.setAttribute('aria-pressed', String(step === selection.step)));
    placementButtons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.poolPlacement === selection.placement)));
    find('position').textContent = `步骤 ${selection.step} / ${model.steps.length - 1} · 将前缀块保存在 ${selection.placement} 段`;
    find('heading').textContent = state.title;
    find('caption').textContent = state.caption;
    find('observation').textContent = state.observation;
    find('prev').disabled = selection.step === 0;
    find('next').disabled = selection.step === model.steps.length - 1;
    find('reset').disabled = selection.step === 0;
    renderRoute(state);
    renderMachines(state);
  }

  function select(next) {
    selection = model.selectStep(selection, next);
    render();
  }
  stepButtons.forEach((button, step) => button.addEventListener('click', () => select(step)));
  placementButtons.forEach(button => button.addEventListener('click', () => {
    selection = model.selectPlacement(selection, button.dataset.poolPlacement);
    render();
  }));
  find('prev').addEventListener('click', () => select(selection.step - 1));
  find('next').addEventListener('click', () => select(selection.step + 1));
  find('reset').addEventListener('click', () => select(0));
  render();
  find('static').hidden = true;
  find('interactive').hidden = false;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function page(number, kind, request) {
    const data = { number, kind, request, validTokens: kind === 'prefix' ? 128 : 1, capacity: 128 };
    if (kind === 'tail') data.token = request === 'R1' ? '100' : '500';
    return data;
  }
  const statusLabels = { pending: '本例尚未到达', tracked: '本次相关页', source: '本步源页', target: '本步目标页', past: '不再跟踪' };
  const steps = [
    {
      title: '两次请求的前 128 个 token 相同',
      caption: 'R1 在阈值为 100 时请求补全。把阈值改为 500 后，编辑器发出 R2。',
      observation: '图中的页号是各实例自己的物理页号。',
      active: [], flow: { kind: 'idle' },
    },
    {
      title: 'R1 未命中缓存，P0 计算全部输入',
      caption: 'P0 计算 R1 的全部 129 个输入 token，将 KV 写入两页。',
      observation: '公共前缀的 128 个位置填满第 3 页。第 4 页只保存 100 对应的 KV。',
      active: ['P0'],
      flow: { kind: 'compute', from: 'R1 输入', fromDetail: '公共前缀 + 100', label: '计算 129 token', initiator: 'P0 执行 Prefill', to: 'P 机 A · P0 HBM', toDetail: '页 3、4' },
    },
    {
      title: 'P0 将前缀块写入 A 段',
      caption: 'P0 只将已填满的第 3 页写入共享池。第 4 页留在 P0，供本次 PD 交接使用。',
      observation: '图中先保存前缀块，再由 D0 读取请求的 KV。这两项操作没有固定的先后顺序。',
      active: ['P0', 'segment-A'],
      flow: { kind: 'store', operation: 'PUT', direction: 'D2H', from: 'P 机 A · P0 HBM', fromDetail: '页 3 · 公共前缀', label: '保存前缀块 · D2H', initiator: 'P0 发起 WRITE', to: 'P 机 A · DRAM', toDetail: 'A 段 · 一份前缀块', source: 'P0', sourcePages: [3], destination: 'segment-A', object: 'prefixBlock' },
    },
    {
      title: 'D0 从 P0 读取 R1 的两页 KV',
      caption: 'D0 发起 READ，把公共前缀和 100 对应的 KV 复制到本机的两页中。两侧的页号可以不同。',
      observation: '本次映射：P0 [3, 4] → D0 [12, 13]。不经过 Store。',
      active: ['P0', 'D0'],
      flow: { kind: 'handoff', operation: 'READ', direction: 'D2D', from: 'P 机 A · P0 HBM', fromDetail: '源页 3、4', label: 'D2D · 两页 KV', initiator: 'D0 发起 READ', to: 'Decode 机 · D0 HBM', toDetail: '目标页 12、13', source: 'P0', sourcePages: [3, 4], destination: 'D0', destinationPages: [12, 13], request: 'R1' },
    },
    {
      title: 'P1 从 A 段读取 R2 的公共前缀',
      caption: 'P1 的显存没有缓存这段前缀。A 段的副本已可读取，P1 将其读入第 9 页。',
      observation: '读取后，前缀块仍在 A 段。图中只追踪 R1 的显存页到交接完成为止。',
      active: ['segment-A', 'P1'],
      flow: { kind: 'store', operation: 'GET', direction: 'rH2D', from: 'P 机 A · DRAM', fromDetail: 'A 段 · 前缀块', label: '读取前缀块 · rH2D', initiator: 'P1 发起 READ', to: 'P 机 B · P1 HBM', toDetail: '目标页 9', source: 'segment-A', object: 'prefixBlock', destination: 'P1', destinationPages: [9] },
    },
    {
      title: 'P1 只计算第 129 个 token（500）',
      caption: 'P1 复用第 9 页的 KV，只处理最后一个 token（500），并将新生成的 KV 写入第 10 页。',
      observation: 'R1 的第 129 个 token 是 100，R2 的是 500。两次请求只能复用此前 128 个位置的 KV。',
      active: ['P1'],
      flow: { kind: 'compute', from: 'R2 的最后一个 token', fromDetail: '500', label: '只计算 1 token', initiator: 'P1 执行剩余 Prefill', to: 'P 机 B · P1 HBM', toDetail: '写入页 10；复用页 9' },
    },
    {
      title: 'D1 从 P1 读取 R2 的两页 KV',
      caption: 'D1 读取公共前缀和 500 对应的 KV，继续为 R2 生成补全结果。',
      observation: '本次映射：P1 [9, 10] → D1 [20, 21]。共享池中仍然只有一份前缀块。',
      active: ['P1', 'D1'],
      flow: { kind: 'handoff', operation: 'READ', direction: 'D2D', from: 'P 机 B · P1 HBM', fromDetail: '源页 9、10', label: 'D2D · 两页 KV', initiator: 'D1 发起 READ', to: 'Decode 机 · D1 HBM', toDetail: '目标页 20、21', source: 'P1', sourcePages: [9, 10], destination: 'D1', destinationPages: [20, 21], request: 'R2' },
    },
  ];

  function freeze(value) {
    if (value && typeof value === 'object') {
      Object.values(value).forEach(freeze);
      Object.freeze(value);
    }
    return value;
  }

  function validatePlacement(placement) {
    if (placement !== 'A' && placement !== 'B') throw new RangeError('Unknown pool placement');
  }

  function getSnapshot(index, placement = 'A') {
    if (!Number.isInteger(index) || index < 0 || index >= steps.length) throw new RangeError('Unknown pool walkthrough step');
    validatePlacement(placement);
    const instances = Object.fromEntries(['P0', 'P1', 'D0', 'D1'].map(name => [name, { status: 'pending', pages: [] }]));
    const store = { prefixBlock: index >= 2 ? { validTokens: 128, replicas: 1, location: placement } : null };
    if (index >= 1 && index <= 3) instances.P0 = { status: index === 3 ? 'source' : 'tracked', pages: [page(3, 'prefix', 'R1'), page(4, 'tail', 'R1')] };
    if (index === 3) instances.D0 = { status: 'target', pages: [page(12, 'prefix', 'R1'), page(13, 'tail', 'R1')] };
    if (index >= 4) {
      instances.P0.status = instances.D0.status = 'past';
      instances.P1 = { status: index === 6 ? 'source' : 'tracked', pages: [page(9, 'prefix', 'R2')] };
    }
    if (index >= 5) instances.P1.pages.push(page(10, 'tail', 'R2'));
    if (index === 6) instances.D1 = { status: 'target', pages: [page(20, 'prefix', 'R2'), page(21, 'tail', 'R2')] };
    const state = { ...steps[index], step: index, placement, instances, store };
    if (placement === 'B' && index === 2) {
      state.title = 'P0 将前缀块写入 B 段';
      state.active = ['P0', 'segment-B'];
      state.flow = { ...state.flow, direction: 'D2rH', label: '保存前缀块 · D2rH', to: 'P 机 B · DRAM', toDetail: 'B 段 · 一份前缀块', destination: 'segment-B' };
    }
    if (placement === 'B' && index === 4) {
      state.title = 'P1 从 B 段读取 R2 的公共前缀';
      state.caption = 'P1 的显存没有缓存这段前缀。B 段的副本已可读取，P1 将其读入第 9 页。';
      state.observation = '读取后，前缀块仍在 B 段。图中只追踪 R1 的显存页到交接完成为止。';
      state.active = ['segment-B', 'P1'];
      state.flow = { ...state.flow, direction: 'H2D', label: '读取前缀块 · H2D', from: 'P 机 B · DRAM', fromDetail: 'B 段 · 前缀块', source: 'segment-B' };
    }
    return freeze(state);
  }

  function selectStep(selection, step) {
    if (!Number.isInteger(step)) throw new RangeError('Unknown pool walkthrough step');
    validatePlacement(selection.placement);
    return freeze({ step: Math.max(0, Math.min(steps.length - 1, step)), placement: selection.placement });
  }

  function selectPlacement(selection, placement) {
    validatePlacement(placement);
    return selectStep({ ...selection, placement }, selection.step);
  }

  return freeze({ steps, statusLabels, getSnapshot, selectStep, selectPlacement });
});
