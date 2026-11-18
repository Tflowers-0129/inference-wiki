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
    node.append(element('span', '', page.kind === 'prefix' ? 'M · 手册 128' : `${page.request} · 尾 1`));
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
      node.append(element('p', 'pool-walk-untracked', instance.status === 'past' ? '已交接；后续驻留未跟踪' : '尚未跟踪到本例的 KV'));
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
    if (state.store.M && state.store.M.location === host) {
      const manual = element('div', 'pool-walk-object is-stored');
      manual.dataset.poolObject = 'M';
      manual.append(element('b', '', 'M ·《手册》'));
      manual.append(element('span', '', '128 token 的 KV · 1 副本'));
      manual.append(element('small', '', state.step === 4 ? '读出后仍在本段' : '可按相同键读取'));
      segment.append(manual);
    } else {
      segment.append(element('p', 'pool-walk-segment-empty', state.store.M ? '本段没有 M' : 'M 尚未保存'));
    }
    if (state.store.warranty.location === host) {
      const warranty = element('div', 'pool-walk-object pool-walk-warranty');
      warranty.dataset.poolObject = 'warranty';
      warranty.append(element('b', '', '《保修条款》'));
      warranty.append(element('span', '', '独立前缀 · 1 副本'));
      warranty.append(element('small', '', '早已存在，始终不动'));
      segment.append(warranty);
    }
    return segment;
  }

  function renderMachines(state) {
    const hostA = machine('P 机 A', 'P0 + 本机共享 DRAM 段', 'pool-walk-host-a');
    hostA.append(instanceCard('P0', state.instances.P0, state.active.includes('P0')), storeSegment('A', state));
    const hostB = machine('P 机 B', 'P1 + 本机共享 DRAM 段', 'pool-walk-host-b');
    hostB.append(instanceCard('P1', state.instances.P1, state.active.includes('P1')), storeSegment('B', state));
    const decode = machine('Decode 机', 'D0、D1 各用独立卡接着生成', 'pool-walk-host-decode');
    ['D0', 'D1'].forEach(name => decode.append(instanceCard(name, state.instances[name], state.active.includes(name))));
    find('machines').replaceChildren(hostA, hostB, decode);
  }

  function renderRoute(state) {
    const route = find('route');
    route.className = `pool-walk-route is-${state.flow.kind}`;
    if (state.flow.kind === 'idle') {
      route.replaceChildren(element('p', '', '从 R1 开始：手册前缀在本地和共享池中都未命中。'));
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
    find('position').textContent = `步骤 ${selection.step} / ${model.steps.length - 1} · M 落 ${selection.placement} 段的推演`;
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
  const page = (number, kind, request) => ({ number, kind, request, validTokens: kind === 'prefix' ? 128 : 1, capacity: 128 });
  const statusLabels = { pending: '本例尚未到达', tracked: '本次相关页', source: '本步源页', target: '本步目标页', past: '不再跟踪' };
  const steps = [
    {
      title: '初态：两条请求，尚无手册 KV 副本',
      caption: 'B 段已有《保修条款》的 KV，它与这两条请求的前缀不同。',
      observation: 'M 是手册公共前缀的对象键简称；页号属于各个实例。',
      active: [], flow: { kind: 'idle' },
    },
    {
      title: 'R1 未命中缓存，P0 计算全部输入',
      caption: 'P0 计算全部 129 个输入 token，写入两页。',
      observation: '页 3 已满；页 4 只有尾 token ① 的 KV。',
      active: ['P0'],
      flow: { kind: 'compute', from: 'R1 输入', fromDetail: '手册 128 + 尾 1', label: '计算 129 token', initiator: 'P0 执行 Prefill', to: 'P 机 A · P0 HBM', toDetail: '页 3、4' },
    },
    {
      title: 'P0 → A 段：保存完整的手册块 M',
      caption: '将页 3 的 KV 复制到共享 Store；尾页 4 不入池。',
      observation: '本例先保存 M；保存与交给 D0 没有强制先后。',
      active: ['P0', 'segment-A'],
      flow: { kind: 'store', operation: 'PUT', direction: 'D2H', from: 'P 机 A · P0 HBM', fromDetail: '页 3 · 手册 KV', label: '保存 M · D2H', initiator: 'P0 发起 WRITE', to: 'P 机 A · DRAM', toDetail: 'A 段 · M 的唯一副本', source: 'P0', sourcePages: [3], destination: 'segment-A', key: 'M' },
    },
    {
      title: 'P0 → D0：交接 R1 的两页 KV',
      caption: 'D0 发起 READ，复制字节到自己的目标页；两侧页号不用相同。',
      observation: '本次映射：P0 [3, 4] → D0 [12, 13]。不经过 Store。',
      active: ['P0', 'D0'],
      flow: { kind: 'handoff', operation: 'READ', direction: 'D2D', from: 'P 机 A · P0 HBM', fromDetail: '源页 3、4', label: 'D2D · 两页 KV', initiator: 'D0 发起 READ', to: 'Decode 机 · D0 HBM', toDetail: '目标页 12、13', source: 'P0', sourcePages: [3, 4], destination: 'D0', destinationPages: [12, 13], request: 'R1' },
    },
    {
      title: 'A 段 → P1：R2 读回同一个 M',
      caption: 'P1 显存未命中；A 段的副本可读，将手册 KV 读入页 9。',
      observation: 'M 仍留在 A 段。R1 交接后的 HBM 驻留不再跟踪。',
      active: ['segment-A', 'P1'],
      flow: { kind: 'store', operation: 'GET', direction: 'rH2D', from: 'P 机 A · DRAM', fromDetail: 'A 段 · M 的手册 KV', label: '读取 M · rH2D', initiator: 'P1 发起 READ', to: 'P 机 B · P1 HBM', toDetail: '目标页 9', source: 'segment-A', key: 'M', destination: 'P1', destinationPages: [9] },
    },
    {
      title: 'P1 只计算第 129 个 token',
      caption: '页 9 的手册 KV 直接复用，尾 token ② 的 KV 写入页 10。',
      observation: '两条请求的手册相同，尾 token 不同；只有手册块共享。',
      active: ['P1'],
      flow: { kind: 'compute', from: 'R2 的最后一个 token', fromDetail: '不同的尾 token ②', label: '只计算 1 token', initiator: 'P1 执行剩余 Prefill', to: 'P 机 B · P1 HBM', toDetail: '写入页 10；复用页 9' },
    },
    {
      title: 'P1 → D1：交接 R2 的两页 KV',
      caption: 'D1 从 P1 读取手册和第二个问题对应的 KV，然后继续生成。',
      observation: '本次映射：P1 [9, 10] → D1 [20, 21]。Store 中的两个对象均不变。',
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
    const store = { M: null, warranty: { label: '保修条款', unchanged: true, location: 'B', replicas: 1 } };
    if (index >= 1 && index <= 3) instances.P0 = { status: index === 3 ? 'source' : 'tracked', pages: [page(3, 'prefix', 'R1'), page(4, 'tail', 'R1')] };
    if (index >= 2) store.M = { key: 'M', validTokens: 128, replicas: 1, location: placement };
    if (index === 3) instances.D0 = { status: 'target', pages: [page(12, 'prefix', 'R1'), page(13, 'tail', 'R1')] };
    if (index >= 4) {
      instances.P0.status = instances.D0.status = 'past';
      instances.P1 = { status: index === 6 ? 'source' : 'tracked', pages: [page(9, 'prefix', 'R2')] };
    }
    if (index >= 5) instances.P1.pages.push(page(10, 'tail', 'R2'));
    if (index === 6) instances.D1 = { status: 'target', pages: [page(20, 'prefix', 'R2'), page(21, 'tail', 'R2')] };
    const state = { ...steps[index], step: index, placement, instances, store };
    if (placement === 'B' && index === 2) {
      state.title = 'P0 → B 段：保存完整的手册块 M';
      state.active = ['P0', 'segment-B'];
      state.flow = { ...state.flow, direction: 'D2rH', label: '保存 M · D2rH', to: 'P 机 B · DRAM', toDetail: 'B 段 · M 的唯一副本', destination: 'segment-B' };
    }
    if (placement === 'B' && index === 4) {
      state.title = 'B 段 → P1：R2 读回同一个 M';
      state.caption = 'P1 显存未命中；B 段的副本可读，将手册 KV 读入页 9。';
      state.observation = 'M 仍留在 B 段。R1 交接后的 HBM 驻留不再跟踪。';
      state.active = ['segment-B', 'P1'];
      state.flow = { ...state.flow, direction: 'H2D', label: '读取 M · H2D', from: 'P 机 B · DRAM', fromDetail: 'B 段 · M 的手册 KV', source: 'segment-B' };
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
