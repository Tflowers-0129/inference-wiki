const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('../dist/pool-walk.js');

for (const placement of ['A', 'B']) {
  test(`${placement} 段放置：共享池只保存一份完整前缀块，读取后副本仍在原段`, () => {
    const states = model.steps.map((_, index) => model.getSnapshot(index, placement));
    assert.equal(states[0].store.prefixBlock, null);
    assert.equal(states[1].store.prefixBlock, null);
    for (const state of states.slice(2)) {
      assert.deepEqual(state.store, { prefixBlock: { validTokens: 128, replicas: 1, location: placement } });
    }
    assert.equal(states[2].flow.object, 'prefixBlock');
    assert.equal(states[4].flow.object, states[2].flow.object);
    assert.deepEqual(states[2].flow.sourcePages, [3]);
    assert.equal(states[2].instances.P0.pages.find(page => page.number === 4).validTokens, 1);
  });

  test(`${placement} 段放置：R2 先读完整前缀，再计算不同的尾 token`, () => {
    const loaded = model.getSnapshot(4, placement);
    const computed = model.getSnapshot(5, placement);
    assert.equal(loaded.flow.object, 'prefixBlock');
    assert.deepEqual(loaded.flow.destinationPages, [9]);
    assert.deepEqual(loaded.instances.P1.pages.map(page => [page.number, page.kind, page.validTokens]), [[9, 'prefix', 128]]);
    assert.deepEqual(computed.instances.P1.pages[0], loaded.instances.P1.pages[0]);
    const tail = computed.instances.P1.pages[1];
    assert.equal(tail.number, 10);
    assert.equal(tail.validTokens, 1);
    assert.equal(tail.capacity, 128);
    assert.equal(tail.token, '500');
    assert.equal(model.getSnapshot(1, placement).instances.P0.pages[1].token, '100');
  });

  test(`${placement} 段放置：两次 PD 交接的源/目标页映射固定，不经过 Store`, () => {
    assert.deepEqual(model.getSnapshot(3, placement).flow.sourcePages, [3, 4]);
    assert.deepEqual(model.getSnapshot(3, placement).flow.destinationPages, [12, 13]);
    assert.deepEqual(model.getSnapshot(6, placement).flow.sourcePages, [9, 10]);
    assert.deepEqual(model.getSnapshot(6, placement).flow.destinationPages, [20, 21]);
    for (const index of [3, 6]) {
      const state = model.getSnapshot(index, placement);
      const flow = state.flow;
      assert.equal(flow.operation, 'READ');
      assert.equal(flow.direction, 'D2D');
      assert.ok(flow.initiator.startsWith(flow.destination));
      assert.ok(flow.to.startsWith('Decode 机'));
      const source = state.instances[flow.source].pages;
      const target = state.instances[flow.destination].pages;
      assert.deepEqual(source.map(page => page.number), flow.sourcePages);
      assert.deepEqual(target.map(page => page.number), flow.destinationPages);
      assert.deepEqual(target.map(({ number, ...content }) => content), source.map(({ number, ...content }) => content));
      assert.equal(source.reduce((sum, page) => sum + page.validTokens, 0), 129);
      assert.equal(flow.object, undefined);
      assert.deepEqual(state.store, model.getSnapshot(index - 1, placement).store);
    }
  });

  test(`${placement} 段放置：上一请求的 HBM 后续驻留不被默认为保留或释放`, () => {
    for (const index of [4, 5, 6]) {
      const state = model.getSnapshot(index, placement);
      for (const name of ['P0', 'D0']) {
        assert.equal(state.instances[name].status, 'past');
        assert.deepEqual(state.instances[name].pages, []);
      }
    }
    assert.equal(model.getSnapshot(6, placement).instances.P1.status, 'source');
    assert.equal(model.getSnapshot(6, placement).instances.D1.status, 'target');
  });
}

test('物理落点决定两跳方向：A 段 D2H/rH2D，B 段 D2rH/H2D', () => {
  const expectations = {
    A: ['D2H', 'rH2D', 'P 机 A · DRAM'],
    B: ['D2rH', 'H2D', 'P 机 B · DRAM'],
  };
  for (const [placement, [putDirection, getDirection, dram]] of Object.entries(expectations)) {
    const put = model.getSnapshot(2, placement).flow;
    const get = model.getSnapshot(4, placement).flow;
    assert.equal(put.direction, putDirection);
    assert.equal(get.direction, getDirection);
    assert.equal(put.from, 'P 机 A · P0 HBM');
    assert.equal(put.to, dram);
    assert.equal(get.from, dram);
    assert.equal(get.to, 'P 机 B · P1 HBM');
    assert.equal(put.destination, `segment-${placement}`);
    assert.equal(get.source, `segment-${placement}`);
  }
  assert.deepEqual(model.getSnapshot(4), model.getSnapshot(4, 'A'));
});

test('切换存储段保留当前步骤，只有前缀块的位置和存取路径发生变化', () => {
  for (let step = 0; step < model.steps.length; step++) {
    const initial = model.selectStep({ step: 0, placement: 'A' }, step);
    const switched = model.selectPlacement(initial, 'B');
    assert.deepEqual(switched, { step, placement: 'B' });
    assert.deepEqual(initial, { step, placement: 'A' });
    const a = model.getSnapshot(initial.step, initial.placement);
    const b = model.getSnapshot(switched.step, switched.placement);
    assert.deepEqual(a.instances, b.instances);
    if (step >= 2) {
      assert.equal(a.store.prefixBlock.location, 'A');
      assert.equal(b.store.prefixBlock.location, 'B');
      assert.equal(a.store.prefixBlock.replicas, 1);
      assert.equal(b.store.prefixBlock.replicas, 1);
    } else {
      assert.deepEqual(a.store, { prefixBlock: null });
      assert.deepEqual(b.store, a.store);
    }
    if (![2, 4].includes(step)) assert.deepEqual(a.flow, b.flow);
    assert.deepEqual(model.selectPlacement(switched, 'A'), initial);
  }
});

test('两种放置下七步均可前进、后退和跳转，重置保留当前放置', () => {
  for (const placement of ['A', 'B']) {
    let selection = model.selectPlacement({ step: 0, placement: 'A' }, placement);
    const reached = [];
    for (let i = 0; i < model.steps.length; i++) {
      reached.push(selection.step);
      assert.doesNotThrow(() => model.getSnapshot(selection.step, selection.placement));
      selection = model.selectStep(selection, selection.step + 1);
    }
    assert.deepEqual(reached, [0, 1, 2, 3, 4, 5, 6]);
    assert.equal(selection.step, 6);
    for (let i = 6; i > 0; i--) selection = model.selectStep(selection, selection.step - 1);
    assert.deepEqual(selection, { step: 0, placement });
    assert.equal(model.selectStep(selection, -1).step, 0);
    assert.deepEqual(model.selectStep(model.selectStep(selection, 4), 0), selection);
  }
});

test('快照独立且不可变，不接受不存在的放置', () => {
  const first = model.getSnapshot(0);
  const last = model.getSnapshot(6, 'B');
  assert.deepEqual(model.getSnapshot(0), first);
  assert.equal(model.getSnapshot(1, 'B').store.prefixBlock, null);
  assert.equal(model.getSnapshot(4, 'B').instances.P1.pages.length, 1);
  assert.ok(Object.isFrozen(last.instances.P1.pages));
  assert.throws(() => last.instances.P1.pages.push({ number: 99 }), TypeError);
  for (const index of [-1, 7, 0.5, NaN]) assert.throws(() => model.getSnapshot(index), RangeError);
  for (const placement of ['C', '', null]) {
    assert.throws(() => model.getSnapshot(0, placement), RangeError);
    assert.throws(() => model.selectPlacement({ step: 0, placement: 'A' }, placement), RangeError);
  }
});
