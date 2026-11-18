const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('../dist/pool-walk.js');

test('共享池只新增手册完整块，读取不搬走副本且不改变独立前缀', () => {
  const states = model.steps.map((_, index) => model.getSnapshot(index));
  assert.equal(states[0].store.M, null);
  assert.equal(states[1].store.M, null);
  for (const state of states.slice(2)) {
    assert.deepEqual(state.store.M, { key: 'M', validTokens: 128, replicas: 1, location: 'C' });
  }
  for (const state of states) assert.deepEqual(state.store.warranty, states[0].store.warranty);
  assert.deepEqual(states[2].flow.sourcePages, [3]);
  assert.equal(states[2].instances.P0.pages.find(page => page.number === 4).validTokens, 1);
});

test('R2 先读完整前缀，再计算不同的尾 token', () => {
  const loaded = model.getSnapshot(4);
  const computed = model.getSnapshot(5);
  assert.equal(loaded.flow.key, 'M');
  assert.deepEqual(loaded.flow.destinationPages, [9]);
  assert.deepEqual(loaded.instances.P1.pages.map(page => [page.number, page.kind, page.validTokens]), [[9, 'prefix', 128]]);
  assert.deepEqual(computed.instances.P1.pages[0], loaded.instances.P1.pages[0]);
  const tail = computed.instances.P1.pages[1];
  assert.equal(tail.number, 10);
  assert.equal(tail.validTokens, 1);
  assert.equal(tail.capacity, 128);
  assert.notEqual(tail.request, model.getSnapshot(1).instances.P0.pages[1].request);
});

test('两次交接各用本次 P/D 页表，保留块内容且不经过 Store', () => {
  assert.deepEqual(model.getSnapshot(3).flow.sourcePages, [3, 4]);
  assert.deepEqual(model.getSnapshot(3).flow.destinationPages, [12, 13]);
  assert.deepEqual(model.getSnapshot(6).flow.sourcePages, [9, 10]);
  assert.deepEqual(model.getSnapshot(6).flow.destinationPages, [20, 21]);
  for (const index of [3, 6]) {
    const state = model.getSnapshot(index);
    const flow = state.flow;
    assert.equal(flow.operation, 'READ');
    assert.ok(flow.initiator.startsWith(flow.destination));
    const source = state.instances[flow.source].pages;
    const target = state.instances[flow.destination].pages;
    assert.deepEqual(source.map(page => page.number), flow.sourcePages);
    assert.deepEqual(target.map(page => page.number), flow.destinationPages);
    assert.deepEqual(target.map(({ number, ...content }) => content), source.map(({ number, ...content }) => content));
    assert.equal(source.reduce((sum, page) => sum + page.validTokens, 0), 129);
    assert.equal(flow.key, undefined);
    assert.deepEqual(state.store, model.getSnapshot(index - 1).store);
  }
});

test('上一请求的 HBM 后续驻留不被默认为保留或释放', () => {
  for (const index of [4, 5, 6]) {
    const state = model.getSnapshot(index);
    for (const name of ['P0', 'D0']) {
      assert.equal(state.instances[name].status, 'past');
      assert.deepEqual(state.instances[name].pages, []);
    }
  }
  assert.equal(model.getSnapshot(6).instances.P1.status, 'source');
  assert.equal(model.getSnapshot(6).instances.D1.status, 'target');
});

test('跳步、回退与重置取得独立快照，不积累后续状态', () => {
  const first = model.getSnapshot(0);
  const last = model.getSnapshot(6);
  assert.deepEqual(model.getSnapshot(0), first);
  assert.equal(model.getSnapshot(1).store.M, null);
  assert.equal(model.getSnapshot(4).instances.P1.pages.length, 1);
  assert.ok(Object.isFrozen(last.instances.P1.pages));
  assert.throws(() => last.instances.P1.pages.push({ number: 99 }), TypeError);
  for (const index of [-1, 7, 0.5, NaN]) assert.throws(() => model.getSnapshot(index), RangeError);
});
