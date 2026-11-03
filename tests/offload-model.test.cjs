const test = require('node:test');
const assert = require('node:assert/strict');
const m = require('../dist/offload-model.js');
test('LRU protects all selected hits before assigning miss slots', () => {
  const p = m.example('lru');
  assert.deepEqual(p.rows.map(r => r.slot), [1, 2, 0, 3]);
  assert.deepEqual(p.slots, [13, 5, 9, 20, 18, 21]);
  assert.deepEqual(p.misses.map(r => r.token), [13, 20]);
  const missFirst = m.lruPlan([2, 5], [0, 1], [13, 2]);
  assert.deepEqual(missFirst.rows.map(r => r.slot), [1, 0]);
});
test('logical IDs, Host rows and resident slots remain distinct', () => {
  const r = m.example('lru').rows.find(r => r.token === 13);
  assert.deepEqual([r.token, r.hostSlot, r.slot], [13, 21, 0]);
});
test('request reassignment and unstable suffix invalidate cached rows', () => {
  assert.equal(m.lruPlan([1, 5], [0, 1], [5], 5).misses.length, 1);
  assert.equal(m.lruPlan([1, 5], [0, 1], [1], Infinity, true).misses.length, 1);
});
test('discrete Resident pages take priority over Hot; only Host misses stage', () => {
  const p = m.example('hisparse');
  assert.deepEqual(p.rows.map(r => r.source), ['resident', 'resident', 'hot', 'host', 'host', 'resident', 'resident', 'resident']);
  assert.deepEqual(p.misses.map(r => r.token), [10, 14]);
  assert.deepEqual(p.rows.map(r => r.slot), [29, 30, 100, 101, 102, 9, 36, 37]);
});
test('duplicate selected rows preserve output order without duplicate DMA', () => {
  const p = m.residencyPlan([10, 10], new Map(), new Map(), 4, 100);
  assert.equal(p.misses.length, 1);
  assert.deepEqual(p.rows.map(r => r.slot), [100, 100]);
});
