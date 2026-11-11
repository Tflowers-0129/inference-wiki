const test = require('node:test');
const assert = require('node:assert/strict');
const { MAX_STEP, stateAt } = require('../dist/kv-decode-model.js');

test('allocation maps position 8 to slot 36 before the KV write', () => {
  const history = stateAt(0);
  const allocated = stateAt(1);
  const written = stateAt(2);
  const emptySlots = [null, null, null, null];

  assert.deepEqual(history.blockTable, [7, 2]);
  assert.deepEqual(history.pages, [
    { physicalBlock: 7, allocated: true, slots: [0, 1, 2, 3] },
    { physicalBlock: 2, allocated: true, slots: [4, 5, 6, 7] },
    { physicalBlock: 9, allocated: false, slots: emptySlots },
  ]);
  assert.equal(history.writeSlot, null);
  assert.equal(allocated.position, 8);
  assert.equal(allocated.blockSize, 4);
  assert.deepEqual(allocated.blockTable, [7, 2, 9]);
  assert.equal(allocated.writeSlot, 36);
  assert.deepEqual(allocated.pages[2], { physicalBlock: 9, allocated: true, slots: emptySlots });
  assert.deepEqual(written.blockTable, allocated.blockTable);
  assert.equal(written.writeSlot, allocated.writeSlot);
  assert.deepEqual(written.pages.slice(0, 2), history.pages.slice(0, 2));
  assert.deepEqual(written.pages[2], { physicalBlock: 9, allocated: true, slots: [8, null, null, null] });
});

test('reads reveal only valid positions in logical block order before output', () => {
  const allPositions = Array.from({ length: 9 }, (_, position) => position);
  const readCounts = [0, 0, 0, 4, 8, 9, 9];
  assert.equal(MAX_STEP, 6);

  for (const [step, count] of readCounts.entries()) {
    const state = stateAt(step);
    assert.equal(state.step, step);
    assert.deepEqual(state.readPositions, allPositions.slice(0, count));
    assert.equal(state.attentionReady, step >= 5);
    assert.equal(state.outputReady, step === MAX_STEP);
  }
});
