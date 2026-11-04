const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../dist/ep-model.js');

test('Every locality keeps forty inputs, eighty tasks and balanced expert work', () => {
  for (const scenario of Object.keys(M.SCENARIOS)) {
    const result = M.compare(scenario);
    assert.equal(result.tokenCount, 40);
    assert.equal(new Set(result.tokens.map(token => token.id)).size, 40);
    assert.equal(result.assignments, 80);
    assert.deepEqual(result.rankTasks, [20, 20, 20, 20]);
    assert.deepEqual(result.expertTasks, Array(8).fill(10));
    assert.equal(result.assignmentMatrix.flat().reduce((sum, count) => sum + count, 0), 80);
    assert.equal(result.localTasks + result.remoteTasks, 80);
  }
});

test('AllGather counts logical source-to-destination copies and excludes the local diagonal from payload', () => {
  for (const scenario of Object.keys(M.SCENARIOS)) {
    const result = M.compare(scenario);
    assert.deepEqual(result.allGather.matrix, Array.from({length: 4}, () => [10, 10, 10, 10]));
    assert.equal(result.allGather.totalCopies, 160);
    assert.equal(result.allGather.localCopies, 40);
    assert.equal(result.allGather.remoteCopies, 120);
    assert.equal(result.allGather.payloadBytes, 960 * 1024);
  }
});

test('Two experts on the same remote rank consume one cross-rank activation per token', () => {
  const result = M.compare('paired');
  assert.equal(result.remoteTasks, 80);
  assert.equal(result.routed.remoteCopies, 40);
  assert.equal(result.routed.localCopies, 0);
  assert.deepEqual(result.routed.matrix, [[0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10], [10, 0, 0, 0]]);
  assert.equal(result.routed.payloadBytes, 320 * 1024);
});

test('Two remote target ranks require two copies while retaining the same eighty expert tasks', () => {
  const result = M.compare('split');
  assert.equal(result.routed.remoteCopies, 80);
  assert.equal(result.routed.totalCopies, 80);
  assert.equal(result.routed.payloadBytes, 640 * 1024);
  assert.ok(result.routed.matrix.every((row, source) => row[source] === 0));
});

test('Mixed local and remote routes count local expert work without remote payload', () => {
  const result = M.compare('mixed');
  assert.equal(result.localTasks, 40);
  assert.equal(result.remoteTasks, 40);
  assert.equal(result.routed.localCopies, 20);
  assert.equal(result.routed.remoteCopies, 40);
  assert.equal(result.routed.totalCopies, 60);
  assert.equal(result.routed.payloadBytes, 320 * 1024);
  assert.ok(result.routed.matrix.every((row, source) => row[source] === 5));
});

test('Local-only single token needs no network payload even with two expert tasks', () => {
  const result = M.countDispatch([{id: 'one', sourceRank: 2, experts: [4, 5]}]);
  assert.equal(result.assignments, 2);
  assert.equal(result.localTasks, 2);
  assert.equal(result.routed.localCopies, 1);
  assert.equal(result.routed.remoteCopies, 0);
  assert.equal(result.routed.payloadBytes, 0);
  assert.equal(result.allGather.remoteCopies, 3);
});

test('Payload scales with hidden dimension and element bytes, independently of expert tasks', () => {
  const tokens = M.buildTokens('split');
  const base = M.countDispatch(tokens);
  const quantized = M.countDispatch(tokens, 4096, 1);
  const wide = M.countDispatch(tokens, 8192, 2);
  assert.equal(base.bytesPerActivation, 8192);
  assert.equal(quantized.routed.payloadBytes * 2, base.routed.payloadBytes);
  assert.equal(wide.routed.payloadBytes, base.routed.payloadBytes * 2);
  assert.equal(wide.assignments, base.assignments);
});

test('Invalid routes cannot silently enter the accounting', () => {
  assert.throws(() => M.buildTokens('unknown'), RangeError);
  assert.throws(() => M.countDispatch([{id: 'a', sourceRank: 0, experts: [1, 1]}]), RangeError);
  assert.throws(() => M.countDispatch([{id: 'a', sourceRank: 4, experts: [0, 1]}]), RangeError);
  assert.throws(() => M.countDispatch([{id: 'a', sourceRank: 0, experts: [0, 8]}]), RangeError);
  assert.throws(() => M.countDispatch([{id: 'a', sourceRank: 0, experts: [0, 1]}, {id: 'a', sourceRank: 0, experts: [2, 3]}]), RangeError);
  assert.throws(() => M.countDispatch([], 0), RangeError);
});
