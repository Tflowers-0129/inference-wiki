'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const model = require('../dist/spec-model.js');

test('each accepted prefix preserves KV/output boundaries and candidate conservation', () => {
  for (let a = 0; a <= 3; a++) {
    const result = model.verification(a);
    assert.deepEqual(result.queryTokens, ['t0', 'd1', 'd2', 'd3']);
    assert.deepEqual(result.queryPositions, [8, 9, 10, 11]);
    assert.deepEqual(result.queryOffsets, [0, 4]);
    assert.equal(result.validKvLength, 9 + a);
    assert.equal(result.invalidTailLength, 3 - a);
    assert.equal(result.validKvLength + result.invalidTailLength, result.provisionalKvLength);
    assert.equal(result.acceptedTokens.length + result.rejectedTokens.length, 3);
    assert.deepEqual(result.emittedTokens, [...['d1', 'd2', 'd3'].slice(0, a), 'x']);
    assert.equal(result.emittedCount, a + 1);
    assert.equal(result.extraToken.position, result.validKvLength);
    assert.equal(result.extraToken.hasTargetKv, false);
    assert.equal(result.extraToken.kind, a === 3 ? 'bonus' : 'correction');
    assert.deepEqual(result.invalidPositions, Array.from({length: 3 - a}, (_, i) => 9 + a + i));
    assert.ok(!result.emittedTokens.includes('t0'), 'previously confirmed anchor is not emitted again');
  }
});

test('eight events follow candidate dependencies before target verification and continuation', () => {
  const frames = model.frames(2);
  assert.deepEqual(frames.map(frame => frame.key), ['anchor', 'draft-1', 'draft-2', 'draft-3', 'metadata', 'verify', 'commit', 'next']);
  assert.deepEqual(frames.map(frame => frame.proposedCount), [0, 1, 2, 3, 3, 3, 3, 3]);
  assert.deepEqual(frames.map(frame => frame.hasMetadata), [false, false, false, false, true, true, true, true]);
  assert.deepEqual(frames.map(frame => frame.hasVerification), [false, false, false, false, false, true, true, true]);
  assert.deepEqual(frames.map(frame => frame.committed), [false, false, false, false, false, false, true, true]);
  for (let a = 0; a <= 3; a++) {
    const variant = model.frames(a);
    assert.equal(variant.length, 8);
    assert.equal(variant[6].result.emittedCount, a + 1);
    assert.equal(variant[7].result.extraToken.hasTargetKv, false);
  }
  assert.throws(() => model.verification(-1), RangeError);
  assert.throws(() => model.verification(4), RangeError);
  assert.throws(() => model.verification(1.5), RangeError);
});

test('next-round K does not change the previous verification or rollback', () => {
  for (let a = 0; a <= 3; a++) {
    const previous = model.verification(a);
    for (let nextK = 0; nextK <= 3; nextK++) {
      const next = model.continuation(a, nextK);
      assert.deepEqual(next.previous, previous);
      assert.equal(next.previous.invalidTailLength, 3 - a);
      assert.equal(next.queryPositions[0], previous.validKvLength);
    }
  }
});

test('continuation metadata addresses each valid query row across the block boundary', () => {
  const expectedSlots = {9: 37, 10: 38, 11: 39, 12: 40, 13: 41, 14: 42, 15: 43};
  for (let a = 0; a <= 3; a++) {
    for (let nextK = 0; nextK <= 3; nextK++) {
      const next = model.continuation(a, nextK);
      assert.deepEqual(next.queryTokens, ['x', ...['e1', 'e2', 'e3'].slice(0, nextK)]);
      assert.deepEqual(next.queryPositions, Array.from({length: nextK + 1}, (_, i) => 9 + a + i));
      assert.deepEqual(next.queryOffsets, [0, nextK + 1]);
      assert.equal(next.seqLength, 10 + a + nextK);
      assert.deepEqual(next.slotMapping, next.queryPositions.map(position => expectedSlots[position]));
      assert.equal(new Set(next.slotMapping).size, nextK + 1, 'query rows use distinct physical slots');
    }
  }
  const boundary = model.continuation(2, 1);
  assert.deepEqual(boundary.queryPositions, [11, 12]);
  assert.deepEqual(boundary.slotMapping, [39, 40]);
  assert.equal(boundary.seqLength, 13);
  assert.equal(model.continuation(2).nextK, 1);
});

test('zero next candidates still schedules the confirmed anchor to build its KV', () => {
  for (let a = 0; a <= 3; a++) {
    const next = model.continuation(a, 0);
    assert.equal(next.previous.extraToken.hasTargetKv, false);
    assert.deepEqual(next.queryTokens, ['x']);
    assert.deepEqual(next.queryPositions, [next.previous.extraToken.position]);
    assert.deepEqual(next.queryOffsets, [0, 1]);
    assert.equal(next.seqLength, next.previous.validKvLength + 1);
  }
  for (const invalidK of [-1, 4, 1.5, NaN]) {
    assert.throws(() => model.continuation(2, invalidK), RangeError);
  }
  assert.throws(() => model.continuation(4, 1), RangeError);
});

test('constructed streams distinguish per-event ITL from request TPOT', () => {
  const expectations = {single: [30, 30, 280, 6], specBurst: [20, 60, 220, 2], buffered: [30, 90, 280, 2]};
  for (const [name, [tpot, gap, last, sampleCount]] of Object.entries(expectations)) {
    const stream = model.streamScenario(name);
    assert.equal(stream.outputTokens, 7);
    assert.equal(stream.ttft, 100);
    assert.equal(stream.tpot, tpot);
    assert.equal(stream.meanChunkGap, gap);
    assert.equal(stream.lastTime, last);
    assert.equal(stream.chunkGaps.length, sampleCount);
    assert.equal(stream.chunkGaps.reduce((sum, value) => sum + value, 0), last - 100);
  }
  assert.deepEqual(model.streamScenario('buffered').internalGenerationTimes, [100, 130, 160, 190, 220, 250, 280]);
  assert.deepEqual(model.streamScenario('buffered').events.map(event => event.count), [1, 3, 3]);
});

test('unobservable within-event timing does not create artificial zero ITL samples', () => {
  const bundled = model.measureStream([{time: 100, count: 7}]);
  assert.equal(bundled.tpot, 0);
  assert.deepEqual(bundled.chunkGaps, []);
  assert.equal(bundled.meanChunkGap, null);
  assert.equal(model.measureStream([{time: 100, count: 1}]).tpot, null);
  assert.equal(model.measureStream([]).tpot, null);
  assert.equal(model.measureStream([]).ttft, null);
  const firstBundled = model.measureStream([{time: 100, count: 3}, {time: 160, count: 4}]);
  assert.equal(firstBundled.tpot, 10);
  assert.deepEqual(firstBundled.chunkGaps, [60]);
  assert.equal(firstBundled.meanChunkGap, 60);
});

test('stream measurement rejects malformed events rather than inventing timings', () => {
  assert.throws(() => model.measureStream([{time: 50, count: 0}]), RangeError);
  assert.throws(() => model.measureStream([{time: 50, count: 1}, {time: 40, count: 1}]), RangeError);
  assert.throws(() => model.measureStream([{time: NaN, count: 1}]), RangeError);
  assert.throws(() => model.streamScenario('unknown'), RangeError);
});
