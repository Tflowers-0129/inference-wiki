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

test('the four events separate draft, metadata, verification, and commitment', () => {
  const frames = model.frames(2);
  assert.deepEqual(frames.map(frame => frame.key), ['draft', 'metadata', 'verify', 'commit']);
  assert.deepEqual(frames.map(frame => frame.hasMetadata), [false, true, true, true]);
  assert.deepEqual(frames.map(frame => frame.hasVerification), [false, false, true, true]);
  assert.deepEqual(frames.map(frame => frame.committed), [false, false, false, true]);
  assert.throws(() => model.verification(-1), RangeError);
  assert.throws(() => model.verification(4), RangeError);
  assert.throws(() => model.verification(1.5), RangeError);
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
