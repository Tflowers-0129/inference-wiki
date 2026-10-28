'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const path = require('node:path');

require(path.join(__dirname, '..', 'dist', 'model.js'));
const {simulate} = globalThis.WikiScheduler;

// Replay observable work with a separate token ledger. The first output token is
// produced by the final prefill chunk; subsequent output tokens require decode.
function checkLedger(result) {
  const {total, input, output, budget, maxSeqs, steps, requests} = result;
  const ledger = Array.from({length: total}, () => ({input: 0, output: 0}));
  let cumulativePrefill = 0;
  let cumulativeDecode = 0;

  for (const step of steps) {
    const label = `step ${step.number}`;
    const decodeIds = ledger.flatMap((r, index) =>
      r.input === input && r.output < output ? [index + 1] : []);
    assert.equal(step.decode, decodeIds.length, `${label}: decode eligibility`);
    for (const id of decodeIds) ledger[id - 1].output++;

    const chunkIds = new Set();
    for (const chunk of step.chunks) {
      assert.ok(Number.isInteger(chunk.id) && chunk.id >= 1 && chunk.id <= total,
        `${label}: valid request id`);
      assert.ok(!chunkIds.has(chunk.id), `${label}: one chunk per request`);
      assert.ok(!decodeIds.includes(chunk.id), `${label}: prefill cannot also decode`);
      chunkIds.add(chunk.id);
      const r = ledger[chunk.id - 1];
      assert.equal(chunk.continued, r.input > 0, `${label}: continued chunk label`);
      assert.ok(Number.isInteger(chunk.tokens) && chunk.tokens > 0,
        `${label}: positive whole prefill tokens`);
      assert.ok(r.input + chunk.tokens <= input, `${label}: no duplicate input work`);
      r.input += chunk.tokens;
      assert.equal(chunk.left, input - r.input, `${label}: remaining input`);
      assert.equal(chunk.complete, r.input === input, `${label}: prefill completion`);
      if (r.input === input) r.output++;
    }

    const chunkTokens = step.chunks.reduce((sum, chunk) => sum + chunk.tokens, 0);
    cumulativePrefill += chunkTokens;
    cumulativeDecode += step.decode;
    assert.equal(step.prefill, chunkTokens, `${label}: prefill token accounting`);
    assert.equal(step.used, step.decode + chunkTokens, `${label}: used token accounting`);
    assert.equal(step.used + step.free, budget, `${label}: token budget conservation`);
    assert.ok(step.used >= 0 && step.used <= budget, `${label}: token budget limit`);
    assert.equal(step.scheduled, decodeIds.length + chunkIds.size,
      `${label}: distinct scheduled sequences`);
    assert.ok(step.scheduled <= maxSeqs, `${label}: scheduled sequence limit`);

    const waiting = ledger.filter(r => r.input === 0).length;
    const prefilling = ledger.filter(r => r.input > 0 && r.input < input).length;
    const decoding = ledger.filter(r => r.input === input && r.output < output).length;
    const complete = ledger.filter(r => r.output === output).length;
    assert.equal(waiting + prefilling + decoding + complete, total,
      `${label}: all requests belong to exactly one state`);
    assert.equal(step.waiting, waiting, `${label}: waiting state`);
    assert.equal(step.prefilling, prefilling, `${label}: prefill state`);
    assert.equal(step.active, prefilling + decoding, `${label}: active state`);
    assert.equal(step.complete, complete, `${label}: complete state`);
    assert.equal(step.admitted, total - waiting, `${label}: admitted state`);
    assert.ok(step.active <= maxSeqs, `${label}: active sequence limit`);
    assert.ok(prefilling <= 1, `${label}: at most one unfinished FIFO prefill`);
    assert.ok(ledger.every(r => r.output >= 0 && r.output <= output),
      `${label}: no excess output`);
    assert.equal(ledger.reduce((sum, r) => sum + r.input, 0), cumulativePrefill,
      `${label}: cumulative input conservation`);
    assert.equal(ledger.reduce((sum, r) => sum + r.output, 0),
      cumulativeDecode + ledger.filter(r => r.input === input).length,
      `${label}: output includes exactly one token per completed prefill`);
  }

  const final = steps.at(-1);
  assert.equal(final.complete, total, 'last step completes every request');
  assert.equal(final.waiting, 0);
  assert.equal(final.active, 0);
  assert.equal(final.prefilling, 0);
  assert.equal(cumulativePrefill, total * input, 'all input tokens computed exactly once');
  assert.equal(cumulativeDecode + total, total * output,
    'all output tokens produced exactly once, including first prefill tokens');
  assert.equal(requests.length, total);
  for (const r of requests) {
    assert.equal(r.inputLeft, 0);
    assert.equal(r.inputComputed, input);
    assert.equal(r.outputLeft, 0);
    assert.equal(r.decoded, output - 1);
    assert.equal(r.done, true);
    assert.equal(r.started, true);
  }
}

for (const output of [64, 1024]) {
  test(`O=${output}: first four rounds show exact chunked prefill budgets`, () => {
    const {steps} = simulate({output});
    assert.deepEqual(steps.slice(0, 4).map(step => ({
      decode: step.decode,
      prefill: step.prefill,
      used: step.used,
      chunks: step.chunks.map(({id, tokens, left}) => ({id, tokens, left})),
    })), [
      {decode: 0, prefill: 8192, used: 8192, chunks: [
        {id: 1, tokens: 4096, left: 0}, {id: 2, tokens: 4096, left: 0},
      ]},
      {decode: 2, prefill: 8190, used: 8192, chunks: [
        {id: 3, tokens: 4096, left: 0}, {id: 4, tokens: 4094, left: 2},
      ]},
      {decode: 3, prefill: 8189, used: 8192, chunks: [
        {id: 4, tokens: 2, left: 0}, {id: 5, tokens: 4096, left: 0},
        {id: 6, tokens: 4091, left: 5},
      ]},
      {decode: 5, prefill: 8187, used: 8192, chunks: [
        {id: 6, tokens: 5, left: 0}, {id: 7, tokens: 4096, left: 0},
        {id: 8, tokens: 4086, left: 10},
      ]},
    ]);
  });

  test(`O=${output}: 200 requests conserve tokens and obey every round's limits`, () => {
    const result = simulate({output});
    checkLedger(result);
    if (output === 64) {
      assert.ok(result.peak < 200, 'short outputs turn over before concurrency reaches 200');
      assert.ok(result.steps.every(step => step.decode < 200));
    } else {
      assert.equal(result.peak, 200, 'long outputs allow concurrency to reach 200');
      assert.ok(result.steps.some(step => step.decode === 200));
    }
  });

  test(`O=${output}: 800 requests preserve sequence slots while full decode batches finish`, () => {
    const result = simulate({output, total: 800, maxSeqs: 200});
    checkLedger(result);
    if (output === 1024) {
      const finishingFullBatches = result.steps.filter(step =>
        step.decode === 200 && step.finishedNow > 0 && step.waiting > 0);
      assert.ok(finishingFullBatches.length > 0, 'fixture exercises finishing a full decode batch');
      assert.ok(finishingFullBatches.every(step => step.chunks.length === 0),
        'completed decode sequences retain their scheduling slots for the current round');
    }
  });
}

test('O=1: prefill produces the entire output without any decode rounds', () => {
  const result = simulate({output: 1});
  checkLedger(result);
  assert.ok(result.steps.every(step => step.decode === 0));
  assert.equal(result.steps.length, 100);
});
