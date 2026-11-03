'use strict';
const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const model = require('../dist/lb-model.js');

function almostEqual(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 1e-10, `${message || 'value'}: ${actual} != ${expected}`);
}

function freezeState(state) {
  for (const value of Object.values(state)) if (Array.isArray(value)) Object.freeze(value);
  return Object.freeze(state);
}

test('count-only and KV pressure policies send the initial request to different ranks', () => {
  const state = model.initialDplbState();
  assert.deepEqual(model.dplbScores(state, 'count'), [8, 4, 2, 5]);
  const kvScores = model.dplbScores(state, 'kv');
  [8, 4, 4.4, 5].forEach((expected, rank) => almostEqual(kvScores[rank], expected));
  assert.equal(model.routeNext(state, 'count').lastTarget, 2);
  assert.equal(model.routeNext(state, 'kv').lastTarget, 1);
  assert.equal(model.dplbScores(state)[1], 4);
  // A busy cache without waiting requests is not penalized by this formula.
  assert.equal(kvScores[3], 5);
});

test('routing preserves ownership and telemetry while updating only the chosen local credits', () => {
  const original = freezeState(model.initialDplbState());
  const next = model.routeNext(original, 'kv');
  assert.deepEqual(next.waiting, [0, 3, 1, 0]);
  assert.deepEqual(next.ownInflight, [0, 1, 0, 0]);
  assert.deepEqual(next.assigned, [1]);
  assert.equal(next.cursor, 1);
  assert.equal(next.lastTarget, 1);
  assert.deepEqual(next.lastScores, model.dplbScores(original, 'kv'));
  assert.deepEqual(next.running, original.running);
  assert.deepEqual(next.kvUsage, original.kvUsage);
  for (const field of ['waiting', 'running', 'kvUsage', 'ownInflight', 'assigned']) {
    assert.notEqual(next[field], original[field]);
  }
  assert.deepEqual(original.waiting, [0, 2, 1, 0]);
  assert.deepEqual(original.ownInflight, [0, 0, 0, 0]);
  assert.deepEqual(original.assigned, []);
  assert.equal(original.lastTarget, null);
});

test('ties scan cyclically and the scan start advances independently of the selected rank', () => {
  const state = model.initialDplbState({waiting: [0, 0, 0, 0], running: [1, 0, 0, 1], cursor: 3});
  const next = model.routeNext(state, 'count');
  assert.equal(next.lastTarget, 1, 'cyclic scan from rank 3 encounters rank 1 before rank 2');
  assert.equal(next.cursor, 0, 'the previous scan start advances, not the chosen rank');
  const noWork = model.initialDplbState({waiting: [0, 0, 0, 0], running: [0, 0, 0, 0], kvUsage: [0, 0, 0, 0]});
  let burst = noWork;
  for (let i = 0; i < 8; i++) burst = model.routeNext(burst, 'kv');
  assert.deepEqual(burst.assigned, [0, 1, 2, 3, 0, 1, 2, 3]);
  assert.deepEqual(burst.waiting, [2, 2, 2, 2]);
  assert.deepEqual(burst.ownInflight, [2, 2, 2, 2]);
  assert.deepEqual(burst.running, [0, 0, 0, 0], 'no invented service completion or scheduling event');
});

test('an in-flight floor protects against stale counts and scales credits by API client count', () => {
  const state = model.initialDplbState({
    waiting: [0, 0, 0, 0], running: [0, 0, 0, 0], kvUsage: [0.5, 0.5, 0.5, 0.5],
    ownInflight: [3, 0, 0, 0], clientCount: 2
  });
  assert.deepEqual(model.dplbScores(state, 'count'), [6, 0, 0, 0]);
  const next = model.routeNext(state, 'count');
  assert.equal(next.lastTarget, 1);
  assert.deepEqual(next.waiting, [0, 2, 0, 0]);
  assert.deepEqual(next.ownInflight, [3, 1, 0, 0]);
  assert.deepEqual(model.dplbScores(next, 'count'), [6, 2, 0, 0]);
});

test('expert placement redistributes the same 80 logical assignments with the same 12 slots', () => {
  const old = model.expertLoads(model.OLD_PLACEMENT);
  const next = model.expertLoads(model.NEW_PLACEMENT);
  assert.deepEqual(old.rankLoads, [48, 16, 8, 8]);
  assert.deepEqual(next.rankLoads, [20, 20, 20, 20]);
  assert.deepEqual(old.replicaCounts, [1, 1, 1, 1, 2, 2, 2, 2]);
  assert.deepEqual(next.replicaCounts, [3, 2, 2, 1, 1, 1, 1, 1]);
  for (const loads of [old, next]) {
    assert.equal(loads.total, 80);
    assert.equal(loads.mean, 20);
    assert.equal(loads.replicaCounts.reduce((sum, count) => sum + count, 0), 12);
    almostEqual(loads.rankLoads.reduce((sum, load) => sum + load, 0), loads.total, 'rank load conservation');
  }
  almostEqual(old.balance, 20 / 48);
  assert.equal(next.balance, 1);
  assert.equal(next.max, 20);
  assert.equal(model.OLD_PLACEMENT.flat().length, model.NEW_PLACEMENT.flat().length);
});

test('the same placement becomes unbalanced when logical expert heat drifts', () => {
  const before = model.expertLoads(model.NEW_PLACEMENT, model.HEAT);
  const after = model.expertLoads(model.NEW_PLACEMENT, model.SHIFTED_HEAT);
  assert.equal(after.total, before.total);
  assert.equal(after.rankLoads[3], 48);
  assert.equal(after.max, 48);
  assert.ok(after.balance < before.balance);
  almostEqual(after.rankLoads.reduce((sum, load) => sum + load, 0), 80);
  assert.deepEqual(after.replicaCounts, before.replicaCounts, 'a heat change does not itself move weights');
});

test('equal replica sharing is a fractional expectation and zero work has no imbalance', () => {
  const placement = [[0, 1], [0, 1], [0, 1]];
  const loads = model.expertLoads(placement, [1, 2]);
  almostEqual(loads.replicaLoads[0], 1 / 3);
  almostEqual(loads.replicaLoads[1], 2 / 3);
  loads.rankLoads.forEach(load => almostEqual(load, 1));
  assert.equal(loads.total, 3);
  assert.equal(model.expertLoads(placement, [0, 0]).balance, 1);
  assert.deepEqual(placement, [[0, 1], [0, 1], [0, 1]]);
});

test('invalid expert layouts cannot lose logical experts or manufacture work', () => {
  const invalidPlacements = [
    [[0, 1], [2]], // unequal rank capacity
    [[0, 0], [1, 2]], // duplicate replica on one rank
    [[0, 1], [0, 1]], // logical expert 2 is missing
    [[0, 1], [2, 3]], // expert index out of range
    [[0, 1], [2, -1]],
    [[0, 1], [2, 1.5]],
    [], [[], []]
  ];
  for (const placement of invalidPlacements) {
    assert.throws(() => model.expertLoads(placement, [3, 2, 1]), RangeError);
  }
  for (const heat of [[-1, 2], [NaN, 2], [Infinity, 2], Array(2), [], [Number.MAX_VALUE, Number.MAX_VALUE]]) {
    assert.throws(() => model.expertLoads([[0, 1]], heat), RangeError);
  }
  assert.throws(() => model.expertLoads([[0]], [0, 0]), /Every logical expert/);
});

test('DPLB rejects malformed snapshots and unsupported policy names', () => {
  for (const overrides of [
    {waiting: [0]}, {waiting: Array(4)}, {running: [0, -1, 0, 0]}, {ownInflight: [0, 0.5, 0, 0]},
    {kvUsage: [0, 0, 0, 1.1]}, {clientCount: 0}, {clientCount: 1.5}, {cursor: 4}, {assigned: [4]}
  ]) {
    assert.throws(() => model.initialDplbState(overrides), RangeError);
  }
  assert.throws(() => model.dplbScores(model.initialDplbState(), 'token'), RangeError);
});

test('the standalone browser script exposes the same public API as CommonJS', () => {
  const context = vm.createContext({});
  vm.runInContext(fs.readFileSync(require.resolve('../dist/lb-model.js'), 'utf8'), context);
  assert.equal(typeof context.LoadBalanceModel.routeNext, 'function');
  assert.equal(context.LoadBalanceModel.routeNext(context.LoadBalanceModel.initialDplbState(), 'kv').lastTarget, 1);
  assert.equal(context.LoadBalanceModel.expertLoads(context.LoadBalanceModel.NEW_PLACEMENT).total, 80);
});
