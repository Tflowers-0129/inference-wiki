const test = require('node:test');
const assert = require('node:assert/strict');
const M = require('../dist/ep-timeline-model.js');

test('DAG scheduling rejects duplicate ids, missing dependencies and cycles', () => {
  const a = {id: 'a', lane: 'x', duration: 1, deps: []};
  assert.throws(() => M.scheduleDAG([a, a]), /unique/);
  assert.throws(() => M.scheduleDAG([{...a, deps: ['missing']}]), /Missing dependency/);
  assert.throws(() => M.scheduleDAG([{...a, release: NaN}]), /Release/);
  assert.throws(() => M.scheduleDAG([{...a, deps: ['b']}, {id: 'b', lane: 'y', duration: 1, deps: ['a']}]), /cycle/);
});

for (const profile of Object.keys(M.PROFILES)) {
  test(`${profile}: every consumer waits for its predecessors; each role remains serial`, () => {
    const comparison = M.buildComparison(profile);
    for (const method of ['mc2', 'fused']) {
      const schedule = comparison[method];
      for (const event of schedule.events) {
        for (const dep of event.deps) assert.ok(event.start >= schedule.byId[dep].end - 1e-8, `${event.id} ran before ${dep}`);
      }
      const lanes = [...new Set(schedule.events.map(event => event.lane))];
      for (const lane of lanes) {
        const events = schedule.events.filter(event => event.lane === lane).sort((a, b) => a.start - b.start);
        for (let index = 1; index < events.length; index++) assert.ok(events[index].start >= events[index - 1].end - 1e-8);
      }
      assert.equal(M.snapshot(schedule, schedule.duration).completed.length, schedule.events.length);
    }
  });
}

test('Host may finish its submissions before Device computation completes', () => {
  const {mc2, fused} = M.buildComparison();
  for (const schedule of [mc2, fused]) {
    const hostEnd = Math.max(...schedule.events.filter(event => event.kind === 'host' && event.rank === 0).map(event => event.end));
    assert.ok(hostEnd < schedule.byId[`${schedule.method}-r0-consume`].start);
    for (const host of schedule.events.filter(event => event.kind === 'host')) {
      assert.ok(host.deps.every(id => schedule.byId[id].kind === 'host'));
    }
  }
});

test('Fused models full Dispatch before SwiGLU, representative-worker GMM1 before GMM2, and two epilogue gates', () => {
  const {fused} = M.buildComparison();
  for (let rank = 0; rank < M.RANKS; rank++) {
    const get = name => fused.byId[`fused-r${rank}-${name}`];
    assert.ok(get('activation-0').deps.includes(get('dispatch-3').id));
    assert.ok(get('activation-0').deps.includes(get('gmm1-2').id));
    assert.ok(get('gmm2-0').deps.includes(get('gmm1-3').id));
    assert.ok(get('gmm2-0').deps.includes(get('activation-0').id));
    assert.ok(!get('gmm2-1').deps.includes(get('activation-1').id));
    assert.ok(get('gmm2-2').deps.includes(get('activation-1').id));
    assert.ok(get('combine-0').deps.includes(get('activation-1').id));
    assert.ok(get('combine-0').deps.includes(get('gmm2-0').id));
  }
});

test('MC2 and Fused preserve the same teaching work in each device role', () => {
  for (const profile of Object.keys(M.PROFILES)) {
    const {mc2, fused} = M.buildComparison(profile, {localFusion: false});
    for (const rank of [0, 1]) for (const kind of ['init', 'dispatch', 'gmm1', 'activation', 'gmm2', 'combine']) {
      const work = schedule => schedule.events.filter(event => event.rank === rank && event.kind === kind).reduce((sum, event) => sum + event.duration, 0);
      assert.ok(Math.abs(work(mc2) - work(fused)) < 1e-8, `${profile}: ${kind}`);
    }
  }
});

test('Input arriving on the peer is a dependency, without a fictitious all-device barrier', () => {
  const {fused} = M.buildComparison('peer');
  assert.ok(fused.byId['fused-r0-dispatch-0'].deps.includes('fused-r1-init'));
  assert.ok(fused.byId['fused-r0-dispatch-0'].start >= fused.byId['fused-r1-init'].end);
  assert.ok(fused.events.filter(event => event.kind === 'gmm2').every(event => event.deps.every(id => fused.byId[id].rank === event.rank)));
});

test('MC2 defaults to four FFN boundaries; disabling local fusion yields five; Fused has one', () => {
  const {mc2, fused, steps} = M.buildComparison();
  assert.equal(mc2.boundaries.length, M.RANKS * 4);
  assert.equal(M.buildComparison('balanced', {localFusion: false}).mc2.boundaries.length, M.RANKS * 5);
  assert.equal(mc2.events.filter(event => event.opaque).length, M.RANKS);
  assert.equal(fused.boundaries.length, M.RANKS);
  assert.ok(steps.every((value, index) => index === 0 || value > steps[index - 1]));
});

test('Fused consumer follows the actual terminal cross-rank sync and unpermute dependency', () => {
  const {fused} = M.buildComparison('peer');
  for (let rank = 0; rank < M.RANKS; rank++) {
    const get = name => fused.byId[`fused-r${rank}-${name}`];
    const peerCombine = fused.byId[`fused-r${1 - rank}-combine-3`];
    assert.ok(get('sync').deps.includes(get('combine-3').id));
    assert.ok(get('sync').deps.includes(peerCombine.id));
    assert.ok(get('unpermute').deps.includes(get('sync').id));
    assert.ok(get('consume').deps.includes(get('unpermute').id));
  }
});

test('MC2 consumes its local Combine API output without a peer Combine completion barrier', () => {
  const {mc2} = M.buildComparison('peer');
  for (let rank = 0; rank < M.RANKS; rank++) {
    const combine = mc2.byId[`mc2-r${rank}-combine`];
    const consume = mc2.byId[`mc2-r${rank}-consume`];
    assert.ok(combine.deps.includes(`mc2-r${rank}-gmm2`));
    assert.ok(combine.deps.includes(`mc2-r${1 - rank}-gmm2`));
    assert.ok(consume.deps.includes(combine.id));
    assert.ok(!consume.deps.includes(`mc2-r${1 - rank}-combine`));
    assert.ok(mc2.events.every(event => event.kind !== 'sync'));
  }
});
