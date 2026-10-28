'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');

require(path.join(__dirname, '..', 'dist', 'pd-model.js'));
const {scenario} = globalThis.WikiPDPlayback;
const ids = ['R1', 'R2'];
const sourceRank = {none: 0, building: 1, held: 2, released: 3};
const targetRank = {none: 0, reserved: 1, receiving: 2, ready: 3, running: 4, released: 5};

function checkRequestLifecycle(frames, id) {
  const prefix = id.toLowerCase();
  const field = suffix => prefix + suffix;
  let previousSource = 'none';
  let previousTarget = 'none';
  let readDone = false;
  let notified = false;
  let done = false;
  let hasCompletedPrefill = false;
  let hasReceivedData = false;

  for (const frame of frames) {
    const label = `${id} at ${frame.key}`;
    const source = frame[field('Source')];
    const target = frame[field('Target')];
    assert.ok(Object.hasOwn(sourceRank, source), `${label}: known source state`);
    assert.ok(Object.hasOwn(targetRank, target), `${label}: known target state`);
    assert.ok(sourceRank[source] >= sourceRank[previousSource],
      `${label}: source lifecycle cannot run backward`);
    assert.ok(targetRank[target] >= targetRank[previousTarget],
      `${label}: destination lifecycle cannot run backward`);
    assert.ok(!readDone || frame[field('ReadDone')], `${label}: reads remain completed`);
    assert.ok(!notified || frame[field('Notified')], `${label}: completion notification persists`);
    assert.ok(!done || frame[field('Done')], `${label}: completion persists`);

    if (source === 'held') hasCompletedPrefill = true;
    if (hasCompletedPrefill && (!frame[field('ReadDone')] || !frame[field('Notified')])) {
      assert.equal(source, 'held', `${label}: source retained through read and notification`);
    }
    if (source === 'released') {
      assert.equal(frame[field('ReadDone')], true, `${label}: no release during remote reads`);
      assert.equal(frame[field('Notified')], true, `${label}: source needs completion notification`);
    }
    if (frame[field('Notified')]) {
      assert.equal(frame[field('ReadDone')], true, `${label}: notification follows remote read`);
    }

    if (target === 'receiving') {
      hasReceivedData = true;
      assert.equal(source, 'held', `${label}: receiving data requires a retained source`);
      assert.equal(frame.wire, 'pulling');
      assert.equal(frame.wireRequest, id);
      assert.equal(frame[field('ReadDone')], false, `${label}: reads still in flight`);
    }
    if (['reserved', 'receiving', 'ready', 'running'].includes(target)) {
      assert.ok(hasCompletedPrefill, `${label}: source prefill precedes direct pull handoff`);
    }
    if (['ready', 'running', 'released'].includes(target)) {
      assert.ok(hasReceivedData, `${label}: ready data must first be transferred`);
      assert.equal(frame[field('ReadDone')], true, `${label}: destination ready only after read`);
    }
    if (target === 'running') {
      assert.equal(frame.dRun, id, `${label}: running destination is used by this request`);
    }
    if (frame.dRun === id) {
      assert.equal(target, 'running', `${label}: decode requires usable destination KV`);
      assert.equal(frame[field('ReadDone')], true, `${label}: no decode of incomplete KV`);
      assert.equal(frame[field('Done')], false, `${label}: completed requests do not decode`);
    }
    if (target === 'released') {
      assert.equal(frame[field('Done')], true, `${label}: destination held through decode completion`);
    }
    if (frame[field('Done')]) {
      assert.equal(target, 'released', `${label}: completed request returns destination reference`);
      assert.notEqual(frame.dRun, id);
      assert.notEqual(frame.pRun, id);
    }

    previousSource = source;
    previousTarget = target;
    readDone = frame[field('ReadDone')];
    notified = frame[field('Notified')];
    done = frame[field('Done')];
  }

  const final = frames.at(-1);
  assert.equal(final[field('Source')], 'released', `${id}: final source reference returned`);
  assert.equal(final[field('Target')], 'released', `${id}: final destination reference returned`);
  assert.equal(final[field('Done')], true, `${id}: user request completes`);
  assert.ok(frames.some(frame => frame[field('Source')] === 'held' && !frame[field('Done')]),
    `${id}: finishing P does not finish the user request`);
}

for (const name of ['normal', 'slow', 'kv']) {
  test(`${name}: request references and destination readiness follow the handoff lifecycle`, () => {
    const {frames} = scenario(name);
    for (const id of ids) checkRequestLifecycle(frames, id);
    for (const frame of frames) {
      assert.ok(['', ...ids].includes(frame.pRun), `${frame.key}: one valid P request`);
      assert.ok(['', ...ids].includes(frame.dRun), `${frame.key}: at most one valid D request`);
      if (frame.pRun) {
        assert.notEqual(frame.pRun, frame.dRun, `${frame.key}: request cannot run P and D together`);
        assert.equal(frame[frame.pRun.toLowerCase() + 'Source'], 'building');
      }
      assert.ok(['idle', 'pulling', 'complete', 'blocked'].includes(frame.wire));
      assert.ok(frame.progress >= 0 && frame.progress <= 100);
      assert.equal(new Set(frame.pQueue).size, frame.pQueue.length);
      assert.equal(new Set(frame.dQueue.map(request => request.id)).size, frame.dQueue.length);
      for (const queued of frame.pQueue) {
        assert.ok(ids.includes(queued));
        assert.notEqual(queued, frame.pRun);
      }
      for (const queued of frame.dQueue) {
        assert.ok(ids.includes(queued.id));
        assert.notEqual(queued.id, frame.dRun);
      }
    }
    assert.equal(frames.at(-1).dRun, '');
    assert.equal(frames.at(-1).pRun, '');
    assert.deepEqual(frames.at(-1).pQueue, []);
    assert.deepEqual(frames.at(-1).dQueue, []);
  });

  test(`${name}: local ready KV permits decode before source reclamation`, () => {
    const {frames} = scenario(name);
    const start = frames.find(frame => frame.dRun === 'R1');
    assert.ok(start);
    assert.equal(start.r1Target, 'running');
    assert.equal(start.r1ReadDone, true);
    assert.equal(start.r1Source, 'held', 'D does not wait for P to reclaim its source');
    assert.equal(start.r1Notified, false, 'no source reclamation acknowledgment gates D');
    const releaseIndex = frames.findIndex(frame => frame.r1Source === 'released');
    assert.ok(releaseIndex > frames.indexOf(start));
  });
}

test('normal: R2 can transfer and become ready while R1 occupies the D slot', () => {
  const {frames} = scenario('normal');
  assert.ok(frames.some(frame => frame.dRun === 'R1' && frame.r2Target === 'receiving'));
  const ready = frames.find(frame => frame.dRun === 'R1' && frame.r2Target === 'ready');
  assert.ok(ready, 'fixture demonstrates independent compute-slot and KV-space availability');
  assert.ok(ready.dQueue.some(request => request.id === 'R2' && request.reason.includes('就绪')));
  assert.equal(ready.r2ReadDone, true);
  assert.equal(ready.kvFull, false);
});

test('kv: insufficient destination space blocks the pull even when the link is idle', () => {
  const {frames} = scenario('kv');
  const blocked = frames.filter(frame => frame.wire === 'blocked');
  assert.ok(blocked.length > 0);
  for (const frame of blocked) {
    assert.equal(frame.dRun, 'R1');
    assert.equal(frame.kvFull, true);
    assert.equal(frame.r2Target, 'none', 'R2 has no destination buffer yet');
    assert.equal(frame.r2Source, 'held', 'R2 keeps its source while waiting for destination');
    assert.equal(frame.r2ReadDone, false);
    assert.equal(frame.wireRequest, 'R2');
    assert.equal(frame.progress, 0, 'blocked pull has not begun copying bytes');
    assert.ok(frame.dQueue.some(request => request.id === 'R2' && request.reason.includes('分配')));
  }
  const firstR2Receive = frames.findIndex(frame => frame.r2Target === 'receiving');
  const r1Release = frames.findIndex(frame => frame.r1Target === 'released');
  assert.ok(firstR2Receive > r1Release, 'R1 target reference is returned before R2 starts pulling');
});

test('slow: R2 finishes prefill during R1 transfer and never returns to building', () => {
  const {frames} = scenario('slow');
  const overlapping = frames.find(frame => frame.r2Source === 'held' &&
    frame.wire === 'pulling' && frame.wireRequest === 'R1');
  assert.ok(overlapping, 'slow scenario exposes source residency after P completes');
  assert.equal(overlapping.pRun, '');
  const firstHeld = frames.findIndex(frame => frame.r2Source === 'held');
  assert.ok(frames.slice(firstHeld).every(frame => frame.r2Source !== 'building'));
  assert.ok(frames.some(frame => frame.key === 'slow-wait'));
  assert.ok(frames.some(frame => frame.key === 'slow-pull'));
});

test('every UI key jump resolves in every scenario', () => {
  const html = fs.readFileSync(path.join(__dirname, '..', 'dist', 'index.html'), 'utf8');
  const keys = [...html.matchAll(/data-pd-jump="([^"]+)"/g)].map(match => match[1]);
  assert.ok(keys.length >= 4, 'reads the actual UI jump controls');
  for (const name of ['normal', 'slow', 'kv']) {
    const {frames} = scenario(name);
    assert.equal(new Set(frames.map(frame => frame.key)).size, frames.length,
      `${name}: event keys are unique`);
    for (const key of keys) {
      assert.ok(frames.some(frame => frame.key === key), `${name}: ${key} exists`);
    }
  }
});
