/* Teaching model only: small address spaces, not runnable SFA parameters. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.OffloadModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  function hostSlot(token, blockTable, blockSize) {
    return blockTable[Math.floor(token / blockSize)] * blockSize + token % blockSize;
  }
  // Protect the whole current selection before choosing an LRU victim.
  function lruPlan(initial, order, topk, stablePrefix = Infinity, requestChanged = false) {
    const slots = initial.map(t => requestChanged || t >= stablePrefix ? null : t);
    const selected = new Set(topk);
    const victims = order.filter(s => !selected.has(slots[s]));
    const misses = [], rows = topk.map(token => {
      let slot = slots.indexOf(token);
      const hit = slot !== -1;
      if (!hit) {
        slot = victims.shift();
        if (slot === undefined) throw new RangeError('selection exceeds cache capacity');
        misses.push({ token, slot, evicted: slots[slot] });
        slots[slot] = token;
      }
      return { token, slot, source: hit ? 'hot' : 'host' };
    });
    return { rows, slots, misses };
  }
  function residencyPlan(topk, residentPages, hot, blockSize, firstFreeSlot) {
    const rows = [], misses = [], staged = new Map(hot);
    let next = firstFreeSlot;
    for (const token of topk) {
      const page = Math.floor(token / blockSize);
      let source, slot;
      if (residentPages.has(page)) {
        source = 'resident'; slot = residentPages.get(page) * blockSize + token % blockSize;
      } else if (staged.has(token)) {
        source = hot.has(token) ? 'hot' : 'host'; slot = staged.get(token);
      } else {
        source = 'host'; slot = next++; staged.set(token, slot);
        misses.push({ token, slot });
      }
      rows.push({ token, slot, source });
    }
    return { rows, misses };
  }
  function example(kind) {
    const blockTable = [7, 3, 8, 5, 9, 6];
    const topk = kind === 'hisparse' ? [1, 2, 6, 10, 14, 17, 20, 21] : [5, 9, 13, 20];
    const plan = kind === 'hisparse'
      ? residencyPlan(topk, new Map([[0, 7], [4, 2], [5, 9]]), new Map([[6, 100]]), 4, 101)
      : lruPlan([2, 5, 9, 11, 18, 21], [0, 1, 2, 3, 4, 5], topk);
    plan.rows.forEach(row => { row.hostSlot = hostSlot(row.token, blockTable, 4); });
    return { kind, topk, ...plan };
  }
  return { hostSlot, lruPlan, residencyPlan, example };
});
