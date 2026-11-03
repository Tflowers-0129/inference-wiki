/* Teaching models: frozen telemetry with local routing credits; expected expert loads, no timing model. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.LoadBalanceModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const HEAT = Object.freeze([24, 16, 8, 8, 8, 8, 4, 4]);
  const SHIFTED_HEAT = Object.freeze([8, 8, 4, 4, 24, 16, 8, 8]);
  const OLD_PLACEMENT = Object.freeze([
    Object.freeze([0, 1, 2]), Object.freeze([3, 4, 5]),
    Object.freeze([6, 7, 4]), Object.freeze([5, 6, 7])
  ]);
  const NEW_PLACEMENT = Object.freeze([
    Object.freeze([0, 1, 2]), Object.freeze([0, 1, 2]),
    Object.freeze([0, 3, 6]), Object.freeze([4, 5, 7])
  ]);

  function nonnegativeInteger(value) {
    return Number.isSafeInteger(value) && value >= 0;
  }

  function validateDplbState(state) {
    if (!state || !Array.isArray(state.waiting) || state.waiting.length < 1) {
      throw new TypeError('DPLB state needs at least one rank');
    }
    const rankCount = state.waiting.length;
    for (const field of ['waiting', 'running', 'ownInflight']) {
      if (!Array.isArray(state[field]) || state[field].length !== rankCount ||
          !Array.from(state[field]).every(nonnegativeInteger)) {
        throw new RangeError(`${field} must contain one nonnegative integer per rank`);
      }
    }
    if (!Array.isArray(state.kvUsage) || state.kvUsage.length !== rankCount ||
        !Array.from(state.kvUsage).every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
      throw new RangeError('KV usage must contain one fraction from 0 to 1 per rank');
    }
    if (!Number.isSafeInteger(state.clientCount) || state.clientCount < 1) {
      throw new RangeError('Client count must be a positive integer');
    }
    if (!nonnegativeInteger(state.cursor) || state.cursor >= rankCount) {
      throw new RangeError('Cursor must identify a rank');
    }
    if (!Array.isArray(state.assigned) ||
        !Array.from(state.assigned).every(rank => nonnegativeInteger(rank) && rank < rankCount)) {
      throw new RangeError('Assigned history must contain valid rank indices');
    }
    return rankCount;
  }

  function initialDplbState(overrides = {}) {
    const state = {
      waiting: [0, 2, 1, 0], running: [8, 2, 1, 5],
      kvUsage: [0.35, 0.4, 0.9, 0.8], ownInflight: [0, 0, 0, 0],
      clientCount: 1, cursor: 0, assigned: [], lastTarget: null,
      ...overrides
    };
    validateDplbState(state);
    return {
      ...state, waiting: [...state.waiting], running: [...state.running],
      kvUsage: [...state.kvUsage], ownInflight: [...state.ownInflight],
      assigned: [...state.assigned]
    };
  }

  function dplbScores(state, policy = 'kv') {
    validateDplbState(state);
    if (policy !== 'count' && policy !== 'kv') throw new RangeError('Unknown DPLB policy');
    return state.waiting.map((waiting, rank) => {
      const base = Math.max(state.clientCount * state.ownInflight[rank], waiting + state.running[rank]);
      const pressure = policy === 'kv' ? waiting * 6 * Math.max(0, state.kvUsage[rank] - 0.5) : 0;
      return base + pressure;
    });
  }

  function routeNext(state, policy = 'kv') {
    const scores = dplbScores(state, policy);
    const rankCount = scores.length;
    let target = state.cursor;
    for (let offset = 1; offset < rankCount; offset++) {
      const rank = (state.cursor + offset) % rankCount;
      if (scores[rank] < scores[target]) target = rank;
    }
    const next = initialDplbState(state);
    // Telemetry is not refreshed and requests do not complete in this model.
    // Waiting includes the local credit used between coordinator snapshots.
    next.waiting[target] += state.clientCount;
    next.ownInflight[target] += 1;
    next.cursor = (state.cursor + 1) % rankCount;
    next.assigned.push(target);
    next.lastTarget = target;
    next.lastScores = scores;
    return next;
  }

  function expertLoads(placement, heat = HEAT) {
    if (!Array.isArray(heat) || heat.length < 1 ||
        !Array.from(heat).every(value => Number.isFinite(value) && value >= 0)) {
      throw new RangeError('Expert heat must contain finite nonnegative values');
    }
    if (!Array.isArray(placement) || placement.length < 1 ||
        !Array.isArray(placement[0]) || placement[0].length < 1) {
      throw new RangeError('Expert placement needs nonempty rank slots');
    }
    const slotsPerRank = placement[0].length;
    const replicaCounts = heat.map(() => 0);
    for (const slots of placement) {
      if (!Array.isArray(slots) || slots.length !== slotsPerRank) {
        throw new RangeError('Every rank must have the same number of expert slots');
      }
      if (new Set(slots).size !== slots.length) {
        throw new RangeError('A logical expert cannot appear twice on the same rank');
      }
      for (const logical of slots) {
        if (!nonnegativeInteger(logical) || logical >= heat.length) {
          throw new RangeError('Expert slots must identify a logical expert');
        }
        replicaCounts[logical] += 1;
      }
    }
    if (replicaCounts.some(count => count === 0)) {
      throw new RangeError('Every logical expert must have at least one physical replica');
    }
    // An expectation under equal sharing, not a guarantee for an individual batch.
    const replicaLoads = heat.map((load, logical) => load / replicaCounts[logical]);
    const rankLoads = placement.map(slots => slots.reduce((sum, logical) => sum + replicaLoads[logical], 0));
    const total = heat.reduce((sum, load) => sum + load, 0);
    if (!Number.isFinite(total)) throw new RangeError('Total expert heat must be finite');
    const max = Math.max(...rankLoads);
    const mean = total / placement.length;
    return {replicaCounts, replicaLoads, rankLoads, total, max, mean, balance: max === 0 ? 1 : mean / max};
  }

  return {HEAT, SHIFTED_HEAT, OLD_PLACEMENT, NEW_PLACEMENT, initialDplbState, dplbScores, routeNext, expertLoads};
});
