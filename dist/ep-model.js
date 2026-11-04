/* Fixed teaching routes; logical dispatch payload only, with no timing model. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.EPCommunicationModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const RANKS = 4;
  const TOKENS_PER_RANK = 10;
  const TOP_K = 2;
  const HIDDEN_SIZE = 4096;
  const BYTES_PER_ELEMENT = 2;
  const SCENARIOS = Object.freeze({
    paired: Object.freeze({label: '同 rank 双专家', description: '每个 token 选择下一 rank 的两位专家；同一激活在目标 rank 内供两次专家计算使用。'}),
    split: Object.freeze({label: '两个目标 rank', description: '每个 token 选择后续两个 rank 各一位专家；两个目标分别需要一份激活。'}),
    mixed: Object.freeze({label: '混合', description: '每个源 rank 的前五个 token 选择本 rank 两专家，后五个 token 选择后续两个 rank 各一位专家。'})
  });

  function matrix() {
    return Array.from({length: RANKS}, () => Array(RANKS).fill(0));
  }

  function buildTokens(scenario = 'paired') {
    if (!Object.hasOwn(SCENARIOS, scenario)) throw new RangeError('Unknown locality scenario');
    return Array.from({length: RANKS * TOKENS_PER_RANK}, (_, index) => {
      const sourceRank = Math.floor(index / TOKENS_PER_RANK);
      const localIndex = index % TOKENS_PER_RANK;
      const next = (sourceRank + 1) % RANKS;
      const afterNext = (sourceRank + 2) % RANKS;
      let experts;
      if (scenario === 'paired') experts = [2 * next, 2 * next + 1];
      else if (scenario === 'mixed' && localIndex < 5) experts = [2 * sourceRank, 2 * sourceRank + 1];
      else experts = [2 * next, 2 * afterNext + 1];
      return {id: `R${sourceRank}:T${localIndex}`, sourceRank, experts};
    });
  }

  function countDispatch(tokens, hiddenSize = HIDDEN_SIZE, bytesPerElement = BYTES_PER_ELEMENT) {
    if (!Array.isArray(tokens)) throw new TypeError('Tokens must be an array');
    if (![hiddenSize, bytesPerElement].every(value => Number.isSafeInteger(value) && value > 0)) {
      throw new RangeError('Payload dimensions must be positive safe integers');
    }
    const bytesPerActivation = hiddenSize * bytesPerElement;
    if (!Number.isSafeInteger(bytesPerActivation * tokens.length * RANKS)) {
      throw new RangeError('Payload exceeds safe integer precision');
    }
    const agMatrix = matrix();
    const routedMatrix = matrix();
    const assignmentMatrix = matrix();
    const expertTasks = Array(RANKS * 2).fill(0);
    const ids = new Set();
    let localTasks = 0;
    for (const token of tokens) {
      if (!token || typeof token.id !== 'string' || !token.id || ids.has(token.id)) {
        throw new RangeError('Every token needs a unique nonempty id');
      }
      ids.add(token.id);
      if (!Number.isInteger(token.sourceRank) || token.sourceRank < 0 || token.sourceRank >= RANKS) {
        throw new RangeError('Token source rank is out of range');
      }
      if (!Array.isArray(token.experts) || token.experts.length !== TOP_K ||
          new Set(token.experts).size !== TOP_K ||
          !token.experts.every(expert => Number.isInteger(expert) && expert >= 0 && expert < RANKS * 2)) {
        throw new RangeError('Each token must select two distinct valid experts');
      }
      for (let destination = 0; destination < RANKS; destination++) agMatrix[token.sourceRank][destination]++;
      const destinations = new Set();
      for (const expert of token.experts) {
        const destination = Math.floor(expert / 2);
        destinations.add(destination);
        assignmentMatrix[token.sourceRank][destination]++;
        expertTasks[expert]++;
        if (destination === token.sourceRank) localTasks++;
      }
      for (const destination of destinations) routedMatrix[token.sourceRank][destination]++;
    }
    function summarize(copies) {
      const localCopies = copies.reduce((sum, row, source) => sum + row[source], 0);
      const totalCopies = copies.flat().reduce((sum, value) => sum + value, 0);
      const remoteCopies = totalCopies - localCopies;
      return {matrix: copies, localCopies, remoteCopies, totalCopies, payloadBytes: remoteCopies * bytesPerActivation};
    }
    const assignments = tokens.length * TOP_K;
    return {
      tokenCount: tokens.length, assignments, assignmentMatrix, expertTasks,
      rankTasks: Array.from({length: RANKS}, (_, rank) => expertTasks[2 * rank] + expertTasks[2 * rank + 1]),
      localTasks, remoteTasks: assignments - localTasks, bytesPerActivation,
      allGather: summarize(agMatrix), routed: summarize(routedMatrix)
    };
  }

  function compare(scenario = 'paired') {
    const tokens = buildTokens(scenario);
    return {scenario, tokens, ...countDispatch(tokens)};
  }

  return {RANKS, TOKENS_PER_RANK, TOP_K, HIDDEN_SIZE, BYTES_PER_ELEMENT, SCENARIOS, buildTokens, countDispatch, compare};
});
