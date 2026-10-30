/* Constructed teaching examples. No acceptance probabilities or hardware timing model. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.WikiSpecPlayback = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const DRAFT_COUNT = 3;
  const BASE_KV_LENGTH = 8;
  const ANCHOR_POSITION = 8;
  const CANDIDATES = ['d1', 'd2', 'd3'];

  function verification(accepted) {
    if (!Number.isInteger(accepted) || accepted < 0 || accepted > DRAFT_COUNT) {
      throw new RangeError('Accepted prefix length must be an integer from 0 to 3');
    }
    const acceptedTokens = CANDIDATES.slice(0, accepted);
    const validKvLength = BASE_KV_LENGTH + 1 + accepted;
    return {
      accepted,
      draftCount: DRAFT_COUNT,
      anchor: {token: 't0', position: ANCHOR_POSITION, alreadyConfirmed: true},
      queryTokens: ['t0', ...CANDIDATES],
      queryPositions: [8, 9, 10, 11],
      queryOffsets: [0, 4],
      baseKvLength: BASE_KV_LENGTH,
      provisionalKvLength: 12,
      validKvLength,
      invalidTailLength: DRAFT_COUNT - accepted,
      invalidPositions: Array.from({length: DRAFT_COUNT - accepted}, (_, i) => validKvLength + i),
      acceptedTokens,
      rejectedTokens: CANDIDATES.slice(accepted),
      emittedTokens: [...acceptedTokens, 'x'],
      emittedCount: accepted + 1,
      extraToken: {
        token: 'x',
        kind: accepted === DRAFT_COUNT ? 'bonus' : 'correction',
        position: validKvLength,
        hasTargetKv: false
      }
    };
  }

  function frames(accepted = 2) {
    const result = verification(accepted);
    return [
      {
        key: 'draft', title: '① Draft：先提出 3 个候选',
        note: 'Target 已有位置 0–7 的 KV。t0 位于位置 8，已确认输出，但尚未作为 Target 输入。Draft 沿 t0 提出 d1、d2、d3；候选还不能交给用户。',
        hasMetadata: false, hasVerification: false, committed: false, result
      },
      {
        key: 'metadata', title: '② 组 Metadata：这一条请求要算 4 行 Query',
        note: '本例把 t0 与 3 个候选打包输入 Target。Query 长度是 4，位置为 8–11，累计偏移为 [0, 4]，本轮序列长度上界为 12。每行仍受因果约束，不能读取后面的候选。KV 写入空间预留不等于候选已接受。',
        hasMetadata: true, hasVerification: false, committed: false, result
      },
      {
        key: 'verify', title: '③ Target：一次前向并行验证候选',
        note: '每行的 logits 预测下一位置：t0 对应 d1，d1 对应 d2，依次类推。在这个贪心示例中，只接受从头连续匹配的前缀；第一处不匹配后，后续候选即使碰巧匹配也不提交。本例手选接受长度，用来观察记账。',
        hasMetadata: true, hasVerification: true, committed: false, result
      },
      {
        key: 'commit', title: '④ 提交输出，并回退无效的逻辑长度',
        note: `提交 ${result.emittedTokens.join('、')}，共 ${result.emittedCount} 个新输出。有效 Target KV 长度为 ${result.validKvLength}；${result.invalidTailLength ? `位置 ${result.invalidPositions.join('、')} 的候选 KV 不再有效。` : '3 个候选全部接受，没有无效候选尾部。'}x 是${accepted === 3 ? '额外的 bonus' : '第一处不匹配位置的纠正'} token，尚无 Target KV；下一轮还要把 x 作为输入。逻辑回退不表示物理块已清零或归还。`,
        hasMetadata: true, hasVerification: true, committed: true, result
      }
    ];
  }

  function measureStream(events, requestStart = 0) {
    if (!Number.isFinite(requestStart)) throw new TypeError('Request start must be finite');
    let previous = requestStart;
    const copy = events.map(event => {
      if (!Number.isFinite(event.time) || event.time < previous || !Number.isInteger(event.count) || event.count < 1) {
        throw new RangeError('Stream events need nondecreasing times and positive integer token counts');
      }
      previous = event.time;
      return {time: event.time, count: event.count};
    });
    const outputTokens = copy.reduce((sum, event) => sum + event.count, 0);
    const chunkGaps = copy.slice(1).map((event, i) => event.time - copy[i].time);
    const firstTime = copy.length ? copy[0].time : null;
    const lastTime = copy.length ? copy[copy.length - 1].time : null;
    return {
      events: copy, outputTokens, firstTime, lastTime,
      ttft: firstTime === null ? null : firstTime - requestStart,
      tpot: outputTokens > 1 ? (lastTime - firstTime) / (outputTokens - 1) : null,
      chunkGaps,
      meanChunkGap: chunkGaps.length ? chunkGaps.reduce((sum, gap) => sum + gap, 0) / chunkGaps.length : null
    };
  }

  function streamScenario(name = 'specBurst') {
    const definitions = {
      single: {
        title: '逐 token 输出',
        note: '构造条件：7 个 token 分别在 100、130、160、190、220、250、280 ms 到达客户端；每次输出事件只含 1 个 token。',
        times: [100, 130, 160, 190, 220, 250, 280], counts: [1, 1, 1, 1, 1, 1, 1]
      },
      specBurst: {
        title: '投机式成组输出',
        note: '构造条件：客户端在 100、160、220 ms 分别收到 1、3、3 个 token。这里只说明成组到达怎样改变指标口径，不代表投机加速的实测值。',
        times: [100, 160, 220], counts: [1, 3, 3]
      },
      buffered: {
        title: '代理缓冲后输出',
        note: '构造条件：内部仍按第一种场景每 30 ms 生成一个 token；代理在 100、190、280 ms 分别发送 1、3、3 个。客户端看到的成组输出也可能来自缓冲，不能据此判定启用了投机。',
        times: [100, 190, 280], counts: [1, 3, 3],
        internalGenerationTimes: [100, 130, 160, 190, 220, 250, 280]
      }
    };
    const definition = definitions[name];
    if (!definition) throw new RangeError('Unknown stream teaching scenario');
    const events = definition.times.map((time, i) => ({time, count: definition.counts[i]}));
    return {
      name, title: definition.title, note: definition.note,
      internalGenerationTimes: definition.internalGenerationTimes ? [...definition.internalGenerationTimes] : null,
      ...measureStream(events)
    };
  }

  return {verification, frames, measureStream, streamScenario};
});
