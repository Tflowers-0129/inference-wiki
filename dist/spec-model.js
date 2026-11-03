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

  function continuation(accepted, nextK = 1) {
    const previous = verification(accepted);
    if (!Number.isInteger(nextK) || nextK < 0 || nextK > DRAFT_COUNT) {
      throw new RangeError('Next draft count must be an integer from 0 to 3');
    }
    const blockSize = 4;
    const blockTable = [7, 2, 9, 10];
    const queryTokens = ['x', ...Array.from({length: nextK}, (_, i) => `e${i + 1}`)];
    const queryPositions = queryTokens.map((_, i) => previous.validKvLength + i);
    const slotMapping = queryPositions.map(position =>
      blockTable[Math.floor(position / blockSize)] * blockSize + position % blockSize);
    return {
      previous, nextK, queryTokens, queryPositions,
      queryOffsets: [0, queryTokens.length],
      seqLength: previous.validKvLength + queryTokens.length,
      blockSize, blockTable, slotMapping
    };
  }

  function frames(accepted = 2) {
    const result = verification(accepted);
    const definitions = [
      ['anchor', '起点', '① 先分清已输出的 t0 与已有 Target KV',
        'Target KV 已覆盖位置 0–7。t0@8 已经确认输出，但 Target 尚未把 t0 作为输入计算它自己的 KV。本轮从这个 t0 开始提出候选。'],
      ['draft-1', '提议 d1', '② Draft 读取 t0，提出 d1',
        'Draft 沿已确认的 t0 提出 d1@9。d1 仍是候选，没有成为 Target 确认的新输出，也没有自己的 Target KV。'],
      ['draft-2', '提议 d2', '③ Draft 读取 d1，继续提出 d2',
        '这条自回归候选链中，d2 依赖刚才的 d1，因此先后生成。Draft 自己维护的状态与下面显示的 Target KV 分开；Target 还没有处理这些候选。'],
      ['draft-3', '提议 d3', '④ Draft 再读取 d2，得到第三个候选',
        '现在候选链是 d1、d2、d3。三个候选是串行提出的；接下来把它们和 t0 一起交给 Target。这里只演示单链提议，不代表所有提议器都逐个生成。'],
      ['metadata', '打包四行', '⑤ 把 t0 和候选打包为一次 Target 输入',
        '输入为 [t0,d1,d2,d3]，位置 8–11，query_start_loc=[0,4]，临时序列长度 12。t0 必须进入这次前向，才能产生用于验证 d1 的分布；每行读取已有历史 KV，以及截至该行的本轮输入，因果掩码屏蔽后面的候选。'],
      ['verify', 'Target 验证', '⑥ Target 一次前向计算四行，再检查候选',
        `t0 行预测 d1，d1 行预测 d2，d2 行预测 d3。这个贪心构造例连续接受 ${accepted} 个候选，${accepted === 3 ? '全部接受后，由 d3 行再采样 bonus x。' : `在第 ${accepted + 1} 个候选处不匹配，从对应的前一输入行采样纠正 Token x；后续候选不提交。`}四行并行前向不等于四个新输出。`],
      ['commit', '提交与回退', '⑦ 提交输出，标出可以继续读取的 Target KV',
        `本轮新输出 [${result.emittedTokens.join(',')}]，共 ${result.emittedCount} 个。有效 Target KV 长度为 ${result.validKvLength}，${result.invalidTailLength ? `位置 ${result.invalidPositions.join('、')} 的候选 KV 不再有效。` : '没有无效候选尾部。'}x@${result.validKvLength} 已确认输出，但尚无自己的 Target KV；t0 不重复发送。`],
      ['next', '接下一轮', '⑧ 把 x 作为下一轮输入，继续生成',
        `下一轮从 x@${result.validKvLength} 开始。下方可把下一轮候选数 K 改成 0、1、3，观察位置和 query 边界。x 的 KV 要等下一次 Target 前向计算后才产生；图中的接续行只表示已经准备好的输入。`]
    ];
    return definitions.map(([key, shortTitle, title, note], i) => ({
      key, shortTitle, title, note, result,
      proposedCount: Math.min(i, 3),
      hasMetadata: i >= 4,
      hasVerification: i >= 5,
      committed: i >= 6
    }));
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

  return {verification, continuation, frames, measureStream, streamScenario};
});
