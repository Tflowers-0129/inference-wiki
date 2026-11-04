/* Source-order teaching scheduler. Durations and the four local segments are illustrative. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.EPTimelineModel = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const PROFILES = Object.freeze({
    balanced: Object.freeze({label: '相近的通信与矩阵时长', submit: .35, init: 1.8, dispatch: 1.4, gmm1: 1.8, activation: .65, gmm2: 1.6, combine: 1.2, sync: .5, unpermute: .6, consume: .8, peer: 0}),
    communication: Object.freeze({label: '通信搬运较长', submit: .35, init: 1.8, dispatch: 2.6, gmm1: 1.3, activation: .65, gmm2: 1.2, combine: 2.1, sync: .5, unpermute: .6, consume: .8, peer: 0}),
    matrix: Object.freeze({label: '矩阵计算较长', submit: .35, init: 1.8, dispatch: 1.0, gmm1: 3.2, activation: .65, gmm2: 2.7, combine: .9, sync: .5, unpermute: .6, consume: .8, peer: 0}),
    peer: Object.freeze({label: '逻辑 rank 1 输入晚到', submit: .35, init: 1.8, dispatch: 1.4, gmm1: 1.8, activation: .65, gmm2: 1.6, combine: 1.2, sync: .5, unpermute: .6, consume: .8, peer: 4})
  });
  const REPO = 'https://github.com/vllm-project/vllm-ascend/blob/main/';
  const KERNEL = 'csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp';
  const QUANT = 'vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py';
  const SOURCES = Object.freeze({
    dispatch: {path: 'vllm_ascend/ops/fused_moe/token_dispatcher.py · 242–275', symbol: 'npu_moe_distribute_dispatch_v2', url: REPO + 'vllm_ascend/ops/fused_moe/token_dispatcher.py#L242-L275'},
    experts: {path: QUANT + ' · 415–482', symbol: 'grouped_matmul_swiglu_quant_weight_nz / npu_dequant_swiglu_quant', url: REPO + QUANT + '#L415-L482'},
    gmm2: {path: QUANT + ' · 539–552', symbol: 'torch_npu.npu_grouped_matmul', url: REPO + QUANT + '#L539-L552'},
    combine: {path: 'vllm_ascend/ops/fused_moe/token_dispatcher.py · 344–350', symbol: 'npu_moe_distribute_combine_v2', url: REPO + 'vllm_ascend/ops/fused_moe/token_dispatcher.py#L344-L350'},
    fused: {path: KERNEL + ' · 216–231', symbol: 'dispatch_ffn_combine / operator()', url: REPO + KERNEL + '#L216-L231'},
    fusedDispatch: {path: KERNEL + ' · 842–867', symbol: 'Dispatch / 每专家就绪 flag', url: REPO + KERNEL + '#L842-L867'},
    fusedGmm1: {path: KERNEL + ' · 450–535', symbol: 'GMM1 / WaitDispatch / Notify', url: REPO + KERNEL + '#L450-L535'},
    fusedGmm2: {path: KERNEL + ' · 555–591', symbol: 'GMM2 / WaitForVector', url: REPO + KERNEL + '#L555-L591'},
    fusedAct: {path: KERNEL + ' · 920–977', symbol: 'SwiGLU / CrossRankSync / Unpermute', url: REPO + KERNEL + '#L920-L977'},
    fusedCombine: {path: KERNEL + ' · 1025–1089', symbol: 'CombineV2 / 专家或 tile 就绪 flag', url: REPO + KERNEL + '#L1025-L1089'}
  });
  const RANKS = 2;
  const SEGMENTS = 4;
  const KIND_LABELS = Object.freeze({host: 'Host 提交', init: '准备 / 量化', dispatch: 'Dispatch', gmm1: 'GMM1', activation: 'SwiGLU / 再量化', localFused: 'GMM1+SwiGLU+Quant（内部不展开）', gmm2: 'GMM2', combine: 'Combine', sync: '跨 rank 完成同步', unpermute: '加权归并', consume: '后继消费'});

  function scheduleDAG(input) {
    if (!Array.isArray(input) || !input.length) throw new TypeError('A nonempty event list is required');
    const events = input.map(event => ({...event, deps: [...(event.deps || [])]}));
    const byId = new Map();
    for (const event of events) {
      if (!event.id || byId.has(event.id)) throw new RangeError('Events require unique ids');
      if (!event.lane || !Number.isFinite(event.duration) || event.duration <= 0) throw new RangeError('Events require a lane and positive finite duration');
      if (event.release !== undefined && (!Number.isFinite(event.release) || event.release < 0)) throw new RangeError('Release must be nonnegative');
      byId.set(event.id, event);
    }
    for (const event of events) for (const dep of event.deps) if (!byId.has(dep)) throw new RangeError(`Missing dependency: ${dep}`);
    const pending = new Set(events.map(event => event.id));
    const laneEnds = new Map();
    const scheduled = new Map();
    while (pending.size) {
      const candidates = events.filter(event => pending.has(event.id) && event.deps.every(dep => scheduled.has(dep)));
      if (!candidates.length) throw new RangeError('The dependency graph contains a cycle');
      const ready = event => Math.max(event.release || 0, ...event.deps.map(dep => scheduled.get(dep).end), laneEnds.get(event.lane) || 0);
      candidates.sort((a, b) => ready(a) - ready(b) || events.indexOf(a) - events.indexOf(b));
      const event = candidates[0];
      const laneReady = laneEnds.get(event.lane) || 0;
      const dependencyReady = Math.max(event.release || 0, 0, ...event.deps.map(dep => scheduled.get(dep).end));
      const start = Math.max(laneReady, dependencyReady);
      const result = {...event, start, end: start + event.duration, laneReady, dependencyReady,
        waitsFor: event.deps.filter(dep => Math.abs(scheduled.get(dep).end - dependencyReady) < 1e-8)};
      laneEnds.set(event.lane, result.end);
      scheduled.set(event.id, result);
      pending.delete(event.id);
    }
    const ordered = events.map(event => scheduled.get(event.id));
    return {events: ordered, duration: Math.max(...ordered.map(event => event.end)), byId: Object.fromEntries(ordered.map(event => [event.id, event]))};
  }

  function buildEvents(method, timing, localFusion) {
    const events = [];
    const boundaries = [];
    function id(rank, name) { return `${method}-r${rank}-${name}`; }
    function add(rank, name, lane, kind, label, duration, deps, extra = {}) {
      const event = {id: id(rank, name), rank, lane: `r${rank}-${lane}`, kind, label, duration,
        deps, source: SOURCES[kind === 'dispatch' || kind === 'init' ? 'dispatch' : kind === 'combine' ? 'combine' : kind === 'gmm2' ? 'gmm2' : 'experts'],
        evidence: '源码调用顺序；角色与分段为教学聚合', assumption: '块宽使用教学时长；不代表真实 trace', ...extra};
      events.push(event);
      return event.id;
    }
    // Host submits independently of device completion. Only the device DAG waits on prior results.
    for (let rank = 0; rank < RANKS; rank++) {
      const names = method === 'mc2' ? localFusion ? ['dispatch', 'local-fused', 'gmm2', 'combine', 'consume'] : ['dispatch', 'gmm1', 'activation', 'gmm2', 'combine', 'consume'] : ['fused', 'consume'];
      let previous = null;
      names.forEach((name, index) => {
        previous = add(rank, `submit-${name}`, 'host', 'host', index === names.length - 1 ? '后继提交' : method === 'fused' ? 'Fused' : `提交 ${index + 1}`,
          timing.submit, previous ? [previous] : [], {release: rank === 1 ? timing.peer : 0,
            operator: name === 'fused' ? 'dispatch_ffn_combine' : name, source: SOURCES[method === 'fused' ? 'fused' : 'experts'], evidence: '教学 Host 发起顺序；未加入逐算子 Host 同步'});
      });
    }
    for (let rank = 0; rank < RANKS; rank++) {
      const peer = 1 - rank;
      const prefix = name => id(rank, name);
      if (method === 'mc2') {
        const init = add(rank, 'init', 'aiv', 'init', '准备', timing.init, [prefix('submit-dispatch')], {operator: 'Dispatch 内准备'});
        const dispatch = add(rank, 'dispatch', 'aiv', 'dispatch', 'D · 整批', SEGMENTS * timing.dispatch, [init, id(peer, 'init')], {operator: 'MC2 Dispatch', group: 'E0…E3'});
        let g1, act;
        if (localFusion) {
          g1 = act = add(rank, 'local-fused', 'aic', 'localFused', 'G1 + S + Quant · 内部不展开', SEGMENTS * (timing.gmm1 + timing.activation), [dispatch, prefix('submit-local-fused')],
            {operator: '_C_ascend.grouped_matmul_swiglu_quant_weight_nz', group: 'E0…E3', opaque: true,
              evidence: '默认局部融合 API；只模拟总区间，未从 Fused-MC2 借用其内部微调度'});
        } else {
          g1 = add(rank, 'gmm1', 'aic', 'gmm1', 'G1 · E0…E3', SEGMENTS * timing.gmm1, [dispatch, prefix('submit-gmm1')], {operator: 'torch_npu.npu_grouped_matmul · GMM1', group: 'E0…E3'});
          act = add(rank, 'activation', 'aiv', 'activation', 'S · 整批', SEGMENTS * timing.activation, [g1, prefix('submit-activation')], {operator: 'torch.ops._C_ascend.npu_dequant_swiglu_quant', group: 'E0…E3'});
        }
        const g2 = add(rank, 'gmm2', 'aic', 'gmm2', 'G2 · E0…E3', SEGMENTS * timing.gmm2, [act, prefix('submit-gmm2')], {operator: 'GMM2', group: 'E0…E3'});
        const combine = add(rank, 'combine', 'aiv', 'combine', 'C · 整批', SEGMENTS * timing.combine, [g2, id(peer, 'gmm2'), prefix('submit-combine')], {operator: 'MC2 Combine', group: 'E0…E3',
          evidence: '调用数据依赖；所需远端专家结果在本模型中保守聚合到齐，不表示源码存在全 rank barrier'});
        const middle = localFusion ? [['local-fused', 'GMM1+S+Quant', [g1]]] : [['gmm1', 'GMM1', [g1]], ['activation', 'SwiGLU', [act]]];
        for (const [key, label, ids] of [['dispatch', 'Dispatch_v2', [init, dispatch]], ...middle, ['gmm2', 'GMM2', [g2]], ['combine', 'Combine_v2', [combine]]]) {
          boundaries.push({id: `${prefix('boundary')}-${key}`, rank, label, events: ids, source: SOURCES[key === 'dispatch' || key === 'combine' || key === 'gmm2' ? key : 'experts'], operator: label, evidence: 'Python 层可见调用边界；非 Host 等待区间'});
        }
      } else {
        const init = add(rank, 'init', 'aiv', 'init', '准备 / 计数', timing.init, [prefix('submit-fused')], {operator: 'InitRouting', source: SOURCES.fused});
        const d = [], g1 = [], g2 = [], c = [];
        for (let index = 0; index < SEGMENTS; index++) {
          d.push(add(rank, `dispatch-${index}`, 'aiv', 'dispatch', `D${index}`, timing.dispatch,
            index ? [d[index - 1]] : [init, id(peer, 'init')], {operator: '按专家 Pull Dispatch / 发布就绪 flag', group: `E${index}`, source: SOURCES.fusedDispatch}));
          g1.push(add(rank, `gmm1-${index}`, 'aic', 'gmm1', `G1·${index}`, timing.gmm1,
            index ? [d[index], g1[index - 1]] : [d[index]], {operator: 'GMM1 / 等待专家输入 flag', group: `E${index}`, source: SOURCES.fusedGmm1,
              evidence: '源码 flag 依赖；代表性 AIC worker 任务序列，非全芯片屏障'}));
        }
        const s0 = add(rank, 'activation-0', 'aiv', 'activation', 'S0 · E0–2', 3 * timing.activation, [d[3], g1[2]], {operator: 'SwiGLU + 再量化 / 第一组', group: 'E0…E2', source: SOURCES.fusedAct});
        const s1 = add(rank, 'activation-1', 'aiv', 'activation', 'S1·E3', timing.activation, [s0, g1[3]], {operator: 'SwiGLU + 再量化 / 第二组', group: 'E3', source: SOURCES.fusedAct});
        for (let index = 0; index < SEGMENTS; index++) {
          g2.push(add(rank, `gmm2-${index}`, 'aic', 'gmm2', `G2·${index}`, timing.gmm2,
            index ? [g2[index - 1], index >= 2 ? s1 : s0] : [g1[3], s0], {operator: 'GMM2 / 等待 epilogue flag', group: `E${index}`, source: SOURCES.fusedGmm2,
              evidence: '同一代表性 AIC worker 先 GMM1 后 GMM2；E2 前等待第二组 epilogue'}));
          c.push(add(rank, `combine-${index}`, 'aiv', 'combine', `C${index}`, timing.combine,
            index ? [c[index - 1], g2[index]] : [s1, g2[index]], {operator: 'Combine / 等待 GMM2 专家就绪', group: `E${index}`, source: SOURCES.fusedCombine}));
        }
        const sync = add(rank, 'sync', 'aiv', 'sync', '跨rank同步', timing.sync, [c[3], id(peer, 'combine-3')], {operator: 'CrossRankSync', source: SOURCES.fusedAct, evidence: '源码在 Combine 后执行 CrossRankSync；教学聚合为对端完成依赖'});
        const unpermute = add(rank, 'unpermute', 'aiv', 'unpermute', '归并', timing.unpermute, [sync], {operator: 'Unpermute / top-k 加权归并', source: SOURCES.fusedAct});
        boundaries.push({id: prefix('boundary-fused'), rank, label: 'dispatch_ffn_combine · 一个外部算子', events: [init, ...d, ...g1, s0, s1, ...g2, ...c, sync, unpermute], operator: 'dispatch_ffn_combine', source: SOURCES.fused,
          evidence: '一个外部自定义算子；内部 AIV 与 AIC 以 flag 协作'});
      }
      const localEnd = prefix(method === 'mc2' ? 'combine' : 'unpermute');
      add(rank, 'consume', 'consumer', 'consume', '后继消费', timing.consume, [localEnd, prefix('submit-consume')], {operator: '模型中下一消费者', source: method === 'mc2' ? SOURCES.combine : SOURCES.fusedAct,
        evidence: method === 'mc2' ? '本地 Combine API 输出就绪后消费；未添加等待 peer Combine 结束的全局屏障' : 'CrossRankSync 和 Unpermute 完成后消费；未指定真实后继算子或 stream'});
    }
    return {events, boundaries};
  }

  function buildComparison(profile = 'balanced', options = {}) {
    if (!Object.hasOwn(PROFILES, profile)) throw new RangeError('Unknown teaching timing profile');
    const timing = PROFILES[profile];
    const localFusion = options.localFusion !== false;
    const result = {};
    for (const method of ['mc2', 'fused']) {
      const input = buildEvents(method, timing, localFusion);
      const schedule = scheduleDAG(input.events);
      const boundaries = input.boundaries.map(boundary => ({...boundary,
        start: Math.min(...boundary.events.map(id => schedule.byId[id].start)),
        end: Math.max(...boundary.events.map(id => schedule.byId[id].end))}));
      result[method] = {...schedule, boundaries, method, label: method === 'mc2' ? 'MC2：多个外部调用' : 'Fused-MC2：一个外部调用'};
    }
    const duration = Math.max(result.mc2.duration, result.fused.duration);
    return {profile, timing, localFusion, ...result, duration, steps: [...new Set([0, ...Object.values(result).flatMap(schedule => schedule.events.flatMap(event => [event.start, event.end]))])].sort((a, b) => a - b),
      assumptions: ['两逻辑 rank，各四个本地专家段 E0…E3；分段数用于读图，不等于设备 tiling。', 'AIV 与 AIC 是角色轨道；代表性 worker 的顺序不能推出全芯片 barrier。', '同一教学工作量、任意教学时长；横轴不能换算为毫秒或预测加速比。']};
  }

  function snapshot(schedule, time) {
    if (!Number.isFinite(time)) throw new TypeError('Time must be finite');
    const clamped = Math.max(0, Math.min(time, schedule.duration));
    return {time: clamped, completed: schedule.events.filter(event => event.end <= clamped),
      active: schedule.events.filter(event => event.start <= clamped && event.end > clamped),
      pending: schedule.events.filter(event => event.start > clamped)};
  }

  return {PROFILES, SOURCES, RANKS, SEGMENTS, KIND_LABELS, scheduleDAG, buildComparison, snapshot};
});
