(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const kind = $('offloadKind');
  if (!kind || !window.OffloadModel) return;
  let phase = 0;
  const phases = Array.from(document.querySelectorAll('[data-offload-phase]'));
  function render() {
    const plan = OffloadModel.example(kind.value), hi = kind.value === 'hisparse';
    $('offloadSetup').textContent = hi
      ? '历史位置 0…22；Resident 页为 0、4、5，两尾页保留。Hot 已有 token 6；位置 10、14 需要从 Source 回填。'
      : '同一请求、稳定前缀覆盖全部条目。slot 0…5 原有 [2,5,9,11,18,21]，LRU 从旧到新按槽号排列。本轮选 [5,9,13,20]。';
    phases.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.offloadPhase) === phase)));
    const resident = plan.rows.filter(r => r.source === 'resident').length;
    const hot = plan.rows.filter(r => r.source === 'hot').length;
    $('offloadCounts').innerHTML = `<span>选中 <b>${plan.topk.length}</b> 行</span><span>页级 Resident <b>${phase ? resident : '—'}</b></span><span>${hi ? 'Hot' : 'resident buffer'} <b>${phase ? hot : '—'}</b></span><span>Host miss <b>${phase ? plan.misses.length : '—'}</b></span>`;
    const labels = { resident: 'Resident 命中', hot: hi ? 'Hot 命中' : 'resident buffer 命中', host: hi ? 'Source miss' : 'Host miss' };
    $('offloadRows').innerHTML = plan.rows.map(row => `<tr class="${phase ? row.source : ''}"><td>${row.token}</td><td>${phase ? labels[row.source] : '待解析'}</td><td>${phase && row.source === 'host' ? row.hostSlot : '—'}</td><td>${phase === 2 || (phase === 1 && row.source !== 'host') ? row.slot : '—'}</td></tr>`).join('');
    $('offloadExplanation').textContent = phase === 0
      ? 'Indexer 给的是序列里的逻辑位置。它还没有回答数据在哪里，也没有保证该行在本轮 Attention 前已经可读。'
      : phase === 1
        ? (hi ? '优先查 Resident，再查 Hot。8 行里 6 行在 Device，只有 10、14 两行生成 Host→Hot 计划。' : '先保护命中的 slot 1、2，再从其他槽里选最旧的 0、3。token 13 的 Host 行为 block_table[3]×4+1=21，和逻辑 ID、目标槽各不相同。')
        : `回填完成后，保持原 TopK 顺序，Attention 读取槽 [${plan.rows.map(r => r.slot).join(', ')}]。${hi ? 'Resident 与 Hot 的物理索引由同一次解析汇合；本层仍须等 miss 数据就绪。' : '仅搬入 token 13、20；resident buffer 变为 [' + plan.slots.join(', ') + ']。'}`;
  }
  kind.addEventListener('change', () => { phase = 0; render(); });
  phases.forEach(button => button.addEventListener('click', () => { phase = Number(button.dataset.offloadPhase); render(); }));
  render();
})();
