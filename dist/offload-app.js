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
      ? '历史位置 0…22。设备仍保存第 0、4、5 页，其中最后两页保留；Hot 已有位置 6。只有 10、14 需要从 Host 搬回。'
      : '工作集第 0…5 行原有 [2,5,9,11,18,21]，第 0 行最久未用。本轮选择 [5,9,13,20]；请求未变，所有旧内容均属于稳定前缀。';
    phases.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.offloadPhase) === phase)));
    const resident = plan.rows.filter(r => r.source === 'resident').length;
    const hot = plan.rows.filter(r => r.source === 'hot').length;
    $('offloadCounts').innerHTML = `<span>选中 <b>${plan.topk.length}</b> 行</span><span>驻留页命中 <b>${phase ? resident : '—'}</b></span><span>${hi ? 'Hot' : '设备工作集'} <b>${phase ? hot : '—'}</b></span><span>需从 Host 搬回 <b>${phase ? plan.misses.length : '—'}</b></span>`;
    const labels = { resident: '驻留页 Resident', hot: hi ? '热行 Hot' : '设备工作集 命中', host: hi ? 'Host Source' : '需从 Host 搬回' };
    $('offloadRows').innerHTML = plan.rows.map(row => `<tr class="${phase ? row.source : ''}"><td>${row.token}</td><td>${phase ? labels[row.source] : '尚未查找'}</td><td>${phase && row.source === 'host' ? row.hostSlot : '—'}</td><td>${phase === 2 || (phase === 1 && row.source !== 'host') ? row.slot : '—'}</td></tr>`).join('');
    $('offloadExplanation').textContent = phase === 0
      ? 'Indexer 只选出了请求中的历史位置。下一步还要查：它们是否已在设备上，若没有，应从 Host 哪里搬回。'
      : phase === 1
        ? (hi ? '先查驻留页 Resident，再查热行 Hot。8 行中已有 6 行在设备，只需把 10、14 从 Host 搬到 Hot。' : '5、9 命中工作集第 1、2 行，保留这两行；13、20 替换最旧的第 0、3 行。位置 13 对应的 Host 行号是 block_table[3]×4+1=21，和设备目标位置不同。')
        : `搬完后仍按原 TopK 顺序，Attention 读取位置 [${plan.rows.map(r => r.slot).join(', ')}]。${hi ? '地址查找同时处理 Resident 和 Hot；本层必须等缺少的数据搬完。' : '仅搬入位置 13、20；工作集变为 [' + plan.slots.join(', ') + ']。'}`;
  }
  kind.addEventListener('change', () => { phase = 0; render(); });
  phases.forEach(button => button.addEventListener('click', () => { phase = Number(button.dataset.offloadPhase); render(); }));
  render();
})();
