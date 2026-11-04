(function () {
  'use strict';
  const M = window.EPCommunicationModel;
  const byId = id => document.getElementById(id);
  const lab = byId('ep-lab');
  if (!M || !lab) return;
  const locality = byId('ep-locality');
  const source = byId('ep-source');
  function payload(bytes) { return `${bytes / 1024} KiB`; }
  function matrixTable(title, data, label) {
    const head = Array.from({length: M.RANKS}, (_, rank) => `<th scope="col">目标 ${rank}</th>`).join('');
    const rows = data.matrix.map((row, rank) => `<tr><th scope="row">源 ${rank}</th>${row.map((value, destination) => `<td${rank === destination ? ' class="ep-local"' : ''}>${value}</td>`).join('')}</tr>`).join('');
    return `<section class="ep-matrix"><h5>${title}</h5><div class="ep-table-scroll"><table aria-label="${label}"><thead><tr><th scope="col">激活份数</th>${head}</tr></thead><tbody>${rows}</tbody></table></div><p><strong>${data.remoteCopies}</strong> 份远端激活 · <strong>${payload(data.payloadBytes)}</strong><br>对角线本地使用 ${data.localCopies} 份</p></section>`;
  }
  function render() {
    const result = M.compare(locality.value);
    byId('ep-scenario').textContent = M.SCENARIOS[result.scenario].description;
    byId('ep-metrics').innerHTML = [
      ['真实输入', `${result.tokenCount} token`, '每 rank 十个'],
      ['有效专家任务', `${result.assignments} 次`, `rank 0～3：${result.rankTasks.join(' / ')}`],
      ['AllGather 远端副本', `${result.allGather.remoteCopies} 份`, payload(result.allGather.payloadBytes)],
      ['定向去重远端副本', `${result.routed.remoteCopies} 份`, payload(result.routed.payloadBytes)]
    ].map(([label, value, detail]) => `<div><span>${label}</span><strong>${value}</strong><small>${detail}</small></div>`).join('');
    byId('ep-matrices').innerHTML = matrixTable('AllGather：每个 rank 获得全部输入', result.allGather, 'AllGather 源到目标激活副本矩阵') + matrixTable('定向路由：同一目标 rank 只传一份', result.routed, '按目标 rank 去重的激活副本矩阵');
    byId('ep-ledger').textContent = `任务守恒：本地专家任务 ${result.localTasks} ＋远端专家任务 ${result.remoteTasks} ＝ ${result.assignments}。专家任务按 top-k 计数；激活副本按目标 rank 计数。定向远端 payload ＝ ${result.routed.remoteCopies} × 4096 × 2 ＝ ${result.routed.payloadBytes.toLocaleString('en-US')} B。`;
    const tokens = result.tokens.filter(token => token.sourceRank === Number(source.value));
    const tokenRows = tokens.map(token => {
      const destinations = [...new Set(token.experts.map(expert => Math.floor(expert / 2)))];
      const remoteCount = destinations.filter(rank => rank !== token.sourceRank).length;
      return `<tr><th scope="row">${token.id}</th><td>${token.experts.map(expert => `E${expert} → rank ${Math.floor(expert / 2)}`).join('；')}</td><td>${destinations.join('、')}</td><td>${remoteCount}</td></tr>`;
    }).join('');
    byId('ep-token-table').innerHTML = `<table><thead><tr><th scope="col">Token</th><th scope="col">两次专家任务</th><th scope="col">去重后的目标 rank</th><th scope="col">远端副本</th></tr></thead><tbody>${tokenRows}</tbody></table>`;
  }
  locality.addEventListener('change', render);
  source.addEventListener('change', render);
  render();
})();
