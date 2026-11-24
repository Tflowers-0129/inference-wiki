/* Single-chain teaching scene. Stage widths are not measured durations. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  root.WikiSpecScene = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const STEPS = ['anchor', 'draft-1', 'draft-2', 'draft-3', 'metadata', 'verify', 'commit', 'next'];
  const STEP_LABELS = ['已确认的 t0', 'Draft 提议 d1', 'Draft 提议 d2', 'Draft 提议 d3', '组装 Target 输入', 'Target 并行验证', '提交与回退', '准备下一轮'];
  const GRID_X = 330;
  const PITCH = 92;
  const CELL_WIDTH = 84;
  const COLORS = {
    anchor: ['#e9f0ff', '#789ada', '#214b96'],
    draft: ['#fff4dc', '#caa355', '#805a17'],
    computed: ['#edf3fa', '#a6bed5', '#385774'],
    valid: ['#e7f5ef', '#77af98', '#195f49'],
    invalid: ['#fff0ed', '#d09a90', '#9c4136'],
    future: ['#f6f8fb', '#bcc8d6', '#66788b']
  };

  function escape(value) {
    return String(value).replace(/[&<>"']/g, character => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    })[character]);
  }

  function text(x, y, value, size = 14, fill = '#263b51', weight = 400, anchor = 'start') {
    return `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}">${escape(value)}</text>`;
  }

  function box(x, y, width, height, fill, stroke, radius = 8, extra = '') {
    return `<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}" ${extra}/>`;
  }

  function token(position, y, value, detail, tone, opacity = 1, current = false) {
    const x = GRID_X + (position - 8) * PITCH;
    const [fill, stroke, ink] = COLORS[tone];
    return `<g class="spec-scene-token${current ? ' is-current' : ''}" opacity="${opacity}">`
      + box(x, y + 3, CELL_WIDTH, 44, fill, stroke, 7, `stroke-width="${current ? 2.4 : 1.2}"${tone === 'invalid' ? ' stroke-dasharray="4 3"' : ''}`)
      + text(x + CELL_WIDTH / 2, y + 22, value, 16, ink, 700, 'middle')
      + text(x + CELL_WIDTH / 2, y + 38, detail, 10.5, ink, 400, 'middle') + '</g>';
  }

  function row(y, title, detail, active = false) {
    return box(28, y, 1044, 50, active ? '#f0f5ff' : '#f8fafc', active ? '#9ab5e3' : '#e2e9f0')
      + text(44, y + 21, title, 14, '#233c57', 700)
      + text(44, y + 39, detail, 11.5, '#5a6f84');
  }

  function render(frame, nextK = 1) {
    const phase = STEPS.indexOf(frame && frame.key);
    const result = frame && frame.result;
    const accepted = result && result.accepted;
    if (phase < 0 || !result || !Number.isInteger(accepted) || accepted < 0 || accepted > 3) {
      throw new RangeError('Scene needs a supported frame and an accepted prefix from 0 to 3');
    }
    if (!Number.isInteger(nextK) || nextK < 0 || nextK > 3) {
      throw new RangeError('Next draft count must be an integer from 0 to 3');
    }
    const proposed = frame.proposedCount === undefined ? Math.min(3, phase) : frame.proposedCount;
    if (!Number.isInteger(proposed) || proposed < 0 || proposed > 3) {
      throw new RangeError('Proposed draft count must be an integer from 0 to 3');
    }
    const hasMetadata = Boolean(frame.hasMetadata) || phase >= 4;
    const hasVerification = Boolean(frame.hasVerification) || phase >= 5;
    const committed = Boolean(frame.committed) || phase >= 6;
    const isNext = frame.key === 'next';
    const validKvLength = 9 + accepted;
    const rejected = 3 - accepted;
    const query = ['t0', 'd1', 'd2', 'd3'];
    const nextQuery = ['x', ...Array.from({length: nextK}, (_, i) => `e${i + 1}`)];
    const parts = [];

    const description = `单条请求的因果候选链。当前为${STEP_LABELS[phase]}。进入本轮前，Target KV 覆盖位置0至7；已确认输出t0位于8，但还没有自身Target KV。`
      + `Draft依次提议3个候选，Target一次处理t0与全部候选。连续接受${accepted}个后，提交${accepted + 1}个新输出，有效KV长度为${validKvLength}。`
      + `x位于${validKvLength}，尚无自身Target KV，下一轮仍须输入x。下方按教学步骤排列Host、设备上的候选生成与目标模型验证，不表示独立物理stream或设备，也不表示实测耗时。`;
    parts.push(`<svg class="spec-scene-svg" xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760" role="img" aria-labelledby="spec-scene-svg-title spec-scene-svg-desc" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif">`
      + `<title id="spec-scene-svg-title">单链投机：${escape(STEP_LABELS[phase])}</title>`
      + `<desc id="spec-scene-svg-desc">${escape(description)}</desc>`
      + '<defs><marker id="spec-scene-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#7b8fa5"/></marker></defs>');
    parts.push(box(1, 1, 1098, 758, '#ffffff', '#d4dfeb', 12));
    parts.push(text(28, 31, '单链投机 · 一条请求 R', 19, '#203b58', 750));
    parts.push(text(1072, 31, `当前：${STEP_LABELS[phase]}　｜　选定 a = ${accepted}`, 13, '#2859a2', 650, 'end'));
    parts.push(box(28, 48, 484, 41, '#edf6f2', '#c9ded4'));
    parts.push(text(44, 74, '进入本轮前：Target KV 已覆盖位置 0–7，共 8 个位置', 13, '#255c48', 600));
    parts.push(box(524, 48, 548, 41, '#edf3ff', '#c9d9f1', 8, phase === 0 ? 'stroke-width="2"' : ''));
    parts.push(text(540, 74, '起点：t0@8 已确认输出，但尚无自己的 Target KV', 13, '#28518f', 650));
    parts.push(text(44, 110, '下方按请求内的位置对齐', 11.5, '#687b8e'));
    for (let i = 0; i < 8; i++) {
      parts.push(text(GRID_X + i * PITCH + CELL_WIDTH / 2, 110, 8 + i, 11.5, '#61768b', 500, 'middle'));
    }

    parts.push(row(120, 'Draft 提议记录', '单条候选链，逐个扩展', phase >= 1 && phase <= 3));
    parts.push(token(8, 120, 't0', '本轮起始 Token', 'anchor', 1, phase === 0));
    for (let i = 1; i <= 3; i++) {
      parts.push(token(8 + i, 120, i <= proposed ? `d${i}` : '·', i <= proposed ? hasVerification ? '曾作为候选' : '候选 · 待验证' : '尚未提出', i <= proposed ? 'draft' : 'future', i <= proposed ? 1 : 0.32, phase === i));
      parts.push(text(GRID_X + i * PITCH - 4, 151, '›', 20, '#b19352', 600, 'middle').replace('<text ', `<text opacity="${i <= proposed ? 1 : 0.25}" `));
    }

    parts.push(row(180, 'Target 本轮 Query', '一次前向；每行预测下一 Token', phase === 4 || phase === 5));
    query.forEach((value, i) => {
      let detail = hasMetadata ? '已打包 · 待验证' : '尚未组装';
      let tone = hasMetadata ? 'computed' : 'future';
      if (hasVerification) {
        detail = i < accepted ? `→ d${i + 1} 接受`
          : i === accepted ? i === 3 ? '→ x（bonus）' : `→ x（拒 d${i + 1}）` : '不用于提交';
        tone = i <= accepted ? 'valid' : 'invalid';
      }
      parts.push(token(8 + i, 180, i <= proposed ? value : '·', detail, tone, hasMetadata ? 1 : 0.3, phase === 4 || phase === 5));
    });

    parts.push(row(240, 'Target KV', '历史位置 0–7 始终保留', phase === 5 || phase === 6));
    query.forEach((value, i) => {
      const valid = i <= accepted;
      const tone = committed ? valid ? 'valid' : 'invalid' : hasVerification ? 'computed' : 'future';
      const detail = committed ? valid ? '有效 KV' : '候选 KV 作废' : hasVerification ? '临时 KV 已计算' : '本轮尚未计算';
      parts.push(token(8 + i, 240, i <= proposed ? value : '·', detail, tone, hasVerification ? 1 : 0.3, phase === 6 && !valid));
    });

    parts.push(row(300, '本轮新提交输出', '不重复提交已经确认的 t0', phase === 6));
    if (committed) {
      for (let i = 0; i < accepted; i++) parts.push(token(9 + i, 300, `d${i + 1}`, '确认 · 有 KV', 'valid'));
      parts.push(token(validKvLength, 300, 'x', accepted === 3 ? 'bonus · 无 KV' : '纠正 · 无 KV', 'anchor', 1, phase === 6));
    } else {
      parts.push(text(GRID_X + 12, 331, '等待 Target 验证完成，再提交接受前缀与 x。', 13, '#8291a2'));
    }

    parts.push(row(360, '下一轮 Query', `新 K = ${nextK}；x 仍要计算自身 KV`, isNext));
    if (committed) {
      nextQuery.forEach((value, i) => parts.push(token(validKvLength + i, 360, value,
        i ? '新候选 · 待验证' : '待输入 · 无 KV', isNext ? i ? 'draft' : 'anchor' : 'future', isNext ? 1 : 0.36, isNext && i === 0)));
    } else {
      parts.push(text(GRID_X + 12, 391, '先确定有效 KV 边界，再从 x 接续；此处尚未执行。', 13, '#8291a2'));
    }
    const boundary = committed
      ? `提交后有效 Target KV 长度：${validKvLength}　｜　回退候选位置：${rejected} 个　｜　x@${validKvLength} 尚无 Target KV${isNext ? `；新 Query 共 ${nextK + 1} 行，尚未运行 Target。` : '。'}`
      : hasVerification ? 'Target 已计算 4 行；验证结果将决定哪些候选的 KV 能在下一轮使用，随后才提交输出。'
        : '候选已被 Draft 提出，并不表示它已经拥有 Target KV，也不表示可以交给客户端。';
    parts.push(text(44, 430, boundary, 12.5, '#3b5d79', 500));

    parts.push('<path d="M28 447H1072" stroke="#d7e2ed"/>');
    parts.push(text(28, 473, '从提议到下一轮：角色泳道', 17, '#203b58', 700));
    parts.push(text(1072, 473, '教学相对时序，非实测', 13, '#8a5d1e', 700, 'end'));
    parts.push(text(28, 496, 'Draft 与 Target 可由同一 Device 执行；下方区分工作角色，不对应独立物理 stream、rank 或设备。', 12, '#60758a'));

    const laneYs = [512, 572, 632];
    laneYs.forEach(y => parts.push(box(28, y, 1044, 50, '#f6f9fc', '#e1e9f1', 6)));
    parts.push(text(44, 534, 'Host', 14, '#334c65', 700), text(44, 553, '控制与输出（示意）', 11.5, '#657b90'));
    parts.push(text(44, 594, 'Device / Draft', 14, '#805a17', 700), text(44, 613, '生成候选（示意）', 11.5, '#657b90'));
    parts.push(text(44, 654, 'Device / Target', 14, '#28518f', 700), text(44, 673, '验证候选（示意）', 11.5, '#657b90'));

    const timeX = stage => 310 + stage * 94;
    parts.push(box(timeX(phase) - 2, 507, 90, 180, '#eaf1ff', '#d8e4f7', 6, 'opacity="0.68"'));
    for (let i = 0; i <= 8; i++) parts.push(`<path d="M${306 + i * 94} 508V687" stroke="#dbe5ef" stroke-dasharray="3 5"/>`);

    function dependency(destination, path) {
      parts.push(`<path d="${path}" fill="none" stroke="#7b8fa5" stroke-width="1.4" marker-end="url(#spec-scene-arrow)" opacity="${phase >= destination ? 0.9 : 0.2}"/>`);
    }
    dependency(1, 'M392 537H398V597H404');
    dependency(2, 'M486 597H498');
    dependency(3, 'M580 597H592');
    dependency(4, 'M674 597H680V537H686');
    dependency(5, 'M768 537H774V657H780');
    dependency(6, 'M862 657H868V537H874');
    dependency(7, 'M956 537H968');

    function task(stage, lane, title, detail, tone) {
      const [fill, stroke, ink] = COLORS[tone];
      const active = stage === phase;
      const x = timeX(stage), y = laneYs[lane] + 6;
      parts.push(`<g class="spec-scene-phase${active ? ' is-current' : ''}" opacity="${stage > phase ? 0.25 : 1}">`
        + box(x, y, 82, 38, fill, active ? ink : stroke, 6, `stroke-width="${active ? 2.4 : 1.2}"`)
        + text(x + 41, y + 16, title, 11.5, ink, 700, 'middle')
        + text(x + 41, y + 30, detail, 10, ink, 400, 'middle') + '</g>');
    }
    task(0, 0, 't0 已确认', '尚无自身 KV', 'anchor');
    task(1, 1, '提议 d1', '输入 t0', 'draft');
    task(2, 1, '提议 d2', '输入 d1', 'draft');
    task(3, 1, '提议 d3', '输入 d2', 'draft');
    task(4, 0, '组装 4 行', '位置 8–11', 'computed');
    task(5, 2, 'Target 前向', '一次验证 4 行', 'anchor');
    task(6, 0, committed ? `提交 ${accepted + 1} 个` : '提交输出', committed ? `有效 KV ${validKvLength}` : '更新有效边界', 'valid');
    task(7, 0, '准备下一轮', `x + ${nextK} 个候选`, 'computed');
    ['起点', 'd1', 'd2', 'd3', '准备输入', '验证', '提交', '下一轮'].forEach((name, i) => {
      parts.push(text(timeX(i) + 41, 706, name, 11.5, i === phase ? '#2859a2' : '#7a8ca0', i === phase ? 700 : 400, 'middle'));
    });
    parts.push(text(28, 733, '横轴按步骤排列，条块宽度不表示耗时。并行验证只指一次 Target 前向中的多行输入，各行仍遵守因果掩码。', 12, '#60758a'));
    parts.push('</svg>');
    return parts.join('');
  }

  function renderCompact(frame, nextK = 1) {
    const phase = STEPS.indexOf(frame && frame.key);
    const result = frame && frame.result;
    const accepted = result && result.accepted;
    if (phase < 0 || !Number.isInteger(accepted) || accepted < 0 || accepted > 3) {
      throw new RangeError('Scene needs a supported frame and an accepted prefix from 0 to 3');
    }
    if (!Number.isInteger(nextK) || nextK < 0 || nextK > 3) {
      throw new RangeError('Next draft count must be an integer from 0 to 3');
    }

    const {proposedCount, hasMetadata, hasVerification, committed} = frame;
    const {queryTokens, queryPositions, validKvLength, emittedTokens, extraToken} = result;
    const isNext = frame.key === 'next';

    function chip(label, position, tone, current = false) {
      return `<span class="spec-web-token spec-web-${tone}${current ? ' is-current' : ''}">${escape(label)}<small>@${position}</small></span>`;
    }

    const draft = queryTokens.map((label, i) => {
      const position = queryPositions[i];
      const proposed = i <= proposedCount;
      return chip(proposed ? label : '待提议', position, i === 0 ? 'anchor' : proposed ? 'draft' : 'future', phase === i);
    }).join('');

    const rows = queryTokens.map((label, i) => {
      const retained = i <= accepted;
      const kv = committed ? retained ? '有效' : '已作废' : hasVerification ? '临时已计算' : '待计算';
      let prediction = hasMetadata ? '待验证' : '待组装';
      if (hasVerification) {
        if (i < accepted) prediction = `→ d${i + 1} 接受`;
        else if (i === accepted) prediction = i === 3 ? '→ x（bonus）' : `→ x，拒 d${i + 1}`;
        else prediction = '不用于提交';
      }
      const tone = committed ? retained ? 'valid' : 'invalid' : 'pending';
      const input = i <= proposedCount ? label : '待提议';
      return `<tr class="spec-web-row-${tone}"><th scope="row">${escape(input)}<small>@${queryPositions[i]}</small></th><td>${escape(kv)}</td><td>${escape(prediction)}</td></tr>`;
    }).join('');

    const output = committed ? emittedTokens.map((label, i) =>
      chip(label, queryPositions[0] + 1 + i, label === 'x' ? 'anchor' : 'valid', label === 'x' && phase === 6)
    ).join('') : '';
    const nextQuery = committed ? [extraToken.token, ...Array.from({length: nextK}, (_, i) => `e${i + 1}`)].map((label, i) =>
      chip(label, extraToken.position + i, isNext ? i ? 'draft' : 'anchor' : 'future', isNext && i === 0)
    ).join('') : '';

    return `<div class="spec-scene-adaptive" data-phase="${escape(frame.key)}">
      <section class="spec-web-chain" aria-label="Draft 候选链">
        <div><h5>Draft 候选链</h5><p>原有 Target KV · 位置 0–7</p></div>
        <div class="spec-web-token-chain">${draft}</div>
      </section>
      <div class="spec-web-comparison">
        <section class="spec-web-target${phase === 4 || phase === 5 ? ' is-active' : ''}">
          <header class="spec-web-card-head"><h5>Target · 四行一次前向</h5><span>${hasVerification ? '验证完成' : hasMetadata ? '已组装' : '待组装'}</span></header>
          <table class="spec-web-query">
            <thead><tr><th scope="col">输入行</th><th scope="col">Target KV</th><th scope="col">预测下一 Token</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </section>
        <div class="spec-web-results">
          <section class="spec-web-commit${phase === 6 ? ' is-active' : ''}">
            <header class="spec-web-card-head"><h5>本轮新输出</h5><span>${committed ? `${emittedTokens.length} 个` : '尚未提交'}</span></header>
            ${committed ? `<div class="spec-web-tokens">${output}</div>` : ''}
            <p>有效 Target KV · 位置 0–${committed ? validKvLength - 1 : 7}${committed ? `（长 ${validKvLength}）` : ''}</p>
          </section>
          <section class="spec-web-next${isNext ? ' is-active' : ''}">
            <header class="spec-web-card-head"><h5>下一轮输入</h5><span>${isNext ? '待执行' : '待准备'}</span></header>
            ${committed ? `<div class="spec-web-tokens">${nextQuery}</div>
            <p><strong>x@${extraToken.position}</strong> 尚无 Target KV</p>` : '<p>提交后，从新采样的 x 接续。</p>'}
          </section>
        </div>
      </div>
    </div>`;
  }

  return {render, renderCompact};
});
