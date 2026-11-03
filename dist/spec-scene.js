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
      + `x位于${validKvLength}，尚无自身Target KV，下一轮仍须输入x。下方按教学步骤排列Host、Device提议角色与Device目标角色，不表示独立物理stream或设备，也不表示实测耗时。`;
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
    parts.push(text(44, 110, '下方按逻辑位置对齐', 11.5, '#687b8e'));
    for (let i = 0; i < 8; i++) {
      parts.push(text(GRID_X + i * PITCH + CELL_WIDTH / 2, 110, 8 + i, 11.5, '#61768b', 500, 'middle'));
    }

    parts.push(row(120, 'Draft 提议记录', '单条候选链，逐个扩展', phase >= 1 && phase <= 3));
    parts.push(token(8, 120, 't0', '本轮起始 anchor', 'anchor', 1, phase === 0));
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
        i ? '新候选 · 待验证' : 'anchor · 无 KV', isNext ? i ? 'draft' : 'anchor' : 'future', isNext ? 1 : 0.36, isNext && i === 0)));
    } else {
      parts.push(text(GRID_X + 12, 391, '先确定有效 KV 边界，再从 x 接续；此处尚未执行。', 13, '#8291a2'));
    }
    const boundary = committed
      ? `提交后有效 Target KV 长度：${validKvLength}　｜　回退候选位置：${rejected} 个　｜　x@${validKvLength} 尚无 Target KV${isNext ? `；新 Query 共 ${nextK + 1} 行，尚未运行 Target。` : '。'}`
      : hasVerification ? 'Target 已计算 4 行；验证结果将决定哪些候选 KV 留在正确路径，随后才提交输出。'
        : '候选已被 Draft 提出，并不表示它已经拥有 Target KV，也不表示可以交给客户端。';
    parts.push(text(44, 430, boundary, 12.5, '#3b5d79', 500));

    parts.push('<path d="M28 447H1072" stroke="#d7e2ed"/>');
    parts.push(text(28, 473, '从提议到下一轮：角色泳道', 17, '#203b58', 700));
    parts.push(text(1072, 473, '教学相对时序，非实测', 13, '#8a5d1e', 700, 'end'));
    parts.push(text(28, 496, 'Draft 与 Target 可由同一 Device 执行；下方区分工作角色，不对应独立物理 stream、rank 或设备。', 12, '#60758a'));

    const laneYs = [512, 572, 632];
    laneYs.forEach(y => parts.push(box(28, y, 1044, 50, '#f6f9fc', '#e1e9f1', 6)));
    parts.push(text(44, 534, 'Host', 14, '#334c65', 700), text(44, 553, '控制与输出角色（示意）', 11.5, '#657b90'));
    parts.push(text(44, 594, 'Device / Draft', 14, '#805a17', 700), text(44, 613, '提议计算角色（示意）', 11.5, '#657b90'));
    parts.push(text(44, 654, 'Device / Target', 14, '#28518f', 700), text(44, 673, '目标计算角色（示意）', 11.5, '#657b90'));

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
    task(1, 1, '提议 d1', '沿 t0 继续', 'draft');
    task(2, 1, '提议 d2', '沿 d1 继续', 'draft');
    task(3, 1, '提议 d3', '沿 d2 继续', 'draft');
    task(4, 0, '组装 4 行', '位置 8–11', 'computed');
    task(5, 2, 'Target 前向', '一次验证 4 行', 'anchor');
    task(6, 0, committed ? `提交 ${accepted + 1} 个` : '提交输出', committed ? `有效 KV ${validKvLength}` : '更新有效边界', 'valid');
    task(7, 0, '组下一轮', `x + ${nextK} 个候选`, 'computed');
    ['起点', 'd1', 'd2', 'd3', '组输入', '验证', '提交', '下一轮'].forEach((name, i) => {
      parts.push(text(timeX(i) + 41, 706, name, 11.5, i === phase ? '#2859a2' : '#7a8ca0', i === phase ? 700 : 400, 'middle'));
    });
    parts.push(text(28, 733, '横轴按步骤排列，条块宽度不表示耗时。并行验证只指一次 Target 前向中的多行输入，各行仍遵守因果掩码。', 12, '#60758a'));
    parts.push('</svg>');
    return parts.join('');
  }

  function renderCompact(frame, nextK = 1) {
    const phase = STEPS.indexOf(frame && frame.key);
    const accepted = frame && frame.result && frame.result.accepted;
    if (phase < 0 || !Number.isInteger(accepted) || accepted < 0 || accepted > 3) {
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
    const verified = Boolean(frame.hasVerification) || phase >= 5;
    const committed = Boolean(frame.committed) || phase >= 6;
    const isNext = frame.key === 'next';
    const validKvLength = 9 + accepted;
    const nextQuery = ['x', ...Array.from({length: nextK}, (_, i) => `e${i + 1}`)];
    const output = [...['d1', 'd2', 'd3'].slice(0, accepted), 'x'];
    const parts = [];
    const description = `单链投机的紧凑纵向图，当前为${STEP_LABELS[phase]}。起点Target KV为位置0至7，t0位于8且尚无自身KV。`
      + `已提出${proposed}个候选，Target一次处理4行输入。连续接受${accepted}个后，有效KV长度为${validKvLength}，新输出是${output.join('、')}。`
      + `x位于${validKvLength}，尚无自己的Target KV；下一轮带${nextK}个新候选，Query尚未执行Target。下方角色泳道与前后依赖均为教学步骤，不代表物理stream、独立设备或实测时间。`;
    parts.push('<svg class="spec-scene-svg spec-scene-compact-svg" xmlns="http://www.w3.org/2000/svg" width="480" height="1050" viewBox="0 0 480 1050" role="img" aria-labelledby="spec-scene-compact-title spec-scene-compact-desc" font-family="system-ui, -apple-system, BlinkMacSystemFont, \'PingFang SC\', \'Microsoft YaHei\', sans-serif">'
      + `<title id="spec-scene-compact-title">单链投机：${escape(STEP_LABELS[phase])}，紧凑视图</title>`
      + `<desc id="spec-scene-compact-desc">${escape(description)}</desc>`);
    parts.push(box(1, 1, 478, 1048, '#ffffff', '#d4dfeb', 12));
    parts.push(text(16, 29, '单链投机 · 请求 R', 20, '#203b58', 750));
    parts.push(text(464, 29, `a = ${accepted}　新 K = ${nextK}`, 13, '#2859a2', 650, 'end'));
    parts.push(box(16, 43, 448, 60, '#edf3ff', '#c9d9f1'));
    parts.push(text(28, 67, '起点：Target KV 已覆盖 0–7，共 8 个位置', 14, '#28518f', 600));
    parts.push(text(28, 89, 't0@8 已确认输出，尚无自己的 Target KV。', 13, '#28518f'));

    const progressLabels = ['起点', 'd1', 'd2', 'd3', '组输入', '验证', '提交', '下一轮'];
    progressLabels.forEach((name, i) => {
      const x = 16 + i * 56;
      const active = i === phase;
      parts.push(`<g opacity="${i > phase ? 0.38 : 1}">`
        + box(x, 117, 42, 27, active ? '#dfeaff' : i < phase ? '#edf6f2' : '#f4f7fa', active ? '#537fca' : '#cbd7e3', 5, `stroke-width="${active ? 2 : 1}"`)
        + text(x + 21, 136, i + 1, 14, active ? '#214b96' : '#597186', 700, 'middle')
        + text(x + 21, 161, name, 12, active ? '#214b96' : '#6c8094', active ? 700 : 400, 'middle') + '</g>');
    });
    parts.push(box(16, 173, 448, 39, '#edf3ff', '#c9d9f1'));
    parts.push(text(28, 198, `${phase + 1}/8 · ${STEP_LABELS[phase]}`, 16, '#28518f', 700));

    function compactToken(x, y, label, detail, tone, opacity = 1, active = false) {
      const [fill, stroke, ink] = COLORS[tone];
      parts.push(`<g class="spec-scene-token${active ? ' is-current' : ''}" opacity="${opacity}">`
        + box(x, y, 104, 54, fill, stroke, 7, `stroke-width="${active ? 2.2 : 1.2}"`)
        + text(x + 52, y + 23, label, 16, ink, 700, 'middle')
        + text(x + 52, y + 44, detail, 12, ink, 400, 'middle') + '</g>');
    }
    parts.push(text(16, 237, 'Draft 逐个提议', 16, '#233c57', 700));
    parts.push(text(464, 237, `已提议 ${proposed}/3`, 13, '#80652f', 600, 'end'));
    compactToken(16, 249, 't0@8', '起始 anchor', 'anchor', 1, phase === 0);
    for (let i = 1; i <= 3; i++) {
      compactToken(16 + i * 112, 249, i <= proposed ? `d${i}@${8 + i}` : '·',
        i <= proposed ? verified ? '曾作为候选' : '候选 · 待验证' : '尚未提出',
        i <= proposed ? 'draft' : 'future', i <= proposed ? 1 : 0.32, phase === i);
    }

    parts.push(text(16, 331, 'Target：四行输入，一次前向', 16, '#233c57', 700));
    parts.push(text(16, 351, '每行只读取历史与截至自身的候选前缀。', 12, '#60758a'));
    parts.push(box(16, 362, 448, 154, '#fbfcfe', '#d7e2ed'));
    parts.push(text(28, 383, '输入行', 12, '#60758a', 600));
    parts.push(text(148, 383, '本行 Target KV', 12, '#60758a', 600));
    parts.push(text(292, 383, '预测下一个 Token', 12, '#60758a', 600));
    ['t0', 'd1', 'd2', 'd3'].forEach((value, i) => {
      const y = 391 + i * 30;
      const valid = i <= accepted;
      let prediction = hasMetadata ? '等待验证' : '尚未组装';
      if (verified) prediction = i < accepted ? `→ d${i + 1} 接受`
        : i === accepted ? i === 3 ? '→ x（bonus）' : `→ x，拒 d${i + 1}` : '不用于提交';
      const kv = committed ? valid ? '有效 KV' : '候选 KV 作废' : verified ? '临时 KV 已计算' : '尚未计算';
      const ink = committed ? valid ? '#195f49' : '#9c4136' : '#385774';
      parts.push(`<g opacity="${hasMetadata ? 1 : 0.38}">`
        + `<path d="M16 ${y - 3}H464" stroke="#e3eaf2"/>`
        + text(28, y + 17, i <= proposed ? `${value}@${8 + i}` : '·', 15, '#263b51', 650)
        + text(148, y + 17, kv, 12.5, ink, committed ? 650 : 400)
        + text(292, y + 17, prediction, 12.5, verified && i <= accepted ? '#195f49' : '#657b90', verified ? 600 : 400) + '</g>');
    });

    parts.push(box(16, 530, 448, 80, committed ? '#eaf6f0' : '#f3f7fb', committed ? '#b8d9ca' : '#d4e0ec'));
    parts.push(text(28, 554, committed
      ? `有效 Target KV：0–${validKvLength - 1}，长度 ${validKvLength}；回退 ${3 - accepted} 位`
      : '有效 Target KV：0–7，长度 8', 14.5, '#255c48', 700));
    parts.push(text(28, 578, committed ? `本轮新输出：[${output.join(', ')}]` : '本轮新输出：尚未提交', 15, '#28518f', 650));
    parts.push(text(28, 599, committed ? `x@${validKvLength} 已确认，但尚无自己的 Target KV。`
      : verified ? '四行已计算，提交时再确定有效 KV 边界。' : '候选已提出，不代表 Target 已计算这些位置。', 12.5, '#536c82'));

    parts.push(text(16, 636, `下一轮 Query · 新 K = ${nextK}`, 16, '#233c57', 700));
    parts.push(text(464, 636, '尚未执行 Target', 12, '#60758a', 600, 'end'));
    if (committed) {
      nextQuery.forEach((value, i) => compactToken(16 + i * 112, 648, `${value}@${validKvLength + i}`,
        i ? '新候选 · 待验证' : 'anchor · 无 KV', isNext ? i ? 'draft' : 'anchor' : 'future', isNext ? 1 : 0.38, isNext && i === 0));
      parts.push(text(16, 723, `query_start_loc = [0, ${nextK + 1}]；seq_len = ${validKvLength + nextK + 1}`, 13, '#536c82'));
    } else {
      parts.push(box(16, 648, 448, 54, '#f8fafc', '#e1e9f1'));
      parts.push(text(28, 680, '提交后，才从 x 接续下一轮输入。', 14, '#8193a4'));
      parts.push(text(16, 723, '新 K 决定新 Query 行数；不会改变上一轮的拒绝数。', 12.5, '#60758a'));
    }

    parts.push('<path d="M16 741H464" stroke="#d7e2ed"/>');
    parts.push(text(16, 766, '当前步骤由谁执行？', 16, '#203b58', 700));
    parts.push(text(464, 766, '教学相对时序，非实测', 12, '#8a5d1e', 700, 'end'));
    const activeLane = [0, 1, 1, 1, 0, 2, 0, 0][phase];
    const currentTasks = [
      ['t0 已确认', 'Target KV 仍只覆盖 0–7'],
      ['Draft 提议 d1', '输入 t0，产生 d1'],
      ['Draft 提议 d2', '输入 d1，产生 d2'],
      ['Draft 提议 d3', '输入 d2，产生 d3'],
      ['组装四行 Target 输入', 't0、d1、d2、d3 一起打包'],
      ['Target 一次前向验证', '四行一批，遵守因果掩码'],
      [`提交 ${accepted + 1} 个新输出`, `有效 Target KV 长度 ${validKvLength}`],
      ['准备下一轮 Query', `x + ${nextK} 个候选；Target 尚未执行`]
    ];
    const laneNames = ['Host', 'Device / Draft', 'Device / Target'];
    const laneTones = ['computed', 'draft', 'anchor'];
    laneNames.forEach((name, lane) => {
      const y = 783 + lane * 53;
      const active = lane === activeLane;
      const [fill, stroke, ink] = COLORS[laneTones[lane]];
      parts.push(box(16, y, 448, 46, active ? fill : '#f6f9fc', active ? stroke : '#e1e9f1', 6, active ? 'stroke-width="2"' : ''));
      parts.push(text(28, y + 19, name, 12.5, ink, 700));
      parts.push(text(28, y + 36, '角色（示意）', 11, '#657b90'));
      if (active) {
        parts.push(text(164, y + 19, currentTasks[phase][0], 13.5, ink, 700));
        parts.push(text(164, y + 36, currentTasks[phase][1], 12, ink));
      } else {
        const inactive = lane === 0 ? '此步的 Host 细节未展开'
          : lane === 1 ? proposed === 3 ? '本轮的三个候选已提出' : '当前步骤未展示 Draft 任务'
            : verified ? '本轮验证已完成' : '等待本轮四行输入准备好';
        parts.push(text(164, y + 28, inactive, 12, '#8495a6'));
      }
    });
    const before = [
      '前轮已确认 t0；Target KV 仍止于 7。',
      't0 已确认，Draft 才能从它继续。',
      '先有 d1，再把它作为 Draft 输入。',
      '先有 d2，再继续扩展候选链。',
      'd1、d2、d3 都已由 Draft 提出。',
      '四行输入、位置与因果边界已准备。',
      'Target 已给出连续接受长度 a。',
      `有效 KV 边界已纠正到 ${validKvLength}。`
    ];
    const after = [
      '沿 t0 逐个提议 d1、d2、d3。',
      'd1 确定后，才用它提议 d2。',
      'd2 确定后，才用它提议 d3。',
      '收齐候选，再组装四行 Target 输入。',
      'Target 将四行作为一批处理。',
      '提交接受前缀与 x，回退无效候选 KV。',
      'x 进入下一轮，仍须计算自身 KV。',
      '新 Query 准备好后，还需执行 Target。'
    ];
    parts.push(text(16, 964, `前置：${before[phase]}`, 12.5, '#536c82'));
    parts.push(text(16, 987, `随后：${after[phase]}`, 12.5, '#536c82'));
    parts.push(text(16, 1015, '角色行不对应物理 stream、rank 或独立设备。', 12, '#718497'));
    parts.push(text(16, 1035, 'Draft 与 Target 可由同一 Device 执行；块宽不表示耗时。', 12, '#718497'));
    parts.push('</svg>');
    return parts.join('');
  }

  return {render, renderCompact};
});
