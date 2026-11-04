/* Rebuild the editable still from the same teaching model used by the player. */
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const modelPath = path.join(root, 'dist/ep-timeline-model.js');
const M = require(modelPath);
const comparison = M.buildComparison('balanced', {localFusion: true});
const revision = crypto.createHash('sha256').update(fs.readFileSync(modelPath)).digest('hex').slice(0, 16);
const W = 1120, H = 1570, L = 184, R = 26, PANEL = 606;
const roles = ['host', 'boundary', 'aiv', 'aic', 'consumer'];
const roleLabels = ['Host 提交', 'Device 调用边界', 'AIV / 搬运与向量', 'AIC / 矩阵角色', '后继消费'];
const palette = {host:['#e8eef8','#385a86'],init:['#fff4df','#9b6513'],dispatch:['#f2ebff','#6736bd'],gmm1:['#eaf1ff','#2458bd'],gmm2:['#eaf1ff','#2458bd'],activation:['#fff2df','#a15c12'],combine:['#e6f4f0','#157565'],sync:['#e6f4f0','#157565'],unpermute:['#e6f4f0','#157565'],consume:['#edf2f7','#475569'],localFused:['#eef1f5','#596779']};
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const x = t => L + t / comparison.duration * (W - L - R);
const y = (panel, rank, role) => 169 + panel * PANEL + rank * 238 + roles.indexOf(role) * 40 + 44;
const label = (s, xx, yy, size=19, fill='#172b4d', weight=400, anchor='start') => `<text x="${xx}" y="${yy}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
const short = e => e.kind==='host' ? '' : e.kind==='sync' ? '↔' : e.kind==='unpermute' ? 'U' : e.kind==='consume' ? '读' : e.kind==='activation' ? (e.id.endsWith('activation-0')?'S0':e.id.endsWith('activation-1')?'S1':'S') : e.kind==='localFused' ? 'G1 + SwiGLU + Quant' : e.kind==='init' ? '准备' : e.label.replace('G1·','1·').replace('G2·','2·').replace('D · 整批','D').replace('C · 整批','C').replace('G2 · E0…E3','G2');
let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title desc" data-model-revision="${revision}"><title id="title">MC2 与自定义 Fused-MC2：源码依赖模拟泳道</title><desc id="desc">非实测。两个逻辑rank，TP1，W8A8 SiLU，每rank四个非空本地专家。普通MC2保留默认局部融合黑盒；自定义Fused-MC2展示按专家Dispatch、GMM1、3加1组SwiGLU、GMM2、Combine及跨rank同步。横轴仅为任意教学推进格，AIV和AIC行是角色聚合，不代表实际stream或全芯片屏障。静态图由同一交互模型生成。</desc><rect x="0" y="0" width="${W}" height="${H}" fill="#f8fafc"/><defs><style>text{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',sans-serif}</style><pattern id="wait" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#fff7ed"/><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" stroke="#b87528" stroke-width="1"/></pattern><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M1 1L7 4L1 7" fill="none" stroke="#aa5c18" stroke-width="1.5"/></marker></defs>`;
svg += label('MC2 与 Fused-MC2：外部边界与内部 flag，分别看什么？',28,44,27,'#172b4d',700);
svg += label('源码依赖模拟 · 稳态 / 两逻辑 rank / TP=1 / W8A8+SiLU / 无共享专家多流',28,78,18,'#475569');
for (const [panel, method] of ['mc2','fused'].entries()) {
  const schedule = comparison[method], top = 100 + panel * PANEL;
  svg += `<rect x="14" y="${top}" width="1092" height="590" rx="10" fill="#fff" stroke="#cbd5e1" stroke-width="1.5"/>`;
  svg += label(method==='mc2' ? 'A. 普通 MC2：默认局部融合保留为整体，多个外部调用接续' : 'B. 自定义 Fused-MC2：一个外部算子内，按专家和 epilogue flag 接续',28,top+31,21,'#172b4d',700);
  svg += label('模拟推进格 →',28,top+62,18,'#475569');
  const tickStep = Math.ceil(comparison.duration / 8);
  for(let t=0;t<=comparison.duration;t+=tickStep) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top+75}" y2="${top+561}" stroke="#e5ebf2" stroke-width="1"/>`;
    svg += label(t,x(t),top+62,18,'#475569',400,'middle');
  }
  for(let rank=0;rank<2;rank++) {
    svg += label(`逻辑 rank ${rank}`,28,y(panel,rank,'host')-23,20,'#172b4d',700);
    roles.forEach((role,i)=>{
      const yy=y(panel,rank,role);
      svg += label(roleLabels[i],28,yy+6,18,'#475569');
      svg += `<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#dce5ef" stroke-width="1.3"/>`;
    });
  }
  for(const e of schedule.events) {
    const opaque=schedule.events.find(q=>q.rank===e.rank&&q.opaque);
    const from=opaque&&e.kind==='combine'?Math.max(e.laneReady,opaque.end):e.laneReady;
    if(['host','consume'].includes(e.kind)||e.start-from<.8)continue;
    const yy=y(panel,e.rank,e.lane.split('-')[1]);
    svg += `<rect x="${x(from)}" y="${yy-12}" width="${x(e.start)-x(from)}" height="24" fill="url(#wait)" stroke="#bd8a4e" stroke-width=".8"/>`;
  }
  for(const e of schedule.events.filter(e=>e.opaque)) {
    const xx=x(e.start), ww=x(e.end)-xx, yy=y(panel,e.rank,'aiv');
    svg += `<rect x="${xx}" y="${yy-12}" width="${ww}" height="24" rx="4" fill="#eef1f5" stroke="#a2adba"/>`;
    svg += label('局部融合内部不展开',xx+ww/2,yy+6,18,'#596779',400,'middle');
  }
  for(const b of schedule.boundaries) {
    const yy=y(panel,b.rank,'boundary'), xx=x(b.start), ww=x(b.end)-xx;
    svg += `<rect x="${xx}" y="${yy-12}" width="${ww}" height="24" rx="4" fill="#fff" stroke="#526986" stroke-width="1.6"/>`;
    const txt=method==='fused'?'dispatch_ffn_combine':b.label==='GMM1+S+Quant'?'G1 + S + Q':b.label==='Dispatch_v2'?'Dispatch':b.label==='Combine_v2'?'Combine':b.label;
    svg += label(txt,xx+ww/2,yy+6,18,'#34516f',600,'middle');
  }
  if(method==='fused') {
    for(const [from,to] of [['dispatch-0','gmm1-0'],['gmm1-2','activation-0'],['activation-1','gmm2-2']]) {
      const a=schedule.byId[`fused-r0-${from}`], b=schedule.byId[`fused-r0-${to}`];
      const ax=x(a.end), bx=x(b.start), ay=y(panel,0,a.lane.split('-')[1]), by=y(panel,0,b.lane.split('-')[1]);
      svg += `<path d="M${ax} ${ay}C${(ax+bx)/2} ${ay},${(ax+bx)/2} ${by},${bx} ${by}" fill="none" stroke="#aa5c18" stroke-width="1.6" marker-end="url(#arrow)"/>`;
    }
  }
  for(const e of schedule.events) {
    const xx=x(e.start), ww=x(e.end)-xx, yy=y(panel,e.rank,e.lane.split('-')[1]);
    const [fill,stroke]=palette[e.kind]||palette.init;
    svg += `<g><title>${esc(e.operator||e.label)}；前驱：${esc(e.deps.join(', '))}。块宽为教学时长。</title><rect x="${xx}" y="${yy-12}" width="${Math.max(ww,2)}" height="24" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="1.3"/>`;
    const txt=short(e);
    if(txt&&ww>txt.length*8+2)svg += label(txt,xx+ww/2,yy+6,e.kind==='activation'?16:18,stroke,600,'middle');
    svg += '</g>';
  }
  svg += label(method==='mc2'?'灰块：局部融合的内部 tiling 未展开；Combine 输出可读后由本地后继消费。':'S0=e0～e2，S1=e3；G2(e2) 前等 S1；↔ 为源码中的跨 rank 完成同步，U 为归并。',28,top+578,18,'#475569');
}
svg += `<rect x="20" y="1328" width="1080" height="212" rx="10" fill="#fff7ed" stroke="#b87528" stroke-width="1.4"/>`;
svg += label('读图边界',38,1360,22,'#9b5714',700);
svg += label('D：输入搬运；1·i / 2·i：专家 i 的 GMM1 / GMM2；S：SwiGLU / 再量化；C：回传。',38,1394,18);
svg += label('斜线只表示该角色等待前驱；Host 小块是提前提交，不是等待 Device 完成。',38,1426,18);
svg += label('AIV / AIC 为教学聚合；每核 GMM1→GMM2 的顺序不等于全芯片 Cube 屏障。',38,1458,18);
svg += label('块宽与终点取自任意教学参数，不能换算为毫秒或实测加速比；交互版可查看全部依赖。',38,1490,18);
svg += label('本图仅展开仓库内 W8A8 自定义 Fused-MC2，不替代 CANN MegaMoE 的内部实现证据。',38,1522,18);
svg += '</svg>\n';
fs.writeFileSync(path.join(root,'dist/assets/ep-overlap-swimlanes.svg'),svg);
process.stdout.write('Rebuilt EP profiling SVG from the teaching event model.\n');
