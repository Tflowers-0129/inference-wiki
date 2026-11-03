(function (root) {
  'use strict';
  const colors = {ink:'#172b4d',muted:'#475569',blue:'#1d4ed8',teal:'#0f766e',amber:'#92400e'};
  const text = (x,y,value,size=19,color=colors.ink,weight=400) => `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" font-weight="${weight}">${value}</text>`;
  const box = (x,y,w,h,fill='#fff',stroke='#cbd5e1') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="${stroke}"/>`;
  const arrow = (x,y,x2,y2,color=colors.teal) => `<path d="M${x} ${y} L${x2} ${y2}" fill="none" stroke="${color}" stroke-width="2"/><path d="M${x2-5} ${y2-7} L${x2} ${y2} L${x2+5} ${y2-7}" fill="none" stroke="${color}" stroke-width="2"/>`;

  function render(accepted=2, mode='triton') {
    if (!Number.isInteger(accepted) || accepted < 0 || accepted > 3) throw new RangeError('accepted must be 0..3');
    if (!['torch','triton'].includes(mode)) throw new RangeError('unknown device implementation');
    const count=accepted+1, rejected=3-accepted, index=accepted;
    const sampled=[...['d1','d2','d3'].slice(0,accepted),'x',...Array(3-accepted).fill('−1')];
    const fused=mode==='triton';
    let s=`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="870" viewBox="0 0 640 870" role="img" aria-labelledby="fusion-title fusion-desc"><title id="fusion-title">${fused?'Triton 融合':'Torch 分步'}：接受 ${accepted} 个候选，接续行索引为 ${index}</title><desc id="fusion-desc">同一条请求有三个候选，Device 有效采样数为 ${count}，拒绝数 ${rejected}，接续索引 ${index}。两种实现都读取 Device Tensor。Triton 在一次 kernel 内计算拒绝数和接续索引；Torch fallback 使用多个设备操作。本图不代表实测耗时，也不表示有效采样计数在这个 kernel 中计算。</desc><style>text{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif}</style>`;
    s+=box(1,1,638,868,'#fbfdff');
    s+=text(22,35,fused?'同一份 Device 数据，一次融合计算':'同一份 Device 数据，分步计算',23,colors.ink,700);
    s+=text(22,65,'单请求 K=3 · 接受 a='+accepted+' · 教学示意，非实测时长',17,colors.muted);
    s+=text(22,104,'① 已在 Device 的输入',20,colors.ink,650);
    [['候选累计数','[3]'],['有效采样数','['+count+']'],['query 边界','[0, 4]']].forEach((v,i)=>{let x=22+i*201;s+=box(x,119,193,72,'#edf5ff','#b9cdec')+text(x+13,146,v[0],17,colors.muted)+text(x+13,175,v[1],21,colors.blue,650);});
    s+=arrow(320,193,320,218);
    s+=text(22,246,'② Host 提交；Device 按依赖执行',20,colors.ink,650);
    if(fused){
      s+=box(22,263,596,184,'#e8f5f1','#94c6b8');
      s+=text(38,293,'一次 Triton launch',20,colors.teal,700);
      s+=text(38,321,'prepare_inputs_padded_kernel',17,colors.teal);
      s+=text(38,355,'K = 候选累计数差分 = 3',19);
      s+=text(38,391,`拒绝数 = K + 1 − ${count} = ${rejected}`,19);
      s+=text(38,427,`接续索引 = query_end − 1 − ${rejected} = ${index}`,19);
    }else{
      s+=box(22,263,596,184,'#f2f6fc','#b9cdec');
      s+=text(38,293,'Torch fallback · 多个 Device 操作',20,colors.blue,700);
      [['差分 / cat','候选累计数 → K = 3'],['减法 / where',`K + 1 − ${count} → 拒绝数 ${rejected}`],['索引计算',`4 − 1 − ${rejected} → 接续索引 ${index}`]].forEach((v,i)=>{let y=304+i*43;s+=box(38,y,563,37,'#fff','#cbd5e1')+text(49,y+25,v[0],17,colors.blue,600)+text(239,y+25,v[1],18);});
      s+=text(38,441,'操作框说明数据依赖，不等于精确 Kernel 个数。',15,colors.muted);
    }
    s+=arrow(320,450,320,473);
    s+=box(22,480,288,61,'#fff7ed','#e5c994')+text(37,505,'num_rejected_tokens',16,colors.muted)+text(37,529,String(rejected),21,colors.amber,700);
    s+=box(322,480,296,61,'#e8f5f1','#94c6b8')+text(337,505,'token_indices_to_sample',16,colors.muted)+text(337,529,String(index)+'  ← 展平 Target 行号',19,colors.teal,700);
    s+=text(22,578,'③ 用索引找到接续状态，配对下一 Token x',20,colors.ink,650);
    s+=text(22,608,'Target 输入 / hidden-state 行',17,colors.muted);
    ['t0','d1','d2','d3'].forEach((v,i)=>{let x=22+i*151;let selected=i===index;s+=box(x,623,143,64,selected?'#e8f5f1':'#f1f5f9',selected?'#0f766e':'#dbe3ed')+text(x+12,648,'行 '+i,16,colors.muted)+text(x+12,675,v,21,selected?colors.teal:colors.ink,650);});
    s+=text(22,719,'Device 采样结果（−1 表示无效尾部）',17,colors.muted);
    sampled.forEach((v,i)=>{let x=22+i*151;let selected=i===index;s+=box(x,733,143,51,selected?'#e8f5f1':'#f1f5f9',selected?'#0f766e':'#dbe3ed')+text(x+12,765,v,21,selected?colors.teal:colors.muted,650);});
    s+=text(22,815,`索引 ${index} 对应 Target 的 ${['t0','d1','d2','d3'][index]} 行；该处采样结果为 x。`,18,colors.teal,650);
    s+=text(22,846,'两种实现都无须为这两个输出先回读 CPU；流依赖仍保留。',17,colors.muted);
    return s+'</svg>';
  }

  const api={render};
  if(typeof module==='object'&&module.exports)module.exports=api;
  root.WikiExecutionLab=api;
  if(typeof document==='undefined')return;
  const byId=id=>document.getElementById(id);
  const mount=byId('fusion-scene');
  if(!mount)return;
  let mode='triton';
  function update(){
    const a=Number(byId('fusion-accepted').value);
    mount.innerHTML=render(a,mode);
    byId('fusion-result').textContent=`接受 ${a} 个候选 → 有效采样数 ${a+1} → 拒绝 ${3-a} 行 → 接续索引 ${a}。`;
    byId('fusion-note').textContent=mode==='triton'?'三段索引计算放进一次 Triton kernel；有效计数已由前序采样路径算好。':'无 Triton 的 fallback 也在 Device 上计算；它用多个 Tensor 操作生成相同的拒绝数和接续索引。';
    for(const value of ['torch','triton'])byId('fusion-'+value).setAttribute('aria-pressed',String(mode===value));
  }
  byId('fusion-accepted').addEventListener('change',update);
  for(const value of ['torch','triton'])byId('fusion-'+value).addEventListener('click',()=>{mode=value;update();});
  update();
})(typeof globalThis!=='undefined'?globalThis:this);
