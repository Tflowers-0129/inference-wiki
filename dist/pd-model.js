/* Event playback, not a wall-clock or throughput simulator.
   Pull-style PD handoff: destination KV readiness and source-reference release
   are separate events. One illustrated decode slot; normal mode can retain both
   requests' destination KV, including room for their growth. */
(function(root){
  'use strict';
  const initial={r1Source:'none',r2Source:'none',r1Target:'none',r2Target:'none',pRun:'',dRun:'',pQueue:['R1','R2'],dQueue:[],wireRequest:'',wire:'idle',progress:0,r1ReadDone:false,r2ReadDone:false,r1Notified:false,r2Notified:false,r1Done:false,r2Done:false,kvFull:false,route:'R1、R2 路由到 P',focus:'R1 / 等待 Prefill'};
  function scenario(name='normal'){
    if(!['normal','slow','kv'].includes(name))throw new Error('Unknown PD teaching scenario');
    const frames=[];let state={...initial};
    function add(key,title,note,patch={}){state={...state,...patch};frames.push({...state,pQueue:[...state.pQueue],dQueue:[...state.dQueue],key,title,note});}
    add('arrival','先进入 P 队列','R1、R2 是两条不同的用户请求。先追踪 R1，再看 R2 怎样与它重叠推进；图里只画一个 D 运行槽，正常场景的目标 KV 足以同时容纳两条请求。');
    add('prefill','P 开始处理 R1','P 读取 R1 的输入并建立 KV。R2 还在 P 队列。这里的一个演示事件不是固定时长，也不是一次引擎调度 step。',{r1Source:'building',pRun:'R1',pQueue:['R2'],focus:'R1 / Prefill'});
    add('p-ready','P 算完，源 KV 继续留着','R1 在 P 端的首 token 路径和交接元数据就绪，代理可以把交接信息交给 D。P 的工作结束了，整条请求还没结束；这里不表示客户端已经收到首 token。',{r1Source:'held',pRun:'',dQueue:[{id:'R1',reason:'等目标 KV 分配'}],route:'R1 的交接元数据送到 D',focus:'R1 / 等目标 KV'});
    add('reserve','D 先准备接收空间','D 为 R1 预留目标 KV。源端和目标端此时都占着相关空间。P 可以继续做 R2，前提是 P 端仍有可用块；P、D 并不共享一组序列预算。',{r1Target:'reserved',r2Source:'building',pRun:'R2',pQueue:[],dQueue:[{id:'R1',reason:'等远端 KV'}],route:'D 按交接元数据发起拉取',focus:'R1 / 目标块已分配'});
    add('pull','R1 的 KV 正在拉取','目标块已经分配，D 把 R1 的 KV 拉到本地。读取未完成前，P 不能覆盖这些源块。P 同时推进 R2，不需要因为 R1 正在传输而整体停机。',{r1Target:'receiving',wireRequest:'R1',wire:'pulling',progress:name==='slow'?25:50,focus:'R1 / 等远端 KV'});
    if(name==='slow'){
      add('slow-wait','传输还没结束，R2 的 P 已完成','这里额外插入一个交接事件，表示 KV 传输比前面的 P 计算慢。源端同时持有 R1、R2 的 KV。能否继续接新 P，取决于还有没有空闲块，不能只看 P 计算单元是否空闲。',{r2Source:'held',pRun:'',progress:50,dQueue:[{id:'R1',reason:'等远端 KV'},{id:'R2',reason:'等目标 KV / 交接'}]});
      add('slow-pull','链路仍在搬，源端仍要 hold','进度只表示演示阶段，不是测量百分比或带宽数据。应在实际 Trace 中对齐拉取耗时、P 源 KV 的驻留和 D 的 remote-KV 等待。',{progress:75});
    }
    add('received','目标 KV 就绪，源端尚未收到完成通知','R1 的远端读取已经结束，D 的目标 KV 可供计算。源端归还引用是另一个事件；不要把“目标数据可用”与“源块已回收”看成同一个状态。',{r1Target:'ready',r2Source:'held',pRun:'',wire:'complete',progress:100,r1ReadDone:true,dQueue:[{id:'R1',reason:'KV 就绪，等运行槽'},{id:'R2',reason:'等目标 KV / 交接'}],focus:'R1 / 目标 KV 就绪'});
    add('decode-start','R1 开始 Decode，同时通知 P','D 可以利用已就绪的本地 KV 继续生成，并向 P 发送“远端读取已完成”通知。这里没有额外要求 D 等待 P 端回收确认才开始计算。',{r1Target:'running',dRun:'R1',wireRequest:'',wire:'idle',progress:0,dQueue:[{id:'R2',reason:'等目标 KV / 交接'}],route:'D → P：R1 远端读取已完成',focus:'R1 / Decode',kvFull:name==='kv'});
    if(name==='kv'){
      add('source-release','P 已归还 R1；R2 卡在目标块分配','这个场景额外假设：R1 占用后，D 的剩余 KV 不足以接收 R2。D 忙不等于传输必停；是目的空间不足，让这次拉取尚未发起。P 继续持有 R2 的源 KV。',{r1Source:'released',r1Notified:true,wire:'blocked',wireRequest:'R2',dQueue:[{id:'R2',reason:'等目标 KV 分配'}],route:'R1 源端引用已归还',focus:'R2 / 目标 KV 不足'});
      add('kv-wait','链路可以空闲，请求仍在等','R2 还没有目标缓冲，增加链路带宽也不能直接解决这一步等待。若目标空间足够，正常场景中 R2 此时可以先传输，再等 D 的计算槽。');
      add('r1-finish','R1 完成，D 归还它的目标引用','R1 的用户请求结束，目标块引用被归还；此例假设回收后足以接收 R2。P 的 R1 源引用早已归还，D 的本地排队不会自动让 P 重新占用它。',{r1Target:'released',r1Done:true,dRun:'',kvFull:false,wire:'idle',wireRequest:'',progress:0,focus:'R1 / 完成'});
      add('r2-reserve','现在才能给 R2 分配目标块','目的空间得到满足后，D 为 R2 分配 KV，并按交接信息发起拉取。此前的等待与实际数据搬运是两个不同的区间。',{r2Target:'reserved',dQueue:[{id:'R2',reason:'等远端 KV'}],route:'D 发起 R2 拉取',focus:'R2 / 目标块已分配'});
      add('r2-pull','R2 开始拉取','R2 的源端与目标端同时驻留。源端保留到本次读取完成通知到达，目标端则继续服务后面的 Decode。',{r2Target:'receiving',wireRequest:'R2',wire:'pulling',progress:50,focus:'R2 / 等远端 KV'});
      add('r2-received','R2 的目标 KV 已就绪','数据就绪后，R2 可以进入 Decode。源端收到读取完成通知后再归还引用；两边的生命周期并不相同。',{r2Target:'ready',r2ReadDone:true,wire:'complete',progress:100,dQueue:[{id:'R2',reason:'KV 就绪，等运行槽'}],focus:'R2 / 目标 KV 就绪'});
      add('r2-decode','R2 开始生成，P 处理读取完成通知','R2 在 D 继续生成后续 token，P 归还 R2 的源端引用。块池与前缀缓存的内容仍可由引擎保留，图里表示请求不再持有这些引用。',{r2Target:'running',r2Source:'released',r2Notified:true,dRun:'R2',dQueue:[],wire:'idle',wireRequest:'',progress:0,route:'D → P：R2 远端读取已完成',focus:'R2 / Decode'});
    }else{
      add('source-release','P 归还 R1，R2 的 KV 可以先传','P 收到 R1 的读取完成通知，归还源引用。虽然 R1 占着示意 D 运行槽，但目标池还有足够空间接收 R2，因此 R2 可以提前拉取。',{r1Source:'released',r1Notified:true,r2Target:'receiving',wire:'pulling',wireRequest:'R2',progress:50,dQueue:[{id:'R2',reason:'等远端 KV'}],route:'R1 源端引用已归还；D 拉取 R2',focus:'R1 Decode / R2 交接'});
      add('r2-received','R2 已传完，等的是 D 计算槽','R2 的目标 KV 就绪，但 R1 还在生成。这是“先传输，后排队”的例子：运行槽与目标 KV 空间不是同一项资源。',{r2Target:'ready',r2ReadDone:true,wire:'complete',progress:100,dQueue:[{id:'R2',reason:'KV 就绪，等运行槽'}],focus:'R2 / 等 D 运行槽'});
      add('r2-source-release','R2 的源引用也已归还','R2 虽然还没开始 Decode，P 已不必继续持有它的源 KV。此后的等待占用 D 目标池，不能一概说成 P 源端仍然被占。',{r2Source:'released',r2Notified:true,wire:'idle',wireRequest:'',progress:0,route:'D → P：R2 远端读取已完成'});
      add('r1-finish','R1 完成，目标引用归还','R1 结束后归还 D 端目标 KV 引用。R2 的目标 KV 已经准备好，下一事件就可以使用空出的示意运行槽。',{r1Target:'released',r1Done:true,dRun:'',focus:'R1 / 完成'});
      add('r2-decode','R2 接续 Decode','R2 不需要再从头 Prefill，使用已交接的本地 KV 接续生成。其具体首 token 处理与长度记账由代理和连接器实现决定。',{r2Target:'running',dRun:'R2',dQueue:[],focus:'R2 / Decode'});
    }
    add('finish','两条请求完成，归还最后的目标引用','P 源端在各次交接确认后归还引用；D 目标端在请求完成后归还引用。这不表示预分配池消失，也不表示前缀缓存内容被立即清空。',{r2Target:'released',r2Done:true,dRun:'',dQueue:[],focus:'R1、R2 / 已完成',route:'本次两条请求均已完成'});
    return {name,frames,title:name==='normal'?'正常交接':name==='slow'?'KV 传输较慢':'D 忙且目标 KV 不足',scope:name==='kv'?'对照条件：D 的可用 KV 不足以同时容纳 R1、R2；先归还 R1 的目标引用，再接收 R2。':'对照条件：目标 KV 足以同时容纳 R1、R2 及后续增长；D 计算忙时仍可先传输 R2。'};
  }
  root.WikiPDPlayback={scenario};
})(globalThis);
