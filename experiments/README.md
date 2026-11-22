# Wiki 实操实验

客户端的文字突然隔很久才刷新，设备一定也慢下来了吗？可以先看表里的 SSE 缓冲实验。它用真实 HTTP 和代理检查发送节奏怎样影响客户端计时。若想查的是设备上发生了什么，可以从 A5 专家通信、上下文并行或 PD 交接的记录中按现象选一项。每项先交代实际跑了什么，再说明结果能支持什么结论。

表中前四项结果采集于 2026-11-04。[A5 结果](results/a5-moe.json)记录对应源码、软件组合和量化条件。`padded-inputs.py` 与 `batch-change.py` 保存的是早期辅助函数结果，覆盖范围止于各自测过的函数。正文 ModelRunnerV2 的输入、采样和状态更新调用链以源码为依据。

上下文并行和 PD 分离实验均采集于 2026-11-05，各自保留配置、结果和复现方法。[上下文并行实验](context-parallel/README.md) · [PD 交接实验](pd-disaggregation/README.md)

2026-11-16 的并行协议实验在服务器 CPU 上核对 DP 协调与 SP 通信。2026-11-18 在 A5 上另采了 HCCL 图探针、SP 两路径与 MC2 eager/replay 对照，结果保留设备既有告警和各自的验证范围。[并行协议实验](parallel-graph/README.md) · [SP 与图回放对照](profiling-ab/README.md)

公开目录提供脚本、数值结果和选定的设备事件。

| 实验 | 实际执行内容 | 已确认的范围 |
| --- | --- | --- |
| [padded-inputs.py](padded-inputs.py) | Ascend950DT 上的实际 Triton kernel、Torch 表达式和独立整数 oracle | 7 组 ×2 次正确性通过；预热后每条 Torch 链 9 个设备任务，Triton 1 个 |
| [batch-change.py](batch-change.py) | 目标安装版本的进度修正辅助函数 | 请求重排、新请求和旧候选参与条件；6 次检查通过，未运行完整 Runner |
| [sse-buffering.py](sse-buffering.py) | Linux loopback 的真实 HTTP/SSE 服务、缓冲代理和客户端 | 4 场景 ×2 轮，输出内容与计数保持，实际事件间隔和 TPOT 口径通过 |
| [a5-moe.py](a5-moe.py) | 相同 W4A8MXFP 权重、输入和路由下的 MC2 拆分链与 CANN MegaMoE | EP2，8/32 行，两个 rank 输出逐元素一致；独立专家计数通过，8 份 NPU trace 已采 |
| [上下文并行](context-parallel/README.md) | PCP 源码方法回放、实际 Q 交换/FIA/DCP 归并、三种双卡 ModelRunnerV2 拓扑 | 16 组布局回放、12 条算子检查通过；三拓扑两轮 greedy tokens 一致，保留 logprob 差异；真实 Perfetto 截图另列来源 |
| [PD 分离系统](pd-disaggregation/README.md) | 四 NPU 上的实际 P TP2、D TP2、官方 HTTP proxy 与 Mooncake；分别运行拉取 GQA/DCP1、逐层推送 MQA/DCP2 | 每组四次测量的 greedy 输出与各自本地参考一致；双 rank READ／逐层 WRITE、D 就绪及请求释放调用有记录；首次 push 预热的生命周期边界单独保留 |
| [DP 协调与 SP 的 CPU 检查](parallel-graph/README.md) | 真实 CPU/Gloo 双进程；固定源码的 DP 协调函数与受控图目录，另做 ReduceScatter／Norm／AllGather 对照 | 历史 CPU DP 共 20 条 rank-case；CPU SP 共 16 条 rank-case；不作为 NPU 性能数据 |
| [DP 的 A5 图探针](parallel-graph/results/a5-npu-20261118.json) | 固定协议 AST＋真实 HCCL/NPUGraph，标量专家教学 payload | 8 次捕获及零输入回放、10 条 rank-case、16 次输出检查；无性能或整模型结论 |
| [SP 与 MC2 图回放前后对照](profiling-ab/README.md) | SP 的两种 HCCL/Norm 路径；同一 MC2 链 eager/replay，分别采两 rank | SP 400 个、MC2 480 个无 profiler 样本，12 份 Trace；带告警条件及回放 flow 缺口分别说明 |

## 特性前后对照的固定流程

以后用机器验证优化特性时，按下列顺序执行并在实验记录中逐项交付。正确性、性能测量和 profiling 各自回答不同的问题：结果是否仍然正确、整体是否改善、执行过程为什么变化。只有一张开启特性后的截图，不能完成前后对照。

| 步骤 | 必须记录的内容 | 检查什么 |
| --- | --- | --- |
| 1. 固定对照条件 | 前后完整命令、源码与软件环境、模型和权重、硬件及拓扑、输入与上下文分布、并发、随机种子；列出特性开关及必要的联动配置 | 除待验证特性外是否可比；无法保持一致的条件如何限制结论 |
| 2. 先核对正确性 | 相同输入的输出、误差或任务计数；涉及 KV、投机或异步时补有效范围与生命周期 | 优化前后是否完成同一件工作；不是只确认进程没报错 |
| 3. 关闭 profiler 测性能 | 预热方式、采样区间、交叉执行顺序、重复次数、原始样本与分布 | 算子/整轮耗时；服务实验另测吞吐、TTFT、ITL/TPOT，写清请求与输出计数口径 |
| 4. 同条件采集前后 Trace | 两套配置的原始采集目录及校验值、采集级别、job/rank/step/layer/microbatch 标记、Host 与设备关联字段 | 确认真正启用了特性、选到了对应路径；分开预热、捕获、稳态回放，记录采集扰动和缺失字段 |
| 5. 画前后 timeline | 同一比较范围、时间单位与比例，相关 rank 的真实计算、通信、拷贝、同步轨道；标出关键事件与依赖 | 哪些任务消失、合并、移动或重叠；等待是否缩短，还是转移到下游；未能归因的空白保留为未知 |
| 6. 核对解释与交付 | 前后配置表、正确性结果、无 profiler 性能表、timeline 及原始截图/事件数据、结论和未确认项 | timeline 的变化能否解释测得的指标；不能把单次采集的条宽比当稳态加速比 |

原始 Trace 保存在实验归档中，公开仓库只放经检查可发布的选定事件、截图和来源校验值。任务结束时逐项说明这六步是否完成。某项资源或证据缺失，可以发布范围明确的局部观察；“性能验证完成”必须有对应性能测量。算子实验不能替代完整模型或服务实验。

### 多 rank timeline 怎样放在一起

涉及 EP、TP/CP、PD、负载均衡或通信重叠时，默认同时检查相关 rank；单 rank 只适合解释已限定的本地行为。按“前后配置 → rank → 实际轨道”组织图，每个 rank 保留同一轮相关的计算、通信、拷贝及 `EVENT RECORD/WAIT`。若问题在 CPU 提交、metadata 回读或 tiling 计划生成，还要采集 Host 线程和 Host↔Device 关联。大规模实验先汇总所有参与 rank，再展示有代表性的 rank 和可能决定完成时间的 rank，说明选择依据，不能只挑最快的一张卡。

先用 job、迭代、层、输入和 microbatch 标记配对，再用 communicator、collective 序号、dtype/count 核对通信。flow ID 通常只用于本文件的 Host→Device 关联，不能自动当成跨 rank 消息编号。**同轮配对不等于时钟对齐**：只有获得时钟转换与误差证据，才共用时间原点、画跨 rank 依赖并讨论谁先到。否则用相同时间比例、各 rank 独立起点的分面图，只比较本地任务组成和区间；图上写明这一限制。

保留真实轨道身份。`Group … Communication` 是通信聚合轨道，不给它编造 stream 编号；MC2 若在同一物理 stream 记录通信和计算，就在该流内用颜色区分；融合算子内部未导出时，不补画想象的内部通信流。短任务可以另画明确标注比例的局部放大图，不能把条宽强行放大却保留原刻度。原生截图用于追溯，事件数据生成的 SVG 用于同尺度阅读，两者分别标明。

### 各章节优先补哪些前后证据

| 章节/特性 | 优先采集与并排展示 | 同时核对的结果 |
| --- | --- | --- |
| 调度、图捕获与回放 | Host 调度/提交、设备执行；跨 DP 时包含忙 rank、空 rank、协调及回退 | 本轮真实/填充 Token 数、图模式与档位、整轮时间和服务指标 |
| metadata 与 tiling 下沉 | Host 长度获取、同步/拷贝、设备计划生成和 Attention；计划更新与使用之间的依赖 | 正确长度及地址、计划生成/复用条件、整轮时间；不能只数 Host 调用 |
| 投机解码 | Draft、Target、接受/拒绝、状态更新及下一轮；必要时联合客户端接收时间线 | 接受长度、有效 KV、实际输出数、吞吐与 ITL/TPOT |
| DPLB/EPLB、EP 融合 | 同轮各 rank 的专家工作量、Dispatch、计算、Combine；EPLB 另含权重搬运 | 同容量、同任务量、布局变化、整个通信组完成时间和服务指标 |
| SP、PCP/DCP | 各 rank 的计算、AllGather/ReduceScatter/AllToAll、拷贝与同步 | 激活/KV 归属、正确性、容量、归并开销与端到端指标 |
| PD 分离 | P/D 两侧计算、传输、接收就绪和页释放；同配置下比较交接方式 | 传输字节、生命周期、TTFT、Decode 间隔、稳态吞吐 |
| KV 布局、稀疏卸载 | 地址/布局转换、索引、Host↔Device 拷贝、Attention、预取流及使用前的等待 | 命中率、搬运字节、有效容量与地址正确性、整轮时间 |

现有 [MC2/MegaMoE 双 rank 图](../dist/assets/a5-mc2-megamoe-measured.svg)和 [DCP1/DCP2 双 rank 图](../dist/assets/cp-dcp-ranks-measured.svg)按保存的设备事件生成，使用同一时间比例、各 rank 独立起点。它们改善已有单次采集的可读性，没有补出跨设备时钟校准或新的服务性能结果。重建命令为 `python3 scripts/render-measured-moe.py` 和 `python3 scripts/render-measured-cp.py`（在仓库根目录运行）。

## 新一轮特性前后对照

[MoE 图回放与 SP 实验](profiling-ab/README.md)使用同一计算函数完成正确性、无 profiler 重复计时和独立 Trace 采集。SP 的 A5 双卡结果已保留两种行数、两个 rank 的原始样本与前后轨道；这次机器存在已记录的 UB 告警，不能作为正常健康机器的性能基线。MoE 的旧脚本只在计时函数内部切换 graph，新的对照脚本同时修正了 profiling 的调用路径；每次真实捕获及回放仍须由结果记录确认。

## 运行条件与命令

设备实验要求匹配的 vLLM-Ascend、PyTorch-NPU、Triton 和 CANN 算子包。A5 的这个 MegaMoE 对照采用 MXFP 路径，不能把自定义 A3 W8A8 INT 实验直接替换进来。完整环境身份见结果 JSON；不同安装版本可能移动模块或改变 ABI。

使用 MindIE 时，先以 coordinator 获得实际物理设备的排他租约，把返回的 `env` 原样传给 remote-dev；单卡实验选择可见设备 0，双卡实验由 torchrun 分配可见设备 0、1。等待前一进程树退出，再开始下一项，完成后释放租约。不要使用已有服务占用的设备。

在该环境中运行：

```sh
python3 padded-inputs.py --output ./captures/padded
python3 batch-change.py --output ./captures/batch-change/result.json
torchrun --standalone --nnodes=1 --nproc_per_node=2 a5-moe.py \
  --tokens 8 32 --repeats 10 --profile-dir ./captures/moe
```

SSE 实验只需要 Python 标准库，不使用模型或 NPU：

```sh
python3 sse-buffering.py --output ./captures/sse/result.json
```

两轮主场景交叉执行 direct→buffer、buffer→direct；另覆盖七项单块和一个输出。输出数由七个已知标签提供，不从 TCP 分片或逐块分词推断。`send_begin/end` 包围 write+flush，不表示对端确认。

## 怎样解释结果

MoE 的正确性先比较相同量化权重的两条实现，并用独立路由计数核对任务守恒；这不等于与预训练 BF16 模型比较精度。NPU event 包围 Python 函数提交，区间可能包含等待 Host 或通信伙伴的空隙；结果中的原始样本和中位数不能写成纯 kernel 算力或服务吞吐收益。

各 rank 的 profiler 初始化不同步，完整采集还含未归属 active scope 的设备任务。active iteration 由原始 `torch_to_npu` flow ID 关联到 Host scope，再关联设备任务；选定的 [泳道数据](results/a5-moe-swimlane.json) 使用每种实现自己的设备首任务作零点，没有跨 rank 对齐。Host／Device 微秒级时间差未经独立校准，Perfetto 导入还存在 flow 告警，因此不据其计算派发延迟。

历史 [padded 正确性结果](results/padded-inputs.json) 与 [任务归属摘要](results/padded-profile.json)、[进度修正结果](results/batch-change.json) 与 [SSE 实际打点](results/sse-buffering.json) 分别保留各自口径。SSE 的单块七项 TPOT=0 是接收边界的退化结果；O=1 时记录为未定义。

第五章另用四张真实 Perfetto 截图配对 tokens32、iteration2 的 MC2／CANN MegaMoE 两个 rank。[逐卡任务与路由结果](results/a5-moe-rank-comparison.json)保留精确 flow、Host／Device 窗口及 71+57=128 次专家任务；[截图来源](results/a5-moe-profiling-screenshots.json)记录源 SHA 和实际显示范围。Level0 没有独立的 HCCL peer 收发轨道，因此不套用第六章的集体通信序号配对，也不从单卡区间判定慢 rank。

Linux 上解析与浏览器查看的完整步骤见 [第五章实操](https://tflowers-0129.github.io/inference-wiki/ep-communication.html#ep-communication-12)。MindStudio Insight 应保留完整 `*_ascend_pt` 目录；本仓库不自动公开原始 trace。

第七章的[环境与本地文件身份](pd-disaggregation/results/environment.json)、[拉取结果](pd-disaggregation/results/pull.json)和[逐层推送结果](pd-disaggregation/results/push.json)分别记录各自配置。拉取设备图查看 D rank0／rank1 的 `HixlBatchGet`，推送设备图查看 P rank0／rank1 的 `HixlBatchPut`；[拉取归属](pd-disaggregation/results/profiling-events-pull.json)与[推送归属](pd-disaggregation/results/profiling-events-push.json)保留连接字段、缺失 flow 端点和未校准时钟等限制。Host 生命周期图另由真实 JSONL 导出到 Perfetto，不混入 NPU kernel。两种实验的 KV head 和 DCP 条件不同，不计算路径间加速比。

## 仍需独立验证的项目

| 章节 | 尚未覆盖的实操 | 必需条件 |
| --- | --- | --- |
| 1 调度与执行 | 实际 Scheduler 连续轮次、图模式与真实捕获形状 | 目标测试夹具；完整模型与匹配 Attention 后端 |
| 2 一次前向 | V2 slot_mapping、Host 精确长度等待、Device tiling plan 的生成、更新与复用 | 对应 Runner／块表 ABI 与逐元素地址 oracle |
| 3 投机解码 | 完整 Draft→Target→提交回退→下一轮的模型执行 | 模型与 proposer 配对；接受结果及 KV 边界取证 |
| 4 负载均衡 | 实际 DPLB 选择回放、EPLB 整数副本映射及同容量关键路径 | 固定入口和遥测；多卡同权重、路由与布局对照 |
| 5 EP 通信 | 完整选择函数、AllGather/AllToAll 容量边界、graph 与服务级指标 | 支持的模型／量化／拓扑和同条件负载 |
| 6 上下文并行 | SP 层间布局与去 padding、预训练长上下文精度、PCP+DCP 叠加、MLA/SFA 与投机侧流、graph 和服务吞吐 | 对应权重、合法拓扑、Attention 后端与同条件负载 |
| 7 PD 分离 | 多主机网络、同配置 pull/push 对比、取消／失败注入、并发块覆盖复用与稳态服务指标 | 匹配缓存布局与 transport、完整生命周期取证、受控多实例负载 |
| 8 KV 布局 | 多 KV heads 的 TP 切分与复制、DCP 所有权、实际 view/stride 与页尾地址 | 同一模型改变 TP/DCP；抓取实际缓存 shape、stride 与地址，核对逻辑字节和分配容量 |
| 9 稀疏 KV | LI+Manage、Copy+SFA 阈值和 HiSparse 页生命周期 | 对应编译产物、合法 backing、Core/Ascend ABI 配对 |

正文明确声明的教学时长、四十 token 路由、缩比缓存和捕获档位不属于设备复现目标。实测采用目标实现合法参数；上述表格不会因为播放器运行正确就改为通过。

## 源码归属

`a5-moe.py` 改编自 [vLLM-Ascend 固定源码的 test_mega_moe.py](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/tests/e2e/nightly/single_node/ops/test_mega_moe.py)。该文件采用 Apache-2.0，许可证保存在 [LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt)。改动包括双卡小形状默认值、独立专家计数 oracle、AB/BA 原始采样及按 rank 的 profiling；普通链与生产权重处理、融合入口均保留。

同一固定提交的 [W4A8MXFP GMM1／激活分支](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py#L368-L441) 说明 SiLU 使用分开的 GMM1 与 SwiGLU/量化，SiTU 的单算子分支另有条件；[生产 MLP 调用方](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/ops/fused_moe/moe_mlp.py#L70-L78) 负责选择对应激活路径。
