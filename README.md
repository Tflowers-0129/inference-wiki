# LLM 推理调度与执行 Wiki

[在线演示](https://tflowers-0129.github.io/inference-wiki/) · [混部播放器](https://tflowers-0129.github.io/inference-wiki/#scheduler) · [PD 播放器](https://tflowers-0129.github.io/inference-wiki/#pdPlayer) · [一次前向](https://tflowers-0129.github.io/inference-wiki/execution.html) · [Tiling 下沉](https://tflowers-0129.github.io/inference-wiki/execution.html#execution-5) · [投机解码实验室](https://tflowers-0129.github.io/inference-wiki/speculation.html#spec-player) · [DPLB 与 EPLB](https://tflowers-0129.github.io/inference-wiki/load-balancing.html) · [EP 通信与计算重叠](https://tflowers-0129.github.io/inference-wiki/ep-communication.html) · [上下文并行与 Attention 归并](https://tflowers-0129.github.io/inference-wiki/context-parallel.html) · [PD 分离系统与 KV 生命周期](https://tflowers-0129.github.io/inference-wiki/pd-disaggregation.html) · [KV 缓存布局与递归状态](https://tflowers-0129.github.io/inference-wiki/kv-cache-layouts.html) · [Decode Offload 与 HiSparse](https://tflowers-0129.github.io/inference-wiki/sparse-offload.html) · [KV 池化与跨机 PD](https://tflowers-0129.github.io/inference-wiki/kv-pooling.html) · [量化推理与算子执行](https://tflowers-0129.github.io/inference-wiki/quantization.html) · [微批次流水与 DBO](https://tflowers-0129.github.io/inference-wiki/dbo.html)

[![Pages deployment](https://github.com/Tflowers-0129/inference-wiki/actions/workflows/pages.yml/badge.svg)](https://github.com/Tflowers-0129/inference-wiki/actions/workflows/pages.yml)

中文教学网页，沿着“混部调度 → 逐轮选图 → 输出长短与稳态测试 → PD 分离与池化 → 抢占与 KV Cache”的顺序展开。混部播放器之后，用第二轮的 `SchedulerOutput` 说明 Runner 如何接到请求进度和块号，再复算一次本地前缀命中。首页也包含三种 PD 交接场景和可编辑 SVG 图解。[图执行说明](https://tflowers-0129.github.io/inference-wiki/#graph-execution) 区分 Decode 比例、本轮形状和捕获档位，并说明显式教学配置与实际后端的适用边界。

网页不依赖前端框架、外部字体或脚本。克隆仓库后即可在本地预览；运行时只需要 Python 3。模型测试另需 Node.js 18 或更新版本。

《从一轮调度到一次前向》从输入组织、算子与访存追到 metadata 同步和跨轮异步调度。《投机解码与流式指标》继续追踪候选验证、KV 有效范围和客户端收到的内容事件。投机章开头的播放器沿八个步骤展示起点、逐个提议、四行验证、提交回退与下一轮输入；默认状态面板随网页宽度排版，可切换完整 SVG 泳道，并配合静态完整流程图和工作坊 profiling 原图阅读。正文用同一位置的 Draft 和 Target 分布算首次拒绝与校正采样，再追到有效计数和下一 Token。实验室另比较逐 Token、成组与代理缓冲三种发送场景。网页不展示具体引擎发行编号。

十二章共用左侧篇章导航，进入后默认展开当前章的目录，点击章标题可折叠；手机通过顶栏菜单打开相同导航。

《从一轮调度到一次前向》沿 ModelRunnerV2 的输入准备、千级 Prefill 分块、KV 分页寻址、Attention、采样和下一轮输入展开。六张 SVG 分别说明输入与 KV 写入位置、层内数据流、拒绝后的状态更新、Attention 计算分块、CPU／设备生成 tiling 计划的区别，以及图回放中的任务等待。正文从 tiling 如何划分计算工作讲起，再说明设备长度、计划生成与多模型迁移的关系；历史输入准备微实验保留在实验档案中，不作为 V2 调用链的证据。

《负载均衡：DPLB 与 EPLB》从请求入口的 DPLB 走到单层 MoE 的 EPLB，用五张 SVG 展开 DP/EP 维度、等请求数与不同工作量、同容量专家副本放置、rank × stream 泳道和权重/映射安全切换。两个交互分别演示固定遥测下的新请求评分与等待请求估计，以及同十二个专家权重存储位置、八十次专家任务下的布局和热度变化。正文追到 DPLB 快照替换、EPLB 逻辑到物理查表与 STAIR 计数汇总。泳道是明确标注的教学构造，不使用工作坊原图推断负载均衡收益，也不把专家计数均衡度写成实际加速比。

《EP 通信与计算重叠：算子选择与融合执行》从专家任务的收发入手，比较 AllGather、AllToAll、MC2 与 MegaMoE 的通信组织和选择条件。源码算例复算 AllToAllV 的 split 数组、专家行展开和结果逆重排。跨 DP 的 FULL_DECODE_ONLY 以四个 rank 的不同 Decode 行数说明捕获集合、模式和档位协调、空 rank dummy 与整组 eager 回退。副本实验室固定四十个 token、八十次专家任务与同一专家布局，只切换目标 rank 分布，区分专家任务、本地使用、远端激活副本与 BF16 逻辑 payload，不预测耗时。模拟 profiling 沿已核查的 W8A8 自定义内核，区分普通 MC2 的默认局部融合与 Fused-MC2 的 AIV / AIC flag 协作；支持单步、事件依赖和源码定位，横轴为教学推进格。A5 实操先以双 rank 同尺度事件图展示 MC2／MegaMoE 的差异，再用四张真实 Perfetto 截图对照 MC2／CANN MegaMoE 的 rank0、rank1，保留专家任务数、flow 归属、异窗定位和解释边界。

《上下文并行：KV 分片与 Attention 归并》先按 TP 与总 KV head 数的关系说明切分与复制，再以 ReduceScatter、逐行 Norm 和 AllGather 解释 SP 的激活分片，随后追踪 PCP 的 query 划分、DCP 的本地块表和有效长度。用 A5 算例对齐 Q AllGather、output/LSE 打包和归并的张量形状与通信 count。A5 实操用 DCP1／DCP2 双 rank 同尺度事件图、真实截图、同轮通信配对和逐卡任务记录对照，区分本地调用链、跨 rank 等待线索与尚未确认的原因。

《PD 分离系统：请求交接与 KV 生命周期》沿 vLLM-Ascend 与 Mooncake 源码追踪目标块分配、传输、就绪与源引用释放。用 A5 上的 129-token GQA 拉取请求复算每个接收 rank 的 READ 地址范围和整页字节量，再用 TP4→TP2 的源码算例说明页内重排。逐层推送用 MQA+DCP2 对照分片、写入事件和完成通知。两条路径保留原生 HIXL 任务与 Host 观察。逐层推送首次预热中，释放调用早于后台写返回，正文保留这次观察的边界。

《KV 缓存布局与递归状态》从 V2 启动时的缓存规格、内存预算和 backend 选择开始，追踪前缀命中、追加页、写时复制、抢占与安全释放，再进入一次 Decode 的 KV 写入位置、分页读取及 BLHNC／LBHNC 步幅。两张总图串起请求生命周期，以及 Dense、MLA、SFA、压缩 DSA 和递归状态各自的读写链。沿 PD 拉取实测的 A5 请求核对每 rank 的 GQA 张量形状、K/V 页大小、块步幅和续算写入位置。正文对照 Kimi K3、GLM-5.3 系列和 DeepSeek 系列中的 MLA、稀疏与压缩 Attention、递归状态，并用 C4 算例复算压缩记录的写入位置、TopK 读取和页内尺度偏移。播放器演示分配物理页、写入 slot_mapping=36 对应的位置，以及从三个物理页读取九个有效位置。[KV 寻址播放器](https://tflowers-0129.github.io/inference-wiki/kv-cache-layouts.html#kv-decode-player)

《稀疏 KV 卸载与 HiSparse》沿完整历史、逻辑 TopK 与设备工作集讲解 Decode Offload；五张 SVG 展示地址空间、A3 泳道、A5 两段融合以及 HiSparse 页生命周期和跨层 gather。缓存实验室可切换 CPU LRU 与 Resident/Hot/Source 解析，逐步查看命中、Host miss 与 Attention 读取的物理位置，正文把同一轮索引和回填配对落到 resolver 输出张量。正文区分已合入框架、待合入算子和 RFC 方向，性能表保留作者报告及测量边界。

《KV 池化：单机混部与跨机 PD》从三台机器的内存位置讲起。[手册 KV 播放器](https://tflowers-0129.github.io/inference-wiki/kv-pooling.html#pool-walk)沿两次请求展示哪张卡读写、共享副本留在哪里、D 在哪几页接着生成。正文把池化与 PD 交接分别对应到连接器、对象键、页表和传输地址，也给出单机混部、逐层传输与 D 本机暂存的对照。

《量化推理：位宽、尺度与算子执行》先用一组数算出量化点积，再沿线性层源码查看权重、激活和尺度如何传给算子。[量化篇正文](content/量化推理与算子执行.md)

《微批次流水：两批 MoE 工作怎样重叠》用两个微批次跟踪 MoE 输入的切分与专家计算，沿泳道看一批在何处交出执行机会、另一批如何接续。[DBO 篇正文](content/微批次流水与DBO.md)

前向篇与投机篇统一使用 V2 的 `idx_mapping`、`num_sampled`、`num_rejected` 和 `post_update` 追踪持久状态。K=3、a=2 的算例区分有效 KV、输出与下一轮 anchor；正文说明谁需要 Host 长度回读、谁生成并使用 Device tiling 计划。Trace 排查要求关联依赖和整轮区间，不把机制图或 kernel 数量写成性能收益。

工作坊素材在格式转换、KV 启动容量和投机章真实设备时间线的对应讲解处采用三张原图。PD 播放器之后用一条实测请求对照 P/D 块表、缓存步幅和传输字节数。正文区分 P/D 实例池、本地 KV 块池和共享 KV 缓存池，以 QPS 推导配比，并分别计算 KDA 和 Lightning Indexer + DSA 的状态、缓存与交接带宽。抢占按分配失败、解除引用、重排队与重算恢复解释；播放器区分本轮执行量与轮末四类请求状态。

[素材来源清单](dist/assets/profile-sources.json) 保存源文件名、上传提交、Git blob 校验值、采用位置及重绘依据；源仓库为私有，站内图片已本地保存，阅读站点不需要源仓库权限。公开仓库和下载包只含挑选的图片与新写分析。

## 实操验证

[实验脚本、结果与待验证清单](experiments/README.md) 保留历史 padded 输入与跨轮进度辅助函数微实验（不作为 V2 主线的执行证据），并覆盖真实 SSE 缓冲、A5 双卡 MC2／CANN MegaMoE，以及上下文并行、[PD 分离系统](experiments/pd-disaggregation/README.md)。上下文并行实验验证 PCP 布局、生产 DCP 归并与三种双卡 ModelRunnerV2 拓扑。PD 分离实验用实际 HTTP 服务、官方 proxy 和 Mooncake 完成两条交接路径。两组实验均保留原生设备截图与 Host／Device 观察边界。[DP 协调与 SP 实验](experiments/parallel-graph/README.md)保留服务器 CPU/Gloo 的数值检查，另以 A5 HCCL/NPUGraph 验证受控档位、空 rank 和回退；标量专家 payload 与完整模型的边界分别说明。教学模拟继续保留原有标识，预训练完整模型、调度、负载均衡和稀疏 KV 的未测项目逐项列出。

机器实操按[特性前后对照的固定流程](experiments/README.md#特性前后对照的固定流程)交付配置、正确性、无 profiler 的重复性能测量、前后 Trace、timeline 与指标核对。涉及多 rank 时同时展示相关计算、通信、拷贝与同步轨道；未校准时钟的数据只能使用标明独立起点的分面图，不能从截图拼接推断跨 rank 等待。

新一轮 [SP 与 MoE 图回放对照](experiments/profiling-ab/README.md)保留 A5 双卡 SP 的 400 条、MC2 eager/replay 的 480 条无 profiler 样本，以及 12 份 Trace；多 rank SVG 展示前后计算、通信与同步，MC2 回放的 flow 导出缺口单独标明。本次 SP 小形状中分片路径更慢，正文保留这一观察及设备既有告警，避免把计算行数减少直接写成性能收益。

## 本地预览

先克隆仓库：

```sh
git clone https://github.com/Tflowers-0129/inference-wiki.git
cd inference-wiki
```

在仓库目录运行：

```sh
python3 scripts/serve.py
```

该命令会先同步正文及图源下载，再把 `dist/` 作为静态站点启动。打开 [Wiki 首页](http://127.0.0.1:4173/)，按 `Ctrl+C` 停止服务。端口被占用时：

```sh
python3 scripts/serve.py --port 4174
```

不需要安装 npm 包，也不需要启动其他项目。

## 现场演示

原有播放器与新增实验室可直接定位：

- [混部调度播放器](http://127.0.0.1:4173/#scheduler)：先逐轮走过前四轮，观察 Decode 如何占用预算、Prefill 如何切块；再切换 64 / 1024 输出，对比并发与排空过程。
- [PD 交接播放器](http://127.0.0.1:4173/#pdPlayer)：先看“正常交接”，观察 R1 正在 Decode 时 R2 可以先传完 KV；再切换“KV 传输较慢”和“D 忙且目标 KV 不足”。

这些动画是教学事件和预算模型。事件间距不表示实测时间，动画不能预测目标硬件上的吞吐或时延。完整假设与待验证问题在正文及网页中。

## 修改正文、图和网页

- `content/LLM推理调度与执行.md`：完整正文，图源路径为 `../dist/assets/`。
- `content/从一轮调度到一次前向.md`、`content/投机解码与流式指标.md`：新增正文。
- `content/负载均衡与专家并行.md`：DPLB、EPLB、rank 泳道与真实 Trace 采集口径。
- `content/EP通信与通算融合.md`：EP 通信路径、算子选择与融合执行。
- `content/上下文并行与Attention归并.md`：TP 的 KV 切分、PCP／DCP 与 Attention 结果归并。
- `content/PD分离系统与KV生命周期.md`：拉取式、逐层推送式请求交接与 KV 生命周期。
- `content/KV缓存布局与递归状态.md` 讲解 KV 形状与物理布局、Indexer 缓存和 Mamba 状态。
- `content/稀疏KV卸载与HiSparse.md`：Decode Offload、A3/A5 算子与社区 HiSparse 接入。
- `content/KV池化与跨机PD.md`：单机混部与跨机 PD 的 KV 池化数据流。
- `content/量化推理与算子执行.md` 讲解量化点积、尺度和线性层的算子调用。
- `content/微批次流水与DBO.md` 讲解微批次切分、专家通信与计算的依赖。
- `dist/ep-model.js`、`dist/ep-app.js`、`dist/ep.css`：专家任务、目标 rank 去重与 Dispatch 逻辑 payload 教学模型；`scripts/ep-lab.html` 为面板源片段。
- `dist/ep-timeline-model.js`、`dist/ep-timeline-app.js`、`dist/ep-timeline.css`：MC2 / Fused-MC2 的源码依赖模拟与 profiling 泳道；`scripts/ep-timeline.html` 为面板源片段。
- `dist/kv-decode-model.js`、`dist/kv-decode-app.js`、`dist/kv-decode.css` 维护 KV 寻址播放器，面板源为 `scripts/kv-decode-player.html`。
- `dist/offload-model.js`、`dist/offload-app.js`、`dist/offload.css`：TopK 缓存和地址教学模型；`scripts/offload-lab.html` 为面板源片段。
- `scripts/build-chapters.py`：用标准库渲染首页之外的受控 Markdown；修改正文后运行同步命令即可更新网页。
- `dist/spec-model.js`、`dist/spec-app.js`、`dist/spec.css`：投机验证与流式指标实验室。
- `dist/chapter.js`、`dist/chapters.css`：新篇章的目录、搜索与排版。
- `scripts/spec-player.html`：投机交互面板源片段。
- `dist/lb-model.js`、`dist/lb-app.js`、`dist/lb.css`：DPLB 请求分发与 EPLB 期望专家任务的交互模型、图形和样式；`scripts/dplb-lab.html`、`scripts/eplb-lab.html` 为面板源片段。
- `dist/assets/*.svg`：矢量图源与 favicon，可直接编辑。
- `dist/assets/profile-*.png`：格式转换、KV 启动容量与设备时间线的原始截图；`profile-sources.json` 保存来源。
- `scripts/image_assets.py`：本地图源校验、PNG/SVG 尺寸与可移植链接处理。
- `dist/index.html`、`dist/style.css`：网页讲解、布局与样式。
- `dist/model.js`、`dist/app.js`：混部调度模型与播放器。
- `dist/pd-model.js`、`dist/pd-app.js`：PD 交接模型与播放器。

修改正文或图源后运行：

```sh
python3 scripts/sync-content.py
```

真实设备事件对照图由 `python3 scripts/render-measured-moe.py` 和 `python3 scripts/render-measured-cp.py` 重建；它们读取已保存的结果 JSON，不运行设备实验。

EP 静态泳道与交互模拟共用模型。修改 `dist/ep-timeline-model.js` 后，先运行 `node scripts/render-ep-profile.cjs` 重建 `dist/assets/ep-overlap-swimlanes.svg`，再同步正文；普通正文编辑不需要这一步。

脚本默认读取本仓库的 `content/`，生成 `dist/source.md`、`dist/execution.md`、`dist/speculation.md`、`dist/load-balancing.md`、`dist/ep-communication.md`、`dist/context-parallel.md`、`dist/pd-disaggregation.md`、`dist/kv-cache-layouts.md`、`dist/sparse-offload.md`、`dist/kv-pooling.md`、`dist/quantization.md`、`dist/dbo.md` 和 `dist/technical-wiki-source.zip`，供页面上的下载链接使用。这些下载生成文件由 `.gitignore` 排除，部署前会重新生成。压缩包正文引用 `figures/`，解压后可直接连图阅读。也可使用 `--manuscript /绝对路径/文稿.md`，图源仍需保存在本仓库 `dist/assets/`。

同步脚本会用对应的 Markdown 重建 `execution.html`、`speculation.html`、`load-balancing.html`、`ep-communication.html`、`context-parallel.html`、`pd-disaggregation.html`、`kv-cache-layouts.html`、`sparse-offload.html`、`kv-pooling.html`、`quantization.html` 与 `dbo.html`。这些生成网页随仓库提交便于直接浏览，应修改 Markdown 和模板后重建；原有 `index.html` 仍按手工图文维护。渲染支持本文使用的段落、标题、平面列表、表格、代码块和 SVG / PNG 图片，不是任意 Markdown 的通用转换器。涉及模型假设时也需检查播放器。

## 检查与测试

```sh
python3 scripts/sync-content.py
python3 scripts/check-site.py
node --test tests/*.test.cjs
```

模型测试覆盖 token 记账、每轮预算与序列上限，PD 交接中的 KV 就绪、传输、源引用保留和释放时序，以及 DPLB 评分/等待请求估计和 EPLB 同容量任务守恒、热度漂移，另覆盖稀疏 KV 逻辑/物理映射、LRU 保护与失效、离散驻留和重复选择去重，以及 EP 路由的任务守恒、目标 rank 去重、本地与远端副本和逻辑 payload，另检查 MC2 / Fused-MC2 的事件依赖、角色串行、局部融合与跨 rank 消费条件。站点检查覆盖十二篇网页的资源、跨页锚点、播放器引用、十二篇下载正文和图文源包，检查教学调用链统一使用 ModelRunnerV2，核对 PNG 头部、包内图片与来源清单字节一致性，并防止具体引擎发行号与提交标识混入公开图文。

## GitHub Pages

GitHub Pages 的部署目录是 `dist/`。仓库中的工作流负责同步正文、检查站点、运行模型测试，再把静态文件部署到 Pages。需要在 GitHub 仓库的 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。

工作流成功后，实际预览地址以仓库 **Actions** 和 **Settings → Pages** 显示的地址为准。资源使用相对路径，支持 GitHub Pages 的仓库子路径。线上演示时，在该地址末尾加上 `#scheduler` 或 `#pdPlayer` 即可直达播放器。

`dist/` 内的 HTML、CSS、JavaScript、SVG 和 PNG 是本站可编辑源码，应提交到仓库；正文与播放器一起维护。

## 更新线上演示

修改、检查后推送到 `main`：

```sh
python3 scripts/sync-content.py
python3 scripts/check-site.py
node --test tests/*.test.cjs
git add .
git commit -m "Update inference wiki"
git push origin main
```

`Deploy Wiki to GitHub Pages` 会自动运行并更新同一个演示地址；可在 [Actions](https://github.com/Tflowers-0129/inference-wiki/actions) 查看结果。PR 只运行检查，不会覆盖线上演示。
