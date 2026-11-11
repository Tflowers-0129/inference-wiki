# LLM 推理调度与执行 Wiki

[在线演示](https://tflowers-0129.github.io/inference-wiki/) · [混部播放器](https://tflowers-0129.github.io/inference-wiki/#scheduler) · [PD 播放器](https://tflowers-0129.github.io/inference-wiki/#pdPlayer) · [一次前向](https://tflowers-0129.github.io/inference-wiki/execution.html) · [Triton 输入准备图解](https://tflowers-0129.github.io/inference-wiki/execution.html#fusion-lab) · [投机解码实验室](https://tflowers-0129.github.io/inference-wiki/speculation.html#spec-player) · [DPLB 与 EPLB](https://tflowers-0129.github.io/inference-wiki/load-balancing.html) · [EP 通信与计算重叠](https://tflowers-0129.github.io/inference-wiki/ep-communication.html) · [上下文并行与 Attention 归并](https://tflowers-0129.github.io/inference-wiki/context-parallel.html) · [PD 分离系统与 KV 生命周期](https://tflowers-0129.github.io/inference-wiki/pd-disaggregation.html) · [KV 缓存布局与递归状态](https://tflowers-0129.github.io/inference-wiki/kv-cache-layouts.html) · [Decode Offload 与 HiSparse](https://tflowers-0129.github.io/inference-wiki/sparse-offload.html)

[![Pages deployment](https://github.com/Tflowers-0129/inference-wiki/actions/workflows/pages.yml/badge.svg)](https://github.com/Tflowers-0129/inference-wiki/actions/workflows/pages.yml)

中文教学网页，沿着“混部调度 → 逐轮选图 → 输出长短与稳态测试 → PD 分离与池化 → 抢占与 KV Cache”的顺序展开。包含混部调度播放器、三种 PD 交接场景和可编辑 SVG 图解。[图执行说明](https://tflowers-0129.github.io/inference-wiki/#graph-execution) 区分 Decode 比例、本轮形状和捕获档位，并说明显式教学配置与实际后端的适用边界。

网页不依赖前端框架、外部字体或脚本。克隆仓库后即可在本地预览；运行时只需要 Python 3。模型测试另需 Node.js 18 或更新版本。

第二、三章为由 Markdown 生成的图解网页：从调度到一次前向（输入组织、metadata、算子与访存、KV 地址、图提交与 Trace、metadata 同步与跨轮异步调度），以及投机解码（候选验证、KV 有效范围、成组输出与 ITL/TPOT）。投机章开头的播放器沿八个步骤展示起点、逐个提议、四行验证、提交回退与下一轮输入；默认状态面板随网页宽度排版，可切换完整 SVG 泳道，并配合静态完整流程图和工作坊 profiling 原图阅读。实验室另比较逐 Token、成组与代理缓冲三种发送场景。网页不展示具体引擎发行编号。

九章共用左侧篇章导航，进入后默认展开当前章的目录，点击章标题可折叠；手机通过顶栏菜单打开相同导航。

第二章另以六张 SVG 说明权重复用与 KV 规模、精确值的消费者、缓冲复用事件、动态 K 接续、跨轮 Token 重排和 Triton padded 输入计算。Triton 图解可切换 Torch / Triton 及接受长度，区分减少 Host 回读与减少设备算子提交；较长的源码对照可在页内展开。

第四章《负载均衡与专家并行》从请求入口的 DPLB 走到单层 MoE 的 EPLB，用五张 SVG 展开 DP/EP 维度、等请求数与不同工作量、同容量专家副本放置、rank × stream 泳道和权重/映射安全切换。两个交互分别演示冻结遥测下的新请求评分与本地信用，以及同十二个物理槽、八十次专家任务下的布局和热度变化。泳道是明确标注的教学构造，不使用工作坊原图推断负载均衡收益，也不把专家计数均衡度写成实际加速比。

第五章《EP 通信与计算重叠：算子选择与融合执行》承接第四章的专家执行，比较 AllGather、AllToAll、MC2 与 MegaMoE 的通信组织和选择条件。副本实验室固定四十个 token、八十次专家任务与同一专家布局，只切换目标 rank 分布，区分专家任务、本地使用、远端激活副本与 BF16 逻辑 payload，不预测耗时。模拟 profiling 沿已核查的 W8A8 自定义内核，区分普通 MC2 的默认局部融合与 Fused-MC2 的 AIV / AIC flag 协作；支持单步、事件依赖和源码定位，横轴为教学推进格。A5 实操另用四张真实 Perfetto 截图对照 MC2／CANN MegaMoE 的 rank0、rank1，保留专家任务数、flow 归属、异窗定位和解释边界。

第六章《上下文并行：KV 分片与 Attention 归并》先说明 TP 对 KV head 的切分与复制，再追踪 PCP 的 query 划分、DCP 的序列分片和局部 Attention 结果归并；A5 实操用 rank0／rank1 的真实截图、同轮通信配对和逐卡任务记录对照，区分本地调用链、跨 rank 等待线索与尚未确认的原因。

第七章《PD 分离系统：请求交接与 KV 生命周期》承接单引擎内部的 KV 分片，沿 vLLM-Ascend 与 Mooncake 源码说明拉取式、逐层推送式的请求交接，追踪目标块分配、传输、就绪与源引用释放。A5 实操分别运行合法的 GQA 拉取与 MQA+DCP2 逐层推送，核对两个来源／接收 rank 的原生 HIXL 任务，并用真实 Host 观察追踪完成和释放；首次预热中释放调用早于后台写返回的记录单独保留，避免把测量中的先后当作实现保证。

第八章《KV 缓存布局与递归状态》从形状和步幅读 BLHNC、LBHNC，并用 GQA、MLA、Indexer 和 Mamba 算例解释缓存容量。模型对照追到 Kimi K3 的 KDA 与 MLA、两种 GLM-5.3 的 IndexShare 与 KeyPool，以及两代 DeepSeek Flash 的压缩页和共享缓存。

第九章《稀疏 KV 卸载与 HiSparse》沿完整历史、逻辑 TopK 与设备工作集讲解 Decode Offload；五张 SVG 展示地址空间、A3 泳道、A5 两段融合以及 HiSparse 页生命周期和跨层 gather。缓存实验室可切换 CPU LRU 与 Resident/Hot/Source 解析，逐步查看命中、Host miss 与 Attention 槽位。正文区分已合入框架、待合入算子和 RFC 方向，性能表保留作者报告及测量边界。

第二、三篇按字段和执行现场展开：先走通五 Token 的输入与 KV 地址，再查 Host/Device 对账、CPU 输入缓冲复用和跨轮提交；投机章沿 K=3、a=2 的同一份账本追到 padded 输入和客户端指标。metadata 部分从投机候选数追到验证后的有效 KV，再切换下一轮 K=0/1/3，对照 query 边界、位置和块寻址；Host / Device 的等待分别解释数据冲突与消费者依赖。每处源码片段注明调用路径，Trace 排查给出可区分的证据，不把机制示例写成实测经验。

工作坊素材在格式转换、KV 启动容量和投机章真实设备时间线的对应讲解处采用三张原图。PD 的交接依赖由播放器与简短说明呈现。正文区分 P/D 实例池、本地 KV 块池和共享 KV 缓存池，以 QPS 推导配比，并分别计算 KDA 和 Lightning Indexer + DSA 的状态、缓存与交接带宽。抢占按分配失败、解除引用、重排队与重算恢复解释；播放器区分本轮执行量与轮末四类请求状态。

[素材来源清单](dist/assets/profile-sources.json) 保存源文件名、上传提交、Git blob 校验值、采用位置及重绘依据；源仓库为私有，站内图片已本地保存，阅读站点不需要源仓库权限。公开仓库和下载包只含挑选的图片与新写分析。

## 实操验证

[实验脚本、结果与待验证清单](experiments/README.md) 覆盖已执行的 padded 输入、跨轮进度辅助函数、真实 SSE 缓冲、A5 双卡 MC2／CANN MegaMoE，以及上下文并行、[PD 分离系统](experiments/pd-disaggregation/README.md)。第二、三、五、六、七章分别补入对应实测；第六章验证 PCP 布局、生产 DCP 归并与三种双卡 ModelRunnerV2 拓扑，第七章用实际 HTTP 服务、官方 proxy 和 Mooncake 完成两条交接路径。两章分别保留原生设备截图与 Host／Device 观察边界。教学模拟继续保留原有标识，预训练完整模型、调度、负载均衡和稀疏 KV 的未测项目逐项列出。

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
- `dist/ep-model.js`、`dist/ep-app.js`、`dist/ep.css`：专家任务、目标 rank 去重与 Dispatch 逻辑 payload 教学模型；`scripts/ep-lab.html` 为面板源片段。
- `dist/ep-timeline-model.js`、`dist/ep-timeline-app.js`、`dist/ep-timeline.css`：MC2 / Fused-MC2 的源码依赖模拟与 profiling 泳道；`scripts/ep-timeline.html` 为面板源片段。
- `dist/offload-model.js`、`dist/offload-app.js`、`dist/offload.css`：TopK 缓存和地址教学模型；`scripts/offload-lab.html` 为面板源片段。
- `scripts/build-chapters.py`：用标准库渲染第二章起的受控 Markdown；修改正文后运行同步命令即可更新网页。
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

EP 静态泳道与交互模拟共用模型。修改 `dist/ep-timeline-model.js` 后，先运行 `node scripts/render-ep-profile.cjs` 重建 `dist/assets/ep-overlap-swimlanes.svg`，再同步正文；普通正文编辑不需要这一步。

脚本默认读取本仓库的 `content/`，生成 `dist/source.md`、`dist/execution.md`、`dist/speculation.md`、`dist/load-balancing.md`、`dist/ep-communication.md`、`dist/context-parallel.md`、`dist/pd-disaggregation.md`、`dist/kv-cache-layouts.md`、`dist/sparse-offload.md` 和 `dist/technical-wiki-source.zip`，供页面上的下载链接使用。这些下载生成文件由 `.gitignore` 排除，部署前会重新生成。压缩包正文引用 `figures/`，解压后可直接连图阅读。也可使用 `--manuscript /绝对路径/文稿.md`，图源仍需保存在本仓库 `dist/assets/`。

同步脚本会从第二章起的 Markdown 重建 `execution.html`、`speculation.html`、`load-balancing.html`、`ep-communication.html`、`context-parallel.html`、`pd-disaggregation.html`、`kv-cache-layouts.html` 与 `sparse-offload.html`。这些生成网页随仓库提交便于直接浏览，应修改 Markdown 和模板后重建；原有 `index.html` 仍按手工图文维护。渲染支持本文使用的段落、标题、平面列表、表格、代码块和 SVG / PNG 图片，不是任意 Markdown 的通用转换器。涉及模型假设时也需检查播放器。

## 检查与测试

```sh
python3 scripts/sync-content.py
python3 scripts/check-site.py
node --test tests/*.test.cjs
```

模型测试覆盖 token 记账、每轮预算与序列上限，PD 交接中的 KV 就绪、传输、源引用保留和释放时序，以及 DPLB 评分/本地信用和 EPLB 同容量任务守恒、热度漂移，另覆盖稀疏 KV 逻辑/物理映射、LRU 保护与失效、离散驻留和重复选择去重，以及 EP 路由的任务守恒、目标 rank 去重、本地与远端副本和逻辑 payload，另检查 MC2 / Fused-MC2 的事件依赖、角色串行、局部融合与跨 rank 消费条件。站点检查覆盖九篇网页的资源、跨页锚点、播放器引用、九篇下载正文和图文源包，核对 PNG 头部、包内图片与来源清单字节一致性，并防止具体引擎发行号与提交标识混入公开图文。

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
