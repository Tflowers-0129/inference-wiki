# KV 缓存布局、Indexer 与递归状态

把同一个物理块交给另一个 Decode worker，有时要逐层取数，有时跨层数据就在同一个连续区间。差别从哪来？要先看层和物理块谁排在外面。到了 DeepSeek，主 Attention 读压缩 latent，Indexer 另存一份历史 Key。Mamba 层则从某个续算边界取递归状态。

读一个 cache shape 时，先确定它在哪个 worker 和缓存组里，再问哪个轴随请求历史增长、下一步的算子实际读什么。

> 正文选取 vLLM 和 vLLM-Ascend 的实际缓存路径。第 8–10 节展开 Kimi K3、两种 GLM-5.3 与两代 DeepSeek Flash，按主干层统计。形状和字节数对应各节链接的实现与配置，图中缩小了维度便于手算。

## 1. BLHNC 和 LBHNC 的五个字母

在上游 vLLM 的 `KVCacheLayout` 里，缓存组的逻辑坐标写成 `[L,B,H,N,C]`。枚举名按**物理内存从外到内**排列，同一个逻辑坐标可以有不同 stride。先以普通 paged Attention 为例。[布局枚举](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/kv_cache_layout.py)

| 轴 | 在普通 KV 里代表什么 | 从哪里得到 |
| --- | --- | --- |
| `L` | 当前分配里的层槽 | 模型在本 worker 的层、缓存组及分配计划 |
| `B` | 设备池中的物理块数 | 分配的块数与 kernel block 切分 |
| `H` | 本 rank 的 KV heads | 模型的 KV heads 与 TP 分片或复制 |
| `N` | 一个 kernel block 内的 token 槽数 | 普通 Attention 的 kernel block size |
| `C` | 每个 head、每个槽的内容宽度 | 例如 K+V 的元素数，或打包后的字节数 |

`B` 是 block，运行中的 batch size 另有自己的口径。`N` 只覆盖页内的位置。长度为 `T` 的请求会通过自己的 block table 找到多页。压缩缓存可用一条状态代表多个 token，届时 `N` 要读成块内存储的状态数。独立的 K/V 视图和混合模型的状态视图则以各自的视图定义为准。[状态数、内容宽度与块视图](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/kv_cache_interface.py)

四个常见名字可以拆成两次选择。最外层选 `LB` 或 `BL`，页内选 `HNC` 或 `NHC`。在旧的布局参数中，`HND` 是 `LBHNC` 的别名，`NHD` 是 `LBNHC` 的别名。FlashInfer kernel 侧的 `HND/NHD` 只表达页内次序，因此传到 FlashInfer 的 `HND` 也可能来自上游的 `BLHNC`。[布局解析和 FlashInfer 映射](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/attention/backends/utils.py)

## 2. shape 相同，为什么读到的地址不同？

取两层、三个物理块、两个 KV heads、每块四槽，每个槽的内容宽度为两个元素。缓存组的逻辑 shape 为 `[2,3,2,4,2]`。每个 `(l,b)` 页是 `2×4×2=16` 个元素，没有 padding。

![同一个物理块 B1 在 LBHNC 中分布在两处，在 BLHNC 中两层相邻。图中同一逻辑坐标对应不同元素偏移。](../dist/assets/kv-layout-block-layer.svg)

假设一条请求的块表为 `[2,1]`，kernel block size 为 4，且未做上下文分片。位置 `p=6` 先定位到逻辑块 `6 // 4 = 1`，查表得到物理块 `b=1`，块内偏移 `n=2`。同一位置在第 1 层、head 1、内容第 0 项，对应坐标 `(l,b,h,n,c)=(1,1,1,2,0)`。

```text
页内元素偏移 = (h × N + n) × C + c = 12

LBHNC 页号 = l × B + b = 4
      元素偏移 = 4 × 16 + 12 = 76
BLHNC 页号 = b × L + l = 3
      元素偏移 = 3 × 16 + 12 = 60
```

取出第 1 层，两种布局都可以向 Python 暴露 `[3,2,4,2]` 的 `[B,H,N,C]` 视图。此时 `LBHNC` 的块 stride 是 16，`BLHNC` 的块 stride 是 32，因为内存里还隔着另一层的一页。PyTorch 的 stride 以元素计，换算字节要乘 `element_size()`。真实视图还要算分配给这层的起点和物理页 padding。上游用 `KVCacheTensor` 的层偏移、块 stride 和 `create_kv_cache_views()` 构造层视图。[分配入口](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/worker/utils.py) · [视图与 stride](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/kv_cache_interface.py)

块表给出物理块 ID，布局给出页内和页间地址。前面的 `p=6` 在两种布局里都查到 B1，变的是 B1 每层数据在内存里的起点。若一个 manager block 切成多个 kernel blocks，backend 看到的 `B` 增多、`N` 减小。本文所查的上游切分视图要求 manager 块 stride 等于未填充的页大小，多层 BLHNC 的跨层步幅通常不满足这一条件。DCP 下还要先确定历史位置的 owner，再找本地块内位置。[块大小协商](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/worker/utils.py) · [切分视图的 stride 条件](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/kv_cache_interface.py#L363-L385) · [上下文并行的 slot mapping](上下文并行与Attention归并.md)

### 你看到的 vLLM BLHNC 和 Ascend LBHNC

这组差异在 PD 场景里很具体。上游 NIXL 文档的普通 Attention 默认用 `LBHNC`。支持跨层连续传输的配置可用 `VLLM_KV_CACHE_LAYOUT=BLHNC`，把同一 B 的各层挨着放，便于将匹配拓扑下的一块跨层数据作为连续区间搬移。单机运行时，vLLM 还要结合各 worker/backend 支持的布局以及连接器偏好选布局。[NIXL 布局与传输矩阵](https://github.com/vllm-project/vllm/blob/ced6857a/docs/features/nixl_connector_compatibility.md) · [上游布局选择](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/attention/backends/utils.py)

Ascend 的 MooncakeConnectorV2 指南要求其 GQA 路径在 P 和 D 两端使用同一个 `LBHNC` 或 `HND` 配置。普通 Ascend Attention 在这个配置下让每层 K/V 各用 `[B,H,N,D]`，其他常规分支可用 `[B,N,H,D]`。这条要求来自连接器与 backend 的组合。核对 P/D 交接时，还要对齐实际 K/V 组件、dtype、kernel block size、每 rank 头和 stride。[Ascend Mooncake 配置](https://github.com/vllm-project/vllm-ascend/blob/c09d3e20/docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md#layout-and-port-configuration) · [Ascend Attention 的 shape](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/attention_v1.py) · [PD 交接生命周期](PD分离系统与KV生命周期.md)

## 3. 普通 Attention 为什么只按 KV head 算？

MHA、GQA、MQA 的历史缓存都可从每个 token 的 K、V 两组向量出发。若每个 KV head 有 `D_k` 维 K、`D_v` 维 V，本 rank 持有 `H` 个 KV heads，逻辑字节数为 `H×(D_k×K元素字节+D_v×V元素字节)`。MHA 的 KV head 数等于 Q head 数。GQA 的多组 Q 各共享一组 K/V，MQA 的全部 Q 共享一组 K/V。

用 Qwen3-8B 做实数账。配置有 36 层、32 个 Q heads、8 个 KV heads、`D_k=D_v=128`。TP4 时每 rank 有 8 个 Q heads 和 2 个 KV heads。BF16 时每层、每 token、每 rank 的 K/V 一共 `2×(128+128)×2=1024 B`。FlashAttention 若用 16 槽的 kernel block，一层一页是 16 KiB。本文所查的 Ascend 普通 Attention 用 128 槽，同样的每 token 宽度得到 128 KiB。该 rank 若持有全部 36 层、4096 个完整历史位置，K/V 有效载荷为 144 MiB。实际设备池的容量还取决于分配块数、块尾、对齐和其他缓存组。[Qwen3-8B 配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) · [每 rank KV heads 的计算](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/config/model.py) · [Ascend kernel block 大小](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/attention_v1.py#L153-L155)

假设本轮调度了五个有效 token。投影后的本地 Q 是 `[5,8,128]`，新 K 和 V 各是 `[5,2,128]`。写入 kernel 逐行用 `slot_mapping` 定位目标页和页内槽。读历史时，Attention 再沿 block table 找到已驻留的页面。这里的五行只决定本轮的输入工作量，持久 KV 的 `B` 来自缓存池。[FlashAttention 的 KV 写入入口](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/attention/backends/flash_attn.py#L1438-L1470) · [从输入行到 slot mapping](从一轮调度到一次前向.md)

同样是 Qwen3-8B，不同 backend 可暴露不同的视图。下面限定普通、等宽的 BF16 K/V，不含量化 scale 和 hybrid page padding。

| 实现 | 每层持久缓存视图 | 算子入口怎么读 |
| --- | --- | --- |
| 上游 FlashAttention | `[B,H,N,2D]` | 转为 `[B,N,H,2D]` 后沿最后一维拆成 K 和 V |
| 上游 FlashInfer | `[B,H,N,2D]` | 按 stride 顺序和 wrapper 契约取 K/V，页内可见 `HND/NHD`，Blackwell TRT-LLM 路径要求 head-major |
| Ascend 普通 Attention | 逻辑描述 `[2,B,N,H,D]`，HND 时为 `[2,B,H,N,D]` | Worker 可绑定成两个四维张量。开头的 `2` 指 K/V，层槽由外部映射持有 |

上游 FlashAttention 与 FlashInfer 的 Python 视图可同形而 stride 不同。Ascend 的 FIA 调用还会把部分布局折成 `[B,N,H×D]`。抓到算子的 `input_layout` 时要看它描述的是 query 还是 cache 参数。CANN 的 query `TND` 是总 token 行、头数、头宽，这里的 `N` 指头数，且 query 仍可以读有 block table 的 paged KV。要恢复缓存的物理层序，需要回到实际分配的 stride。[FlashAttention 拆分](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/attention/backends/flash_attn.py) · [FlashInfer 转接](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/attention/backends/flashinfer.py) · [Ascend FIA 输入与 cache 视图](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/attention_v1.py) · [CANN TND 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnFusedInferAttentionScoreV3.md)

在支持的 TP 切分中，Q/KV heads 一般随 TP 分配。如果总 KV heads 比 TP rank 数少，worker 仍至少持有一个 KV head，多 rank 可能保存同一份头。这样再加 TP 就不一定继续缩小每卡 KV。DCP 则可能切同一个头的历史位置。计算容量时分别取本地 `H` 和本地历史长度。[TP 头复制实现](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/config/model.py) · [TP、PCP 与 DCP](上下文并行与Attention归并.md)

## 4. MLA 的 576 维缓存从哪里来？

DeepSeek-V3 和 R1 的 Attention 有 128 个 Query 头。每个头展开后的 Q/K 有 128 维内容和 64 维 RoPE，V 有 128 维。要算压缩 KV 的容量，得回到投影入口。每个 token 的 KV 投影先产生 512 维 latent 和 64 维共享 RoPE Key，vLLM 的 MLA 缓存把这两部分作为一条历史记录，`num_kv_heads=1`、`head_size=512+64=576`。模型配置中的 `num_key_value_heads=128` 和这个缓存 spec 描述的是不同层次。[DeepSeek-V3 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json) · [DeepSeek-R1 配置](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/config.json) · [vLLM MLA spec](https://github.com/vllm-project/vllm/blob/ced6857/vllm/model_executor/layers/attention/mla_attention.py#L1347)

看一条记录怎样参与计算。512 维 latent 同时提供内容 Key 和 Value 的信息。展开计算时，`kv_b_proj` 从 latent 生成各头的内容 K 和 V。吸收投影的路径可以让 Q 先变换到 latent 空间，与缓存中的 latent 打分，再把 Attention 的 latent 输出投影回各头的 V 空间。64 维 RoPE Key 单独参与位置相关的打分。持久容量因此按每位置一份 512 维 latent 和一份 64 维 Key 计算，BF16 合计 `(512+64)×2=1152 B`。[MLA 投影与前向](https://github.com/vllm-project/vllm/blob/ced6857/vllm/model_executor/models/deepseek_v2.py#L1094) · [vLLM MLA 计算说明](https://github.com/vllm-project/vllm/blob/ced6857/vllm/model_executor/layers/attention/mla_attention.py)

同样的 576 个元素，在 vLLM 的合并视图中可以读作 `[B,N,576]`，绑定缓存时压掉了大小为 1 的 KV 头轴。Ascend ModelRunnerV2 的普通拆分视图则是 `[B,N,1,512]` 和 `[B,N,1,64]`。源码变量里的 `k_cache` 对应 latent，`v_cache` 对应 RoPE Key。Ascend FlashMLA 也可以直接绑定合并视图。比较两个 dump 时，要同时核对 backend、dtype、stride 和缓存 spec。[vLLM 绑定视图](https://github.com/vllm-project/vllm/blob/ced6857/vllm/model_executor/layers/attention/mla_attention.py#L715) · [Ascend KV 视图](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/worker/v2/attn_utils.py#L2043)

## 5. DeepSeek-V3.2 的 Indexer 另存了什么？

V3.2 仍用 `512+64` 的 MLA 主缓存，同时增加 Lightning Indexer。Indexer 给当前 `T_q` 条 query 生成 `[T_q,64,128]` 的索引 Q 和 `[T_q,64]` 的头权重。历史的每个位置只存一条 128 维索引 K，供 64 个索引 Q 头共享。Indexer 汇总分数，给每条 query 选择至多 `K_top=2048` 个历史位置，再由主 Attention 读取这些位置的 MLA 记录。[DeepSeek-V3.2 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) · [官方 Indexer 实现](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py#L435)

这里有三种会一起出现在 trace 里的张量。主 KV 存内容，Indexer K 存检索向量，TopK 的 `[T_q,K_top]` 行存本批选中的位置编号。假设一个请求的 block table 为 `[1,3]`，块长 `N=128`。Indexer 选中请求内位置 130 时，逻辑块号为 1，块内偏移为 2，对应物理块 3 内偏移 2 的行，紧密排布时的槽号为 `3×128+2=386`。具体 kernel 可以消费物理槽，或接收逻辑索引和 block table 后寻址。随着 query 变化，下一轮可以选择不同位置，主 KV 和 Indexer K 的持久容量都要按保留的历史计算。[CUDA FlashMLA 的索引转换](https://github.com/vllm-project/vllm/blob/ced6857/vllm/v1/attention/backends/mla/flashmla_sparse.py#L803) · [Ascend SFA 的索引与块表参数](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/sfa_v1.py#L229)

| 实现与缓存格式 | 缓存张量视图 | 每位置有效字节 |
| --- | --- | --- |
| Ascend SFA，主 KV 与 Indexer K 均 BF16 | 主 KV `[B,N,1,512]` 和 `[B,N,1,64]`，Indexer K `[B,N,1,128]` | 主 KV 1152 B，Indexer 256 B |
| Ascend SFA，主 KV C8，Indexer 另开 LI C8 | 主 KV 是 `[B,N,1,656]` 的打包记录。Indexer K 为 `[B,N,1,128]`，另有 `[B,N,1,1]` 的 scale | 主 KV `512+128+16=656 B`。Indexer FP8 加 FP32 scale 为 132 B，INT8 加 FP16 scale 为 130 B |
| CUDA FlashMLA Sparse，主 KV `fp8_ds_mla`，FP8 Indexer | 主 KV 是 `[B,N,656]` 的打包视图，Indexer 另有 `[B,N,132]` 的字节视图 | 主 KV `512+16+128=656 B`，Indexer `128+4=132 B` |

Ascend SFA 用 `additional_config.enable_sparse_sfa_c8` 开启主 KV C8，并从 `cache_dtype` 取得缓存 dtype。Indexer 的 LI C8 另看 `attention_config.indexer_kv_dtype`。两边都写出 656 字节时，CUDA 的这份 FlashMLA 记录依次存 latent、scale、RoPE，Ascend SFA 的打包顺序是 latent、RoPE、scale。表内按有效载荷计字节，实际分配还要看 page padding、存储块与 kernel 块的对应关系，以及是否启用 DCP 的 Indexer 复制。[Ascend C8 开关](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/sfa_v1.py#L860-L882) · [Ascend 规格与分配](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/worker/v2/attn_utils.py#L215) · [Ascend C8 打包](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/sfa_v1.py#L1657) · [CUDA FlashMLA 格式](https://github.com/vllm-project/vllm/blob/ced6857/vllm/v1/attention/backends/mla/flashmla_sparse.py#L77) · [CUDA Indexer 规格](https://github.com/vllm-project/vllm/blob/ced6857/vllm/model_executor/models/deepseek_v2.py#L637)

以表中第一行为例，取 Ascend SFA 的 kernel 块长 `N=128`，分配 `B=4` 个块，先忽略对齐和复制。普通 SFA 分别为主 KV 与 Indexer 建张量，它们可以在同组共用 block ID。于是同一个 block ID 在两套张量中分别取到主记录和索引记录。[Ascend SFA 块长](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/sfa_v1.py#L402) · [Indexer 分组语义](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/core/kv_cache_interface.py#L295)

```text
一层主 KV       = 4 × 128 × (512 + 64) × 2 = 576 KiB
一层 Indexer K  = 4 × 128 × 128 × 2        = 128 KiB
一层合计        = 704 KiB
```

这笔账计的是 512 个物理位置的容量。实际写入多少、哪些位置属于同一个请求，由序列长度与块表确定。完整主 KV 卸载后还有另外一层地址映射。[稀疏 KV 卸载与 HiSparse](稀疏KV卸载与HiSparse.md)

换到 GLM 或 DeepSeek-V4 系列，还要查哪层写入 Indexer、一个存储槽代表几个原始位置。第 9、10 节分别沿这两条线展开。

## 6. Mamba 层的 block 里保存什么？

全 Attention 的缓存沿历史位置保存 K/V。Mamba2 层沿序列推进时，因果卷积要接上最近 `w−1` 行输入投影，SSM 要接上截至该位置的递归状态。给下一段输入一个有效的续算边界，就可以从这两份状态继续更新。Gated DeltaNet（GDN）也保存卷积状态和递归状态，递归张量的维度由它自己的 key 和 value 头定义。[Mamba2 形状计算](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/layers/mamba/mamba_utils.py#L195-L220) · [GDN 形状计算](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/layers/mamba/mamba_utils.py#L274-L295)

### 同样存两份状态，为什么 shape 不同？

取一个 TP rank 上一层、一个续算边界，使用 vLLM 的 `SD` 卷积布局，关闭投机和 ReplaySSM。下表是单份状态的 shape。若切换到 `DS`，交换卷积状态的两个轴。[卷积布局](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/layers/mamba/mamba_utils.py#L25-L50)

| 类型 | `conv_state` | 递归状态 `ssm_state` |
| --- | --- | --- |
| Mamba1 | `[w−1, I/TP]` | `[I/TP, N_s]` |
| Mamba2 | `[w−1, (I + 2·G_eff·N_s)/TP]` | `[H_s/TP, P, N_s]` |
| Gated DeltaNet | `[w−1, (2·Hk·Dk + Hv·Dv)/TP]` | `[Hv/TP, Dv, Dk]` |

`w` 是卷积核宽度，`I` 是 Mamba 的中间投影宽度，`N_s` 是 SSM 状态维。Mamba2 的 `H_s`、`P` 是 SSM 的头数和每头宽度，`I=H_s·P`。`G_eff` 是按 TP 分配后计入组复制的有效组数，卷积投影包含输入和两份按组组织的 SSM 参数。GDN 的 `Hk/Hv` 和 `Dk/Dv` 是 key/value 的头数和每头宽度，卷积覆盖 Q、K、V 三份投影。[Mamba1、Mamba2 和 GDN 的实现](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/layers/mamba/mamba_utils.py#L179-L295)

### 用两个模型复算一页的有效载荷

- **Nemotron-H-8B-Base-8K 的 Mamba2 层。** 取 TP=4，`H_s=128`、`P=64`、`N_s=128`、`G_eff=8`、`w=4`。按 BF16 卷积和 vLLM 为该模型自动解析出的 FP32 SSM dtype，`conv_state=[3,2560]` 占 `3×2560×2=15,360 B`，`ssm_state=[32,64,128]` 占 `32×64×128×4=1,048,576 B`，合计 `1,063,936 B`。该模型的 GQA 层另有按历史位置增长的 K/V。[NVIDIA 模型配置](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json) · [vLLM 模型接入](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/models/nemotron_h.py#L755-L811) · [NemotronH 的 dtype 解析](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/models/config.py#L660-L687)
- **Qwen3.5-27B 的 GDN 层。** 同为 TP=4，`Hk=16`、`Hv=48`、`Dk=Dv=128`、`w=4`。卷积同样是 `[3,2560]`，按 BF16 算为 `15,360 B`。递归状态是 `[12,128,128]`，模型配置指定 FP32，vLLM 在 `mamba_ssm_cache_dtype=auto` 时采用它，因此占 `12×128×128×4=786,432 B`。两份合计 `801,792 B`。混合模型还包含 16 层全 Attention，各自按其 GQA 配置存 K/V。[Qwen 模型配置](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) · [vLLM 的 dtype 解析](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/models/config.py#L801-L825)

这两个数字都对应单 rank、单层、单个续算边界。`MambaSpec.shapes` 和 `dtypes` 描述页里的状态。`mamba_cache_dtype` 决定卷积 dtype，`mamba_ssm_cache_dtype` 可单独决定递归 dtype。页数则由缓存模式决定。本文所查的上游容量估算在关闭投机和额外 Prefill 检查点时，`none` 按一页计，`align` 按两页计，`all` 按 `ceil(max_model_len / block_size)` 页计。这里每页是一份续算边界，Attention 的历史则逐位置保存。[dtype 计算](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/layers/mamba/mamba_utils.py#L97-L110) · [MambaSpec 的页容量与请求页数](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/v1/kv_cache_interface.py#L993-L1057)

### 物理页比状态大时，下一行在哪？

在 vLLM-Ascend 的 MRv2 混合缓存路径中，`_reshape_mamba_kv_cache()` 按物理页大小计算状态行数 `B`，再把上述 Qwen 单行 shape 变成 `[B,3,2560]` 和 `[B,12,128,128]`。每个 view 的第 0 轴 stride 是 `物理页字节数 / 该 view 的元素字节数`。`storage_offset` 从 backing storage 起点计入原始分配偏移与首个页内组件偏移。stride 和 `storage_offset` 都以各自 dtype 的元素数计。[Ascend MRv2 状态 view](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc8aa2/vllm_ascend/worker/v2/attn_utils.py#L1668-L1686) · [stride 与 offset 构造](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc8aa2/vllm_ascend/worker/v2/attn_utils.py#L783-L811) · [Ascend Qwen3.5 部署](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Qwen3.5-27B-Qwen3.6-27B.html)

取 `801,792 B` 的 Qwen 单行载荷，假设物理页为 `800 KiB=819,200 B`。卷积从页首开始，递归状态从页内第 `15,360 B` 开始。两个 view 的偏移都以 backing allocation 起点为零。

| view | shape | stride（元素） | storage offset（元素） |
| --- | --- | --- | --- |
| 卷积 BF16 | `[B,3,2560]` | `[409600,2560,1]` | 0 |
| 递归 FP32 | `[B,12,128,128]` | `[204800,16384,128,1]` | 3840 |

两份 view 的第 1 行都从相应的第 0 行跨过 `819,200 B`，跳过页尾的 `17,408 B`。这是用于对账的示例页大小，目标运行时应读实际 `MambaSpec` 和张量 metadata。Ascend GDN 的部分 Prefill 路径按状态索引 gather 指定行，再把更新后的状态 scatter 回相同分配，状态索引指向要续算的快照。带压缩、卸载或 PD 连接器的路径还要核对自身的分配与传输约定。[GDN 读取与写回](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc8aa2/vllm_ascend/ops/gdn.py#L828-L875) · [Ascend 布局兼容范围](https://docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/non_contiguous_kv_cache.html#8-compatibility-and-limitations)

## 7. 拿到一个服务，怎样确认实际缓存？

先用模型配置列出**每类层**。同一模型里可以同时有 GQA 和 GDN。Attention cache 也可能分组使用不同的窗口和压缩率。

1. 查本 rank 的层、TP、DCP 和实际 cache dtype。普通 Attention 读 `num_key_value_heads` 和 head dim。MLA 读 `kv_lora_rank` 和 RoPE 维度。混合模型再读每类状态的 `shapes` 与 `dtypes`。
2. 记下 Runner 选中的 backend、对应的 `KVCacheSpec`、manager block size 和 kernel block size。上游 CUDA 会按 MLA/普通 Attention、硬件、dtype、block size 等条件选兼容 backend，也可以在支持时用 `--attention-backend FLASH_ATTN` 显式选择。Ascend 的平台选择函数按普通 Attention、MLA、Sparse 等入口匹配，并考虑硬件和配置。MLA 的 Prefill 计算还可能选单独的实现。[vLLM 后端选择与支持矩阵](https://docs.vllm.ai/en/latest/design/attention_backends/) · [Ascend 平台选择函数](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/platform.py)
3. 在 cache 完成分配和绑定后，挑每组一层读取 `shape/stride/dtype/storage_offset`。再沿该层的 KV 写入与 Attention 或状态更新入口，查看真正交给 kernel 的 view。上游 GPU Runner 在 `init_kv_cache()` 中分配缓存，整理 kernel block 视图后绑定到层。Ascend MRv2 构造各层视图的入口在 `worker/v2/attn_utils.py`。[vLLM 绑定位置](https://github.com/vllm-project/vllm/blob/5df63a72/vllm/v1/worker/gpu/attn_utils.py#L396-L444) · [Ascend 分配与视图](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/worker/v2/attn_utils.py)

下面的临时检查代码接收单层已经绑定的张量，或者 K/V、latent/RoPE、conv/SSM 等张量列表。它只读取 Python 侧的张量 metadata。若同时开着 PD，在两边各留一份同组记录，并记录连接器真正传的组件。

```python
def describe_cache(name, cache):
    views = cache if isinstance(cache, (tuple, list)) else (cache,)
    for i, t in enumerate(views):
        print(
            name, i, t.device, t.dtype,
            "shape", tuple(t.shape),
            "stride", t.stride(),
            "offset", t.storage_offset(),
            "bytes_per_elem", t.element_size(),
        )
```

拿到记录后，可以用 `storage_offset + Σ(坐标 × stride)` 算出已知槽位相对 backing storage 起点的元素偏移。`numel() × element_size()` 算这个 view 中的逻辑载荷。看设备分配时，沿 cache plan 核对每页字节、块数和共享 backing storage。同一块 backing 可以被多个层、组件或组引用，统计占用时按实际分配去重。[上游页容量与视图构造](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/kv_cache_interface.py) · [Ascend 非连续页说明](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/non_contiguous_kv_cache.html)

## 8. Kimi K3 怎样同时保存 MLA 历史和 KDA 状态？

取公开的 `moonshotai/Kimi-K3` 文本主干。配置中的 93 层分为 24 层带输出门的全量 MLA 和 69 层 Kimi Delta Attention（KDA）。Ascend 部署指南选用 `Eco-Tech/Kimi-K3-w4a8`，该权重的配置保留相同的层型与维度。统计缓存时，先按层型分开，MLA 按保留的历史位置计，KDA 按保留的续算边界计。[Moonshot 模型配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) · [Ascend W4A8 模型配置](https://www.modelscope.cn/models/Eco-Tech/Kimi-K3-w4a8/resolve/master/config.json) · [Ascend 部署指南](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Kimi-K3.html)

**MLA 每位置保存一份 576 维记录。** 其中 512 维是 KV latent，另有 64 维共享 Key。主干走 NoPE，64 维 Key 保存在缓存中，计算时跳过旋转。CUDA 的 K3 MLA 以 `use_sparse=False` 选取兼容后端，KV cache 的 `H=1`，绑定时压掉这条轴。BF16 的视图是 `[B,N,576]`，每位置 `1152 B`。普通 FP8 的形状相同，每位置 `576 B`。主干的 MLA 直接读历史，持久缓存分为 MLA 历史与 KDA 状态两组。[CUDA 主干选层](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/models/kimi_k3/nvidia/model.py#L909-L959) · [CUDA MLA 后端选择](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/models/kimi_k3/nvidia/mla.py#L316-L349) · [缓存 spec 与绑定视图](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/models/kimi_k3/nvidia/mla.py#L412-L434)

**KDA 每个续算边界保存卷积历史与递归矩阵。** 模型配置给出 96 个头、每头宽 128、短卷积核宽 4。沿第 6 节的 `SD` 布局，取 TP=8，关闭投机与额外状态记录。每个 rank 的本地 KDA 头数为 12，卷积把 Q、K、V 的投影通道排在一起，递归矩阵为每个本地头保留一份。按该路径的自动 dtype，卷积使用模型的 BF16，递归状态使用 FP32。[K3 状态入口](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/models/kimi_k3/nvidia/kda.py#L540-L574) · [KDA dtype 与 shape 计算](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/model_executor/layers/mamba/mamba_utils.py)

| 单层、单 rank、单个续算边界 | shape | 有效载荷 |
| --- | --- | --- |
| BF16 卷积状态 | `[4−1,3×12×128] = [3,4608]` | 27,648 B |
| FP32 递归状态 | `[12,128,128]` | 786,432 B |

一份 KDA 状态合计 `814,080 B = 795 KiB`。分配多个状态页后，两个 view 分别带上页轴 `B`。这个页轴枚举续算边界。前缀匹配配置和投机模式会改变保留的状态页或页内字段，计算分配量时还要核对两类缓存的 block size 与页步长。[vLLM K3 前缀缓存实现说明](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-27-k3.md) · [MambaSpec 页容量](https://github.com/vllm-project/vllm/blob/ced6857afa0e/vllm/v1/kv_cache_interface.py#L993-L1057)

vLLM 的 K3 部署配方在 Hopper 的一条路径上选择 `FLASHMLA`，Blackwell 的 FP8 路径则给出 `TOKENSPEED_MLA`，并在 `mla_prefill_backend` 指定 Prefill 实现。混合分配中，物理状态页和前缀匹配粒度也分别记账。配方中的 `--prefix-match-unit 128` 细化匹配边界，KDA 每份状态的 shape 仍由本地头数、卷积核宽和 dtype 决定。[vLLM K3 部署配方](https://github.com/vllm-project/recipes/blob/main/models/moonshotai/Kimi-K3.yaml)

vLLM-Ascend 的 ModelRunnerV2 在层构造时分别选 `AscendKimiMLAAttention` 和 `AscendKimiK3DeltaAttention`。若 MLA 走 BF16 拆分视图，两份缓存是 `[B,N,1,512]` 和 `[B,N,1,64]`。满足 FlashMLA 条件时可直接绑定合并视图。KDA 使用 Ascend 的 metadata 与计算入口，继承 KDA 的状态 shape，MRv2 再按页步长绑定卷积和递归视图。[Ascend 主干选层](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/models/kimi_k3.py#L514-L554) · [FlashMLA 选择条件](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/attention/mla_v1.py#L1067-L1087) · [Ascend KDA 入口](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/ops/kimi_kda.py#L179) · [MRv2 缓存视图](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bd/vllm_ascend/worker/v2/attn_utils.py#L2010-L2070)

## 9. GLM-5.3 与 GLM-5.3-Flash 的缓存为什么不同？

### GLM-5.3 的 78 层都读 MLA，21 层计算 TopK

`zai-org/GLM-5.3` 走 `GlmMoeDsaForCausalLM`。只计主模型，78 层均用稀疏 MLA。主缓存每个历史位置存 `512` 维 latent 和 `64` 维 RoPE Key，BF16 一条 `1152 B`。Indexer 给每条 query 生成 `32×128` 的索引 Q，历史每位置存一条 128 维索引 K，每条 query 取至多 `2048` 个位置。[GLM-5.3 配置](https://huggingface.co/zai-org/GLM-5.3/blob/main/config.json) · [vLLM 模型注册](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/model_executor/models/registry.py#L119-L124)

用 0 起始层号看，0、1、2、6、10，直到 74，共 21 层创建 Indexer 并算 TopK。其余 57 层复用前面 `full` 层选出的索引，随后仍从**各自**的 MLA 历史缓存读取所选位置。在 vLLM 的常规 DSA 分支和 Ascend MRv2 对应缓存分支，Indexer K 只在这 21 个 `full` 层分配。按 Ascend BF16 主 KV 与 BF16 Indexer 的有效载荷计，主模型每保留一个历史位置的逻辑载荷为 `78×1152 + 21×256 = 95,232 B`。[vLLM 创建 Indexer 的条件](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/model_executor/models/deepseek_v2.py#L1134-L1184) · [Ascend Indexer 分配条件](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/worker/v2/attn_utils.py#L347-L383)

在普通 Ascend SFA 的 BF16 路径，主缓存视图为 `[B,N,1,512]` 和 `[B,N,1,64]`，有 Indexer 的层另有 `[B_i,N_i,1,128]`。打包 C8 路径把主缓存合成 `[B,N,1,656]`，其中一条含 512 B latent、128 B RoPE 和 16 B 尺度。Indexer 量化另用 `--attention-config.indexer_kv_dtype` 选择。当前 Ascend 指南在 950DT 示例中对主 KV 与 Indexer 各选 FP8，在 A3 W8A8C8 示例中各选 INT8。上游 CUDA 选到兼容的 `FLASHMLA_SPARSE` 或 `FLASHINFER_MLA_SPARSE` 时，BF16 主 KV 绑定为 `[B,N,576]`，常规 FP8 Indexer 使用 `[B_i,N_i,132]` 字节视图。[Ascend 部署与参数](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/GLM5.2.html) · [SFA C8 字节计算](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/attention/utils.py#L184-L195) · [CUDA Indexer 规格](https://github.com/vllm-project/vllm/blob/ced6857/vllm/model_executor/models/deepseek_v2.py#L637) · [vLLM 后端支持矩阵](https://docs.vllm.ai/en/latest/design/attention_backends/)

### GLM-5.3-Flash 的 11 层稀疏 MLA 旁边还有 34 层 KDA

`zai-org/GLM-5.3-Flash` 走 `Glm5NextForConditionalGeneration`。它的 45 个文本主模型层由 11 层稀疏 MLA 和 34 层 Kimi Delta Attention（KDA）组成。稀疏层的 `qk_rope_head_dim=0`，主缓存每个历史位置只存 512 维 latent，BF16 为 `1024 B`。上游 NoPE MLA 绑定 `[B,N,512]` 的 BF16 视图。Ascend SFA 的 NoPE C8 用 `[B,N,1,528]`，每条有 512 B latent 和 16 B 尺度。两种路径的主缓存都按 token 位置寻址。[Flash 配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) · [vLLM MLA 投影](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/models/glm5next/nvidia/attention.py#L425-L460) · [Ascend NoPE SFA](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/attention/sfa_v1.py#L1722-L1757)

KeyPool 的索引 Q 仍为每 query `32×128`，`index_kpool=4` 则把每四个连续历史位置的索引 K 汇成一条 128 维 Key。完整 pool 上按 `2048/4=512` 个 pool 的预算选取，再展开成主 Attention 消费的 token 位置。未满的最后一个 pool 还有最多三条因果可见的尾部位置。比如当前 query 已到位置 10，两个完整 pool 都能入选，分别展开为 0–3 和 4–7，追加尾部的 8–10 后，Attention 读的是这 11 个位置对应的主 MLA 记录。[vLLM KeyPool 规格](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/models/glm5next/nvidia/attention.py#L78-L156) · [选 pool 后展开位置](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/model_executor/layers/sparse_attn_indexer_kpool.py#L534-L590) · [Ascend 追加尾部位置](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py#L26-L50)

| 稀疏层的状态 | 每个逻辑块或请求存什么 | 什么时候增长 |
| --- | --- | --- |
| NoPE MLA | BF16 下每位置 512 维。一页的 `N` 是主缓存的 token 槽数 | 随保留历史逐位置增长 |
| 完整 KeyPool | 一页覆盖 128 个原始位置时，实际存 `N_i=32` 条。上游 FP8 视图是 `[B_i,N_i,132]`，每条含 128 B Key 和 4 B 尺度。Ascend BF16 的 Key 是 `[B_i,N_i,1,128]`，FP8 路径另有 `[B_i,N_i,1,1]` 尺度视图 | 每攒满四个位置写入一条 |
| 尚未完成的 KeyPool | 每请求留一份原始 K 和 gate 向量。上游所查 NVIDIA 实现用 BF16 `[B,2,4,128]`，Ascend 主线用 FP32 `[B,R,256]`，`R=4+投机 lookahead` | 按请求和投机预算留固定 ring |

表中 `B`、`B_i` 取各自的组，尾部 view 的 `B` 对应请求状态槽。一个逻辑块可对应多张 Indexer kernel 页，核对 shape 时用每张页实际覆盖的原始位置数除以 4。单请求关闭投机时，尾部状态为 `2×4×128×2=2048 B`（所查 NVIDIA 实现），或 `4×256×4=4096 B`（Ascend 主线）。尾部用来续攒下一条完整 pool，和本轮交给 Attention 的位置编号各有自己的生命周期。[上游完整池与尾部视图](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/models/glm5next/nvidia/attention.py#L159-L205) · [上游 FP8 Indexer](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/models/glm5next/nvidia/attention.py#L273-L298) · [Ascend 完整池与尾部规格](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/glm5next/kv_cache.py#L28-L42) · [Ascend dtype 与组件](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/glm5next/attention.py#L111-L127)

KDA 层则沿第 6 节的续算边界记账。配置为 64 个 KDA heads、head dim 128、短卷积宽 4。用 TP4、`SD`、无投机、BF16 卷积和 FP32 递归状态举例，每层每 rank 的卷积 shape 是 `[3,6144]`，递归状态是 `[16,128,128]`，合计 `3×6144×2 + 16×128×128×4 = 1,085,440 B`。三份卷积投影同属该层的状态视图，34 层单个续算边界合计约 `35.195 MiB/rank`，另计稀疏层的历史、tail、对齐与额外检查点。[Flash 配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json) · [vLLM KDA 形状接入](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/models/glm5next/nvidia/kda.py#L158-L180) · [Ascend KDA 形状与 dtype](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/glm5next/kda.py#L200-L250)

vLLM 的 GLM-5.3-Flash 配方在 Hopper 用 BF16 主 KV，在 Blackwell 可用 FP8。上游所查的 Hopper 选择器会优先尝试支持 512 维 NoPE 的 `FLASHINFER_MLA_SPARSE_SM90`，还要通过安装环境和缓存格式的校验。Ascend 主线指南的 MRv2 单节点示例里，950DT 同时用 `--kv-cache-dtype fp8` 和 `--attention-config.indexer_kv_dtype fp8`。A3 则用 `--kv-cache-dtype int8`，保留默认 BF16 KeyPool，并显式设置 `--mamba-ssm-cache-dtype bfloat16`。可从这三个参数分别确认主 KV、完整池和 KDA 递归状态的精度，tail 则沿实际实现的 cache spec 读取。[vLLM 部署指南](https://github.com/vllm-project/vllm/blob/8d902b41/recipes/models/zai-org/GLM-5.3-Flash.yaml) · [CUDA 候选顺序](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/platforms/cuda.py#L82-L153) · [Ascend 主线 MRv2 部署与参数](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/docs/source/tutorials/models/GLM5.3-Flash.md#L240-L337)

## 10. DeepSeek-V4-Flash 与 V4.1-Flash 怎么分配缓存？

前面算 V3.2 的 Indexer 时，一条历史位置对应一条索引 K。到了 V4 系列，先要问这条记录覆盖多少原始位置，再问哪些层实际写入这份缓存。两次判断分别决定页里的行数和需要分配几份页。

### DeepSeek-V4-Flash 的 C4、C128 和滑窗各存几行？

V4-Flash 的 43 个主干层各用 128 位置的滑窗 KV（SWA）。其中 2 层只用滑窗，21 层另存 C4 历史，20 层另存 C128 历史。C4 每 4 个原始位置生成一条压缩主 KV 和一条 Indexer K。Indexer 最多选 512 条可见的压缩记录。C128 每 128 个原始位置生成一条压缩主 KV，Attention 读取可见的压缩历史。主 KV 的语义宽度是 512，包括 448 维 NoPE 和 64 维 RoPE。[V4-Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d707/config.json) · [上游缓存构造](https://github.com/vllm-project/vllm/blob/ced6857/vllm/models/deepseek_v4/attention.py#L281-L367) · [vLLM 的 V4 Attention 说明](https://vllm.ai/blog/2026-04-24-deepseek-v4)

取 vLLM CUDA 的 FlashMLA 路径。主 KV 的逻辑块覆盖 256 个原始位置，所以 C4 页有 64 条记录，C128 页有 2 条。滑窗另用 64 个位置一页。FlashMLA 的 `fp8_ds_mla` 每条记录对应 448 B FP8 NoPE、128 B BF16 RoPE 和 8 B 尺度区。C4 页先放 64 条数据，再放 64 份尺度区，合计 `64×(576+8)=37,376 B`，随后按后端要求对齐。CUDA 绑定的 SWA、C4、C128 主 KV 视图依次是 `[B,64,584]`、`[B,64,584]`、`[B,2,584]`。C4 的 Indexer 另有 `[B,64,68]` 的 MXFP4 字节视图，或 `[B,64,132]` 的 FP8 字节视图，分别含 4 B 尺度。各组的 `B` 按该组分配读取。[CUDA 逻辑块长](https://github.com/vllm-project/vllm/blob/ced6857/vllm/models/deepseek_v4/sparse_mla.py#L47-L62) · [主 KV spec 与绑定](https://github.com/vllm-project/vllm/blob/ced6857/vllm/models/deepseek_v4/attention.py#L834-L903) · [滑窗 spec](https://github.com/vllm-project/vllm/blob/ced6857/vllm/v1/attention/backends/mla/sparse_swa.py#L72-L130) · [FlashMLA 的页内地址](https://github.com/vllm-project/vllm/blob/ced6857/vllm/models/deepseek_v4/common/ops/cache_utils.py#L67) · [Indexer 的两种格式](https://github.com/vllm-project/vllm/blob/ced6857/vllm/models/deepseek_v4/attention.py#L987-L1004)

CUDA 的 FlashInfer 普通行格式直接把主 KV 和 SWA 暴露成 `[B,N,512]`。BF16 每行 1,024 B，per-tensor FP8 每行 512 B。格式由 backend 与 cache dtype 共同决定。[V4 的 CUDA 格式解析](https://github.com/vllm-project/vllm/blob/ced6857/vllm/models/deepseek_v4/attention.py#L94-L129)

Ascend DSA 用自己的块几何。取部署指南里的 `--block-size 32`，SWA、C4 和 C128 的视图各有 `N=32` 个存储行。它们的页分别覆盖 32、128 和 4,096 个原始位置，C4 的 Indexer 页同样有 32 行。A5 选 FP8 KV 时，主 KV 和 SWA 的视图为 `[B,32,1,608]`，每行包含 448 B NoPE、128 B RoPE、14 B 尺度区和 18 B 填充。Indexer K 为 `[B,32,1,128]` FP8，另有 `[B,32,1,1]` FP32 scale，每行合计 132 B。A3 的 BF16 主 KV／SWA 为 `[B,32,1,512]`，每行 1,024 B，Indexer 的同形 K 和 scale 改用 INT8 与 FP16，每行 130 B。[Ascend 部署配置](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/docs/source/tutorials/models/DeepSeek-V4-Flash.md#L195-L258) · [块尺寸表](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/layer/attention/layer.py#L33-L57) · [压缩页与主 KV spec](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/layer/attention/layer.py#L208-L235) · [滑窗记录](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/deepseek_v4/model.py#L117-L153) · [Indexer dtype 与 shape](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/deepseek_v4/indexer.py#L98-L158)

### DeepSeek-V4.1-Flash 的四份全局 KV 由谁写、由谁读？

V4.1-Flash 的 40 个主干层都用 128 位置的滑窗。按从零开始的层号，0、1 层只用滑窗。全局主 KV 只由 2、8、14、20 层写入。前三个 source 每 2 个原始位置生成一条记录，20 层每个位置生成一条，并供后续 Decoder 层读取。Index 的计算有 8 个 source 层，但历史 Indexer K 仍由那 4 个 KV source 写入。比如 24 层可以用自己的索引 Query 和第 20 层的历史索引 K 重新选择位置，然后读取第 20 层的全局主 KV。模型在训练时采用这套跨层共享关系，推理按配置读取 source 层生成的数据。[V4.1-Flash 配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8/config.json) · [vLLM 的 source 解析](https://github.com/vllm-project/vllm/blob/63ac0a4e/vllm/models/deepseek_v41/attention.py#L375-L422) · [Indexer K 的拥有者](https://github.com/vllm-project/vllm/blob/63ac0a4e/vllm/models/deepseek_v41/attention.py#L502-L544) · [CSA2 跨层共享](https://arxiv.org/html/2609.19969v1#S2.SS3.SSS1)

论文给出的 NVFP4 主 KV 将 512 个值存成 256 B，再为每 16 个值存 1 B E4M3 scale，每条 288 B。MXFP4 Indexer 每条是 64 B 数据加 4 B 尺度。于是四个 source 的全局有效载荷平均为 `(288+68)×(3/2+1)=890 B` 每原始 token。这个口径计全局主 KV 和 Indexer K，活跃滑窗、压缩器状态、分配对齐和额外算子视图要按部署实现另算。[V4.1 技术报告](https://arxiv.org/html/2609.19969v1#S2.SS4.SSS4) · [vLLM 的全局 KV 与滑窗说明](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html#swa-bounded-replay)

对于全局页，`N=逻辑块覆盖位置数/压缩率`。V4.1 的 Ascend 部署示例用 `--block-size 128`，所以 C2 与 C1 的 `N` 分别是 64 和 128。表内形状是绑定给模型的视图，字节数是记录格式所占空间。[Ascend V4.1 部署参数](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md#L271-L291)

| V4.1 部署路径 | 滑窗 | 全局主 KV | Indexer K 与 scale |
| --- | --- | --- | --- |
| vLLM CUDA，SM100 FlashMLA `nvfp4_ds_mla`，MXFP4 Indexer | `[B,32,528]`，512 B FP8 加 16 B 尺度 | `[B,N,288]`，NVFP4 288 B | `[B,N,68]`，MXFP4 68 B |
| Ascend A5 packed DSA | `[B,128,1,544]`，512 B FP8 加 32 B BF16 尺度 | `[B,N,1,320]`，256 B FP4 加 64 B BF16 尺度 | `[B,N,1,64]`、`[B,N,1,4]`，合计 68 B |
| Ascend A3 DSA，SWA 选 BF16 | `[B,128,1,512]`，BF16 1,024 B | `[B,N,1,512]`，BF16 1,024 B | `[B,N,1,128]` INT8、`[B,N,1,1]` FP16，合计 130 B |

CUDA 还提供 `fp8_ds_mla` 的全局记录和 FlashInfer 的普通 BF16／FP8 行。读到同一个 checkpoint 名，要结合所选 backend 和生效的 cache dtype 确定记录格式。[CUDA 格式解析](https://github.com/vllm-project/vllm/blob/63ac0a4e/vllm/models/deepseek_v41/attention.py#L181-L243) · [CUDA SWA 与全局规格](https://github.com/vllm-project/vllm/blob/63ac0a4e/vllm/models/deepseek_v41/attention.py#L568-L633)

用 A5 的记录宽度复算四份全局 KV 和 Indexer K，得到 `(320+68)×(3/2+1)=970 B` 每原始 token。A5 的 packed DSA 还为第 20 层分配 `[B,16,1,544]` 的 folded Indexer 视图，按每 8 个位置重新排布已量化的 K 和 scale，供候选块选择消费。这份视图的空间要单独计入。[Ascend 格式定义](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/deepseek_v41/cache_config.py#L40-L129) · [Ascend 绑定与折叠视图的偏移](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/worker/v2/attn_utils.py#L1862-L1897)

还要分清分配器里的复用。Ascend 将同一个 source 的全局 KV、Indexer 和可选 folded Indexer 放进同一物理页的不同区段。滑窗页或压缩器状态在持有别的物理块号时也可以使用这份 backing，页 stride 由这一槽可能装入的最大内容决定。模型的跨层复用按 source 层追，显存容量按物理槽、页 stride 和块数追。启用 SWA bounded replay 后，前缀命中可重算滑窗，运行中仍需给当前滑窗分配可写的页。[Ascend 分配器](https://github.com/vllm-project/vllm-ascend/blob/1af7c5fc/vllm_ascend/models/deepseek_v41/cache_config.py#L154-L213) · [SWA bounded replay](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html#swa-bounded-replay)
