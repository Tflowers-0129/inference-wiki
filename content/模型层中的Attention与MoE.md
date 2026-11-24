# 一条 Token 怎样走过模型的一层：Attention、FFN 与 MoE

假设你正在续写 `SELECT name FROM`，这一轮只处理请求 A 的一个新位置。Runner 已经把这行放进输入，也给了模型读写历史所需的地址。走进模型第 12 层后，这一行要回答两个问题：前面的哪些内容和我有关？带上这些信息后，用哪套权重变换我？前一个问题由 Attention 或递归模块处理，后一个由 FFN 处理。MoE 把 FFN 分成多套专家权重，每行只选择其中一部分。

本章数的是模型文本主干的层和逻辑张量。手算按单行、无 TP 切分、无图补齐展开。遇到真实部署，再看模型构造、backend 实例和各 rank 的本地形状。[从调度到模型输入的过程](从一轮调度到一次前向.md)

## 一行 `[1,4096]` 怎样穿过 Qwen3-8B 的一层？

先看一层最容易顺着代码读的模型。Qwen3-8B 有 36 层，隐藏宽度 4096。每层都用 GQA 和稠密 FFN。这里读上游 vLLM 的 `Qwen3DecoderLayer`，忽略 TP 和残差的融合实现，沿一行看形状。[模型配置](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) · [层构造和 `forward()`](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen3.py#L165-L236)

| 调用 | A 的一行发生了什么 |
| --- | --- |
| `qkv_proj` | 从 4096 个输入元素投影出 Q `[1,32,128]`，新 K 和 V 各 `[1,8,128]` |
| `q_norm`、`k_norm`、`rotary_emb` | 在对应的头和位置上准备 Q、K，然后交给内层 Attention |
| `attn` → `o_proj` | 新 K/V 按写槽保存。每组四个 Q heads 共用一个 KV head，查询各自可见的历史。Attention 输出按 32 个 heads 拼接，再投回 `[1,4096]` |
| `gate_up_proj` → `act_fn` → `down_proj` | 同一行映射成两份 12288 元素，一份做 SiLU，再逐元素乘另一份。所得 `[1,12288]` 投回 `[1,4096]` |

在 Python 层，`Qwen3Attention.forward()` 把 Q、K、V 交给 `self.attn(q,k,v)`。分页后端才用 metadata 完成 KV 写入和读取。当前层的残差跟着这条行传递，FFN 不再查询请求的 KV 页。上游把部分残差相加和下一次 norm 合并，并携带 `residual`。表里数的是每步的逻辑形状。[Qwen3 投影与 Attention 调用](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen3.py#L140-L163) · [稠密 MLP 的乘门与降维](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen2.py#L80-L114) · [KV 页和写槽](KV缓存布局与递归状态.md)

把代码停在两步之间。Attention 返回后，FFN 的输入还要过一次 `post_attention_layernorm`。下面是该层前向的后半段，`residual` 和 FFN 输出一起返回给后续层。[层内调用的实际顺序](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen3.py#L228-L236)

```python
hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
return hidden_states, residual
```

**试一试。** 暂且假设 BF16，每个元素 2 B。这一层新增的一个历史位置占多少逻辑 KV？若令同样的 32 个 Q heads 各自拥有独立 K/V，又是多少？

<!-- DETAILS: 核对 GQA 的字节数 -->

GQA 是 `2×8×128×2=4096 B`。若改为 Q/KV 各 32 heads，则是 `2×32×128×2=16384 B`。都只数该层的一个新位置，实际设备页还可能补齐，TP 下也要检查每 rank 的本地 heads。

<!-- /DETAILS -->

## Attention 选历史，MoE 选权重。两次 TopK 各选了什么？

换成 Qwen3-30B-A3B，同一行先经过 GQA（32 个 Q heads、4 个 KV heads），再进入 `Qwen3MoeSparseMoeBlock`。构造时，层把 `GateLinear` 交给 `FusedMoE`。执行时，MoE 收到一行，gate 给它产生 128 个专家分数，再按配置选 8 个专家并汇总加权结果。每个专家使用自己的 FFN 权重，这个检查点的专家中间宽度是 768；混合后的结果仍对应原来的一行，隐藏宽度 2048。四行在同一层都走 MoE 时是 `4×8=32` 次路由专家任务。几条输入选中同一个专家时，可以合在一组矩阵乘里。有 EP 时，再按物理专家所在的 rank 分发并归还结果。[模型配置](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) · [Router 与专家执行](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen3_moe.py#L172-L247) · [层里的 Attention 与 MoE](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen3_moe.py#L346-L415) · [EP 的 Dispatch 与 Combine](EP通信与通算融合.md)

DeepSeek-V3.2 在同一层里能看到两次选择。先取它的第 12 层，索引从 0 开始。前三层用稠密 FFN，第 12 层已是 MoE。它的 DSA 用 Lightning Indexer 为当前行挑历史位置，最多 2048 个，再用选出的索引读取历史的 MLA 缓存。MoE Router 对 Attention 之后的表示打分，从 256 个路由专家中选 8 个，另外还有一个共享专家。若当前共有 4132 个可见位置而 Indexer 选满，Attention 使用的是其中 2048 个位置；MoE 的 8 表示该行要交给的专家数，此外共享专家也处理该行。一个输出仍回到本行。[V3.2 模型配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) · [构造 Attention、稠密层和 MoE 层](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/deepseek_v2.py#L1250-L1356) · [Ascend SFA 读取 TopK 索引](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/sfa_v1.py#L2272-L2311)

![DeepSeek-V3.2 的同一输入行先按历史位置查 MLA KV，再按专家 ID 选权重并归并回同一行。示例中 4132 个可见位置、最多选 2048 个，MoE 从 256 个路由专家选 8 个，另有 1 个共享专家。](../dist/assets/model-layer-branches.svg)

所以看到 `TopK` 时先看它输出的 ID 被谁消费。如果作为 `topk_indices` 交给 Sparse Flash Attention，它要和逻辑历史、物理页对上。如果 Router 输出的是专家 ID 和权重，执行器接下来关心的是专家位置、消息和每个专家分到了几行。把这两份输出分别记下来，就能区分是历史读取变化了，还是同一行的专家计算变了。[稀疏 KV 的索引与回填](稀疏KV卸载与HiSparse.md) · [逻辑专家到物理槽](负载均衡与专家并行.md)

## 历史越长，下一行一定要读更多 KV 吗？

在 GQA 的全量 Attention 中，一行 Q 要与本请求因果可见的所有位置计算。MLA 把每位置写到缓存里的 K/V 表示压成 latent 及模型所需的 Key 分量，能减少每位置的缓存规模。**是否还要读遍历史**看模型有没有再做位置选择。例如 Kimi K3 的全量 MLA 仍读可见历史，DeepSeek-V3.2 的稀疏 MLA 先筛位置再读主 KV。Indexer 自己也要处理候选历史，主 Attention 读得少还要加上索引工作。[Kimi 选层源码](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/kimi_k3/kimi_k3.py#L691-L723) · [Attention 与辅助缓存拆分](KV缓存布局与递归状态.md)

DeepSeek-V4.1-Flash 还能把“谁存历史”和“谁做这一层的选择”分开。40 层各有 128 位置滑窗，层 2、8、14、20 另外持有全局 KV 和 Indexer K。看层 24 和层 26：层 24 用自己的索引 Q 查层 20 的 Indexer K，得到本次选择，再读层 20 的全局 KV。层 26 取层 24 算好的 TopK，仍读层 20 的全局 KV。当前层仍处理自己的滑窗。这里可以核对三个层号：**窗口的层、全局 KV 的所有者、索引结果的生产者**。外层 `hc_pre`/`hc_post` 在 Attention 和 FFN 前后分别处理残差流，源码里的两次调用都值得保留。[层型与所有者配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8/config.json) · [模型层角色](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/deepseek_v41/model.py#L604-L646) · [取已有索引及读取全局 KV](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/dsa_v41.py#L660-L707) · [层内 `hc_pre`、Attention、FFN 和 `hc_post`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/deepseek_v41/model.py#L1105-L1137)

另一条路是维护跨 Token 的状态。拿 Qwen3.5-35B-A3B 的 Gated DeltaNet（GDN）层来说，当前行使用之前保存的短卷积状态和递归矩阵，产生输出，同时更新状态供下一行。Kimi 的 KDA 和 GLM-5.3-Flash 的 KDA 也有对应的卷积与递归状态，参数和形状各看自己的实现。混合模型里隔一段仍有 GQA 或 MLA，逐层分开估算页表、全量历史和状态缓冲。[Qwen3.5 的层构造](https://github.com/vllm-project/vllm/blob/c410543d9/vllm/model_executor/models/qwen3_5.py#L123-L183) · [递归状态形状与 dtype](KV缓存布局与递归状态.md)

## 看一眼模型配置，能猜出这一层的 A 和 FFN 吗？

按具体检查点读 `layer_types` 和 FFN 的选层条件。表中层数指文本主干，另接的投机层不计入。模型结构确定计算与状态的语义，后端和并行方式由运行配置再决定。

| 检查点 | A 怎样接触历史 | FFN 怎样处理当前行 |
| --- | --- | --- |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) | 36 层 GQA，Q/KV heads 为 32/8 | 各层稠密，SwiGLU 的中间宽度 12288 |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json) | 48 层 GQA，Q/KV heads 为 32/4 | 每层 MoE，128 选 8 |
| [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json) | 30 层 GDN、10 层 GQA | MoE，256 选 8，另有共享专家分支 |
| [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json) | 61 层稀疏 MLA，Indexer 的历史位置上限 2048 | 前 3 层稠密，其余为 256 选 8、另加 1 个共享专家 |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8/config.json) | 40 层各有 128 位置滑窗，部分层写或借全局 KV | MoE，384 选 6，另有 1 个共享专家 |
| [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66/config.json) | 24 层全量 MLA、69 层 KDA | 首层稠密，其余为 Latent MoE，896 选 16，另有 2 个共享专家 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/46a050be/config.json) | 11 层稀疏 NoPE MLA、34 层 KDA | `mlp_layer_types` 指定 3 层稠密、42 层 MoE，MoE 为 288 选 8、另有 1 个共享专家 |

比如看到 GLM-5.3-Flash 的 `layer_types[layer_idx] == "linear_attention"`，可知这一层构造 KDA，下一步还要单独看 `mlp_layer_types[layer_idx]`。Kimi 也是先实例化 KDA 或 MLA，再用 `is_moe_layer_idx()` 给 `self.mlp` 选实现。到了 Kimi 的 Latent MoE 里，`gate` 看的是原始 7168 维。路由分支用 `routed_input_transform` 把输入投到 3584 维，专家结果经 `routed_output_transform` 投回 7168 维。共享分支从原宽度计算。若只比较相同元素精度下的一行逻辑表示，3584 维是 7168 的一半。真实 Dispatch 的字节数还取决于后端是否传这份表示、远端副本、精度、scale 和 padding。[GLM 选择 A 与 FFN](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/glm5next/model.py#L320-L379) · [Kimi 选层](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/kimi_k3/kimi_k3.py#L691-L746) · [Kimi Router、共享分支与 Latent MoE](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/kimi_k3/kimi_k3.py#L382-L498)

## `attention/` 里的文件多，当前层到底进哪个？

先查模型层构造出的类和选项，再查平台选择器。比如 Ascend 的通用表会读 `use_mla`、`use_sparse`、`use_compress`。普通稠密 fallback 走 `attention_v1.py`，MLA、SFA、压缩 DSA 分别有对应实现。满足硬件、包、dtype、模型及执行模式的条件时还有优先路径。DeepSeek-V4.1 的层则构造专用缓存角色与实现，不能直接由这个 fallback 表推出。[Ascend Attention 后端选择](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/platform.py#L330-L353) · [V4.1 层构造](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/deepseek_v41/model.py#L1007-L1037)

| 看到的文件 | 接着在调用里找什么 |
| --- | --- |
| [`attention_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/attention_v1.py) | 稠密的 KV 写入、分页 Attention 或 FIA 分支。核对 batch 状态、页表、query 布局和有效长度 |
| [`flash_attention_v3.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/flash_attention_v3.py)、[`fa3_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/fa3_v1.py)、[`fa_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/fa_v1.py) | 先确认实际类名和包。这里有不同的 Flash Attention 接口，metadata 与图更新要读选中的实现 |
| [`mla_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/mla_v1.py)、[`flashmla.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/flashmla.py) | latent/Key 缓存布局、Prefill 和 Decode 的不同计算组织、精确长度的来源 |
| [`sfa_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/sfa_v1.py)、[`dsa_v1.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/dsa_v1.py)、[`dsa_v41.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/attention/dsa_v41.py) | Indexer 选出的历史位置交给谁，主 KV 或压缩 KV 由哪层持有，该位置怎样转成地址 |
| [GLM 的 `kda.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/models/glm5next/kda.py)、[Kimi 的 `kimi_kda.py`](https://github.com/vllm-project/vllm-ascend/blob/913f9d79f/vllm_ascend/ops/kimi_kda.py) | 从模型的递归 Attention 模块追状态更新内核，查卷积和递归状态的 shape、dtype、批次映射 |

最后读具体的 `forward()` 调用和 metadata。以设备 tiling 的 FA3 为例，光知道这一层是 GQA，还不能回答 CPU 会不会等长度。还要找到选中的 backend 类、`seq_lens` 的生产者、`get_scheduler_metadata()` 更新的计划和 Attention 消费的位置。涉及投机、CP、KV 量化或 Offload 时，这些依赖都可能变。[跟 4132 走一遍 FA3 计划和回放](从一轮调度到一次前向.md)

## Trace 里同一行突然变慢，先查历史还是专家？

先记录这行的 `request_id`、本轮 batch 和层号。低成本地做两种对照。一组保持批行数，改历史长度，记录 A 的实际后端、可见长度、Indexer 的候选与 TopK、读到的 KV 或状态。另一组固定历史长度与后端，对照专家路由，记录每行的 TopK、每个目的 rank 的任务数、Dispatch 与 Combine 及其等待。要单独验证一种专家布局时，应固定逻辑路由和权重。完整模型里上下文会改变激活和路由，所以跨请求比较还要同时记录两侧的工作量。

最终在同一轮找出哪个阶段延后了下一层或采样。A 侧 `TopK=2048` 只能说明一个选择上限，FFN 侧 `top_k=8` 也只说明每行的专家任务数。实际受限于哪段，需要该模型、设备和合法配置的采样。本章的形状、字节数及层内顺序来自配置、源码和手算。[前向的 Trace 排查](从一轮调度到一次前向.md) · [专家通信的设备对照](EP通信与通算融合.md)
