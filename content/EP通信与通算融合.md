# EP 通信与计算重叠：算子选择与融合执行

第四章把同一层的专家任务摊到了多个 rank。现在先固定放置和路由，假设四个 rank 各承担二十次任务：工作量已经接近，为什么一次 MoE 前向仍可能慢？

还缺一段执行过程。专家权重在目标 rank 上，输入激活要送过去，计算结果还要回到原 token 所在的行。下面先追这次往返，再看 AllGather、AllToAll、MC2 和 MegaMoE 分别改变哪一段。Prefill 与 Decode 的区别也放回这条数据路径里解释。

> 本章以 Ascend 的 EP 实现为主。教学算例固定四个 rank、TP=1、无 PCP、八个逻辑专家且没有冗余副本，每 rank 两个专家；四十个 token 使用 top-2，产生八十次专家任务。这个算例与第 7 节模拟泳道均为教学构造；第 11 节另列 A5 实测。源码选择条件对应本章核查的实现，不是硬件的通用最优策略。

## 1. Router 选出两个专家以后，究竟要发什么？

先看 rank 0 上的一行隐藏状态 `x_i`。它对应一条请求的某个位置，形状可记为 `[H]`。Router 为它选出逻辑专家 E2、E5，并给出权重 `w2、w5`。按当前放置，E2 在 rank 1，E5 在 rank 2。

目标专家需要 `x_i` 的数值，才能执行自己的 FFN。单独发送 token ID 并不能替代这一层的输入：此前各层已经把它变成了当前隐藏状态。通信还需要足够的路由与还原信息，例如目标专家、原行关联、计数和权重；各实现放在哪个张量、在哪一步应用权重，要沿调用方核对。

```text
原行 i 的 x_i
  ├─ E2 / rank 1 → f_E2(x_i)
  └─ E5 / rank 2 → f_E5(x_i)
原行 i 的结果：y_i = w2 × f_E2(x_i) + w5 × f_E5(x_i)
```

![一行隐藏状态按 top-2 路由到专家，计算结果按原行关联归还并加权](../dist/assets/ep-token-roundtrip.svg)

图中的两条专家路径不会产生两个新的输出 token。它们是同一行在一个 MoE 层里的两次专家计算。Combine 后仍要恢复这行的结果，后续层才能继续；新输出 token 则还要经过后面的模型层、输出头和采样。

第四章讨论的逻辑专家到物理副本映射，在这里确定目标。放置变化会改变数据去向，但不能丢掉原行身份。输入重排后，结果若回到了另一行，即使形状和地址都合法，计算也已经错了。[通信与专家计算的调用边界](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py) · [路由、计数与结果还原](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/token_dispatcher.py)

## 2. AllGather 与 AllToAll：数据怎样找到持有专家的 rank？

先取本章 TP=1 的四 rank 布局，每个 rank 初始持有十行输入。AllGather 路线让各 rank 获得全部四十行，再由各 rank 只计算自己持有的专家。某行没有选中本 rank 的专家时，本地对应贡献为零；收尾通过求和与分片，把各处的专家贡献合并并返回原行归属。

在这里，AllGather 与 ReduceScatter 要连起来读。前者收集输入，后者合并并归还输出；只画“所有 token 都送到所有卡”，会漏掉计算怎样被限制在本地专家，以及结果怎样回去。

```text
AllGather 输入与路由信息
  → 按本地专家筛选、排序和分组计算
  → 还原 token 行并形成各 rank 的贡献
  → ReduceScatter 合并结果，返回本地原行
```

Ascend 的 `PrepareAndFinalizeWithAllGather` 把跨 rank 的收集和收尾放在 prepare / finalize。普通 DP 路径先按各 DP 最大行数补齐，再收集 hidden states、router logits，最后 ReduceScatter 并裁掉 padding；`TokenDispatcherWithAllGather` 内部主要做本地专家路由与还原。启用序列并行、TP 或 PCP 时，通信组与行布局还会变化，不能把这张四 rank 示意直接当成所有拓扑的调用链。[prepare / finalize 与并行组](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/prepare_finalize.py)

另一条思路是先按目标专家所在的 rank 组织输入，只把相应激活送往需要它的目的地。AllToAll 类路径完成这种定向交换，专家计算后再按反向关联返回结果。不同源、目的之间的行数可以不同，因此实现还要维护计数、偏移、分片长度和逆重排。

![全量聚合后本地筛选，与按目标 rank 组织输入及回传结果的两条数据路径](../dist/assets/ep-exchange-paths.svg)

少发不需要的副本，是定向交换值得比较的原因；代价则包括路由排序、消息大小管理和结果还原。当前 Ascend 的 `TokenDispatcherWithAll2AllV` 会组织专家计数、交换计数、取得 CPU 侧 split 数组，再完成输入重排、通信和逆重排。某些分支的控制与同步成本，也要计入这条路径。[AllToAllV 的预处理与回传](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/token_dispatcher.py)

因此，两条路线要同时比较数据量和组织这些数据的成本。AllGather 的集合操作规整，但会让没有选中本地专家的行也到达本 rank；定向交换减少这类副本，却需要处理不规则路由。哪条更快，不能只凭通信名称下结论。

### AllToAllV 交换哪张张量，怎样把结果送回原行？

沿 Ascend 的 `TokenDispatcherWithAll2AllV`，设源 rank 的隐藏状态为 `[T,H]`，物理专家 ID 和路由权重均为 `[T,K]`。`_preprocess` 对专家 ID 做直方图，并按目标 rank 求和得到 `input_splits`。各源 rank 再 AllGather 这份直方图，形成 `[EP,E]` 的计数表，`E` 是本次路由使用的物理专家总数。接收者取本地专家的列，沿专家轴求和得到按来源排列的 `output_splits`，沿来源轴求和得到传给 grouped GEMM 的 `group_list`。[直方图与 split 数组](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L643-L686)

取下一节“选下一个 rank 的两位专家”的路由。rank 0 的十行都选 E2、E3，两个专家在 rank 1。rank 0 的 `input_splits=[0,20,0,0]`。rank 1 从 rank 0 接收二十行，所以 `output_splits=[20,0,0,0]`，本地专家计数为 `[10,10]`。`npu_moe_token_permute` 按 `topk_ids.numel()` 生成二十行，原来的每行激活在两个专家的区间各出现一次。[源端重排与 dispatch](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L502-L584) · [展开行数与排序参数](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L607-L647)

接收缓冲按来源 rank 分段。多个来源命中同一位专家时，接收侧用第二次重排把它们归到同一个连续专家段，供 grouped GEMM 读取。Combine 先撤销这次重排，再交换 `input_splits` 与 `output_splits` 的收发角色做反向 AllToAll。每个源 rank 最后用自己保存的逆重排索引和 `topk_weights` 归并，恢复 `[T,H]`。这条路径由源 rank 保存权重，并在回收结果时消费。[接收端专家重排](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L688-L725) · [反向交换](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L586-L605) · [还原原行和加权](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L727-L749)

## 3. 同样八十次专家任务，为什么通信量还会变？

把八个专家固定为 `rank 0: E0/E1`、`rank 1: E2/E3`，依次类推。每个源 rank 有十个 token。下面三组构造始终有四十个 token、八十次专家任务，而且每个 rank 都承担二十次任务：

| 构造路由 | 每个 token 选择什么 | 定向教学模型的远端激活副本 |
| --- | --- | --- |
| 同一目标 rank 的双专家 | 选下一个 rank 的两位专家 | 40 份 |
| 两个目标 rank | 分别选后续两个 rank 的一位专家 | 80 份 |
| 混合本地与远端 | 每个源前五个 token 选本地双专家，后五个各选后续两个 rank | 40 份 |

这里给定“每个 token 对每个目标 rank 只发一份”的教学规则。同一目标 rank 收到一份激活后，可供该 rank 的两次专家计算使用；本地专家不需要跨 rank 传入。真实后端是否这样去重、在什么阶段展开专家行，应另查实现。

AllGather 对照让每个 token 到达另外三个 rank，远端副本总数固定为 `40×3=120`。选 `H=4096`、BF16，每份激活为 `4096×2=8192 B=8 KiB`，则输入侧远端逻辑 payload 为：

```text
AllGather 对照：       120 × 8 KiB = 960 KiB
同一远端 rank 双专家：  40 × 8 KiB = 320 KiB
两个远端 rank：         80 × 8 KiB = 640 KiB
混合本地与远端：        40 × 8 KiB = 320 KiB
```

<!-- EP_LAB -->

这本账只数输入激活送达远端目的地的逻辑副本。不含路由元数据、量化尺度、padding、协议、ReduceScatter、Combine 或链路跳数；集合通信的中转也可能使实际链路字节不同。本地使用同样需要设备访存。因此 `960/320=3` 只能比较这个给定口径的数据量，不能写成三倍加速。

按第 2 节的 Ascend AllToAllV 实现，同一远端 rank 上的双专家各有一行。沿用 BF16、`H=4096`，关闭传输量化，这组路由的 Dispatch 激活量应按 `40×2×8 KiB = 640 KiB` 推导。若专家输出也为 BF16、宽度仍为 4096，源端在归并前还会接回同等大小的结果。这个数对应传输描述的行数，实测链路流量还需计入计数收集与通信实现。[Dispatch 与 Combine 的实际调用](https://github.com/vllm-project/vllm-ascend/blob/1bf75989/vllm_ascend/ops/fused_moe/token_dispatcher.py#L522-L605)

专家任务数相同，目的 rank 的分布仍可能不同。回到第四章，EPLB 解决的计算分布，与这里要比较的源到目的通信分布，已经是两张需要对齐的账。

## 4. Prefill 与 Decode，为什么经常落到不同路径？

普通单 token Decode 中，一条请求通常只增加一行 query；Prefill 的一个 chunk 可以带来很多行。这会改变本轮输入量，以及分给每个专家的矩阵行数。投机验证又可能让 Decode 请求一次带来多行，所以选择算子时仍要看实际 batch。

对一个专家，设收到 `n_e` 次有效专家任务，FFN 的矩阵乘可以按这个行数组织。较大的 `n_e` 有机会摊薄权重读取和提交成本；很小的 `n_e` 则更容易暴露启动、路由和通信延迟。这是分析形状的起点，实际效率还受量化、分组 GEMM、对齐和设备实现影响。

这也解释了为什么 Decode 值得比较更大的融合边界：一轮里多次较短调用的提交、就绪通知与中间处理，可以成为显著开销。Prefill 同样可能从融合和分块流水获益，但大输入还要检查传输吞吐、接收缓冲与大矩阵计算是否匹配。最终需要比较的是同一形状下的完整执行路径。

AllGather 和定向交换也会随形状改变成本。输入多时，要检查全量聚合带来的副本和缓冲压力；输入少时，要检查计数交换、重排、小消息与同步是否占据主要时间。EP 规模和 top-k 又决定一行可能需要到达多少个 rank，节点内与跨节点链路也不能混作同一带宽。

所以“Prefill 使用 AllGather 居多”需要限定到具体后端与策略。有些 Ascend 路径确实会在较大 batch 回到 AllGather，另一些则回到 AllToAll；支持的全链融合路径也可能用于 Prefill。P/D 标签帮助理解负载，真正的选择还要继续追到 token 计数、容量与硬件分支。[运行时通信选择](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ascend_forward_context.py)

## 5. 沿选择函数看：这轮为什么用了这个算子？

先从 `select_moe_comm_method()` 的共同入口读。非 MoE 不需要选择这条专家通信路径；没有启用 EP，或 EP 组只有一个成员时，会进入 AllGather 类实现，这个类名不表示单成员组还要发生跨卡交换。LoRA 与 EP 的组合则先选择 AllToAll，随后才轮到硬件策略。

下面概括本章核查实现的主要分支，省略模型维度校验等前置条件：

| 硬件策略 | 主要选择条件 | 超出普通 MC2 容量后的路径 |
| --- | --- | --- |
| A2 | 每设备专家数不超过 24、计算出的 EP 规模至少 16，且本轮计数在 MC2 容量内，才走 MC2；否则 AllGather | AllGather |
| A3 | 先检查可用的 CANN MegaMoE；再检查已启用的旧融合路径及 EP≤32 条件；未选融合时，容量内走 MC2 | AllToAll |
| A5 | 可用融合路径优先，MTP drafter 还受量化布局限制；未选融合时，容量内且跨 DP 并行规模大于 1，走 MC2 | 源码的 `world_size_across_dp`≤top-k 时 AllGather，否则 AllToAll |

表中是框架采用的策略，不能据此证明 A2 的 AllGather 在任意大输入上最优，也不能把 A3 的容量分支当成所有硬件的行为。底层算子支持某种产品，也不保证当前框架已把它接进该产品的选择路径。[硬件策略注册](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/device/hardware_profile.py) · [三个选择函数](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ascend_forward_context.py)

这里的“本轮计数”还需要对账。Runner 会传入本轮 batch 的 token 计数，并在相关路径取跨 DP 最大值；图 padding、TP 切分和每请求 query 长度又影响后续形状。它不是八十次专家任务，也不能不经核对就当成所有算子实际消费的有效行数。

<!-- DETAILS: 源码对照：用户配置值与运行时融合分支 -->

本章核查的 `enable_fused_mc2` 用户配置有三个取值：0 关闭融合，1 请求旧融合路径，2 请求 CANN MegaMoE。配置初始化会把 2 归一化成内部值 1，再由运行时算子包可用标志区分两条路线。因此后续看到 `enable_fused_mc2 == 1`，还要一起看该标志和 `use_cann_megamoe()`，不能直接认定它只会调用旧算子。

可用性还包括硬件能力、模型、EP 规模和量化等检查。配置值表示请求采用某条路径，不替代实际调用验证；本章也没有在目标机器上执行这些启动组合。[配置校验与归一化](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ascend_config.py) · [融合路线的调用方](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py)

<!-- /DETAILS -->

## 6. 从源码看 MC2 与 Fused-MC2 的关系

`MC2CommImpl` 与 `FusedMC2CommImpl` 是 `MoECommMethod` 下的两条实现路线，复用 MC2 的输入准备和收尾布局。普通路线沿基类的 `fused_experts()` 分别调用 Dispatch、专家 MLP、Combine；Fused 路线覆盖这个方法，直接调用更大范围的融合算子。接口名里已经有 `fused`，并不能证明整个专家过程都在同一个 Kernel 里。[两条通信实现](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py#L136-L196)

### 6.1 普通 MC2，默认也有局部融合

先固定 W8A8、SiLU、默认开启的 `fusion_ops_gmmswigluquant`。普通 MC2 的主要外部调用是：

```text
npu_moe_distribute_dispatch_v2
  → grouped_matmul_swiglu_quant_weight_nz 〔GMM1 + SwiGLU + Quant〕
  → npu_grouped_matmul                  〔GMM2〕
  → npu_moe_distribute_combine_v2
```

专家权重以列表存放时还可能走对应的 tensor-list 接口。只有关闭上述局部融合，才会把中间部分展开为 GMM1、`npu_dequant_swiglu_quant`、GMM2 三个调用。因此普通 MC2 的默认泳道不能直接画成五个完全分离的 Kernel。[W8A8 的局部融合与非融合分支](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py#L415-L552) · [Dispatch / Combine 调用](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/token_dispatcher.py#L242-L275)

这块局部融合也有不同内部调度：其普通 tiling 先完成主要 GMM 再做向量计算；split-workspace tiling 则通过两份工作区允许下一片矩阵计算与上一片激活、量化交叠。选择受形状和工作区限制影响。下面的默认模拟把它保留成不透明的整体，不借用 Fused-MC2 的 flag 来补画内部时序。[局部融合的两个内核分支](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/gmm/grouped_matmul_swiglu_quant/op_kernel/grouped_matmul_swiglu_quant.cpp#L73-L90) · [tiling 选择](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/gmm/grouped_matmul_swiglu_quant/op_host/grouped_matmul_swiglu_quant_tiling.cpp#L219-L246)

Python 中这几个调用顺序提交，后一个设备消费者要等前一个结果。当前普通 routed-expert 主线没有在 Python 里把 Dispatch、MLP、Combine 分别切到三条流；Host 也不必每次等 Device 算完再提交下一项。共享专家的可选多流事件属于另一条协作路径，应与本节分开。[提交与事件位置](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py#L154-L184)

### 6.2 Fused-MC2，还有自定义与 CANN 两个落点

结合第 5 节的配置归一化，再追 `FusedMC2CommImpl.fused_experts()`，可以区分：

| 路线 | 主要入口 | 执行边界 |
| --- | --- | --- |
| 普通 MC2 | `TokenDispatcherWithMC2`、`apply_moe_mlp()` | Dispatch、MLP、Combine 是不同外部调用；MLP 内可局部融合 |
| 自定义 Fused-MC2 | `torch.ops._C_ascend.dispatch_ffn_combine` | 输入路由、专家 FFN、回传与还原收进一个自定义算子 |
| CANN MegaMoE | `_apply_cann_mega_moe()` 调用算子包接口 | 另一条全链融合实现，使用对应对称通信缓冲与接口契约 |

Fused 的两条路线都获得权重、路由、概率和通信上下文，但不能把它们的内部调度直接等同。下节能逐行展开的，是仓库内 W8A8 自定义 `dispatch_ffn_combine`；CANN MegaMoE 的内部实现不由这段 C++ 代码证明。[Fused 的两条调用分支](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py#L563-L615) · [CANN MegaMoE 算子说明](https://gitcode.com/cann/ops-transformer/tree/master/mc2/mega_moe)

![分离的语义阶段、普通 MC2 局部边界与 Fused-MC2 全链边界；具体局部融合随实现变化](../dist/assets/ep-fusion-boundaries.svg)

图中的五段是语义工作，不用于数 Kernel。全链融合扩大的是跨 Dispatch、专家计算、Combine 的协作范围；普通 MC2 已有的局部融合仍需保留在比较条件里。DeepGEMM 的同名 Mega MoE 则属于自己的实现，不能据名称互相证明硬件支持或性能。[DeepGEMM Mega MoE](https://github.com/deepseek-ai/DeepGEMM#mega-moe)

## 7. 按实际 flag 关系模拟两种方式的 profiling

这里沿作者自绘图中的“交换与计算能否接续”继续往下追，固定到自定义 W8A8 内核。入口声明 `KERNEL_TYPE_MIX_AIC_1_2`，AIC 分支执行矩阵计算，AIV 分支执行 `DispatchAndCombine()`；这不是 Python 新建一条通信流后，让两个普通算子自由并行。[混合核入口](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine.cpp#L22-L31)

### 7.1 AIV 与 AIC，各自按什么顺序走？

把同类核的工作聚合起来，可以先读成两行：

```text
AIC 每核：GMM1 的本地专家循环 → 等第一组激活结果 → GMM2 的本地专家循环
AIV：路由/量化与计数交换 → 按专家 Dispatch → 两组 SwiGLU/再量化
     → 跟随 GMM2 进度回传 → 跨 rank 同步 → unpermute / 加权归并
```

AIC 的顺序是每个核自身的控制流。源码没有在 GMM1 和 GMM2 之间插一个“全设备所有 Cube 都齐了”的屏障，因此聚合图不能用来断言所有 AIC 完成 GMM1 后，任何 AIC 才能开始 GMM2。[AIC / AIV 分支](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L216-L231)

这条自定义路径用 HCCL 上下文取得本地及远端 window 地址。Dispatch 通过 AIV 的搬运路径，把 peer GM 的激活经 UB 搬到本地 GM；Combine 再把结果写回来源 rank 的窗口。不能据 HCCL 上下文的存在，额外画一条未经确认的独立 HCCL 通信 stream。[通信窗口](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/utils/hccl_shmem.hpp#L84-L145) · [输入搬运与就绪通知](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L842-L867)

### 7.2 四个本地专家，哪些位置真的允许交叠？

为了把等待位置逐项标出来，下面另取两个逻辑 rank，每 rank 四个非空本地专家，TP=1、W8A8+SiLU、关闭共享专家多流，只看稳态；它与第 3 节四 rank 的副本计数实验是两个算例。本地专家记作 e0～e3，图中 G1、S、G2 分别表示第一段矩阵、SwiGLU/再量化、第二段矩阵。

| 交接位置 | 源码中的等待条件 | 四专家示意中的表现 |
| --- | --- | --- |
| Dispatch → GMM1 | 每个本地专家输入完成后发布对应就绪 flag，GMM1 消费前等待 | G1(e0) 可以与 D(e1) 等后续输入搬运交叠 |
| GMM1 → 第一组 S | AIV 先走完全部 Dispatch，再等第一组 GMM1 结果 | `epilogueGranularity=E-1=3`，S0 处理 e0～e2，可与 G1(e3) 交叠 |
| 第一组 S → GMM2 | 当前 AIC 跑完自己的 GMM1，再等第一组向量结果 | G2(e0) 开始；AIV 继续处理第二组 S1(e3) |
| 第二组 S → 后续 GMM2 | `groupIdx+1==epilogueGranularity` 时再次等 flag | 等待发生在 G2(e2) 前，不能擅自挪到 e3 前 |
| GMM2 → Combine | AIV 处理完两组 S 后，再按 GMM2 专家进度等待与回传 | C(e0) 可以与 G2 的后续专家段交叠 |

这里的分组、等待点来自这条 W8A8 内核，不是任意 token 小块的通用规则。[GMM1 等待与两次通知](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L450-L535) · [GMM2 的第二次等待](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L555-L591) · [激活分组与回传入口](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L920-L977)

本 rank 的输入行数 M 与 top-k 的乘积 `M×topK≤4096` 时进入 CombineV2，超过时进入 CombineV1。下面模拟按小输入分支聚合专家进度；真实 CombineV2 继续按 AIC 的 tile 分工和更小的行切片搬运。模拟既没有逐核展开，也没有把四个专家段当成真实 tiling 数。[Combine 分支选择](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L254-L258) · [更细粒度回传](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L1025-L1089)

<!-- EP_TIMELINE -->

默认对照保留普通 MC2 的 GMM1+SwiGLU+Quant 局部融合，只把这个算子的总区间画出来。取消勾选后，才展示关闭局部融合的受控教学配置。两种情况都把 Host 提交与 Device 执行分开，不要求 Host 等每个算子完成才提交下一项。

![MC2 与自定义 Fused-MC2 的源码依赖泳道，保留普通 MC2 默认局部融合，并展开自定义算子的 AIV / AIC 协作](../dist/assets/ep-overlap-swimlanes.svg)

播放器和静态图使用相同教学时长假设，横轴为自定推进格。块宽、输入晚到和同角色串行都是模型条件，不是实测 Kernel 时长或 stream 编号；默认普通 MC2 局部融合的内部 tiling 未展开，所以也不能拿两图终点计算性能收益。可点击事件，核对其源码位置、直接前驱和下游消费者。

结果回传后还要满足原行输出条件。这条自定义内核明确在 Combine 后进行 `CrossRankSync()`，然后 unpermute 并按概率合并；普通 MC2 的输出则在自身 Combine 接口结果可读后交给下游，不能凭空为它补上同名的显式屏障。融合扩大了协作边界，也保留了这些依赖。[同步与加权还原](https://github.com/vllm-project/vllm-ascend/blob/main/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp#L957-L977)

## 8. 算子内流水、共享专家多流与 DBO，重叠的是同一件事吗？

上一图按专家和激活分组展开了同一专家路径内部的依赖。另一类机会来自独立工作：例如满足模型与实现条件的共享专家计算，可以与 routed experts 的部分通信交错。分析时要先找出两边什么时候产生输入、什么时候需要合并结果，再看能否在各自的流上执行。

DBO 则把 batch 拆为两个 microbatch，用一份的计算覆盖另一份的通信。每份仍要保留自己的输入边界、KV 地址和执行状态；它与第二章讲的 Host 提前安排后续批次，也不是同一个控制位置。vLLM 的 DBO 设计围绕 MoE 通信设置交接点，让两个 microbatch 在通信与计算之间切换。[DBO 设计](https://docs.vllm.ai/en/latest/design/dbo/)

因此可以分别记录三种粒度：算子内部的数据块、同一前向的独立分支、两份 microbatch。把时间线上的通信和计算放在不同的行，只说明它们被分开记录；要声称发生重叠，还应看到输入已就绪、执行区间实际交叠，并且没有在消费点把全部等待重新暴露出来。

支持范围也要一起核对。本章所查 Ascend 配置在启用融合 MC2 路线时会关闭共享专家多流选项，不能把二者分别报告的收益直接相加。拆 microbatch 还可能改变每专家的 GEMM 行数和 padding，前后对照需要记录这些变化。[融合与共享专家多流配置](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ascend_config.py)

## 9. 选中了融合路径，还需要满足哪些条件？

先查量化和布局。通信可以携带量化后的激活及尺度，FFN 又要消费对应的权重格式、尺度与激活函数。CANN MegaMoE 对不同产品支持的组合并不相同；框架还需准备匹配的权重列表、通信缓冲和接口参数。某个 BF16 示例能跑，不能证明同模型的另一量化格式也会走相同内核。[CANN 算子契约](https://gitcode.com/cann/ops-transformer/tree/master/mc2/mega_moe) · [量化与接口适配](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_utils.py)

再把两种容量分开：每 rank 原始输入行数，和 Dispatch 后该 rank 收到的行数。后者受所有源 rank 的路由影响，可能远大于本地输入量；第四章的热点专家即使不会改变全局任务总数，也会改变接收端压力。

当前 CANN 路径的对称缓冲按一致的输入容量建立，另有接收容量设置。`mega_moe_max_tokens` 属于接收侧口径，不能拿它当作本地原始输入上限。输入超过已分配容量时，调用方会报错；选择函数优先选中融合分支，也不保证超量后自动切回 AllToAll。[对称缓冲与输入容量检查](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/fused_moe/moe_comm_method.py)

最后查通信拓扑与实际瓶颈。节点内链路、跨节点 RoCE、通信域规模，以及通信任务与矩阵计算竞争的设备资源，都会影响可以覆盖多少时间。若链路持续是最长的一段，提升计算吞吐甚至可能让通信尾部更明显。融合能否缩短整轮，需要在相同模型、精度、路由和拓扑下对照，不能从算子名字推断。

## 10. 怎样用 Trace 解释一次选择和一次收益？

沿同一轮、同一层同时保存“选择依据”和“执行证据”。先记录每 rank 的真实输入行数、padding 后行数、query 长度、top-k、专家任务数，以及本轮选中的通信类和实际算子。P/D 标签可以保留，但它不足以恢复这些字段。

| 要解释的问题 | 需要对齐的证据 | 能排除的误判 |
| --- | --- | --- |
| 为什么这轮走 AllGather / AllToAll / MC2 / 融合路线 | 选择函数输入、硬件策略、容量、量化和有效配置 | 仅凭请求处于 Decode，就认定一定用了 MegaMoE |
| 通信是否真的变少 | 源到目的行数、去重与本地聚合规则、传输格式和链路字节 | 把八十次专家任务直接换成八十份网络载荷 |
| 长通信区间在做什么 | 发送前输入就绪、对端到达、通信事件与消费者等待 | 把等对端生产数据的时间全记成链路慢 |
| 是否形成通算流水 | 相关 rank 的通信与计算区间、分块就绪、消费事件 | 把异步 API 返回或两条 stream 当成重叠收益 |
| 融合后服务是否更快 | 同条件的 MoE 层时间、整轮时间、吞吐和 TTFT / ITL / TPOT | 只凭 Kernel 数减少，就认定端到端时延下降 |

对比 AllGather 与定向交换时，固定逻辑路由、专家放置和有效工作量；对比普通 MC2 与全链融合时，还要固定量化、拓扑与可比较的输入形状。若拆批、padding 或专家副本同时改变，就把这些变化单独列出来，否则无法归因。

通信与计算重叠以后，不能再把 Trace 中各区间时长直接相加当作整轮时间。应追到下一消费者开始前，最后一个尚未满足的依赖：它可能是远端输入、专家结果、回传尾部或缓冲复用。正式采集也要避免随手打印 Device Tensor，防止加入新的 Host 回读。

下面把这份账本放到 A5 上运行。前面的字节数仍是给定路由下的教学计算，第 7 节仍解释自定义 W8A8 内核的依赖；实测单独记录对应的硬件、量化与算子入口。

## 11. A5 实操：MC2 拆分链与 CANN MegaMoE

### 11.1 先让同一份权重和路由算出相同结果

本次使用 Ascend950DT 的两张空闲卡，EP=2、TP=1，八个专家，每 rank 四个，top-k=2，隐藏维度与 FFN 中间维度均为 1024。输入为 BF16，权重与激活采用 W4A8MXFP 路径，激活函数为 SiLU，执行模式为 eager。每 rank 分别输入 8、32 行；不加载模型 checkpoint，也不包含 Attention、Router 网络或在线调度。[固定源码、环境与结果](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/results/a5-moe.json)

脚本使用目标实现的权重处理函数，从同一份量化权重分别生成普通 GMM 与 MegaMoE 所需布局。两条链共享输入、top-k 专家 ID 和概率；融合侧直接调用生产 `_apply_cann_mega_moe()` 入口，普通侧按当前测试拆开 Dispatch、GMM1、SwiGLU/量化、GMM2、Combine。权重转换、通信组和对称缓冲初始化放在测量外。本次验证了这些算子的执行，没有覆盖整个配置选择函数。[可复现脚本与来源](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/a5-moe.py)

四组 rank／形状结果逐元素一致，最大绝对误差和相对 RMSE 均为 0。独立计数 oracle 先统计各源 rank 的专家 ID，再跨 rank 求和、取本 rank 的专家区间：8 行时两 rank 分别承担 12、20 次任务，总计 `2×8×2=32`；32 行时为 71、57 次，总计 `2×32×2=128`。这也说明同一次对照可以固定不均衡路由，先把算子差异与布局变化分开。

### 11.2 真实 trace 里减少了哪些外部任务？

两种形状、两个 rank 的 active iteration 都观察到下列设备任务。这里的五段来自本次 W4A8MXFP 拆分链；第 6.1 节的 W8A8 默认局部融合仍是另一组条件。

目标源码的 W4A8MXFP+SiLU 生产 MLP 同样先执行 GMM1，再执行 SwiGLU/量化和 GMM2；其中 GMM1 与激活、量化的单算子分支要求 SiTU 等额外条件，本次 SiLU 不触发。这里仍使用直接拆分参考链，没有覆盖生产 prepare／finalize、完整 wrapper 或其他分支。[固定量化实现与来源](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/README.md#源码归属)

| 本次 MC2 拆分参考链 | 本次 CANN MegaMoE |
| --- | --- |
| `MoeDistributeDispatchV3` | `MegaMoe` |
| `aclnnGroupedMatmulWeightNz_GroupedMatmul_GroupedMatmul` | 同一个融合任务内，外部 trace 不展开内部阶段 |
| `SwigluGroupQuant` | 同上 |
| 第二个 GroupedMatmul | 同上 |
| `MoeDistributeCombineV3` | 同上 |

名称还需按层对齐：Python 调用的是 `npu_moe_distribute_dispatch_v2/combine_v2`，Host trace 记录 `aclnnMoeDistributeDispatchV4/CombineV4`，设备任务名则带 V3。它们属于不同接口层，不能仅凭设备矩形的名字反推 Python 配置。

<!-- COLLAPSIBLE_FIGURE: 展开 rank 0 实测事件复绘 · SVG，非查看器截图 -->

![A5 实测事件复绘：同一 rank 的 MC2 五个设备任务与单个 CANN MegaMoE，各自首任务归零，不是跨 rank 同步时间线](../dist/assets/a5-mc2-megamoe-measured.svg)

图选取每 rank 32 行、rank 0 的第二个 active iteration，保留实际 stream 61 与设备时间，两个实现各自以首任务为零。普通链的首尾跨度为 449.64 µs，五个任务的时长和为 42.87 µs；融合任务为 32.52 µs。留白只表示这条流没有记录到所选专家链的设备任务，尚未逐段归因，不能全部标成网络等待。这是一对带 profiling 的执行快照，不是延迟中位数或跨 rank 同步时间线。[可复算的泳道数据](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/results/a5-moe-swimlane.json)

另在采集外预热，按 MC2→MegaMoE→MegaMoE→MC2 顺序取样，每种方式各 20 个 NPU event 区间，得到以下中位数：

| 每 rank 输入行数 | rank | MC2 区间 / µs | MegaMoE 区间 / µs |
| --- | --- | --- | --- |
| 8 | 0 | 702.99 | 169.75 |
| 8 | 1 | 708.32 | 178.91 |
| 32 | 0 | 770.40 | 160.30 |
| 32 | 1 | 776.58 | 170.31 |

event 包围的是 Python 函数提交的整条链，可能包含设备等待 Host 后续提交及通信伙伴的空隙。它反映这组固定小输入的 eager 执行区间，不是纯 kernel 时长；独占的是这两张设备，同机其他卡仍有服务。大矩阵、graph replay、其他量化或完整模型都需要重新对照。外部只有一个 `MegaMoe` 矩形，也不足以证明第 7 节自定义内核的每个 flag 在 A5 CANN 中怎样发生。

### 11.3 把同一轮的 rank 0、rank 1 放在一起看

单个 rank 的快照只能解释本地任务。判断两卡如何共同完成同一轮，先要补齐参与者，再检查窗口是否真的对应。下面复用同次双卡实验的每 rank 32 行输入，选择明确标记的 `wiki/MC2/tokens32/iteration2` 与 `wiki/MegaMoe/tokens32/iteration2`；权重、输入和路由与第 11.1 节相同，没有另外挑选一组更均衡的数据。

本组 Level0 导出没有独立的 HCCL 收发轨道、`communication.json` 或可用的集体通信序号。配对依据是同一双进程 job、实现、形状、显式迭代标签及一致的调用顺序；随后在各自文件内沿精确 flow ID 找到设备任务。MC2 与 MegaMoE 各自创建通信组，不能把两条实现的相似编号当成同一个 communicator，也不能把某个 flow ID 当作跨 rank 的消息 ID。[双 rank 原始任务、窗口与正确性记录](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/results/a5-moe-rank-comparison.json)

先看 MC2。两卡都是 Dispatch、GMM1、SwiGLU/量化、GMM2、Combine 五个外部设备任务，实际记录在 `Stream 61` 上。任务序列相同，不表示每段耗时、提交间隙或有效专家任务数相同。

![MC2 · rank 0：tokens32/iteration2 的 Host 提交与五个关联设备任务](../dist/assets/profile-a5-mc2-rank0.png)

![MC2 · rank 1：按 flow 定位同一 iteration2 的设备窗口，保留五个任务](../dist/assets/profile-a5-mc2-rank1.png)

再看 CANN MegaMoE。两卡各观测到一个外部 `MegaMoe` 任务；这确认了本次调用的融合边界，内部 Dispatch、矩阵与向量计算、回传怎样交错，仍没有从这一级采样展开。

![CANN MegaMoE · rank 0：tokens32/iteration2 的单个融合设备任务](../dist/assets/profile-a5-megamoe-rank0.png)

![CANN MegaMoE · rank 1：通过 flow 关联同一 iteration2 的融合设备任务](../dist/assets/profile-a5-megamoe-rank1.png)

| 同一 iteration2 的记录 | rank | 本 rank 专家任务数 | 外部设备任务数 | 关联设备首尾跨度 |
| --- | --- | --- | --- | --- |
| MC2 | 0 | 71 | 5 | 449.64 µs |
| MC2 | 1 | 57 | 5 | 870.36 µs |
| CANN MegaMoE | 0 | 71 | 1 | 32.52 µs |
| CANN MegaMoE | 1 | 57 | 1 | 299.76 µs |

71 与 57 是 token-expert assignment 数。rank 0 的四位本地专家分别收到 `[15,18,24,14]`，rank 1 为 `[14,15,13,15]`，满足 `71+57=2×32×2=128`。拆分链把这些计数作为 `group_list` 传给两次 grouped GEMM。以 rank 0 第一个专家为例，GMM1 对 15 个有效行做 `[15,1024] × [1024,2048] → [15,2048]`，2048 列包含 gate 与 up 两部分。SwiGLU/量化生成 1024 列供 GMM2 使用，最终仍产生 15 行 1024 维的专家贡献。这里写的是有效计算形状，量化权重和尺度采用脚本中各算子需要的存储布局。[拆分参考链的权重与调用](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/a5-moe.py#L86-L225)

本次脚本为 MegaMoE 配置每 rank 32 个原始输入行和 128 个接收行的上限。后者按两个源各 32 行、每行至多两份送到同一目标 rank 预留，实际收到的仍是 71、57。两条实现最终按各自的原始输入行归还 `[32,1024]`，脚本在同一源 rank 上比较输出。[输入和接收容量的配置](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/a5-moe.py#L106-L128)

表中的单次设备跨度与第 11.2 节的 NPU event 中位数是两种口径。rank 0 的 32.52 µs 不能代表整个 EP 组；rank 1 的 Dispatch、Combine 区间较长，也不能仅凭较长矩形断言对端迟到、链路慢或本卡专家计算更慢。通信和内部同步包含在这些设备任务里，本采集没有恢复每条 peer 消息、producer-ready 或消费者等待的完整因果链。上述数值不用于计算稳态或服务级加速比。

这里还有一个定位问题：rank 1 的 Host scope 与其关联设备任务在原始导出轴上相距较远，不能只放大 Host 范围、再把落在范围外的设备任务删掉。图中 rank 1 展示的是精确 flow 找到的设备窗口；rank 0 则保留能同时看到提交与设备链的窗口。每张图独立缩放，画面左边界不是统一起跑线；这些 Host→Device 时间差也不作为已校准的派发延迟。[真实截图来源与选中设备窗口](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/results/a5-moe-profiling-screenshots.json)

完整采集还含未归到 active scope 的设备任务，不能凭“第几个矩形”直接选择迭代。图中只解释经 flow 归属到 iteration2 的任务；其他可见区间不混入本轮计数，也不把留白一概标成网络等待。本次 A5 W4A8MXFP CANN 图与第 7 节的自定义 W8A8 源码模拟继续分开，单个 `MegaMoe` 记录不证明后者的每个 AIV/AIC flag 已在本次测得。

### 11.4 在 Linux 解析，再用浏览器看泳道

本次已在服务器的 Linux aarch64 上运行 Perfetto Trace Processor，读取真实 `trace_view.json` 并执行 SQL；随后通过 SSH 转发，把服务器本地解析器接到浏览器 Perfetto UI，实际展开了 Host 线程和 Ascend Hardware 的 stream 泳道。Linux 无桌面也能完成解析，不要求先安装 GUI。官方工具提供 Linux arm64 预编译版本；本次服务器直连二进制下载失败，采用本地下载、核对 SHA256 后传入的方式。[官方下载脚本](https://raw.githubusercontent.com/google/perfetto/main/tools/trace_processor) · [命令行接口](https://perfetto.dev/docs/reference/trace-processor-cli)

把 `TRACE_FILE` 换成自己的文件，已验证的解析与服务命令为：

```sh
TRACE_FILE=/path/to/ASCEND_PROFILER_OUTPUT/trace_view.json
./trace_processor_shell_linux_arm64 query "$TRACE_FILE" \
  'SELECT count(*) AS slice_count FROM slice'
./trace_processor_shell_linux_arm64 server http \
  --ip-address 127.0.0.1 --port 9001 "$TRACE_FILE"
```

本地另开终端运行 `ssh -N -L 9001:127.0.0.1:9001 <SSH别名>`，打开 Perfetto UI，选择匹配版本及“YES, use loaded trace”，即可使用远端解析结果。服务只监听 loopback。小 trace 也可以直接在浏览器打开本地文件；Perfetto 本地查看默认不上传 trace。[远端解析与大 trace](https://perfetto.dev/docs/visualization/large-traces) · [本地数据处理说明](https://perfetto.dev/docs/getting-started/system-tracing)

查看时先定位实验的 `record_function` 范围，再沿原始 flow ID 关联外部调用与设备任务，检查 active iteration 与预热是否混入。本次八份 MoE trace 都能导入 Perfetto，但均出现 flow 的 enclosing-slice 告警；原始导出中也存在 Host→Device 终点早于起点数微秒的情况。因此保留原始 JSON 关联，细粒度派发延迟暂不从这组时间差计算。设备内部阶段或跨 rank 等待，还需要更细的采样及独立时钟校验。

需要 Ascend 专项分析时，MindStudio Insight 提供 Linux x86_64、aarch64 GUI 包；无桌面服务器需配合桌面转发或 VNC。保留完整 `*_ascend_pt` 目录，既可看 Timeline，也保留算子和通信分析所需的 CSV／JSON；只拿走一张截图会丢掉这些证据。[Linux 安装条件](https://msinsight.readthedocs.io/zh-cn/26.1.0/zh/install_guide/mindstudio_insight_install_guide.html) · [采集目录导入](https://www.hiascend.com/doc_center/source/en/mindstudio/830/GUI_baseddevelopmenttool/MindStudioInsight/Insight_userguide_0032.html)
