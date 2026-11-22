# KV 池化：单机混部与跨机 PD

你在编辑器里写 SQL，光标前的最后一行是 `WHERE latency_ms > 100`。编辑器发起一次代码补全。你把阈值改成 `500`，又触发一次补全。两次请求都发给同一个模型，第一次由机器 A 处理，第二次分到了机器 B。B 的显存里没有这段 SQL 的 KV，还得把整段输入再算一遍吗？

假设分词后，两次输入的前 128 个 token 完全相同，末尾的 `100` 和 `500` 各占一个 token。因果 Attention 在计算前 128 个位置时看不到后面的阈值。末尾的阈值改了，这段前缀的 KV 仍能复用。设每页正好放 128 个位置。如果 A 已经把这一页的 KV 存进共享池，B 就可以读回整页，只计算末尾的 `500`。

跟着第二次请求走一遍时，留意两个读 KV 的人。先是读入新请求的 P1 从池里找以前算过的前缀，再是负责生成的 D1 向 P1 取同一次请求的完整输入 KV。两次读取需要的数据范围不同。

## 两个 Prefill 实例共用一个 KV 池

机器 A 运行 Prefill 实例 P0，机器 B 运行 Prefill 实例 P1。它们各用一张 NPU，计算输入 token 在各层 Attention 的 Key 和 Value，也就是 KV。计算使用卡上的显存（HBM）。A 和 B 还各自拿出一段主机内存（DRAM），注册到同一个 Mooncake Store，供两个实例保存和查找前缀 KV。

![P0 和 P1 各用一张卡计算，两台 P 机器各贡献一段主机 DRAM 给同一个 Mooncake Store。另有一台 Decode 机器运行 D0 和 D1。](../dist/assets/kv-pool-machines.svg)

假设 P0 算出的前缀 KV 保存在 A 贡献的内存段。P1 可以用相同前缀的对象键查到这个副本，再从 A 的主机内存复制到自己的显存。本例在池中只保存一份副本。 [Mooncake 内嵌客户端与共享内存段](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/docs/source/design/store/mooncake-store.md#L25-L43)

D0 和 D1 是另一台机器上的两个 Decode 实例，各用一张卡。它们从处理本次请求的 P 实例接收 KV，再生成补全内容。本例中，Decode 机器给 Store 贡献的容量为 0。

<!-- DETAILS: 两台机器怎样向同一个 Store 注册内存？ -->

P0 和 P1 的 Worker 各自通过 `AscendStoreConnector` 接入 Mooncake。两边的 `master_server_address` 指向同一个 Master，`global_segment_size` 都大于 0，各自向 Store 注册一段本机 DRAM。

假设 `mooncake_master` 运行在 A。Master 记录对象键和副本的位置。写入时，Store 可以选择 A 或 B 提供的内存段，因此 Master 的地址和 KV 的存放地址要分别查。 [Ascend 配置说明](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L145-L155) · [Worker 注册容量](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L550-L573) · [Master 的职责](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/docs/source/deployment/mooncake-store-deployment-guide.md#L5-L14)

<!-- /DETAILS -->

## 从第一次写入，到第二次命中

第一次请求记为 R1，输入以 `100` 结尾，由 P0 和 D0 处理。第二次请求 R2 以 `500` 结尾，由 P1 和 D1 处理。先看池中副本保存在 A 段的情况。

<!-- POOL_WALK -->

**R1 没有命中缓存。** P0 计算全部 129 个输入 token，把公共前缀的 KV 写进显存第 3 页，把 `100` 的 KV 写进第 4 页。接下来，有两项操作可以分别进行。

- 为了让后来的请求复用前缀，P0 把第 3 页复制到 A 的 DRAM。写入完成并成功提交后，Store 中的前缀块就可读了。这次复制从卡到本机内存，方向是 **D2H**（Device → Host）。第 4 页只有一个有效位置，本例按完整块保存，因此只向池中写入第 3 页。
- 为了生成当前补全，D0 需要这一条请求的全部输入 KV。假设 D0 本地也没有命中，它先分配自己的第 12、13 页，再从 P0 的第 3、4 页分别读入前缀和 `100` 的 KV。交接就绪后，D0 开始 Decode。这次是卡到卡的 **D2D**（Device → Device）。 [双连接器分工](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L245-L251) · [D 端分配与拉取](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1947-L2009)

**R2 在 P1 的显存中没有命中，却在共享池里找到了前 128 个 token 的 KV。** 此时数据仍在 A 的 DRAM。P1 在自己的卡上分配第 9 页，从 A 把这份 KV 读进来。这次是远端主机内存到设备的 **rH2D**（remote Host → Device）。P1 随后用 `500` 作为输入做前向，Attention 读取已载入的前缀，`500` 自己的 KV 写入第 10 页。 [查询与可复用长度](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L325) · [Store 按地址读入](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L748-L767)

P1 读完后，A 段里的副本仍可供其他请求使用。P1 把数据从 A 的 DRAM 直接读入 B 的显存，无需再向 B 的 DRAM 写一份。D1 此时还没有 R2 的 KV。它从 P1 的第 9、10 页读到自己的第 20、21 页，然后接着生成。D0 不参与 R2 的交接。即使 P0 原来的显存页已被其他请求复用，P1 仍可从共享池取得这段前缀。 [完整块保存条件](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L1153-L1204) · [P 与 D 的页映射](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1013-L1064)

![池中副本在 A 段。P0 将公共前缀写入本机 DRAM，P1 从 A 读回前缀，D1 再从 P1 拉取 R2 的两页 KV。](../dist/assets/kv-pool-pd.svg)

上例中，P0 写本机内存，P1 跨机读取。把副本最初的存放位置改成 B，两次复制的方向就会对调。在播放器第 2 步和第 4 步切换存放位置，可以分别看到写入和读回的差别。

| 池中副本的位置 | R1：P0 写入公共前缀 | R2：P1 读回公共前缀 |
| --- | --- | --- |
| 机器 A 的 DRAM 段 | A 的 P0 显存 → A 的 DRAM，**D2H** | A 的 DRAM → B 的 P1 显存，**rH2D** |
| 机器 B 的 DRAM 段 | A 的 P0 显存 → B 的 DRAM，**D2rH** | B 的 DRAM → B 的 P1 显存，**H2D** |

两种配置都只保存一个池内副本。实际写入时，即使优先选本机，A 段空间不足也可能让 Store 选中 B 段。副本的位置直接影响 P0 的写入和 P1 的读回。R2 接下来仍由 D1 从 P1 的显存拉取 KV。 [Ascend 设置放置偏好](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L626-L631) · [Master 判断是否优先本机](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/mooncake-store/src/master_service.cpp#L5190-L5197) · [首选段分配失败后的尝试](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/mooncake-store/include/allocation_strategy.h#L377-L435)

一条较长的 SQL 可能跨好几页。假设两次输入开头相同的部分有 384 个 token，占三整页，后面仍跟一个不同的 token。P1 从输入开头匹配。如果它的显存中没有前缀，但池中三页都可用，就读回三页，只计算最后一个 token。如果只有前两页可用，就复用 256 个 token，剩下的 129 个需要计算。如果 P1 本地已有可复用的前缀页，对应部分还可以省去从 Store 读回的传输。 [连续前缀与多分片查询](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1132-L1218)

## 复用上一次请求的 KV，传递本次请求的 KV

R2 一共读了两次 KV，读取的范围不同。P1 从共享池取回 P0 为 R1 算过的公共前缀，省去重复的 Prefill。接下来 D1 要处理的是 R2，除了公共前缀，还需要 P1 刚为 `500` 算出的 KV。即使池中没有命中任何前缀，P 算完后仍要和 D 交接。本例按默认配置关闭了 D0、D1 的 Store 读写，两个 D 各自从对应的 P 拉取数据。 [连接器组合](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L245-L251) · [D 端独立开关](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L1603-L1616)

| 部署 | 后来的请求怎样复用前缀 | 当前请求怎样从 Prefill 转入 Decode |
| --- | --- | --- |
| 单机混部，开共享池 | 本卡没命中，可以从 Store 读回 | 同一实例在自己的显存页上接着算 |
| 跨机 PD，只开直接交接 | 各实例可用自己的本地前缀缓存 | D 从选中的 P 接收自己缺的 KV |
| 跨机 PD，再开共享池 | P1 可读回 P0 存入共享池的前缀块 | D1 从 P1 接收本次请求缺少的 KV |

**试一试。** 关掉共享池，P1 和 D1 的本地缓存也都未命中。第二次请求的 P1 要计算多少个输入 Token？D1 还要接收几页？

<!-- DETAILS: 核对答案 -->

P1 要处理全部 129 个输入 Token。D1 仍需要从 P1 接收两页，第一页存公共前缀，第二页保存 `500` 的 KV。关掉池改变了 P1 能省掉多少输入计算，这个例子里 P 和 D 仍然分开，交接照常进行。

<!-- /DETAILS -->

在单机混部中，同一个实例可以用同一张卡完成一次请求的 Prefill 和 Decode。如果原先的显存页已被分配给其他数据，而前缀副本还在本机 DRAM，下一条请求依然能复用它。第一条请求保存前缀时执行 **D2H**，第二条请求复用时执行 **H2D**。 [混部部署示例](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L416-L477)

<!-- COLLAPSIBLE_FIGURE: 看单机混部时的写入和读回 -->
![原来的显存页已被复用，混部实例从本机 DRAM 读出前缀 KV，写入新分配的显存页。](../dist/assets/kv-pool-colocated.svg)

## P1 查到前缀之后，还要读回数据

查到 128 个 token 可以复用时，P1 得到的是命中长度。要让 Attention 能读第 9 页，还得分配显存、找到副本并完成复制。各实例有自己的调度器和 Worker，P0 和 P1 通过各自的客户端访问同一个 Store。

| 谁 | 在第二条请求里具体做什么 |
| --- | --- |
| P1 的 Scheduler（调度器） | 确认本地已命中的长度，向连接器询问还能复用多少，再为请求安排本地页。页号 9 属于 P1。 |
| P1 的 AscendStoreConnector、Worker 中的查询线程 | 在查询阶段组合前缀键，查 Store 中哪些块可用，向调度器返回命中长度。随后执行侧 Worker 按本地页表准备 K、V 的目标地址，向 Store 客户端发起 GET。 |
| Mooncake Store 的 Master 和提供内存的 Worker | Master 记录前缀块对应的键和副本所在的内存段。本例的副本在 A 段，KV 字节保存在 A 的 DRAM。 |
| Mooncake 的 Transfer Engine（传输引擎） | 客户端取得副本描述后提交 READ，传输引擎把 A 段的前缀 KV 复制到 B 上 P1 指定的显存地址。复制完成后，P1 才能使用这一页。 |

同一套部署里还有一条独立的调用路径。vLLM 的 **MultiConnector** 把两个连接器接到实例上。**AscendStoreConnector** 按对象键访问共享池，**MooncakeConnectorV1** 负责本次 PD 拉取，由 D 根据请求指定的 P 端点和源页号，将 KV 读入自己的目标页。第一条路径查找可供其他请求复用的前缀，第二条路径传输当前请求继续生成所需的 KV。两者都可以使用 Mooncake 传输数据。[调度器和 Worker 入口](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py#L190-L201) · [查询消息与键](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/lookup_key.py#L59-L88) · [Store 的存在性查询](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L600-L631) · [客户端获取副本描述](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/mooncake-store/src/real_client.cpp#L8764-L8778) · [传输引擎的地址与方向](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/mooncake-store/src/transfer_task.cpp#L1219-L1242)

## 同一页 KV 在各台机器上的地址怎样算？

P1 用**对象键**查找副本，无需知道这页 KV 在 P0 的显存里曾经是哪一页。实际复制时，传输引擎才需要源地址和目标地址。取单层普通 Attention，每个实例用一张卡，关闭投机解码。显存页和共享缓存块都容纳 128 个 token，每个位置的 K 和 V 各有一个 64 维的 BF16 头。一个 K 页占 128 × 64 × 2 B = **16 KiB**，加上同样大小的 V 页共 **32 KiB**。各卡的 K、V 分开存放，各自的物理页连续排列。下表的页号取自前面的例子。

| 存放位置 | 这一页 K 的起始地址 | 这一页 V 的起始地址 |
| --- | --- | --- |
| A，P0，显存页 3 | P0 的 K 基址 + 3 × 16 KiB | P0 的 V 基址 + 3 × 16 KiB |
| 机器 A 贡献给 Store 的 DRAM 段 | 前缀副本在 A 段中的起始地址 s | s + 16 KiB |
| B，P1，显存页 9 | P1 的 K 基址 + 9 × 16 KiB | P1 的 V 基址 + 9 × 16 KiB |
| Decode 机，D1，显存页 20 | D1 的 K 基址 + 20 × 16 KiB | D1 的 V 基址 + 20 × 16 KiB |

如果 Store 最初选中 B 段，表中第二行就换成副本在 B 段的地址，P1 的目标仍是显存第 9 页。**P1 从池中读 32 KiB，D1 从 P1 读 64 KiB。** P1 只取一页公共前缀，D1 却要接收第 9、10 两页。本例的普通 Attention 路径按整页传输，即使第 10 页只有 `500` 这一个位置有效，也会复制整页。两页中的 129 个有效位置实际占 129 × 64 × 2 B × 2 = **33,024 B**。D1 按有效长度读取这些位置。 [Worker 按块表计算分量地址](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L477-L517) · [D 端按步幅组织整页 READ](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1013-L1064)

<!-- DETAILS: 对象键、PUT 和 GET 在源码中怎样对应 -->

两个 P 访问同一 Store，并使用相同的租户命名空间（本例沿用默认的 `tenant_id`）。普通完整块的哈希包含前一块的哈希、本块的 token IDs 和适用的额外信息，例如 cache salt。Ascend 还会在对象键中加入模型名、DCP、head/TP、PP、缓存组、缓存角色和缓存类型。两个请求能否匹配同一个键，取决于前缀 token 是否相同，以及模型、分片和缓存配置是否兼容。[Mooncake 租户配置](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L158-L160) · [vLLM 块哈希](https://github.com/vllm-project/vllm/blob/ba15685a/vllm/v1/core/kv_cache_utils.py#L703-L732) · [Ascend 对象键](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L167)

本例的非逐层路径中，查询调用 `batch_is_exist`。存入时，客户端取得放置结果后用 WRITE 把 K、V 片段放到副本相应偏移，成功提交对象。读回时，Worker 将键、本卡地址和长度交给 `batch_get_into_multi_buffers`，客户端取得副本描述并发起 READ。不同的硬件和传输配置可能经中间缓冲区。D2rH、rH2D 描述的是起点和终点。[Store PUT / GET](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L715-L767) · [提交可读对象](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/mooncake-store/src/client_service.cpp#L3418-L3426) · [Ascend transport 的传输模式](https://github.com/kvcache-ai/Mooncake/blob/35f41d98/docs/source/design/transfer-engine/transport/ascend_direct_transport.md#L99-L109)

<!-- /DETAILS -->

## 逐层传输与 Decode 端暂存

前面的 64 KiB 按整页统计。模型有多层时，P1 还可以按层把 KV 交给 D1。假设 D1 已经准备好目标页。P1 写好第 1 层可发送的 KV 后就可以提交传输，继续算后面的层。这样有机会缩短最后一层算完后的交接等待。Ascend 主干已有实验性的 Mooncake V2 逐层推送，D 等本次交接所需的发送端完成后才报告就绪。若前缀副本在 A 段，P1 从 A 读前缀仍走 rH2D。[Ascend 已合入的逐层推送 PR](https://github.com/vllm-project/vllm-ascend/pull/20940) · [两端的完成条件](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md#L65-L126)

**池化本身也可以逐层。** 此时分层的是共享 DRAM 段与 P 的显存之间的存取。例如 P1 算当前层时预取下一层的 KV，当前层开算前仍要等自己的数据就绪。社区 RFC 进一步提出让加载、PD 交接、消费确认和块回收按相同粒度配合，以便更早腾出工作页。具体回收哪一侧的资源仍是 RFC 的开放问题。截至 2026 年 11 月 18 日，上游逐层直写 PR 仍在评审。另有一份 Ascend 草案让 P 发布到 DRAM 池、D 从池取回，复用同一批对象来做 PD 交接。[池化逐层读写](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/mooncake_layerwise_adaptation_and_optimization.md#L43-L82) · [含现状图与规划图的 RFC](https://github.com/vllm-project/vllm/issues/61787) · [上游逐层直写 PR](https://github.com/vllm-project/vllm/pull/62504) · [经 DRAM 池交接的草案](https://github.com/vllm-project/vllm-ascend/pull/21000)

<!-- DETAILS: 经过 Decode 机主机内存的两跳交接 -->

选用 `MooncakeD2RHConnectorV1` 时，D 的 Worker 可以先把 P 的 KV 拉入 D 机器注册的 Host 暂存区，再从这台机器的 DRAM 载入本卡 HBM。对应到前面的部署，数据依次经过 **P 机 B 上的 P1 显存 → Decode 机的主机暂存区 → 同一台 Decode 机上的 D1 显存**。第一跳从 P1 看是 D2rH，第二跳从 D1 看是同机 H2D。暂存区由 D 的 Worker 注册，服务于本次交接。若 D 的主机内存中已有有效副本，对应块就可省去第一跳，已有本地 HBM 前缀也可减少第二跳。首跳完成后可通知 P 释放源页。所有依赖的 H2D 结束后，D 才解除 Host 页引用。[两跳与两级命中](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/developer_guide/Design_Documents/disaggregated_prefill.md#L161-L189) · [官方验证的硬件与模型](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md#L969-L983)

![独立的 PD 两跳配置。图中的 P 机对应运行 P1 的机器 B。第一跳写入 Decode 机的主机暂存区，第二跳从暂存区读入 D1 的显存。](../dist/assets/kv-pool-d2rh-staging.svg)

<!-- /DETAILS -->

相邻章节记录了多卡服务中实际观察到的传输与完成时序。本章两台 Prefill 机器共享 DRAM 的配置还没有做服务性能对照。[PD 分离系统与 KV 生命周期](PD分离系统与KV生命周期.md)

## 池化对首个输出、吞吐和内存占用的影响

回到两次 SQL 补全。先只关闭共享池，保留原来的 P→D 交接方式和请求路由，P0 和 P1 都没有命中本地缓存。两个 P 各为 129 个输入位置计算 KV，共 258 个。再打开共享池，让 R2 在 R1 的前缀块已写好、可读时到达。P0 仍计算 129 个位置，P1 只计算 1 个，合计 130 个。P1 处理最后一个位置时，Attention 仍要读取前 128 个位置的 KV。

两条请求中有一条从池里读到了前缀，按请求算是 1/2。若按两条请求提交的全部输入计算，复用的是 128/258，约 49.6%。这里数的是输入位置，没给每个位置假定相同的计算时间。vLLM 还可以按引擎记录 Prefill 实际新计算 KV 的 token 数，以及连接器查询和命中的 token 数。核对 258 和 130 时，只统计 P0、P1。D 端交接后的计算要单独看。[vLLM 服务指标与计数单位](https://docs.vllm.ai/en/latest/usage/metrics/)

**用户等第一个输出的时间（TTFT）。** P1 少算了 128 个位置，代价是先查共享池，再把 32 KiB 读进显存。只有省下的计算和排队时间超过首个输出之前新增的等待，用户才会更早看到补全。本例由 D1 返回输出，它仍要从 P1 接收两页，共 64 KiB。比较开关前后的 TTFT，要从客户端发出请求计到收到第一个非空输出，再对照 P 的排队、前缀读回、Prefill 和 P→D 交接的时间。可能并行的步骤应按实际先后分析。[前缀缓存的作用范围](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) · [客户端时延指标](https://docs.vllm.ai/en/latest/benchmarking/cli/#understanding-the-latency-metrics)

**首个输出之后的间隔。** 在这个 PD 配置下，D1 后续仍要逐步生成同样数量的输出 token，P 上省下的是输入计算。在单机混部中，如果先前的停顿来自较长的 Prefill 占用了同一张卡，减少重复 Prefill 有机会缩短同批其他请求的输出间隔。反过来，读回 KV 的拷贝如果需要与计算争用带宽，也可能拉长间隔。可以分别看客户端相邻输出事件的间隔（ITL），以及每条请求从首个到最后一个输出的时间除以后续输出 token 数（TPOT）。输出只有一个 token 时，后者没有定义。[混部时的输出间隔](LLM推理调度与执行.md) · [流式返回的计时口径](投机解码与流式指标.md)

**系统每秒能完成多少请求。** 如果请求原先主要排在 P 端，共享前缀又经常命中，P 少做重复计算，就可能更早处理下一条请求。如果瓶颈已经在 D 端或传输链路，P 更快完成也可能只是让请求更早进入下一处队列。固定到达率时，如果两组都能及时处理全部请求，每秒完成的请求数可能一样，差别体现在 TTFT。要比较服务容量，可以逐档提高到达率，观察在相同的成功率、TTFT 和 TPOT 要求下，各自能持续完成多少请求，同时记录每秒实际生成的输出 token 数。[服务压测的负载与指标](https://docs.vllm.ai/en/latest/benchmarking/cli/)

**主机内存和传输带宽。** 本例的单层前缀在池中占 32 KiB 的 KV 数据，A、B 为 Store 注册的 DRAM 容量还要单独计算。P1 仍需分配显存页供 Attention 读取。播放器里的一次 PUT 和一次 GET 各传 32 KiB，把副本放在 A 段时只有 GET 跨机，放在 B 段时只有 PUT 跨机。两种放置下，两个 D 各自的 64 KiB 交接都还在。这些是按 KV 内容计算的字节数，链路上的实际流量还包含传输协议开销。当前缀很少复用或副本在下次命中前已被淘汰，保存这些块仍会占用主机内存和写入带宽。[Store 的内存配置](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L145-L155) · [Store 的 PUT 和 GET](https://github.com/vllm-project/vllm-ascend/blob/0c99d3fc/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L715-L767)

做服务对照时，两组使用同一模型、硬件、P/D 配比和路由，重放相同的输入、到达时刻和输出长度。模型预热后，让两组从相同的本地缓存状态开始。开启共享池的一组分别统计首次写入和随后命中的请求，以免一批很快的命中掩盖了未命中请求的等待。先关闭 profiler，重复测量客户端 TTFT 和从发出请求到收到最后一个输出的端到端时延，报告中位数和 P95、P99，同时统计吞吐。再按相同负载采集 Trace，核对查找、读回或交接是否让后续计算等待。这两条 SQL 请求能算出少算的位置和传输的字节数，实际时延和吞吐变化仍需上述对照测量。[实验对照流程](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/README.md#特性前后对照的固定流程)
