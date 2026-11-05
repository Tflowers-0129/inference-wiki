# PD 分离系统：请求交接与 KV 生命周期

第六章把一条请求的 Attention 分到多个 rank，最后归并结果。现在把边界再向外移：Prefill 由一套引擎完成，Decode 交给另一套引擎。两边各有 Scheduler、Worker 和 KV 块池，一条请求怎样从 P 走到 D？

第一章已经介绍 PD 的资源组织。本章跟着同一条请求，把“P 算完了”拆成几个具体事件：源 KV 已写入、D 已分配目标块、传输完成、D 可以计算、P 可以归还源块。先看拉取，再看逐层推送，最后回到 A5 上逐 rank 核对这些事件。

> 源码主线是 vLLM-Ascend 的 `MooncakeConnectorV1` 与 `MooncakeLayerwiseConnector`，配对实际使用的 vLLM Scheduler 和 ModelRunnerV2。前者指向 `kv_p2p/mooncake_connector.py`；另一个拆分目录中的 `MooncakePullConnector` 不是本章实验路径。固定源码、Mooncake 二进制来源边界、脚本和结果见[实验材料](https://github.com/Tflowers-0129/inference-wiki/tree/main/experiments/pd-disaggregation)。机制图为教学图，真实采集在后面的实操部分。

## 1. 交出去的是什么？

取普通 GQA 请求：prompt 为 513 个 Token，P TP2、D TP2，PCP、DCP 均为 1。P 的 rank 0、1 与 D 的 rank 0、1 是四个不同 Worker；两边的 rank 0 不能当成同一个进程。P 的块号属于 P 的分配器，D 必须先从自己的块池分配空间。

假设同一个逻辑块在 P 的物理块号为 7，在 D 为 19。交接要把这一层、这一 KV head、这个逻辑范围的数据，从 P 的块 7 写到 D 的块 19。把 `7` 直接放进 D 的 Attention block table，会指向 D 自己的块 7，数据身份就错了。第六章的并行布局也继续生效：两端 head、位置、块大小与数据格式必须兼容，必要的重排由连接器负责。

请求还有模型输入、采样参数、请求身份与生成进度。HTTP 请求负责把这些信息送给另一套引擎；KV 数据另走 Worker 间的传输。不能只复制一份 KV 张量，就认为 D 已经获得可继续执行的完整请求状态。[拉取式连接器](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) · [逐层连接器](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

两套引擎也没有共用一份本轮 Token 预算。P 是否能开始 Prefill，由 P 自己的调度与 KV 容量决定；D 是否能接收、接收后何时执行，由 D 的分配和调度决定。PD 请求路由发生在实例之间，TP 通信仍发生在各实例内部。

## 2. Proxy、连接器与 Mooncake 各管哪一段？

沿源码往下，有三种不同的信息交换：

| 参与者 | 交换的信息 | 对这条请求的作用 |
| --- | --- | --- |
| Proxy 与 P/D HTTP 服务 | prompt、采样参数、请求身份、`kv_transfer_params` | 选择实例、发起两端请求、向客户端转发 D 输出 |
| 连接器 Scheduler 与 Worker，以及 Worker 侧信道 | 本地/远端块号、布局、内存描述、完成通知 | 分配、生成传输计划、把完成状态送回 Scheduler |
| Mooncake Transfer Engine | 已注册缓冲区之间的 READ / WRITE | 实际搬运 KV payload |

Proxy 返回了 HTTP 元数据，不表示 KV 已到 D；连接器返回了“外部可命中多少 Token”，也不表示这些 Token 已经可以被 Attention 读取。后面要分别找出对应的完成条件。

同名字段还要看它由谁构造、交给谁消费：

| 交接信息 | 拉取式：P 返回给 D | 逐层推送式：D 返回给 P |
| --- | --- | --- |
| `remote_block_ids` | P 的源块号 | D 的目标块号 |
| `remote_engine_id`、请求身份 | 定位保留源 KV 的 P 请求 | 定位等待接收的 D 请求 |
| `remote_host`、`remote_port` | P 的连接器侧信道 | D 的连接器侧信道 |
| 交接后的工作 | D 发起 READ | P 按层发起 WRITE |

`remote` 是接收这份 metadata 的一方所看到的远端。不能把两种方式的同名字段解释成固定属于 P 的地址。[拉取字段构造](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) · [推送字段构造](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

本章这条 P2P 路线由 `GlobalTE` 在 Worker 内初始化 Transfer Engine，选择 `P2PHANDSHAKE` 与 `ascend` transport。它直接用于 P/D 交接，没有启动 Mooncake Store。连接器的 `kv_port` 侧信道与 TE 发布的 RPC 端口也不同，不能把它们画成同一个数据入口。[TE 初始化与注册](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py)

Mooncake 的 Python `batch_transfer_sync_read/write` 把本地缓冲区、远端地址、长度和 READ/WRITE 操作交给 C++；参照源码在提交后等待 batch 完成，成功才返回 0。连接器可以把这个同步调用放到后台线程，使引擎继续推进其他工作。因此“相对 Scheduler 异步”“后台线程执行”和“调用 TE async API”要分开读。本章连接器使用的是同步批量读写接口。[Mooncake Python/C++ bridge](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-integration/transfer_engine/transfer_engine_py.cpp)

## 3. 拉取式：P 先算好，D 再来取

官方拉取式 proxy 先选 P。`build_prefill_request` 给 P 填入 `do_remote_decode=True`，把 P 请求的 `max_tokens`、`min_tokens` 都设为 1，并关闭这一段 HTTP 的流式返回。P 完成这次本地请求后，proxy 才拿着返回的交接信息选择 D。[Proxy 请求构造与实例选择](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

![拉取式先选 P，P 保留源块，D 分配后读取；就绪和源释放分别推进](../dist/assets/pd-pull-handoff.svg)

### P 返回元数据，源块仍被请求持有

普通 FullAttention 路径中，P 处理完整 prompt。请求因输出长度限制结束后，连接器 Scheduler 的 `request_finished` 检查 `do_remote_decode` 与结束原因，取出需要交接的 prompt 块。只要有源块，就返回 `delay_free_blocks=True`，并把请求放入待发送完成的跟踪集合。

这时 HTTP 可以返回，但上游 Scheduler 会保留请求的块引用。返回的 `kv_transfer_params` 包含远端请求/引擎身份、P 块号、侧信道入口和并行布局。proxy 把它放进 D 请求；KV payload 留在 P 的已注册缓存里，等待 D 读取。[`request_finished` 与交接字段](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) · [Scheduler 的 `_free_request`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

若 D 尚未选出、没有目标块或迟迟不能准入，P 已经算好的 KV 仍占着引用。此时 P 的“计算结束”与“缓存可供下一请求使用”之间出现了一个驻留窗口。这个窗口不能从 P 的 Prefill kernel 耗时里读出来。

### D 先占空间，再进入接收等待

D 的连接器通过 `get_num_new_matched_tokens` 告诉 Scheduler：哪些 prompt Token 预计可以从 P 获得，并返回异步加载标记。Scheduler 先分配目标 slots，再通过 `update_state_after_alloc` 记录本地块号与 P 的信息，生成发给 Worker 的传输计划。

请求随后进入 `WAITING_FOR_REMOTE_KVS`。这时账本里的 `num_computed_tokens` 可以已经计入预计加载的数量，但状态仍禁止它直接进入正常计算。这个字段描述调度进度；设备完成要等 Worker 回报，不能单看数值已经达到 prompt 长度。[D 的匹配/分配接口](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) · [外部 KV 分配与等待状态](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

Worker 的接收线程取得 P 的内存描述与 TE 入口，按两端块号构造传输范围，再调用 `batch_transfer_sync_read`。READ 的数据方向是 P→D；API 的 local buffer 参数属于发起读取的 D。源码中的 `src_list`、`dst_list` 名称不能替代 READ 的地址语义。布局需要转换时，接收完成还要经过相应重排。

### 读完以后，有两条独立的完成路径

D 的本地接收任务全部结束后，Worker 报告 `finished_recving`；执行器聚合必要 rank 的回报，再交给 D Scheduler。Scheduler 才解除远端 KV 等待，恢复这条请求参与后续调度的资格。

另一条路径发回 P：D 通知对应来源自己已不再读取这次 KV，P Worker 汇总完成状态，经自己的执行器与 Scheduler，最终调用 `_free_blocks` 归还源请求的引用。D 的本地就绪回报与 P 的释放通知由不同线程、不同调度轮次处理，不能画成同一个原子时刻。[接收任务、DONE 与完成跟踪](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py) · [多 rank 完成聚合](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/utils.py)

所以，rank 0 的一次 read 成功，只能证明该次本地读取完成。是否还有其他来源、其他本地任务或 rank 尚未完成，必须继续看任务计数与引擎聚合结果。

## 4. P 算出的首 Token，为什么 D 还会重算？

这里需要回到 proxy 的真实行为。P 虽被要求输出一个 Token，拉取式 proxy 只把返回的 `kv_transfer_params` 交给 D，没有把 P 采样的 Token 追加到 D 的 prompt，也没有先把 P 的文本交给客户端。元数据里的 `last_token_id` 出现了，不代表后续路径实际消费了它。

在本章的普通 FullAttention 路径中，D 收到完整 prompt KV 后，上游 `_update_waiting_for_remote_kv` 发现 `num_computed_tokens == num_tokens`，会退回一个位置。D 重算最后一个 prompt Token，得到 logits，再采样客户端的第一个输出。这条路径不能按“P 贡献第一个输出，D 继续生成剩下的输出”记账。[Proxy 的交接处理](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) · [末 Token 重算](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

513 个 prompt Token 的简化账本是：P 写好 prompt 的 KV 并触发交接；D 加载这份历史，恢复调度时将已算数退回 512；下一次计算包含位置 512，随后采样新 Token。临时退回调度进度，不意味着整份历史 KV 都被丢弃重算。

Mamba 或压缩状态还有 P 端截断 prompt 等额外处理，因为状态不一定能通过重写最后一个 KV slot 恢复。那是另一条状态契约，本章普通 GQA 的算例不把它拼进来。

## 5. 逐层推送式：D 先占位，再让 P 开始

拉取式要等 P 完成后，才知道它的源块。逐层推送则需要提前知道写到哪里，因此交接顺序反过来：proxy 先选 D，D 分配目标块，再把目标信息送回 proxy 的 MetaServer 接口；此时 proxy 才选 P 并派发 Prefill。[逐层 proxy](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)

![逐层推送先让 D 分配目标块，再派发 P，按层写入后收齐 DONE](../dist/assets/pd-layerwise-handoff.svg)

### MetaServer 交接的是目标位置

D 请求携带 `do_remote_prefill=True` 与 MetaServer 地址。`update_state_after_alloc` 在块分配之后，回报 D 的请求身份、目标块号、块大小、并行配置及本地已命中的前缀范围。proxy 把这些目标信息附到 P 请求里；若一个 API 请求展开为多个子请求，先收齐对应子请求的目标信息，再统一派发这次 Prefill。

这里的 MetaServer 是 proxy 暴露的请求交接接口，不是 Mooncake Store 的对象目录，也不搬运 KV。P 后续仍要通过 Worker 侧信道取得 D 的内存描述与 TE 入口。proxy 的 `release_prefiller_kv` 等函数调整入口评分计数，不能据名字认作 NPU 分配器已经释放块。[目标分配与 `_access_metaserver`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

### 一层何时允许发送？

P 的 Scheduler 根据本轮计划生成发送 metadata；ModelRunnerV2 在前向前绑定 metadata，并准备逐层状态。Attention 写入该层 KV 后，Ascend hook 在计算 stream 记录事件。`save_kv_layer` 把该事件、层身份和块映射放入后台发送队列。

发送线程先等待这个 KV 写入事件，再调用 `batch_transfer_sync_write`。主计算可以继续执行后续算子或后续层，后台线程处理已经准备好的层。若某条 Attention 路径没有提前发出写入 hook，连接器会在较晚的位置记录备用事件，允许重叠的窗口也相应改变。[KV 写入通知](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/attention/utils.py) · [逐层发送与写入事件](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

Chunked Prefill 又多一条边界：非最终 chunk 只发送已经覆盖的完整块，最后一个 chunk 才包含末尾未满块。Scheduler 根据计划更新的 `local_computed_tokens`、`local_transferred_tokens` 用于构造这些范围；它们不是 TE 已完成传输的回执。不能看字段叫 transferred，就直接把同一时刻画成远端已有效。

### 按层传，不等于 D 按层开始 Decode

P 的最终层、最终 chunk 传输完成后，向对应 D 发送 DONE。D 按请求记录唯一来源集合，达到映射要求的来源数后，才把请求加入完成集合；重复 DONE 不会重复增加来源数。Worker 再把外部请求身份映射回引擎内部身份，报告 `finished_recving`。

本章核查的 `wait_for_layer_load` 为空。D 仍由整条请求的远端 KV 完成状态解除等待，随后重算末 prompt Token 并继续生成。逐层优化首先改变 P 侧“哪一层什么时候能发”，不能据此推断 D 已在第 0 层刚到时启动同一请求的第 0 层 Decode。[按来源跟踪与 `get_finished`](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

## 6. 两种方式怎样管理源块与目标块？

先把“块归还”与“内存注销”分开。连接器在 KV 池创建后注册底层 storage，内存描述可以服务多条请求。请求完成时，分配器解除的是它对物理块的引用；池张量和 TE 注册仍可继续存在。只有独立的资源清理路径才会断开 peer 并注销注册区。[内存注册与注销](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py)

![已注册 KV 池跨请求存在，源请求引用与目标块有效性各有生命周期](../dist/assets/pd-kv-ownership.svg)

对 D 来说，目标块依次经历已分配、正在写入、加载完成、参与计算。取消请求会终止它的业务输出，但后台写入可能仍在进行；Scheduler 对尚在等待远端 KV 的已终止请求保留目标块，等完成回报后再释放，避免新请求复用仍被写入的地址。[取消与延迟释放](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

P 的两条实现则有直接可见的差异：

| 检查点 | 拉取式 | 本章核查的逐层推送式 |
| --- | --- | --- |
| P `request_finished` | 有待交接源块时返回延迟释放 | 返回不延迟释放 |
| P 发送完成回报 | D 通知后，经 Worker 聚合报告 `finished_sending` | `get_finished` 的发送完成集合为空 |
| `wait_for_save` | 不能用函数名代替实际后台完成条件 | 函数体为空，没有等待发送队列 |
| 源端后续处理 | 源端完成跟踪进入 Scheduler | 连接器不额外延迟，继续走 P 请求结束及分配器 fence 路径 |

因此不能把拉取式的 hold/ACK 释放协议直接画到逐层推送图上。在这一逐层快照中，连接器没有要求普通 P 请求的 `_free_blocks` 调用必定晚于全部后台写完成。分配器仍可能按在途模型 step 延后实际归还；已有的层 buffer reuse 事件也只保护相应复用路径。需要继续区分“释放调用返回”“实际解除块引用”和“下一请求覆写地址”。本次实操记录前一项与写返回的先后，没有覆盖全部实际复用条件。[连接器实现](https://github.com/vllm-project/vllm-ascend/tree/main/vllm_ascend/distributed/kv_transfer/kv_p2p) · [分配器延后释放路径](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)

## 7. 失败、超时和取消，分别结束了什么？

数据读取完成、读取成功、源端不再需要保留数据，是不同条件。拉取式接收线程即使读取失败，也会在终结该次读取后通知 P；D 的无效块或失败请求信息另外报告给 Scheduler。后续重算还是终止，要看失败类型与引擎配置，不能把 DONE 当成 KV 正确的证明。

拉取式还为被 D 拒绝、尚未正式接收的请求准备空接收通知，让 P 有机会解除源引用。P 的超时跟踪是故障保底，会在交接长期没有正常终结时结束持有；超时本身不证明 D 已拿到完整数据。[拉取失败处理与超时跟踪](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py)

逐层推送也有 FAILED 消息与目标块错误报告，但本章快照中的 Worker 没有把失败请求同时并入接收完成集合，也没有补齐请求级 `failed_recving` 回报；配对 Scheduler 的恢复又要等接收终结。不能仅凭代码里出现 FAILED，就认定错误一定会自动转为本地重算。MetaServer 请求的异常重试、DONE 控制消息的重试，也不能合起来称为整条交接已具备有界超时恢复。[逐层失败与 DONE 路径](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

客户端断开连接也不等于所有设备传输立即停止。proxy 取消 HTTP task、清理路由计数，与后台传输终结和块引用释放仍有各自的条件。[Proxy 取消处理](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py)

排查时至少保留请求身份、P/D 所在实例、目标是否分配、还有哪些来源未完成、源/目标请求引用是否已归还。Proxy 队列清了、HTTP task 取消了、块回到分配器、TE 注册区注销了，分别是不同的事实。

## 8. A5 上，实际跑通了哪两条路径？

这次在同一台 A5 主机上使用四张健康、空闲的 NPU，按排他租约串行运行两套 PD 系统。两边都是实际 HTTP 服务、官方 proxy、ModelRunnerV2 与 Mooncake 原生传输，没有用 D 本地重算代替加载成功。采集结束后已确认实验进程退出并释放设备。

模型是固定随机种子的两层 Qwen2：hidden=512、Q heads=8、head_dim=64、BF16。两组缓存与并行条件如下：

| 项目 | 拉取式 | 逐层推送式 |
| --- | --- | --- |
| 连接器 | `MooncakeConnectorV1` | `MooncakeLayerwiseConnector` |
| P / D 的 TP | 2 / 2 | 2 / 2 |
| P / D 的 DCP | 1 / 1 | 2 / 2 |
| KV heads | 2，按 TP 分开 | 1，按 DCP 分历史位置 |
| KV 布局 | 实际 GQA 布局 | 连续的独立 K/V 张量 |
| block size / CP interleave | 128 / 128 | 128 / 128 |

每 Worker 固定 32 MiB KV 池预算，模型长度上限 2048，本轮 Token 预算 1024，最大序列数 4，eager，关闭 prefix caching。prompt 取 129、513，各重复两轮，每次输出四个 Token。每组先在同配置 P 上运行普通本地请求作为参考，再预热并采集 PD 请求。[复现命令与结果](https://github.com/Tflowers-0129/inference-wiki/tree/main/experiments/pd-disaggregation)

两组四次测量的输出 Token IDs 均与各自参考一致；所选 Token 的 logprob 最大绝对差，拉取式约为 0.00766，逐层推送式约为 0.00789，没有声称逐位相同。随机权重验证交接与续算的一致性，不验证预训练质量。两组 KV head 数与 DCP 条件不同，也不据此比较 pull、push 的耗时或加速比。[拉取结果](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/pd-disaggregation/results/pull.json) · [推送结果](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/pd-disaggregation/results/push.json)

### 为什么没有直接照抄一组启动参数？

首次拉取式把 P、D 的设备可见集分开，两个 rank 都在 TE READ 建连时超时。随后使用 2 MiB 对齐缓冲区做最小探针：固定物理 pair、UB 配置和 QoS，仅改变进程可见集，四卡都可见时 READ/WRITE 与全缓冲区检查通过，分开可见时 READ 失败。通过引擎已有的 `--device-ids` 保留四卡可见、分别绑定实例后，完整交接通过。这个结果只定位到本次可见性条件，没有定案 HIXL 内部根因。

首次逐层推送采用普通 GQA/DCP1，却在注册 KV 时被拒绝：packed K/V 的块间 stride 触发 component 分支，而该分支不接受这种非 MLA 布局。改用前一章已解释的 MQA+DCP2，沿现有分配器取得连续 K/V，并将 block size 与 interleave 都设为 128，才跑通本次推送。这里没有修改生产连接器绕过校验；“模型使用 FullAttention”本身不足以保证任意物理布局都受支持。[分配分支](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/v2/attn_utils.py) · [逐层注册校验](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)

## 9. 拉取式多 rank：真正的 READ 在哪里？

下面两张是 Linux 原生 Perfetto 解析器与浏览器查看器的真实截图，分别选中同一次 513 Token 测量请求的 D rank 0、1。`AscendCL@hixlOpBatchRead` 是 CANN 调用，`HixlBatchGet` 是对应设备任务；依据同文件的 `connection_id` 字段配对，并用请求观察记录确认归属。

![拉取式 D rank0 的真实 CANN Read 与 HixlBatchGet](../dist/assets/profile-pd-pull-d-rank0.png)

![同请求 D rank1 的真实 CANN Read 与 HixlBatchGet](../dist/assets/profile-pd-pull-d-rank1.png)

| 该次请求的接收者 | 读取的块数 | TE 调用中的字节量 | 选中设备 Get 任务 |
| --- | --- | --- | --- |
| D rank 0 | 5 | 327,680 B | 约 47.409 μs |
| D rank 1 | 5 | 327,680 B | 约 45.766 μs |

这里每 rank 的一块跨两层 KV 为 `2 层 × 2 份 K/V × 128 位置 × 1 个本地 KV head × 64 维 × 2 B = 65,536 B`。513 个 Token 覆盖五块，传输按块包含末尾 padding；不能拿 513 行的有效 KV 字节数代替实际描述符总长度。129 Token 请求对应两块、每 rank 131,072 B。[逐请求块号、字节量与任务证据](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/pd-disaggregation/results/profiling-events-pull.json)

两张图各自缩放，左边缘没有对齐成同一个同步点。设备 Get 的持续时间也不是整次 TE 调用、排队或端到端交接耗时。选中任务的 HostToDevice flow 只有终点，不能凭查看器补出一根完整的因果箭头；这里使用的是调用与设备任务的相同连接字段。

### READ 成功后，P 与 D 各自推进

下图把同一请求的 Host 观察记录放到一个共享的主机 monotonic 轴上。它由真实 JSONL 的 begin/end 导出为 Chrome trace，再由 Perfetto 显示；没有添加 NPU kernel，也没有把设备时间换算到这根 Host 轴。rankS 表示引擎 Scheduler，rankU 表示日志尚未确定 rank，均不改称 rank0。

![拉取式真实 Host 观察：两 rank 读取、D 就绪与 P 释放调用](../dist/assets/profile-pd-pull-host-lifecycle.png)

观察记录包含两个 D rank 的读取返回、完成通知与 D 状态推进，以及 P 的释放调用。它可以用于核对各自调用的先后，不能把两套 Scheduler 的完成压缩成一个时刻。观察包装、日志记录和采集本身有开销，本项目没有用这根轴测量无插桩服务的延迟。[Host 导出脚本与生命周期结果](https://github.com/Tflowers-0129/inference-wiki/tree/main/experiments/pd-disaggregation)

## 10. 逐层推送多 rank：为什么发送量也不同？

这组使用 MQA+DCP2。按 128 个位置交织，513 个全局位置覆盖五个全局页：rank 0 持有第 0、2、4 页，rank 1 持有第 1、3 页。两端 TP、DCP 布局相同，因此 P rank 0 写对应 D rank 0，P rank 1 写对应 D rank 1。

下面是同一次 513 Token 测量请求的 P rank 0、1，图中聚焦最后一层。每份完整采集包含四条请求、每请求两层，共八次实际 `HixlBatchPut`，在 CANN 侧对应 `AscendCL@hixlOpBatchWrite`；普通 TP 的 HCCL 通信仍是另一路工作，不能把它当作跨 P/D 的 KV payload。

![逐层推送 P rank0 的真实 HIXL 写入任务](../dist/assets/profile-pd-push-p-rank0.png)

![同请求 P rank1 的真实 HIXL 写入任务](../dist/assets/profile-pd-push-p-rank1.png)

| 513 Token 请求的来源 | 每层发送的本地页数 | 每层 TE 字节量 | 两层合计 |
| --- | --- | --- | --- |
| P rank 0 | 3 | 98,304 B | 196,608 B |
| P rank 1 | 2 | 65,536 B | 131,072 B |

每层一页是 `2 份 K/V × 128 位置 × 1 head × 64 维 × 2 B = 32,768 B`。这个 3:2 来自 DCP 位置所有权与尾页，不能解释成专家路由负载不均。129 Token 时，两 rank 每层各传一页；四次测量合计，P rank 0 发送 524,288 B，rank 1 发送 393,216 B，所有观察到的 TE 写调用均成功返回。[逐层字节量与多 rank 完成证据](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/pd-disaggregation/results/profiling-events-push.json)

源码的分块路径中，还要按来源判断最终 chunk：某一来源已经覆盖自己的最后一页，可以完成本来源的最后一层与 DONE；另一来源还可能有尾页要处理。D 等的是映射要求的必要来源，不要求所有 rank 在同一个全局 chunk 边界发出相同消息。本轮两种长度均一次 P 调度完成，没有实测这个跨 chunk 场景。两张设备图的区间也不构成未校准的跨卡共同时间轴。

### 首次预热记录：释放调用返回，不代表所有后台写已返回

本次首次 129 Token 预热中，P 的 `_free_blocks` / `_free_request_blocks` 调用已经返回，最后一次 TE 写约在 191 ms 后才返回。下图来自这次真实 Host 观察记录，保留两个来源 rank、Scheduler 与完成处理，共用同一主机 monotonic 轴。

![逐层推送首次预热的真实 Host 记录：P 释放调用返回早于最后写返回](../dist/assets/profile-pd-push-host-lifecycle.png)

这与第 6 节的接口差异一致：逐层连接器返回“不额外延迟释放”，没有复用拉取式的 `finished_sending` hold 协议。但分配器可能继续使用 deferred-free fence；本次没有记录实际 `block_pool.free_blocks` 或下一请求覆写地址。因此图只能确认这些调用的先后，不能宣称物理块已经被覆盖，也不能直接判定输出错误。

该预热输出仍与本地参考一致。四次测量请求则都是四个写调用返回后，P 才开始释放调用。两种先后都保留在结果中；少量正常请求成功，既不能证明任意负载下的复用安全，也不能用首次建链的长写调用代表稳态传输耗时。[预热观察与边界](https://github.com/Tflowers-0129/inference-wiki/blob/main/experiments/pd-disaggregation/results/lifecycle-summary-push.json)

## 11. 在 Linux 上怎样复查这些图？

实际采集使用 Ascend profiler 的 CPU+NPU activities。Worker 处于 daemon 进程，采集目录需在进程退出后用 `torch_npu.profiler.profiler.analyse` 离线解析；本次先得到原始采集，再生成每 rank 的 `trace_view.json`，最后由 Linux Perfetto 原生解析器打开。

```sh
trace_processor_shell --httpd --http-ip-address 127.0.0.1 \
  --http-port 9001 /private/capture/ASCEND_PROFILER_OUTPUT/trace_view.json
```

通过 SSH 将该 loopback 端口转发到本机后，在 Perfetto UI 选择已载入的 trace。拉取图搜索 `HixlBatchGet`，推送图搜索 `HixlBatchPut`；再用保存的请求身份、层号、字节量与连接字段核对，不能只挑最长的条形当作目标请求。Host 生命周期图使用另行导出的真实观察记录，图注与来源清单区分这两种文件。

本次导出的 Python TE 后台 `record_function` 没有进入 NPU trace，但 JSONL 保留了这些调用，原生 CANN/HIXL 任务也可单独定位。导入器还报告部分完整事件相互重叠并移到 overflow track，选中的 HostToDevice flow 存在缺端；这些限制与原文件校验值一并保存。空白、缺失 scope 或较长调用，都需要补充证据才能解释为等待、瓶颈或异常。[查看步骤、脚本与来源](https://github.com/Tflowers-0129/inference-wiki/tree/main/experiments/pd-disaggregation)
