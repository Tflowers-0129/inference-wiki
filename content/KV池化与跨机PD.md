# KV 池化：单机混部与跨机 PD

一条请求结束后，下一条请求可能带着相同的系统提示词或文档。已有的 KV 如果还在本地，就能直接复用；如果本地已经没有，而外部池里还保留着副本，就可以把它取回来，少算这部分前缀。

这里的“池化”指 **KV Cache Pool**。图中会同时出现实例自己的 NPU KV 页池，以及由 `AscendStoreConnector` 接入的外部共享前缀池。多个 P、D 实例组成的实例池，则负责承接和分配请求。[当前功能与部署指南](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L3-L18)

> 两图依据 2026-11-18 核查的 vLLM Ascend 主干源码，采用 Mooncake、普通全 Attention、非逐层路径。图表示逻辑数据流；节点摆放和箭头长度不表示实测耗时。单机图选 Store 与推理同机部署，跨机图选官方 `MultiConnector` 基础组合。

## 单机混部：同一实例计算，池里保留前缀副本

![单机混部池化：先查本地命中，再将池中额外命中的 KV 载入本地页；符合条件的新前缀块写入 Store](../dist/assets/kv-pool-colocated.svg)

沿图看一次复用：Scheduler 先拿到本地命中长度，再查询外部池；Worker 把**额外命中**的 KV 载入已经分配的本地页。剩余输入继续 Prefill，随后由同一实例 Decode。Attention 读取的是 NPU 本地 KV，外部池扩大的是可复用前缀的存储范围。[命中量与目标页分配](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L247-L402) · [Worker 载入](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py#L1384-L1536)

保存也有条件：普通路径按完整传输块保存前缀，同一份请求元数据不会同时安排普通 GET 与 PUT；默认不保存 Decode 新增 KV。把副本写进池，也不等于立即释放活动请求的 HBM 页。[保存条件](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L1129-L1183) · [Decode 保存开关](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L530-L569) · [副本与 HBM 占用](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L382-L414)

## 跨机 PD：共享前缀池与 P→D 交接各走一条路

![跨机 PD 池化：P 读写共享 Store，D 发起 READ 从 P 接收 KV；请求和交接元数据经过 Proxy](../dist/assets/kv-pool-pd.svg)

Proxy 先把请求交给 P，P 从池中复用前缀并补算缺失部分。P 返回源 KV 位置等交接信息，Proxy 再把这些信息带给 D。**D 分配自己的目标页，发起 READ；KV 的数据方向是 P→D。** 载入完成后，D 才读取这些页继续生成。[Proxy 的 P/D 顺序](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L982-L1025) · [D 目标页与传输安排](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1947-L2009) · [READ 调用](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1034-L1064)

这条基础组合中，`AscendStoreConnector` 负责 P 与共享前缀池的读写，`MooncakeConnectorV1` 负责 P→D 的点对点交接；KV 不需要先存入 Store 再交给 D。扩成多个 P 时，各 P 可以查同一个池，但复用还要求模型、缓存布局和键空间等兼容，并不按本地页号互认。[官方双连接器配置](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L247-L349) · [池对象键](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/metadata.py#L118-L160)

## 对照当前源码时，先认这几个名字

| 位置 | 当前入口 | 图中的职责 |
| --- | --- | --- |
| 单机混部 | `AscendStoreConnector`，`kv_role=kv_both` | 同一实例从池取前缀、向池存副本 |
| 跨机 PD | `MultiConnector` | 组合池化与 P2P 两条路径 |
| 共享前缀池 | `AscendStoreConnector`，`backend=mooncake` | 按键查询，通过 GET / PUT 搬运 KV |
| P→D 交接 | `MooncakeConnectorV1` | D 从 P 读取本次请求需要的 KV |

[混部配置](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L416-L477) · [Store 的 PUT / GET 实现](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py#L715-L767)

D 的池读写默认关闭；`consumer_is_to_load`、`consumer_is_to_put`、`save_decode_cache` 提供可选能力，具体支持条件需要另行核对。Mooncake 的 SSD 层也是可选配置，图中只画 DRAM。旧设计文档仍使用 `MooncakeStoreConnector` 等名称，配置应以当前指南和实现为准。[默认开关](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py#L81-L95) · [可选参数](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L1603-L1616) · [SSD 配置](https://github.com/vllm-project/vllm-ascend/blob/f85f77a9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#L132-L168)

两图完成的是源码机制核对，尚未进行这组池化配置的机器实测；TTFT、吞吐与容量收益仍待同条件对照。P/D 页分配、就绪与释放的细节，继续看[PD 分离系统与 KV 生命周期](PD分离系统与KV生命周期.md)。
