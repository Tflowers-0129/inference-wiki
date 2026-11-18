# DP 图协议与 SP 集体通信实验

这里保存 2026-11-16 的两组 CPU/Gloo 实验，以及 2026-11-18 完成的 A5 双卡 HCCL／NPUGraph 探针。DP 实验直接执行固定源码的协调函数；NPU 分支再用标量专家教学计算检查捕获、回放和 eager 回退。SP 实验执行真实 CPU 集体通信和 RMSNorm。这些结果不构成完整模型、Ascend FlashComm 或 NPU 图性能测量。

| 实验 | 实际执行与计数 | 结果 |
| --- | --- | --- |
| [DP 协调源码探针](parallel_graph_probe.py) | 2 次独立启动 × 2 rank × 5 场景，共 20 个 rank-case | [协议结果](results/dp-protocol-cpu.json)全部符合预期；没有捕获设备图 |
| [SP 集体通信探针](sp_gloo_probe.py) | 2 种 dtype × 2 种有效行数 × 2 组输入 × 2 rank，共 16 个 rank-case | [SP 结果](results/sp-gloo.json)中，两条路径逐元素一致，并通过独立数值 oracle |
| DP 探针的 `--device npu` 分支 | 1 次启动 × 2 rank；8 次捕获和零输入回放、10 个 rank-case、16 次场景输出检查 | [A5 NPU 结果](results/a5-npu-20261118.json)全部通过；场景输出含 8 次图回放、8 次 eager，未计时或采集 profiling |

[CPU 执行清单](results/manifest.json)记录 2026-11-16 的脚本校验值、计数、环境与执行边界，其结构见 [manifest schema](run-manifest.schema.json)。A5 NPU 结果单独保存本次脚本、源文件和 AST 校验值，以及逐 rank 的捕获与输出检查。公开结果不包含服务器地址、凭据、设备进程列表或私有路径。

## DP：先核对协议，再检查设备图

实验从 [固定来源](sources.json)提取 `CUDAGraphMode`、`BatchExecutionDescriptor`、`_is_compatible`、`CudaGraphManager.dispatch`、`_resolve_effective_loras`、`DPSyncState`、`sync_cudagraph_and_dp_padding` 和 `dispatch_cg_and_sync_dp` 的原始 AST。函数体保持原样，使用真实 PyTorch CPU 张量和双进程 Gloo AllReduce。

测试夹具提供 `[1,2,4,8]` 的 FULL capture catalogue、每请求一行的 Decode 描述符和 DP CPU group；`should_skip_dp_coordination` 固定为 false。不运行原模块导入、模型初始化、实际 capture catalogue 构建，也不覆盖 DBO、LoRA、投机或 varlen decode。因此，结果验证所选源码在这些输入下的协调行为，不能证明目标安装版本的完整 Runner 已捕获或回放成功。

| 输入场景 | 实际调度行数 | 进入协议的行数 | 两个 rank 的结果 |
| --- | --- | --- | --- |
| 不等长 Decode | `[3,5]` | `[3,5]` | `FULL`，各 8 行 |
| 一侧空闲 | `[0,5]` | `[1,5]`，空侧由夹具提供一行 dummy | `FULL`，各 8 行 |
| 第二侧有 Prefill | `[3,5]` | `[3,5]`，第二侧 uniform query count 为 `None` | `NONE`，各保留 3、5 行 |
| 超出捕获上限 | `[3,9]` | `[3,9]` | `NONE`，各保留 3、9 行 |
| 全部空闲 | `[0,0]` | `[0,0]` | `NONE`、0 行，无同步状态 |

这里的空侧 dummy 是输入条件。实验没有执行 Scheduler 或 `execute_dummy_batch`，不能把夹具的这一行记成模型实际调度出的 token。eager 的 `num_tokens_across_dp` 保留本地长度；后续通信算子如需 padding，应单独记录，不能反过来改写这项协议结果。

实际环境为 Linux 服务器、Python 3.11.6、PyTorch 2.15.0+cu132，设备参数为 CPU。包名包含 CUDA 后缀不表示实验初始化或使用了 CUDA/NPU。固定源码身份与完整文件及 AST 校验值保存在结果中。

复现 CPU 源码探针：

```sh
python3 fetch_sources.py --output ./source
OMP_NUM_THREADS=1 python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 parallel_graph_probe.py \
  --device cpu --source-dir ./source --output ./captures/dp-cpu
```

`fetch_sources.py` 只下载清单中的 vLLM Apache-2.0 文件并核对 SHA256；遇到不匹配的已有文件会停止。提取后的函数在真实 CPU group 上运行，但 capture catalogue 仍是受控夹具。[vLLM DP 协调源码](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/worker/gpu/dp_utils.py) · [图描述符与分派源码](https://github.com/vllm-project/vllm/blob/ced6857a/vllm/v1/worker/gpu/cudagraph_utils.py)

## SP：哪些计算可以留在行分片上？

输入模拟 Row Parallel Linear 在 TP2 上的两份部分和，每个 rank 持有 `[N,H]`，`H=8`。对照路径先 AllReduce 得到完整激活，再运行 RMSNorm；SP 路径先 ReduceScatter 得到本 rank 的行，再在这些行上做 RMSNorm，最后 AllGather 恢复完整结果。RMSNorm 沿完整 hidden 维计算，平方均值采用 FP32，输出转回输入 dtype，`epsilon=1e-6`，`gamma[c]=(c+1)/8`。collective 张量仍采用所选 dtype。

有效行数分别取 4 和 5。5 行先补到 6 行，ReduceScatter 后每 rank 是 `[3,8]`，分别拥有有效行 0–2 和 3–4，最后一个 padding 行保持为零。该场景每 rank 的 RMSNorm 元素数从 `6×8=48` 降到 `3×8=24`；这只是张量工作量计数，不是时延或峰值显存测量。

FP32、BF16 各用两组确定性输入。两条集体通信路径逐元素一致；独立 Python FP64 oracle 用 `math.fsum` 和 `sqrt` 计算归约与归一化。所有场景中，FP32 最大绝对误差为 `1.1780110709e-7`，BF16 为 `0.0036120666`。BF16 比较的相对、绝对容差均为 0.004，FP32 均为 `1e-6`，原始结果逐项保存。

实验没有改变 Attention 的 KV 所有权，也没有运行 Ascend FlashComm 的 MoE prepare/finalize。它说明这种行归一化与分片组合在给定输入下等价，不能据此推导整模型性能或 KV 容量。

复现 SP 实验：

```sh
OMP_NUM_THREADS=1 python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 sp_gloo_probe.py \
  --output ./captures/sp-gloo
```

## NPU：捕获、回放与 eager 回退

本次运行使用 Python 3.12.16、PyTorch 2.10.0+cpu、torch_npu 2.10.0.post7 和两张 Ascend950DT。DP 模式与容量仍由固定 `ced6857a` 源码的原始 AST 在 CPU/Gloo 组上协调；设备计算另用 HCCL。两个 rank 按相同顺序为 `[1,2,4,8]` 各建一张 `NPUGraph`，共 8 次捕获，每张图随后都通过零输入回放检查。

图内先 AllGather 激活和目标专家编号，再由每 rank 的标量专家处理属于自己的行，最后 AllReduce 合并。每行有 4 个 FP32 值，两个专家分别把输入乘以 1 和 2。回放前在固定缓冲区内改变 Token 值与目标专家，随后按独立 Python 参考结果检查输出；padding 必须为零，缓冲地址必须保持不变。

上表五种 DP 场景各运行一次，共 10 个 rank-case，模式与容量全部符合预期。全空闲场景只检查协议返回，不启动设备计算；其余四种场景分别使用两组输入，共 16 次逐 rank 输出检查，全部通过。其中 8 次调用容量 8 的图回放，另 8 次执行 Prefill／超档的 eager 路径。容量 1、2、4 只验证了捕获和零输入回放；非零输入的图回放只覆盖容量 8，不能据此宣称其他容量的动态输入已经验证。

eager 场景的协议返回仍保留 `[3,5]` 或 `[3,9]`。为了运行这个固定形状的 HCCL 教学计算，脚本另外把每侧通信输入补到 5 或 9 行，结果中以 `payload_capacity_per_rank` 单独记录。这不是图协议要求 eager 输入统一补齐。空侧的一行 dummy 也由夹具提供，没有执行 Scheduler 或真实 Runner 的 dummy 前向。

设备运行前已有 UB 总线告警 `81AF8009`，此次通过不能证明设备健康。实验没有调用完整 ModelRunnerV2、Attention、MC2、MegaMoE 或真实 EP 模型，也没有采集 profiling 或重复时延样本，不报告性能收益。[逐 rank 捕获、场景输出与源校验值](results/a5-npu-20261118.json)

复现前先核对设备健康状态，并取得两张设备的排他租约或已验证的专属预留，将可见设备环境原样传给运行进程。本次没有启用 `--profile`：

```sh
OMP_NUM_THREADS=1 python3 -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 parallel_graph_probe.py \
  --device npu --source-dir ./source --output ./captures/npu \
  --reservation-id "$RESERVATION_ID"
```

外层进程管理器须设置超时，失败后确认两个 worker 和子进程退出再释放租约。`--reservation-id` 要求提供外部资源预留标识，但不验证租约或建立排他性。完整 ModelRunnerV2、真实 EP 算子和不同 DP 到达状态仍需另做输出正确性与性能对照。

## 九章的实验补充顺序

优先级 A 对应本轮主题所需的新证据；B 用于把现有机制解释补成目标环境验证；C 是已有基础实测之外的扩展。各行是后续实验设计，不是已取得的结果。

| 章节 | 已有证据与缺口 | 建议实验和判据 | 顺序 |
| --- | --- | --- | --- |
| 1 调度与执行 | 教学调度播放器；缺完整调度与图命中关联 | 同一小 MoE 的 eager／FULL_DECODE_ONLY，逐轮记录 Scheduler 行数、DP 协议、实际 capture/replay 和输出；含空 DP、Prefill 混入、超档 | A，与第 5 章共用 |
| 2 一次前向 | 已有 padded-inputs Triton 正确性和任务数；缺 metadata 与 tiling 的独立代价 | 固定 Runner、Attention 后端和输入，分别记录 Host metadata、D2H 精确值等待、tiling 更新和设备执行；不把 kernel 数减少写成去除了所有 Host 工作 | A |
| 3 投机解码 | 已有进度辅助函数、真实 SSE；缺完整 Draft/Target KV 链 | 固定 proposer 和随机种子，记录不同接受长度的有效 KV、候选临时槽、下一轮输入及实际流式输出 | B |
| 4 负载均衡 | 同容量教学模型与实现证据；缺真实重排关键路径 | 固定输入、权重、冗余预算，对比重排前后每层整数专家任务、最忙 rank、迁移窗口和稳态；先校验输出 | B |
| 5 EP 通信 | 已有双卡 MC2/MegaMoE eager 实测及本文 HCCL／NPUGraph 教学探针；缺真实模型的跨 DP 图 | 运行真实模型，核对每 rank 相同模式、捕获档位、通信方法、collective 顺序和有效 token mask | A，与第 1 章共用 |
| 6 上下文并行 | 已有 PCP/DCP、三种双卡小 Runner；本文补了 CPU SP | 同权重 TP2 eager 对比 SP 开关，测 NPU RS/AG 与 Norm 范围、峰值激活、输出；另用长上下文验证 DCP，分别记 KV 与激活容量 | A |
| 7 PD 分离系统 | 已有拉取、逐层推送、真实 HIXL；异构 TP 页重排是源码算例 | P TP4→D TP2，核对 KV head 对应、每段地址、有效行与续算输出；注入传输失败核对实际复用边界 | C |
| 8 KV 缓存布局 | 已有 GQA 页视图、步幅与传输字节；其他模型布局未同样实测 | 固定 MHA/GQA/MQA 配置比较 TP1/2/4 的真实 KV shape、stride 和池容量；DCP 开关另测，不拿分配总量代替有效 payload | B，可复用第 6 章 |
| 9 稀疏 KV 与 HiSparse | 现有 A5 性能来自 PR 作者，尚无项目实测 | 先验证 LI+Manage / Copy+SFA 算子；覆盖首次初填、稳态重复 miss、512/513 分支与 tail；再配对 HiSparse Core/Ascend 框架测量 | B |

涉及设备的各项都需要匹配的模型／量化／后端与独占 NPU。对照需保持相同输入和测量边界。本文 CPU 正确性结果不替代这些条件，也不为这些未执行项目生成时延或吞吐数字。
