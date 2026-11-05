# 上下文并行：源码回放与 A5 实操

本目录对应 Wiki 的 TP／PCP／DCP 章节。实验在 2026-11-05 的 Ascend950DT 上完成，分为源码方法回放、Attention 算子对照，以及真实 ModelRunner V2 的小模型运行。原始 trace 和随机模型权重留在实验环境，公开目录仅保存脚本、脱敏结果和可复算的比较。

## 固定环境与来源

- vLLM-Ascend：[f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e](https://github.com/vllm-project/vllm-ascend/tree/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e)。
- vLLM：[5df63a7203cdb9328d824a27b682e77c7b9928be](https://github.com/vllm-project/vllm/tree/5df63a7203cdb9328d824a27b682e77c7b9928be)。
- 本次硬件与软件包身份见 [environment.json](results/environment.json)；小模型的实际配置见各 `tiny-*.json` 的 `model_config`。
- 正文各文件的固定提交入口见 [sources.json](results/sources.json)。

算子脚本调用安装版本的 CANN FIA、GroupCoordinator 和注册 DCP 算子，没有把自己编写的归并公式替换成生产实现。完整 Runner 脚本也使用同一安装版本；教学 TP 表另作算术计算，不声称执行了 `QKVParallelLinear`。

| 脚本 | 实际执行内容 | 结果 |
| --- | --- | --- |
| [cp_layout_lab.py](cp_layout_lab.py) | 从固定源码提取未经修改的 PCP 分片方法 AST，在受控 receiver 上做 CPU 回放 | 8 组 ×2 次，query 覆盖、唯一归属与 head-tail 算例通过；[layout.json](results/layout.json) |
| [cp_attention_lab.py](cp_attention_lab.py) | 两 rank 的完整 KV Attention 对照与 DCP 分片计算、真实 Q-gather、输出/LSE 交换和归并 | 两 rank ×3 case ×2 次，共12条检查通过；[attention.json](results/attention.json) |
| [tiny_cp_runner.py](tiny_cp_runner.py) | 同一个两层随机 Qwen2 checkpoint，经 ModelRunner V2 运行三个并行拓扑 | 每拓扑两轮、4种输入长度；生成 Token ID 与选中 Token 的 logprob 分别比较 |
| [compare_results.py](compare_results.py) | 读取公开 JSON，在 CPU 上复算跨配置比较 | [tiny-comparison.json](results/tiny-comparison.json) |

## 运行前提

设备工作负载必须使用空闲、由自己持有排他租约的两张 A5。使用 MindIE 时，把 coordinator 返回的 `env` 原样传给 remote-dev；等待前一进程树退出后再运行下一项，完成后释放租约。脚本里的可见设备0、1对应租约设备，不是主机物理卡号的承诺。

环境需要匹配的 PyTorch-NPU、CANN、Triton-Ascend、vLLM 和 vLLM-Ascend。下面的命令在已进入该环境、已获得设备租约的前提下运行，不会安装或修改其他服务。不同版本的类名、方法或算子 ABI 可能需要重新审计。

本次运行另设置 `HCCL_CONNECT_TIMEOUT=120`、`OMP_NUM_THREADS=4`、`MAX_JOBS=2`；`ASCEND_RT_VISIBLE_DEVICES`取两张租约设备。没有强制指定 Attention backend，实际后端由目标安装版本选择。

## PCP 分片源码回放

这一步只使用 Python 标准库，既不导入目标模块，也不初始化 NPU。`--source` 指向已固定版本的源码文件：

```sh
python3 cp_layout_lab.py \
  --source /path/to/vllm-ascend/vllm_ascend/worker/v2/pcp_manager.py \
  --output ./captures/layout.json
```

脚本仅编译 `AscendPCPManager._iter_rank_chunks` 的原始 AST，receiver固定 `is_decode_sharded=True`，覆盖 PCP-only、DCP=1 分支。L8/P2得到 rank0的[0,1,6,7]与rank1的[2,3,4,5]；另覆盖不整除、短输入、零行和混合请求。源文件、节点原文与 AST 都保存 SHA256。

结果中的 `teaching_tp_memory` 是独立教学算术：普通分离 K/V、Hq16、Hkv2、D128、BF16、单层16K上下文，TP1/2/4每rank有效KV分别为16/8/8 MiB。它不计块padding、尺度、池预留或权重，也不是设备显存测量。

## 两卡 Attention 与归并

```sh
torchrun --standalone --nnodes=1 --nproc_per_node=2 cp_attention_lab.py \
  --output ./captures/attention --lengths 512 4096 --profile
```

固定 batch4、Hq8、Hkv1、head_dim128、BF16和128位置交错粒度。普通 TP 每rank持有4个query heads及完整KV；DCP沿生产GroupCoordinator收齐8个query heads，只计算本地历史KV，再通过注册 `dcp_a2a_fused` 交换和归并，输出恢复4头。另以相同BF16输入在CPU FP64计算完整Attention oracle。

本次DCP对FP64最大绝对误差为0.0013071429472392704，对普通TP输出最大绝对差为0.001953125，归并LSE对FP64最大绝对差为8.47031419226596e-7。所有检查均使用脚本内声明的容差，不能把它们描述为逐位相等。

不等softmax质量的反例中，局部输出分别为10和0，普通平均为5；按LSE加权后的实际BF16结果为9.0，CPU参考为9.001291271667876。参考略偏离理论9来自输入BF16舍入。

L4096时，每rank完整K+V输入tensor为8 MiB，本地2048位置为4 MiB。对照进程同时持有完整与分片输入，所以这是张量payload差异，不能据此报告进程HBM减半。普通路径与DCP路径均独立预热后采集；实际BF16形状走 `sfa_dcp_exchange` 优化分支，并额外验证通用FP32 `return_lse=True` 路径。

## ModelRunner V2 的三个拓扑

先启用 MRV2，在同一个 `--root` 下依次运行，让三种拓扑复用同一份 checkpoint：

```sh
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 tiny_cp_runner.py --root ./captures/tiny --tp 2 --pcp 1 --dcp 1
python3 tiny_cp_runner.py --root ./captures/tiny --tp 2 --pcp 1 --dcp 2 --profile
python3 tiny_cp_runner.py --root ./captures/tiny --tp 1 --pcp 2 --dcp 1 --profile
```

脚本用固定种子生成随机 Qwen2，词表256、hidden512、两层、8个query heads、1个KV head。它没有训练，生成内容只用于执行链和并行归并检查，不评估模型能力。三份结果的 checkpoint SHA256相同；复现时还需保持软件组合一致，不能假定跨版本随机初始化仍产生同一文件。

引擎固定 eager、每rank32 MiB KV预算、max_model_len2048、max_num_seqs4、max_num_batched_tokens512，关闭前缀缓存并启用chunked prefill。四条输入长度为64、129、256、513，每条贪心生成4个Token，预热后再记录两轮；DCP2与PCP2另采集完整Runner profiling，baseline只记录正确性结果。

| 配置 | 两轮生成ID与baseline | 选中Token logprob相对baseline最大绝对差 |
| --- | --- | --- |
| TP2／PCP1／DCP1 | 全部一致 | 0 |
| TP2／PCP1／DCP2 | 全部一致 | 0.01552438735961914 |
| TP1／PCP2／DCP1 | 全部一致 | 0.0077381134033203125 |

baseline为TP2／PCP1／DCP1第一轮的16个输出位置。相同ID并不表示逐位数值一致；PCP配置自身两轮的选中Token logprob也有最大0.007948875427246094的差异。这里仅比较上报的选中Token概率，没有比较完整词表logits或更大测试集。

固定32 MiB预算下，TP2/DCP1的物理block_size为128、512个块，对应逻辑KV容量65,536位置；DCP2的effective block_size为256，对应131,072位置。物理预留预算相同；这些字段来自引擎配置，不代表本次实际填满了这些位置，`kv_cache_max_concurrency`也不是压测得到的并发承载能力。

公开结果保留各轮原始输出与wall_seconds，但profiling、初始化缓存状态和两轮样本数都影响墙钟时间，不用于宣布吞吐或延迟收益。跨配置差异可重新计算：

```sh
python3 compare_results.py --results ./results --output ./comparison-recomputed.json
```

## 真实 profiling 的解释范围

章节里的截图直接来自查看器，并对照同次运行的rank0、rank1。算子对照选中scope为 `wiki/TP2_DCP1/length4096/iteration2` 与DCP2对应名称；iteration0为预热。实际DCP2轨迹含Q AllGather、本地FIA、pack、AlltoAll、merge及布局转换，另有流事件；不能把RECORD/WAIT全部当作计算kernel计数。

两rank采自同一host，独立保存导出与时钟转换记录；Host换算偏移很接近，但没有本实验确认的Host/device或跨设备对齐误差上界，部分flow终点甚至早于起点。Perfetto能导入这些trace，但部分flow缺少enclosing slice，并有重叠slice告警。因此用原始flow ID做本trace内操作关联，以同组操作序列配对两个rank，不用端点微秒差计算派发延迟，也不把各文件首事件分别归零当作同步开始。HCCL AICPU父区间与传输子任务可能嵌套，不能将各层区间直接相加。

[rank-comparison.json](results/rank-comparison.json)保存两rank算子窗口、各自对应head范围的正确性结果，以及完整PCP捕获的24对Attention。PCP没有显式global-step/layer字段，配对由模型调用顺序重建，并用同一communicator、操作序号、dtype和count交叉核对。首个scope两卡的通信为`_752_2_1`/`_752_3_1`，第九个为`_752_34_1`/`_752_35_1`。这些序号用于本次配对，不是跨运行固定编号。

第九个scope两rank都有CPU FIA调用，但rank1未观测到关联设备FIA。完整PCP捕获的CPU FIA均30次，设备FIA记录为30/22；这不能直接解释为空query、未执行或丢采样。若要确认原因，需补有效query/请求位置及明确step/layer标记。DCP2中rank1的collective导出区间更长、在导出轴上却更早进入，也不能因此把它判作slowest rank。当前结果用于机制和观测边界对照，不用于认定straggler或性能收益。

[截图来源清单](results/profiling-screenshots.json)保存实际查看器、窗口、裁切与源文件校验值；[任务清单](results/profiling-events.json)保留截图对应算子窗口的完整任务名、stream/通信轨道及原始flow关联。图片中很短的kernel标签可能被截断，可对照这份清单。原始trace、随机权重及服务器环境日志不随公开目录发布。

## 代码归属与许可证

`cp_attention_lab.py` 的通信组初始化参考固定源码的 [两卡DCP测试](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_dcp_a2a.py)，为新增query all-gather启用了device communicator。该脚本保留Apache-2.0声明，许可证见 [LICENSE-APACHE-2.0.txt](../LICENSE-APACHE-2.0.txt)。目录名含A3不等于本次A5运行的设备声明，本次设备由环境盘点和真实结果确定。

直接调用的生产实现为 [DCP共享调用路径](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/attention/context_parallel/common_cp.py)、[注册DCP交换／归并算子](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/ops/triton/dcp/dcp_a2a.py)及 [PCP方法源文件](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/worker/v2/pcp_manager.py)。源码回放不导入整个PCP模块；完整Runner结果单独列出，二者的验证范围不同。
