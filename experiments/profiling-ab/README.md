# MoE 图回放与 SP 的前后对照

这里用两项小实验分别回答：同一条 MoE 算子链从 eager 改为图回放后，提交和执行怎样变化；把归一化移到行分片上后，通信与计算怎样变化。SP 与 MC2 eager/replay 的 A5 双卡对照已完成，结果和轨道见下文。两项均保留设备既有告警与微实验范围；MC2 回放 Trace 还保留 flow 导出缺口。

| 脚本 | 固定条件 | A → B 的唯一变化 | 能验证的范围 |
| --- | --- | --- | --- |
| [moe_ab.py](moe_ab.py) | 两个 EP rank；同一实现、量化权重、输入、专家选择及路由权重 | eager 调用 → 同一算子链的 `NPUGraph.replay()` | A5 W4A8MXFP MoE 算子链的正确性、提交开销与设备执行 |
| [sp_ab.py](sp_ab.py) | TP2 部分和输入；BF16、H=4096、同一 gamma 和 epsilon；有效行数分别为 128、129 | AllReduce→完整 RMSNorm → ReduceScatter→本地 RMSNorm→AllGather | 实际 HCCL 和 NPU RMSNorm 的两条等价微路径 |

MoE 实验不经过 Scheduler、ModelRunnerV2、Attention 或跨 DP 图模式协调，因此不能证明 `FULL_DECODE_ONLY` 在完整模型中生效。SP 实验不调用 FlashComm 的 MoE prepare/finalize，不测 KV 分片，也不证明整模型加速。它的 CPU 模式另用 Gloo 和 TorchNorm，只用于检查脚本与数值，不能作为 NPU 结果。

## 先确认设备和环境

两项 NPU 实验都需要两张设备的排他使用窗口。先由资源协调器核对物理设备、运行中的容器及设备进程，取得租约，把它返回的可见设备环境原样传给执行进程。`LOCAL_RANK=0/1` 指这两张可见设备，不是固定的物理卡号。`--reservation-id` 只记录外部预留标识，脚本不取得租约，也不验证排他性。

启动前保存 `npu-smi` 健康状态、告警和进程信息，并核对目标容器的实际软件身份。已有告警必须随实验记录保存，不能把资源租约成功当作设备健康证明。先运行短时 smoke；发生设备或通信错误就停止并保留失败记录，不通过 reset、清告警或停止其他服务来继续实验。每项作业使用独立输出目录和唯一 job ID；外层管理器必须设进程树超时，退出后确认两个 worker 及子进程都结束，再运行下一项或释放租约。

SP 需要匹配的 PyTorch、PyTorch-NPU、CANN/HCCL，以及 `torch_npu.npu_rms_norm`、AllReduce、ReduceScatter、AllGather。MoE 还需要 A5 MXFP 格式、vLLM-Ascend 的 MC2/MegaMoE 接口、自定义算子和图捕获支持。即使指定 `--implementation mc2`，MoE 脚本也会先运行两种 eager 实现做正确性对照，因此仍需要 MegaMoE 算子。

可优先核对已有 [A5 环境记录](../context-parallel/results/environment.json)中的组合，但不能把历史版本直接写成新作业的安装身份。两脚本均动态记录包版本、可取得的源码身份和脚本 SHA256；CANN、驱动或硬件信息缺失时，由资源记录补齐。原始输出和 Trace 先归档，公开前检查端点、文件路径及设备身份等内容。

### MoE 自定义算子的注册时机

这次 MoE 首次进入 eager 正确性检查时报出 `y dtype must be hifloat8`。配对核查显示，新旧容器的十份相关文件校验值相同，匹配的 custom 算子源码支持实验所用的 MXFP8。固定实现会在扩展初始化时读取 custom OPP 路径；在设备初始化后才设置路径，可能无法改变此前的算子查找状态。

受控重跑保持脚本、输入、权重和其他启动字段相同，只在 Python 启动前把匹配的 vendor 目录放到 `ASCEND_CUSTOM_OPP_PATH` 前面，随后两 rank 的 eager、捕获与回放检查通过。没有把实验改成 HIF8，也没有改全局环境。失败进程缺少 loaded-library map，所以这组证据支持注册时机的解释，不声称已直接追踪每一次库选择。[脱敏的启动对照与文件哈希](results/moe-preflight.json)

复现这个固定实现时，先确认实际包中的 vendor 目录，再在启动 Python 之前设置：

```sh
VLLM_ASCEND_OPP_VENDOR=/path/to/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH="$VLLM_ASCEND_OPP_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

这两个环境设置只作用于当前 shell 及其子进程。其他安装组合须读取对应包的环境配置，不能照搬一条服务器绝对路径。

## 运行命令

以下命令在本目录、已激活的目标 Python 环境中执行。`RESERVATION_ID` 由外部资源管理器提供；示例 job ID 每次换一个新值。Linux 的 `timeout` 是外层保护，不能替代作业结束后的进程核对。

先做两项短测，分别运行，等前一项完全退出后再开始下一项：

```sh
OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 600s \
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  moe_ab.py --implementation mc2 --smoke \
  --job-id moe-smoke-001 --reservation-id "$RESERVATION_ID" \
  --output ./captures/moe

OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 600s \
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  sp_ab.py --device npu --warmup 1 --samples 2 \
  --job-id sp-smoke-001 --reservation-id "$RESERVATION_ID" \
  --output ./captures/sp-smoke-001
```

两者都会先验证正确性。短测样本只用于确认路径可运行，不能据此发布稳定性能。MoE 的 `--smoke` 将预热改为 3 次、每块采样 2 次、ABBA 循环 1 次；若另开 profiling，活动迭代为 2 次。SP 没有 `--smoke` 参数，用 `--warmup 1 --samples 2` 明确缩短采样。

短测通过且设备仍满足运行条件后，使用新目录做正式采样。下列 `--profile` 会在无 profiler 的计时完成之后，另起独立的 Trace 采集循环：

```sh
OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 1800s \
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  moe_ab.py --implementation mc2 --tokens 32 --experts 8 --topk 2 \
  --hidden 1024 --intermediate 1024 --activation silu \
  --warmup 20 --repeats 30 --rounds 2 \
  --profile --profile-iterations 3 --profile-level Level1 \
  --job-id moe-ab-001 --reservation-id "$RESERVATION_ID" \
  --output ./captures/moe

OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 1800s \
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  sp_ab.py --device npu --warmup 10 --samples 50 \
  --profile --profile-level Level1 \
  --job-id sp-ab-001 --reservation-id "$RESERVATION_ID" \
  --output ./captures/sp-ab-001
```

MoE 的 `--implementation mega` 改为比较 MegaMoE 自己的 eager 与 replay；`both` 分别完成两组比较。每组内部保持实现不变，不能把 MC2 eager 与 MegaMoE replay 的差别全部归因于图回放。`--profile-level` 默认采用 Level1；SP 的 CPU 模式不传 NPU profiler 配置。

没有可用 NPU 时，可单独检查 CPU 路径：

```sh
OMP_NUM_THREADS=1 timeout --signal=TERM --kill-after=30s 300s \
  python3 -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  sp_ab.py --device cpu --warmup 1 --samples 2 \
  --job-id sp-cpu-001 --output ./captures/sp-cpu-001
```

CPU 结果明确记录 `device=cpu`、`collective_backend=gloo`，NPU event 耗时为空。历史 [sp_gloo_probe.py](../parallel-graph/sp_gloo_probe.py)及其 H=8 结果保持不变；本脚本的 H=4096 是另一组实验条件。

## 先核对正确性，再看计时

MoE 先从相同随机权重生成两种实现所需的量化布局，用相同输入和路由比较 MC2 与 MegaMoE。独立的路由计数检查每个专家收到的整数任务数；数值比较要求相对 RMSE 小于 0.03、余弦相似度大于 0.999。这是两条量化算子链之间的比较，不是预训练模型的精度评估。

随后才捕获所选实现，并把 replay 输出与该实现自己的 eager 输出比较。图比较默认 `--graph-rtol 0 --graph-atol 0`，专家任务计数必须精确相等。捕获或首次回放失败会记入 `capture` 和 `failure_stage` 并退出，不切到 eager 生成一份冒充 replay 的结果。计时及 profiling 结束后还会复查输出、输入内容和缓冲地址。

SP 为每个行数使用两组确定性输入。129 个有效行补到 130 行，两条路径都保留相同的 padding。NPU 用实际 `npu_rms_norm`；独立 Python FP64 oracle 以 `math.fsum` 和 `sqrt` 计算归约和归一化，gamma 采用实际 BF16 值。两路径与 oracle 的绝对、相对容差均为 0.004；另记录路径间最大误差、是否逐元素相等，并要求 padding 输出为零。所有输入组合通过后才进入计时。

| 项目 | MoE | SP |
| --- | --- | --- |
| 无 profiler 顺序 | 每轮按 eager→replay→replay→eager 四个块执行，即 ABBA | 偶数配对为 AB，奇数配对为 BA；每对用相同输入 |
| 示例采样量 | 每块 30 次、2 轮；每 rank、每实现、每模式 120 次 | 每种行数、每 rank、每模式 50 次 |
| 窗口外工作 | 权重准备、图捕获、预热、对齐及结果汇总 | 输入恢复、预热、设备同步、控制组对齐及结果写盘 |
| 设备计时 | `npu_event_us`，微秒 | `npu_event_ms`，毫秒；CPU 模式为空 |
| Host 计时 | `host_submit_us` 只包围执行提交，不含结束同步 | `host_ms` 包围本地执行至同步完成，包含提交和等待 |

NPU event 区间可能包含 Host 提交空隙及等待通信伙伴的时间，不能直接叫作纯 kernel 耗时。两脚本的 Host 指标边界不同，也不能直接横向比较。SP 分片后 Norm 的输入行数减少，不代表整条路径必然更快：ReduceScatter 与 AllGather 的代价也在 B 路径内。

## Trace 必须来自计时的那条路径

历史 [a5-moe.py](../a5-moe.py) 的 `--graph` 只在 `measure()` 内把局部 `fn` 改成 `graph.replay()`；后面的 profiling 循环仍调用原来的 `decomposed` 或 `fused`。因此，给旧脚本加 `--graph` 并不能证明其 Trace 是图回放。本目录不改变历史结果的含义。

新 MoE 脚本把 eager 和 replay 各包装成一个 `Execution` 对象。捕获后的回放验证、无 profiler 计时和 profiling 使用同一组对象；replay 对象调用已捕获的图，并持有对应输出缓冲。这个调用关系消除了旧脚本的路径错位，但图回放是否成功、Trace 是否记录到对应设备任务，仍要检查真实回执与 Host↔Device 关联。

SP 则在计时和 profiling 中调用相同的 `forward(case, mode, input_variant)`。profiling 额外加上 AR、RS、Norm、AG 的 Host 标记，计算函数不变。两脚本都先完成无 profiler 样本，再分别采集各模式，不能把 Trace 内的时长混入正式性能样本。

MoE 的输出在 `<output>/<job-id>/rank0|rank1/result.json`，Trace 在同 rank 的 `profiles/<implementation>/<mode>/`。SP 的输出直接在 `<output>/rank0.json` 和 `rank1.json`，Trace 在 `profiles/N128|N129/A|B/rank0|rank1/`。MoE 示例的活动迭代为 1～3，SP 为 1、2，迭代 0 用于 profiler 预热。

配对时先核对两个 rank 的 job ID、脚本、环境与配置，再按实现/模式/行数/迭代选取相同语义的运行；SP 还要核对 `input_variant`。随后沿各文件中的 Host scope、flow 和设备任务确认执行内容，并在有信息时核对 communicator、collective 序号、dtype 和 count。只有 scope 名带着 replay，不足以证明设备任务来自回放。

**同一迭代不等于时钟已对齐。** 两个 rank 的 Trace 使用各自时钟，未经校准不能直接相减，也不能仅凭截图左右位置判定谁先到达。画前后时间线时保持相同时间比例、各 rank 独立起点；展示实际相关的计算、通信和等待轨道。MoE 的 `matched_rank_max_us` 是同一配对中本地 event 时长的最大值，不是已校准的多设备共同起止时间。无法归因的空白仍保留为未知。

## 本次结果与未覆盖范围

### SP

2026-11-18 的 A5 双卡实验先完成 smoke，再运行每组 50 次正式计时和 Level1 profiling。两个 rank 共保留 8 条正确性记录、400 条无 profiler 样本，以及 8 份 Trace。A/B 输出逐元素相同，相对 FP64 oracle 的最大绝对误差为 0.0038676165，补齐行保持零；实际计时路径保留完整 130 行输出，并没有额外执行去 padding。

[可公开结果](results/sp-ab.json)保留原始计时样本、校验、实际软件身份、8 份源 Trace 的 SHA256、选定事件索引和关联依据。原始 Trace 在私有实验目录归档；公开 JSON 不含主机地址、进程号或私有目录。[双 rank 轨道图](../../dist/assets/sp-measured-rank-streams.svg)选取 N128、活动迭代 2，四个分面使用相同时间比例，各 rank/模式独立起点。未关联到调用的窗口内事件另作标识，不能补画跨 rank 的等待因果。

N128 下，A 的两个 rank 中位数分别为 148.92、147.32 µs，B 为 200.17、200.45 µs；N129 的 A 为 158.59、158.95 µs，B 为 208.23、209.13 µs。这是无 profiler 的本地 NPU event 区间，不能称作纯 kernel 时间或跨设备共同区间。该小形状中分片路径更慢，不能仅从 Norm 行数减半推断收益。

机器已有 `81AF8009` UB 总线告警，当前结果未隔离告警影响。因此，本次数据可用于核对执行内容与报告该环境的观察，不能作为正常健康 A5 的性能基线，也不能外推到完整 FlashComm 模型。

在仓库根目录重建公开数据与 SVG：

```sh
python3 experiments/profiling-ab/analyze_sp.py --input /path/to/private/sp-profile
python3 scripts/render-measured-sp.py
```

分析器读取私有原始采集目录；SVG 只读已经公开的结果 JSON，不访问设备。

### MC2 eager 与真实图回放

两个 rank 都通过 eager 交叉数值检查、独立专家计数、首次图回放及计时/profiling 后复查。正式配置为每 rank 32 Token、8 专家、TopK=2、H/I=1024、W4A8MXFP、SiLU，路由固定，总任务数为 128；两侧分别处理 60、68 次专家任务。每种模式每 rank 120 个无 profiler 样本，合计 480 个。

| rank | eager NPU event 中位数 | replay NPU event 中位数 | Host 提交中位数：eager → replay |
| --- | --- | --- | --- |
| 0 | 842.36 µs | 65.54 µs | 770.7 → 17.8 µs |
| 1 | 813.88 µs | 54.50 µs | 779.7 → 21.5 µs |

[公开结果](results/moe-graph-ab.json)保留全部样本、capture 状态和校验；[双 rank 对照图](../../dist/assets/moe-graph-rank-streams.svg)展示两种模式各三次活动执行的完整窗口。replay 的导出日志报告 ACL→NPU flow 关联失败，部分 ID 又在重复执行中复用，因此没有把设备任务强行配到某个 Host iteration。额外放大图显式标注比例，不与总览混用。

这个小形状的 eager 区间有明显的 Host 提交开销；表中区间下降不等于 GMM 算力增加或整模型同倍加速。机器仍有已记录的 UB 告警。正式作业正常结束、所有后代进程退出后才释放双卡预留；原始 Trace 与运行回执另行私有归档。重绘命令为 `python3 scripts/render-measured-moe-graph.py`。重新分析原始 MoE 数据时，同时传入原始 supervisor stdout，以核对 flow 导出错误：

```sh
python3 experiments/profiling-ab/analyze_moe.py \
  --input /path/to/private/moe-job \
  --launch-log /path/to/private/job/stdout.log
```

| 项目 | 当前文档中的状态 |
| --- | --- |
| MC2 同实现 eager/replay 的正式计时及 Trace | 完成；回放 Host↔Device flow 关联有导出缺口 |
| SP 两路径的正式 NPU 计时及配对 Trace | 完成；保留既有设备告警与微实验范围 |
| 完整 ModelRunnerV2 跨 DP `FULL_DECODE_ONLY` | 不在这两项微实验范围内 |
| 完整模型 FlashComm 的输出、内存和服务性能 | 不在这两项微实验范围内 |
