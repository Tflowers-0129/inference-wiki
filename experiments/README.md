# Wiki 实操实验

这些脚本把第二、三、五、六、七章里的部分机制放到真实运行环境中核对。前四项结果采集于 2026-11-04；固定源码身份、软件组合和量化条件保存在 [A5 结果](results/a5-moe.json)。上下文并行实验采集于 2026-11-05，另有[独立环境与复现说明](context-parallel/README.md)。PD 分离同日运行，[独立说明](pd-disaggregation/README.md)记录拉取式与逐层推送式的实际配置、请求结果及多 rank 证据。公开目录只保留脚本、数值结果及选定设备事件，不包含服务器端点、凭据或整份原始 trace。

| 实验 | 实际执行内容 | 已确认的范围 |
| --- | --- | --- |
| [padded-inputs.py](padded-inputs.py) | Ascend950DT 上的实际 Triton kernel、Torch 表达式和独立整数 oracle | 7 组 ×2 次正确性通过；预热后每条 Torch 链 9 个设备任务，Triton 1 个 |
| [batch-change.py](batch-change.py) | 目标安装版本的进度修正辅助函数 | 请求重排、新请求和旧候选参与条件；6 次检查通过，未运行完整 Runner |
| [sse-buffering.py](sse-buffering.py) | Linux loopback 的真实 HTTP/SSE 服务、缓冲代理和客户端 | 4 场景 ×2 轮，输出内容与计数保持，实际事件间隔和 TPOT 口径通过 |
| [a5-moe.py](a5-moe.py) | 相同 W4A8MXFP 权重、输入和路由下的 MC2 拆分链与 CANN MegaMoE | EP2，8/32 行，两个 rank 输出逐元素一致；独立专家计数通过，8 份 NPU trace 已采 |
| [上下文并行](context-parallel/README.md) | PCP 源码方法回放、实际 Q 交换/FIA/DCP 归并、三种双卡 ModelRunnerV2 拓扑 | 16 组布局回放、12 条算子检查通过；三拓扑两轮 greedy tokens 一致，保留 logprob 差异；真实 Perfetto 截图另列来源 |
| [PD 分离系统](pd-disaggregation/README.md) | 四 NPU 上的实际 P TP2、D TP2、官方 HTTP proxy 与 Mooncake；分别运行拉取 GQA/DCP1、逐层推送 MQA/DCP2 | 每组四次测量的 greedy 输出与各自本地参考一致；双 rank READ／逐层 WRITE、D 就绪及请求释放调用有记录；首次 push 预热的生命周期边界单独保留 |

## 运行条件与命令

设备实验要求匹配的 vLLM-Ascend、PyTorch-NPU、Triton 和 CANN 算子包。A5 的这个 MegaMoE 对照采用 MXFP 路径，不能把自定义 A3 W8A8 INT 实验直接替换进来。完整环境身份见结果 JSON；不同安装版本可能移动模块或改变 ABI。

使用 MindIE 时，先以 coordinator 获得实际物理设备的排他租约，把返回的 `env` 原样传给 remote-dev；单卡实验选择可见设备 0，双卡实验由 torchrun 分配可见设备 0、1。等待前一进程树退出，再开始下一项，完成后释放租约。不要使用已有服务占用的设备。

在该环境中运行：

```sh
python3 padded-inputs.py --output ./captures/padded
python3 batch-change.py --output ./captures/batch-change/result.json
torchrun --standalone --nnodes=1 --nproc_per_node=2 a5-moe.py \
  --tokens 8 32 --repeats 10 --profile-dir ./captures/moe
```

SSE 实验只需要 Python 标准库，不使用模型或 NPU：

```sh
python3 sse-buffering.py --output ./captures/sse/result.json
```

两轮主场景交叉执行 direct→buffer、buffer→direct；另覆盖七项单块和一个输出。输出数由七个已知标签提供，不从 TCP 分片或逐块分词推断。`send_begin/end` 包围 write+flush，不表示对端确认。

## 怎样解释结果

MoE 的正确性先比较相同量化权重的两条实现，并用独立路由计数核对任务守恒；这不等于与预训练 BF16 模型比较精度。NPU event 包围 Python 函数提交，区间可能包含等待 Host 或通信伙伴的空隙；结果中的原始样本和中位数不能写成纯 kernel 算力或服务吞吐收益。

各 rank 的 profiler 初始化不同步，完整采集还含未归属 active scope 的设备任务。active iteration 由原始 `torch_to_npu` flow ID 关联到 Host scope，再关联设备任务；选定的 [泳道数据](results/a5-moe-swimlane.json) 使用每种实现自己的设备首任务作零点，没有跨 rank 对齐。Host／Device 微秒级时间差未经独立校准，Perfetto 导入还存在 flow 告警，因此不据其计算派发延迟。

第二章的 [padded 正确性结果](results/padded-inputs.json) 与 [任务归属摘要](results/padded-profile.json)、第三章的 [进度修正结果](results/batch-change.json) 与 [SSE 实际打点](results/sse-buffering.json) 分别保留各自口径。SSE 的单块七项 TPOT=0 是接收边界的退化结果；O=1 时记录为未定义。

第五章另用四张真实 Perfetto 截图配对 tokens32、iteration2 的 MC2／CANN MegaMoE 两个 rank。[逐卡任务与路由结果](results/a5-moe-rank-comparison.json)保留精确 flow、Host／Device 窗口及 71+57=128 次专家任务；[截图来源](results/a5-moe-profiling-screenshots.json)记录源 SHA 和实际显示范围。Level0 没有独立的 HCCL peer 收发轨道，因此不套用第六章的集体通信序号配对，也不从单卡区间判定慢 rank。

Linux 上解析与浏览器查看的完整步骤见 [第五章实操](https://tflowers-0129.github.io/inference-wiki/ep-communication.html#ep-communication-11)。MindStudio Insight 应保留完整 `*_ascend_pt` 目录；本仓库不自动公开原始 trace。

第七章的[环境与本地文件身份](pd-disaggregation/results/environment.json)、[拉取结果](pd-disaggregation/results/pull.json)和[逐层推送结果](pd-disaggregation/results/push.json)分别记录各自配置。拉取设备图查看 D rank0／rank1 的 `HixlBatchGet`，推送设备图查看 P rank0／rank1 的 `HixlBatchPut`；[拉取归属](pd-disaggregation/results/profiling-events-pull.json)与[推送归属](pd-disaggregation/results/profiling-events-push.json)保留连接字段、缺失 flow 端点和未校准时钟等限制。Host 生命周期图另由真实 JSONL 导出到 Perfetto，不混入 NPU kernel。两种实验的 KV head 和 DCP 条件不同，不计算路径间加速比。

## 仍需独立验证的项目

| 章节 | 尚未覆盖的实操 | 必需条件 |
| --- | --- | --- |
| 1 调度与执行 | 实际 Scheduler 连续轮次、图模式与真实捕获形状 | 目标测试夹具；完整模型与匹配 Attention 后端 |
| 2 一次前向 | slot_mapping、多 KV 组融合、Host 精确值等待 | 对应 Runner／块表 ABI 与逐元素地址 oracle |
| 3 投机解码 | 完整 Draft→Target→提交回退→下一轮的模型执行 | 模型与 proposer 配对；接受结果及 KV 边界取证 |
| 4 负载均衡 | 实际 DPLB 选择回放、EPLB 整数副本映射及同容量关键路径 | 固定入口和遥测；多卡同权重、路由与布局对照 |
| 5 EP 通信 | 完整选择函数、AllGather/AllToAll 容量边界、graph 与服务级指标 | 支持的模型／量化／拓扑和同条件负载 |
| 6 上下文并行 | 预训练长上下文精度、PCP+DCP 叠加、MLA/SFA 与投机侧流、graph 和服务吞吐 | 对应权重、合法拓扑、Attention 后端与同条件负载 |
| 7 PD 分离 | 多主机网络、同配置 pull/push 对比、取消／失败注入、并发块覆盖复用与稳态服务指标 | 匹配缓存布局与 transport、完整生命周期取证、受控多实例负载 |
| 9 稀疏 KV | LI+Manage、Copy+SFA 阈值和 HiSparse 页生命周期 | 对应编译产物、合法 backing、Core/Ascend ABI 配对 |

正文明确声明的教学时长、四十 token 路由、缩比缓存和捕获档位不属于设备复现目标。实测采用目标实现合法参数；上述表格不会因为播放器运行正确就改为通过。

## 源码归属

`a5-moe.py` 改编自 [vLLM-Ascend 固定源码的 test_mega_moe.py](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/tests/e2e/nightly/single_node/ops/test_mega_moe.py)。该文件采用 Apache-2.0，许可证保存在 [LICENSE-APACHE-2.0.txt](LICENSE-APACHE-2.0.txt)。改动包括双卡小形状默认值、独立专家计数 oracle、AB/BA 原始采样及按 rank 的 profiling；普通链与生产权重处理、融合入口均保留。

同一固定提交的 [W4A8MXFP GMM1／激活分支](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py#L368-L441) 说明 SiLU 使用分开的 GMM1 与 SwiGLU/量化，SiTU 的单算子分支另有条件；[生产 MLP 调用方](https://github.com/vllm-project/vllm-ascend/blob/f9c8e0bdb263d7a8c0270536f1ea00fa7373a84e/vllm_ascend/ops/fused_moe/moe_mlp.py#L70-L78) 负责选择对应激活路径。
