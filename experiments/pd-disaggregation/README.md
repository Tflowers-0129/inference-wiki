# PD 请求交接与 KV 生命周期实验

本目录对应第七章《PD 分离系统：请求交接与 KV 生命周期》。实验使用实际 vLLM-Ascend HTTP 服务、官方 proxy、Mooncake 原生传输与 NPU 模型执行；教学机制图与真实采集分别标注。

## 固定源码与二进制身份

[results/sources.json](results/sources.json) 保存固定 vLLM-Ascend、配对 vLLM、Mooncake 参照 tag、文件 SHA256 与具体主张的代码位置。正文的 main 链接用于导航；本次结论以清单中的固定源码为准。

安装元数据将原 wheel 声明为 `mooncake-transfer-engine 0.3.13`，其中的 archive hash 来自 `dist-info/direct_url.json`。该声明不是对原 wheel 的独立重算，也不能证明安装的 native binary 与参照 tag 同源。运行环境另记录安装目录中指定 native 文件的校验值。

## 运行对象与校验

`run_pd_lab.py` 创建固定随机种子的两层 Qwen2 checkpoint，并附一个本地 WordLevel tokenizer。模型配置为 vocab=256、hidden=512、intermediate=1024、Q heads=8、head_dim=64、BF16。拉取式使用 KV heads=2、DCP1；逐层推送使用 KV heads=1、DCP2。随机权重用于检查请求交接与续算一致性，不验证预训练模型质量。

P、D 独立启动 Scheduler、Worker 和 KV 池，两端 TP2。模型长度上限 2048，本轮 Token 预算 1024，最大序列数 4，每 Worker KV 池预算 32 MiB；eager、关闭 prefix caching、启用平台默认 Chunked Prefill。两端池预算是预留容量，不是实际传输量或运行中有效 KV 占用。两组 checkpoint/并行布局不同，不能比较它们的耗时来评价 pull 与 push 的性能。

拉取式选择 `MooncakeConnectorV1` 与 `load_balance_proxy_server_example.py`；逐层推送式选择 `MooncakeLayerwiseConnector` 与 `load_balance_proxy_layerwise_server_example.py`。不使用 mock connector，不用 D 本地重算冒充 KV 传输成功。默认 KV 加载失败策略为 fail。

每种方式先在同配置 P 上完成普通本地请求，建立 colocated greedy 参考，再经 proxy 预热。采集阶段使用 prompt 长度 129、513，各重复两次、生成四个 Token，检查输出 Token IDs 与参考一致，并记录所选输出 Token 的 logprob 最大绝对差；脚本阈值为 0.05。该阈值只检查这组有限输出，不是完整 logits 精度界。

## 执行命令

先通过 MindIE coordinator 探测空闲健康设备并取得排他租约，将返回的环境传给作业。以下物理设备列表仅为调用形式；必须改成自己的租约设备，原始日志、HTTP 响应、权重和 profiler 输出保存在 Wiki 之外。

```sh
python run_pd_lab.py --root /private/experiment-root \
  --mode pull --devices 1,2,3,6 --p-tp 2 --d-tp 2 \
  --device-selection shared-visible --kv-heads 2 --dcp 1 --profile

python run_pd_lab.py --root /private/experiment-root \
  --mode push --devices 1,2,3,6 --p-tp 2 --d-tp 2 \
  --device-selection shared-visible --kv-heads 1 --dcp 2 \
  --base-port 18310 --kv-base-port 26400 --profile
```

`--device-selection split-visible` 为每套服务只暴露其设备子集；`shared-visible` 保留所选租约设备可见，并用引擎的 `--device-ids` 显式选择本实例。可见性、协议环境与实际通过的执行方式需以结果清单为准，不能把命令示例当作任意 A5 环境都已验证。

launcher 仅为自己的子进程设置通信环境，使用 loopback 单机控制面，不改主机网络或驱动。P/D 各用不同侧信道端口；TE 入口由原生库分配。离开单机时应按部署网络设置地址与网卡，不能照搬 loopback。

### 两个实际遇到的启动/传输条件

首次拉取式将 P/D 设备可见集分开，两个 rank 的 TE READ 均建连超时。独立的 2 MiB 对齐缓冲区探针在相同物理 pair、相同 UB 与 QoS=1 下，仅改变可见集：所有进程可见四卡时 READ/WRITE 与全缓冲区内容检查通过，分开可见时 READ 建链失败、没有进入 WRITE。随后通过框架已有的 `--device-ids` 选择实例设备，完整拉取式通过。这个结果定位到本次可见性条件，未证明 HIXL 内部根因或其他环境的通用修复。

首次逐层推送采用 GQA/DCP1，启动时被连接器拒绝：默认 packed K/V 的 block stride 触发 component transfer，而该分支不接受这种非 MLA 布局。MQA+DCP2 使用现有 allocator 的连续独立 K/V 分支，两端显式 block size=128、CP interleave=128，完整推送通过。未 patch 生产连接器或 allocator 来绕过检查。

## 观察器与真实 profiling

`instrument_pd.py` 为正常 CLI 与 spawn Worker 安装观察包装。它不改生产方法的返回值、异常或传输 payload，也不额外同步设备；记录 JSONL 并通过正常 `torch.profiler.record_function` 添加 `wiki/<mode>/<role>/rankN/...` scope。观察器有开销，因此本目录不据此比较服务吞吐或优化收益。

请求身份经过同一非秘密盐的哈希；P/D 内部身份通过生产 metadata、`request_map` 和实际 `get_external_request_id` 调用建立显式映射。分析器不靠截字符串或按调用顺序猜同一请求。rankS 是 Scheduler，rankU 是尚未确认的 rank，均不改名为 rank0。

`/start_profile` 与 `/stop_profile` 控制已安装 Ascend profiler。它使用 CPU+NPU activities 与 Level1，保留四个 Worker 的原始采集。自定义 scope 是本次观察器添加的 Host 标记，不能认作 CANN 原生通信算子；后台线程 scope 是否进入导出文件，需要逐文件检查。

```sh
python analyze_lifecycle.py --root /private/experiment-root/pull \
  --clock-domain same-host-monotonic --output /private/pull-summary.json
```

只有确认同一主机、同一 boot 的进程记录，才允许用 `same-host-monotonic` 比较 Host 先后。NPU 多设备轨迹需要另外检查时钟元数据；独立缩放的截图不能凭左边缘对齐推断跨 rank 等待或 straggler。

分析保留分配、源延迟释放、TE 调用/字节量/返回码、DONE、D 状态转换、块释放调用以及缺失记录。比较逐层 P 的释放与最后写返回时，报告的是被观察调用区间，不把 `_free_blocks` 返回当成某块已经被另一请求覆写。

## Linux 查看器

可在 Linux 用官方 Perfetto `trace_processor_shell` 载入导出的 `trace_view.json`，通过只绑定 loopback 的 HTTP RPC 与浏览器 UI 查看原始轨迹。SSH 转发只用于本地阅读；结束后停止查看器与转发。

```sh
trace_processor_shell --httpd --http-ip-address 127.0.0.1 \
  --http-port 9001 /private/capture/ASCEND_PROFILER_OUTPUT/trace_view.json
```

先用下面的 SQL 确认有哪些观察 scope，再在真实时间线中选中与请求对应的区间。未捕获到某个后台 scope，应与 JSONL 和原生任务记录交叉核对，不能把图中的空白直接解释成等 KV。

```sql
SELECT name, ts, dur, track_id
FROM slice
WHERE name GLOB 'wiki/*'
ORDER BY ts;
```
