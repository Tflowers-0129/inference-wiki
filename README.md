# LLM 推理调度与执行 Wiki

[在线演示](https://tflowers-0129.github.io/inference-wiki/) · [混部播放器](https://tflowers-0129.github.io/inference-wiki/#scheduler) · [PD 播放器](https://tflowers-0129.github.io/inference-wiki/#pdPlayer) · [一次前向](https://tflowers-0129.github.io/inference-wiki/execution.html) · [投机解码实验室](https://tflowers-0129.github.io/inference-wiki/speculation.html#spec-player)

[![Pages deployment](https://github.com/Tflowers-0129/inference-wiki/actions/workflows/pages.yml/badge.svg)](https://github.com/Tflowers-0129/inference-wiki/actions/workflows/pages.yml)

中文教学网页，沿着“混部调度 → 逐轮选图 → 输出长短与稳态测试 → PD 分离 → 抢占与 KV Cache”的顺序展开。包含混部调度播放器、三种 PD 交接场景、11 张可编辑 SVG 图解和自测。[图执行说明](https://tflowers-0129.github.io/inference-wiki/#graph-execution) 区分 Decode 比例、本轮形状和捕获档位，并说明显式教学配置与实际后端的适用边界。

网页不依赖前端框架、外部字体或脚本。克隆仓库后即可在本地预览；运行时只需要 Python 3。模型测试另需 Node.js 18 或更新版本。

新增两篇由 Markdown 生成的图解网页：从调度到一次前向（输入组织、metadata、算子与访存、KV 地址、图提交与 Trace），以及投机解码（候选验证、KV 有效范围、成组输出与 ITL/TPOT）。投机实验室可逐步播放接受/回退，并比较逐 Token、成组与代理缓冲三种发送场景。网页不展示具体引擎发行编号。

## 本地预览

先克隆仓库：

```sh
git clone https://github.com/Tflowers-0129/inference-wiki.git
cd inference-wiki
```

在仓库目录运行：

```sh
python3 scripts/serve.py
```

该命令会先同步正文及图源下载，再把 `dist/` 作为静态站点启动。打开 [Wiki 首页](http://127.0.0.1:4173/)，按 `Ctrl+C` 停止服务。端口被占用时：

```sh
python3 scripts/serve.py --port 4174
```

不需要安装 npm 包，也不需要启动其他项目。

## 现场演示

原有播放器与新增实验室可直接定位：

- [混部调度播放器](http://127.0.0.1:4173/#scheduler)：先逐轮走过前四轮，观察 Decode 如何占用预算、Prefill 如何切块；再切换 64 / 1024 输出，对比并发与排空过程。
- [PD 交接播放器](http://127.0.0.1:4173/#pdPlayer)：先看“正常交接”，观察 R1 正在 Decode 时 R2 可以先传完 KV；再切换“KV 传输较慢”和“D 忙且目标 KV 不足”。

这些动画是教学事件和预算模型。事件间距不表示实测时间，动画不能预测目标硬件上的吞吐或时延。完整假设与待验证问题在正文及网页中。

## 修改正文、图和网页

- `content/LLM推理调度与执行.md`：完整正文，图源路径为 `../dist/assets/`。
- `content/从一轮调度到一次前向.md`、`content/投机解码与流式指标.md`：新增正文。
- `scripts/build-chapters.py`：用标准库渲染这两篇受控 Markdown；修改正文后运行同步命令即可更新网页。
- `dist/spec-model.js`、`dist/spec-app.js`、`dist/spec.css`：投机验证与流式指标实验室。
- `dist/chapter.js`、`dist/chapters.css`：新篇章的目录、搜索与排版。
- `scripts/spec-player.html`：投机交互面板源片段。
- `dist/assets/*.svg`：11 张独立图源与 favicon，可直接编辑。
- `dist/index.html`、`dist/style.css`：网页讲解、布局与样式。
- `dist/model.js`、`dist/app.js`：混部调度模型与播放器。
- `dist/pd-model.js`、`dist/pd-app.js`：PD 交接模型与播放器。

修改正文或图源后运行：

```sh
python3 scripts/sync-content.py
```

脚本默认读取本仓库的 `content/`，生成 `dist/source.md`、`dist/execution.md`、`dist/speculation.md` 和 `dist/technical-wiki-source.zip`，供页面上的下载链接使用。这些下载生成文件由 `.gitignore` 排除，部署前会重新生成。压缩包正文引用 `figures/`，解压后可直接连图阅读。也可使用 `--manuscript /绝对路径/文稿.md`，图源仍需保存在本仓库 `dist/assets/`。

同步脚本会从新增的两篇 Markdown 重建 `execution.html` 与 `speculation.html`。这两份生成网页随仓库提交便于直接浏览，应修改 Markdown 和模板后重建；原有 `index.html` 仍按手工图文维护。渲染支持本文使用的段落、标题、平面列表、表格、代码块和 SVG 图片，不是任意 Markdown 的通用转换器。涉及模型假设时也需检查播放器。

## 检查与测试

```sh
python3 scripts/sync-content.py
python3 scripts/check-site.py
node --test tests/model.test.cjs tests/pd-model.test.cjs
```

模型共 22 项测试，覆盖 token 记账、每轮预算与序列上限，以及 PD 交接中的 KV 就绪、传输、源引用保留和释放时序。站点检查覆盖三篇网页的资源、跨页锚点、播放器引用、三篇下载正文和可编辑源包，并防止具体引擎发行号与提交标识混入公开图文。

## GitHub Pages

GitHub Pages 的部署目录是 `dist/`。仓库中的工作流负责同步正文、检查站点、运行模型测试，再把静态文件部署到 Pages。需要在 GitHub 仓库的 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。

工作流成功后，实际预览地址以仓库 **Actions** 和 **Settings → Pages** 显示的地址为准。资源使用相对路径，支持 GitHub Pages 的仓库子路径。线上演示时，在该地址末尾加上 `#scheduler` 或 `#pdPlayer` 即可直达播放器。

`dist/` 内的 HTML、CSS、JavaScript 和 SVG 是本站可编辑源码，应提交到仓库；正文与播放器一起维护。

## 更新线上演示

修改、检查后推送到 `main`：

```sh
python3 scripts/sync-content.py
python3 scripts/check-site.py
node --test tests/*.test.cjs
git add .
git commit -m "Update inference wiki"
git push origin main
```

`Deploy Wiki to GitHub Pages` 会自动运行并更新同一个演示地址；可在 [Actions](https://github.com/Tflowers-0129/inference-wiki/actions) 查看结果。PR 只运行检查，不会覆盖线上演示。
