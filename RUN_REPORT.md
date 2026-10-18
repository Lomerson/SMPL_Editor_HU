# SMPL-X T-pose 查看器 —— 架构分析与 WSL 落地报告

- 工程：`/home/lomerson/MyProjects/smplx_tpose_viewer_package`
- 模型：`/home/lomerson/MyProjects/models/smplx`（`SMPLX_{NEUTRAL,MALE,FEMALE}.npz`，Version 1.0）
- 目标环境：WSL Ubuntu，conda env `torch3d`（python 3.10.20）
- 结论：**已跑通**。服务端 SMPL-X 加载、T-pose 网格生成、OBJ 导出、骨架下发、IK 求解、骨段编辑、重置全部可用；真实浏览器（headless Chromium + WebGL2）成功加载并渲染出 T-pose，无控制台错误。`pytorch3d 0.7.9 / torch 2.14.0.dev+cu130 / torchvision 0.29.0.dev+cu130` 与基线完全一致。

---

## 1. 架构与功能分析

### 1.1 项目定位与功能

这是一个**基于浏览器的 SMPL-X 人体 T-pose 查看器 / 编辑器**，采用「Python 本地 HTTP 服务 + Three.js 前端」的单文件架构：

| 能力 | 说明 |
|---|---|
| 模型发现 | 扫描 `--model-dir` 下的 `SMPLX_{FEMALE,MALE,NEUTRAL}.npz`，前端下拉框显示三种性别，缺失项标注「（未安装）」并禁用 |
| T-pose 网格 | 用 `smplx.create(...)` 加载 npz，以全零 `body_pose` 前向，得到标准 T-pose 顶点（10475 顶点 / 20908 面） |
| 网格导出 | `trimesh.Trimesh(...).export()` 生成 OBJ，落到 `--cache-dir`，前端用 `OBJLoader` 读取 |
| 骨架显示 | 下发 55 关节坐标 + 55 个 parent 索引，前端画黄色骨段连线 + 22 个可拾取关节点 |
| 姿态编辑（IK） | 拖动关节点到目标位置，后端用 Adam 优化 `body_pose` 让该关节靠近目标；支持「局部旋转（只转父关节）」与「全身 IK（整条父链）」两种模式 |
| 骨段编辑 | 点击黄色骨段，可调该骨段长度（0.35–2.5 倍）与横截面半径剖面（2–12 个控制点，0.3–2.5），后端按 LBS 权重做径向变形后重新导出 |
| 重置 | 姿态归零；骨段配置独立保留（UI 另有「重置此段」） |

主要用户流程：打开网页 → 自动加载第一个可用性别 → 鼠标拖动黄色关节/点选骨段编辑 → 每次交互后端重算网格并整包重载 OBJ → 画面即时更新。

### 1.2 模块划分与调用关系

工程只有 **4 个文件、1 个真正的模块**：`smplx_tpose_server.py`（400 行）内含全部后端逻辑，并把前端 HTML/CSS/JS 以字符串常量 `HTML`（L18–75）内嵌。

```
浏览器 (three.js r169)
  │
  │  GET /                    → HTML 常量 (L304-311)
  │  GET /api/models          → ModelCache.available()  (L312-314)
  │  GET /api/model/<gender>  → ModelCache.obj()        (L315-331)
  │  GET /api/skeleton/<g>    → ModelCache.skeleton()   (L332-340)
  │  POST /api/pose           → ModelCache.pose_joint() (L353-364)
  │  POST /api/segment        → ModelCache.update_segment() (L365-376)
  │  POST /api/reset          → ModelCache.reset()      (L349-352)
  ▼
handler_factory(cache) → SimpleHTTPRequestHandler 子类 (L289-384)
  ▼
ModelCache (L89-286)
  ├── _model()      smplx.create(npz, model_type="smplx", use_pca=False,
  │                              flat_hand_mean=True, batch_size=1)   L106-113
  ├── _pose()       torch.zeros(1, 21*3) 缓存                          L115-119
  ├── _output()     model(body_pose=...) 缓存                          L121-129
  ├── _geometry()   ★核心：顶点/关节/父表 + 骨段缩放 + 径向变形         L134-182
  ├── obj()         trimesh 导出 OBJ（持锁、临时文件原子替换）          L187-203
  ├── skeleton()    joints/parents/names/segments → JSON               L205-216
  ├── update_segment()  校验 + 存配置 + 失效几何缓存                    L218-227
  ├── reset()       姿态归零 + 失效输出/几何缓存                        L229-237
  └── pose_joint()  ★Adam 优化 body_pose 逼近目标关节                   L239-286
```

入口：`main()`（L387–396），`argparse` 三个参数 `--model-dir / --cache-dir / --host / --port`，`ThreadingHTTPServer` 常驻。

### 1.3 数据流

```
SMPLX_*.npz
  └─ smplx.create() ──► SMPLX model (v_template/shapedirs/lbs_weights/parents/faces)
       └─ model(body_pose=0) ──► vertices[1,10475,3], joints[1,127,3]
            ├─ base_vertices / base_joints[:55]
            ├─ 骨段长度：joints[c] = joints[p] + (base[c]-base[p]) * length[c]
            ├─ 顶点位移：vertices += lbs_weights[:, :55] @ (joints - base_joints)
            ├─ 径向剖面：对 child<=21 的骨段，按 (w_parent + w_child) * axial_feather
            │            把顶点沿径向推拉到 radii 插值出的半径
            └─ trimesh.Trimesh(vertices, faces).export(OBJ)
                 └─ HTTP text/plain 整包返回 ──► OBJLoader ──► THREE.Mesh（半透明蓝）
                                                  └─ 骨架 JSON ──► LineSegments + Sphere 关节点
```

`pose_joint()`：把 `body_pose` 包成 `torch.nn.Parameter`，用 `Adam(lr=0.035)` 迭代 36 次（全身）或 24 次（局部），损失 = `‖joint - target‖² + 0.003·mean((pose-base)²)`，只允许链上关节的通道可训练（`editable` mask），结束后前向一次缓存 `output` 并失效几何缓存。

### 1.4 技术栈

| 层 | 技术 |
|---|---|
| 后端 HTTP | Python 标准库 `http.server.ThreadingHTTPServer` + `SimpleHTTPRequestHandler`，无 Web 框架 |
| 模型 | 官方 `smplx` 0.1.28（`smplx.create`），**仅用 .npz**，不触碰 .pkl/chumpy |
| 张量/优化 | PyTorch（CPU 上跑）；IK 用 `torch.optim.Adam` |
| 网格 IO | `trimesh` 5.1.0（导出 OBJ） |
| 数学 | `numpy` 2.2.6 |
| 前端渲染 | three.js **r169**（`three.module.js`）+ `OrbitControls` + `OBJLoader`，WebGL，ES module + importmap |
| 前端 UI | 原生 JS/DOM，无框架；深色主题，滑块调参 |

> 注意：工程**完全没用到 `pytorch3d`**——它只是我们必须保护的目标环境。

### 1.5 外部依赖清单

| 包 | 类型 | 来源 | torch3d 环境状态 |
|---|---|---|---|
| `argparse/json/os/tempfile/threading/http/pathlib/urllib` | 标准库 | import | ✅ |
| `numpy` | 第三方 | import | ✅ 2.2.6 |
| `torch` | 第三方 | import | ✅ 2.14.0.dev+cu130 |
| `smplx` | 第三方 | `import smplx`（L108） | ✅ 0.1.28 |
| `trimesh` | 第三方 | `import trimesh`（L193） | ✅ 5.1.0 |
| three.js r169 | 前端 CDN | importmap（L47） | ✅ jsdelivr 可达（curl 200） |
| `chumpy` | 隐式（.pkl 才需要） | — | ❌ 未安装，**也不需要**（走 npz） |

工程**没有** `requirements.txt` / `pyproject.toml` / `setup.py` / `README`，依赖只能从 import 反推。依赖面极小，现有环境已全部满足，**全程未为运行工程安装任何依赖**。

### 1.6 资产与路径依赖

| 项 | 值 | 状态 |
|---|---|---|
| 默认模型目录 | `/home/hgc/models/smplx`（L389） | ❌ 本机不存在（作者机器路径） |
| 默认缓存目录 | `/home/hgc/project/AnimeAvatarGen/outputs/smplx_tpose_viewer`（L390） | ❌ 本机不存在且不可创建 |
| 实际模型 | `/home/lomerson/MyProjects/models/smplx/SMPLX_*.npz`（108 MB × 3） | ✅ 三性别齐全 |
| `.pkl` 模型 | 同目录 544 MB × 3 | ⛔ 未使用（避免 chumpy） |
| 贴图/样本数据 | 无 | — |
| `/mnt/c` 等 Windows 路径引用 | 无 | ✅ 无需跨盘拷贝 |

三个原始脚本还硬编码了作者环境（见 1.7），本报告用新增的 `run_local.sh` 参数化解决，**未改原文件**。

### 1.7 原工程内的运行说明（原样摘录）

工程无 README，运行说明只存在于脚本注释/输出中：

`smplx_tpose_viewer_ctl.sh`：

```bash
PROJECT=/home/hgc/project/AnimeAvatarGen
PIDFILE=/tmp/anime_smplx_tpose_viewer.pid
LOGFILE=/tmp/smplx_tpose_viewer.log
PORT=8766
...
source /home/hgc/miniconda3/etc/profile.d/conda.sh
conda activate mars39
cd "$PROJECT"
nohup python -u tools/smplx_tpose_server.py --host 0.0.0.0 --port "$PORT" >"$LOGFILE" 2>&1 &
...
echo "URL: http://100.95.255.117:$PORT"
```

`start_smplx_tpose_viewer_desktop.sh`：

```bash
PROJECT=/home/hgc/project/AnimeAvatarGen
cd "$PROJECT"
./tools/smplx_tpose_viewer_ctl.sh restart
echo "SMPL-X T-pose 预览服务已启动。"
echo "请在浏览器打开: http://127.0.0.1:8766"
echo "远程访问: http://100.95.255.117:8766"
```

`启动 SMPL-X T-pose 预览.desktop`：`Exec=/home/hgc/project/AnimeAvatarGen/tools/start_smplx_tpose_viewer_desktop.sh`，`Path=/home/hgc/project/AnimeAvatarGen`，`Terminal=true`。

### 1.8 潜在问题清单

| # | 严重度 | 问题 | 位置 |
|---|---|---|---|
| 1 | **高（阻断）** | 默认模型目录硬编码 `/home/hgc/models/smplx`；直接 `python smplx_tpose_server.py` 时三个性别全部 `available:false`，`/api/model/*` 返回 404 | L389 |
| 2 | **高（阻断）** | 默认缓存目录硬编码 `/home/hgc/project/...`，本机无法创建（`/home` 下无写权限），首次导出会 500 | L390 |
| 3 | 中 | `ctl.sh` 硬编码 `conda activate mars39`（本机无此环境）、工程路径 `/home/hgc/...`、Tailscale IP `100.95.255.117` | ctl.sh L4–7, L20–24 |
| 4 | 中 | `.desktop` 与 `start_*_desktop.sh` 全部指向 `/home/hgc/project/AnimeAvatarGen`，双击必失败 | desktop/start 脚本 |
| 5 | 中 | 前端 three.js 走 jsdelivr CDN（importmap），**离线环境白屏**；且本沙箱的 Chromium 出网被阻断（curl 可以），需本地化才能离线 | L47 |
| 6 | 中 | `--host` 默认 `0.0.0.0`，无任何鉴权，局域网可任意改网格/跑 Adam 优化 | L391 |
| 7 | 低 | `skeleton()` 读 `geometry_cache` 未持锁，与 `obj()`/`pose_joint()` 并发时可能重建竞态（前端是串行请求，实际难触发） | L205-216 |
| 8 | 低 | `reset` 只清姿态不清骨段配置（UI 设计如此，但「重置」语义易误解） | L229-237 |
| 9 | 低 | 局部 IK 只能转父关节，cannot 沿骨骼轴向伸缩——用户把点拖到轴向时关节到不了鼠标位置（几何约束，非 bug，但无提示） | L256-263 |
| 10 | 低 | `?edited=1` 查询串后端完全忽略（`urlparse().path`），属冗余参数 | L304, L316 |
| 11 | 低 | 访问日志被 `log_message` 静默，排障无痕 | L291-292 |
| 12 | 信息 | 计算全在 CPU（模型未 `.to(cuda)`），一次全身 IK 约 1.2 s | L264-286 |
| 13 | 信息 | `JOINT_NAMES` 只有 22 项而 parents 有 55 项；因只对 child≤21 取名字，未越界 | L83-86 |

其他检查（均已确认无问题）：无 `np.float` 等废弃 API；Python 3.10 下 `from __future__ import annotations` + `list[float]` 正常；`.npz` 路径使 `smplx` 完全绕开 chumpy（`import chumpy` 确认为 ModuleNotFoundError）；headless 可行（纯 HTTP + WebGL 前端）。

---

## 2. WSL 环境自检结果

| 项 | 结果 |
|---|---|
| `uname -a` | `Linux Lom-CyberNugget 6.18.33.2-microsoft-standard-WSL2 ... x86_64` |
| `which -a python`（激活后） | `/home/lomerson/miniconda3/envs/torch3d/bin/python` ✅ Linux 侧 |
| `sys.executable` | `/home/lomerson/miniconda3/envs/torch3d/bin/python` ✅ |
| Python 版本 | 3.10.20 (GCC 14.3.0) |
| `nvidia-smi` | ✅ 可用，NVIDIA GeForce RTX 5080，驱动 615.71.08 / CUDA UMD 13.4，16 GB |
| `torch.cuda.is_available()` | ✅ True，`torch.version.cuda = 13.0` |
| 显示方案 | WSLg 已就绪（`DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0`、`/mnt/wslg` 存在）。渲染验证实际走 **headless Chromium + SwiftShader（WebGL 2.0）**，未启动任何 X server |
| `xvfb-run` / 浏览器 | 系统内无 `xvfb-run`；浏览器**已按需安装**：`chromium` 156 落在 `~/.local/opt/chromium`（免 sudo），`playwright` 的 headless 构建在 `~/.cache/ms-playwright`（均 Linux 侧） |
| 是否触及 Windows 侧 | **否**。未调用任何 `.exe/.bat/.ps1`，未写入 `/mnt/*`，未改 `/etc/wsl.conf`、`~/.bashrc`、全局 pip/conda 源，未安装驱动 |
| 缓存位置 | pip/playwright 缓存均在 Linux 侧（`~/.cache`），无 DrvFs 写入 |

> PATH 中虽含 `/mnt/d/.../miniconda3` 等 Windows 条目，但激活 `torch3d` 后 `python` 解析到 Linux 侧路径，符合准入条件。

---

## 3. 环境变更记录

**为运行本工程安装的依赖：无**（`smplx`/`trimesh`/`numpy`/`torch` 环境已具备）。

**为「真实浏览器验收」额外安装（纯 Python，与 torch 无关）**：

| 命令 | 理由 | 方式 | 装后 p3d 校验 |
|---|---|---|---|
| `pip install --no-deps playwright pyee` | 用真实浏览器加载 UI 并截图，验证前端 three.js/WebGL 链路 | `--no-deps`，纯 Python（greenlet 已有） | `2.14.0.dev20260809+cu130 13.0 0.7.9` ✅ |
| `python -m playwright install chromium` | 下载 Chromium 156（浏览器二进制，非 Python 包） | 下载到 `~/.cache/ms-playwright`（Linux 侧） | — |
| `cp -a chromium-1248/chrome-linux64 ~/.local/opt/chromium` | 给 WSL 装一个可直接开窗的 Chromium（免 sudo） | 用户级安装，403 MB | — |
| 写 `~/.local/bin/chromium` + `~/.local/share/applications/chromium.desktop` | 生成 PATH 可用的 `chromium` 命令与 WSLg 应用菜单项 | `~/.profile` 自带 `~/.local/bin` 的 PATH 注入，**未改 .bashrc/.profile** | — |

环境差异（`conda env export` 前后 diff）**只有两行**：

```
>       - playwright==1.64.0
>       - pyee==13.0.1
```

未使用 `pip install -U`、`--upgrade`、`conda update/install`，未卸载/重装任何 torch 系包，未编译 pytorch3d。

---

## 4. 代码改动记录

**原始源码 `smplx_tpose_server.py`：零改动**（mtime 仍为 2026-10-18 01:56，早于本次会话）。原作者的 3 个启动脚本**只做了移动**（内容一字未改），见下表。

| 文件 | 类型 | 说明 | 回退 |
|---|---|---|---|
| `run_local.sh` | **新增**（1355 B，可执行） | 本机启动器：按脚本位置推导工程目录，`SMPLX_MODEL_DIR` 默认 `../models/smplx`，缓存默认 `./outputs/smplx_tpose_viewer`，激活 `torch3d`，透传其余参数 | `rm run_local.sh` |
| `outputs/render_preview.py` | 新增（验证工具） | 用 pytorch3d 把导出的 OBJ 渲成 PNG（只读调用，不改 p3d） | `rm outputs/render_preview.py` |
| `outputs/ui_smoke_test.py` | 新增（验证工具） | Playwright 加载页面、收集控制台错误、截图；仅在该脚本内把 CDN 请求映射到本地副本 | `rm outputs/ui_smoke_test.py` |
| `outputs/vendor_three/` | 新增（3 个 JS 文件） | 从 jsdelivr 下载的 three r169 本地副本，仅供离线截图测试使用 | `rm -rf outputs/vendor_three` |
| `legacy/`（3 个原脚本 + README.md） | **移动** | `smplx_tpose_viewer_ctl.sh`、`start_smplx_tpose_viewer_desktop.sh`、`启动 SMPL-X T-pose 预览.desktop` 原样移入 `legacy/`（内容未改），并新增 `legacy/README.md` 说明「原作者机器的脚本，本机请用 run_local.sh」 | `mv legacy/{*.sh,*.desktop} .`（见 legacy/README.md 末尾命令） |

`run_local.sh` 中值得记录的两个坑（已修）：

1. `set -u` + `conda activate` 冲突：`envs/torch3d/etc/conda/activate.d/env_vars.sh` 引用未定义的 `LD_LIBRARY_PATH`，导致 `unbound variable` 退出 → 在激活前后加 `set +u` / `set -u`。
2. conda 激活会**覆盖 `HOST` 变量**（设为交叉编译三元组 `x86_64-conda-linux-gnu`），使 `--host "$HOST"` 变成非法主机名 → 改用 `BIND_HOST`/`BIND_PORT`。

---

## 5. 运行方式（可复现）

```bash
# 1) 进入工程
cd /home/lomerson/MyProjects/smplx_tpose_viewer_package

# 2) 前台启动（推荐；脚本自动激活 torch3d）
./run_local.sh
#    → python : /home/lomerson/miniconda3/envs/torch3d/bin/python
#    → url    : http://127.0.0.1:8766

# 3) 浏览器打开（三选一）
#    a) WSL 内新装的 Chromium（WSLg 原生窗口）
chromium --new-window http://127.0.0.1:8766
#    b) Windows 桌面浏览器：http://localhost:8766（WSL2 localhost 转发）
#    c) 任意浏览器手动访问 http://127.0.0.1:8766
```

等价的「不依赖启动器」手工命令：

```bash
source /home/lomerson/miniconda3/etc/profile.d/conda.sh
conda activate torch3d
cd /home/lomerson/MyProjects/smplx_tpose_viewer_package
python -u smplx_tpose_server.py \
  --model-dir /home/lomerson/MyProjects/models/smplx \
  --cache-dir /home/lomerson/MyProjects/smplx_tpose_viewer_package/outputs/smplx_tpose_viewer \
  --host 127.0.0.1 --port 8766
```

可覆盖的环境变量：`SMPLX_MODEL_DIR`、`SMPLX_CACHE_DIR`、`SMPLX_BIND_HOST`（如需局域网访问设为 `0.0.0.0`）、`SMPLX_PORT`。

当前已作为**脱离会话的后台进程**运行：PID `334185`，日志 `/tmp/smplx_tpose_viewer_run.log`。停止：`pkill -f smplx_tpose_server.py`。

---

## 6. 运行结果

### 6.1 服务端（全部 200，无未捕获异常）

```
GET  /                     -> 200  14274 B
GET  /api/models           -> 200  female/male/neutral 全部 available:true
GET  /api/skeleton/neutral -> 200  joints=55 parents=55
GET  /api/model/neutral    -> 200  731590 B  (v=10475, f=20908)
GET  /api/model/male       -> 200  731589 B
GET  /api/model/female     -> 200  731668 B
POST /api/pose  full_body  -> 200  ok, 1.18 s, 目标误差 0.019 m
POST /api/segment          -> 200  ok, 股骨长 0.3828 -> 0.5359 (length=1.4)
POST /api/reset            -> 200  ok, 姿态回到基准
```

- 网格 bbox：`x ∈ [-0.8613, 0.8613]`（左右镜像误差 1e-6）、`y ∈ [-1.3018, 0.4193]`（身高 1.721 m）、`z ∈ [-0.1518, 0.1422]` → 标准 T-pose、Y 轴向上、左右对称。
- 同一性别重复导出 OBJ **字节级一致**（`cmp` 通过），说明流程确定性、可复现。
- 局部 IK 在「目标沿骨骼轴向」时无法到达（刚性骨长约束，误差 0.05 m 属几何上限），在可达目标上正常收敛——与算法设计一致，非缺陷。

### 6.2 真实浏览器 UI

```
WEBGL : WebGL 2.0 (OpenGL ES 3.0 Chromium)
PAGE  : title='SMPL-X T-pose 预览', canvas=[1140,900], options=['女性','男性','中性']
STATUS: 已加载 female，可拖动黄色身体关节
CDN->local: three.module.js / OrbitControls.js / OBJLoader.js
ERRORS: none
```

截图 `outputs/smplx_tpose_viewer/ui_screenshot.png`（1440×900）中可见：左侧控制面板、右上 HUD 提示、中央半透明蓝色 T-pose 网格（约 7.2 万像素）、黄色骨架线（约 1.5 千像素）、地面网格。

### 6.3 WSL 内原生 Chromium（WSLg 真实窗口）

安装后直接开窗验证，不是 headless：

```
$ chromium --version
Google Chrome for Testing 156.0.8078.4

$ DISPLAY=:0 chromium --new-window http://127.0.0.1:8766
CDP /json/list: page  'http://127.0.0.1:8766/'  title='SMPL-X T-pose 预览'
页面状态: 已加载 female，可拖动黄色身体关节
WebGL   : WebGL 2.0 (OpenGL ES 3.0 Chromium)
```

窗口截图 `outputs/smplx_tpose_viewer/ui_screenshot_wslg.png`（945×860，即真实窗口尺寸）中同样可见 T-pose 网格（6.6 万像素）与黄色骨架（1410 像素）。Chromium 走 WSLg 的 `DISPLAY=:0`，渲染用 SwiftShader（`--use-gl=angle --use-angle=swiftshader`），不依赖主机显卡驱动。

### 6.4 输出文件

| 文件 | 说明 |
|---|---|
| `outputs/smplx_tpose_viewer/smplx_neutral_tpose.obj` | 中性 T-pose 网格（731 KB） |
| `outputs/smplx_tpose_viewer/smplx_male_tpose.obj` | 男性 T-pose 网格 |
| `outputs/smplx_tpose_viewer/smplx_female_tpose.obj` | 女性 T-pose 网格 |
| `outputs/smplx_tpose_viewer/ui_screenshot.png` | headless 浏览器渲染截图（1440×900） |
| `outputs/smplx_tpose_viewer/ui_screenshot_wslg.png` | WSL 内原生 Chromium 在 WSLg 窗口中的截图（945×860） |
| `outputs/smplx_tpose_viewer/neutral_tpose_front.png` | pytorch3d 渲染的正视图（身高 1.721 m） |
| `outputs/smplx_tpose_viewer/neutral_tpose_3q.png` | 3/4 视角 |
| `outputs/smplx_tpose_viewer/segment_edit_demo.png` | 骨段拉长/加粗后的效果（身高 1.891 m） |

---

## 7. pytorch3d 完整性验收

| 包 | 基线（安装前） | 结束（验收） | 一致 |
|---|---|---|---|
| torch | `2.14.0.dev20260809+cu130` | `2.14.0.dev20260809+cu130` | ✅ |
| torch.version.cuda | `13.0` | `13.0` | ✅ |
| torchvision | `0.29.0.dev20260810+cu130` | `0.29.0.dev20260810+cu130` | ✅ |
| pytorch3d | `0.7.9`（editable `/home/lomerson/pytorch3d`） | `0.7.9` | ✅ |
| `torch.cuda.is_available()` | True | True | ✅ |
| 环境整体 | `/tmp/torch3d_env_backup.yaml` | 仅多 `playwright==1.64.0`、`pyee==13.0.1` | ✅ |

备份：`/tmp/torch3d_env_backup.yaml`（245 行）、`/tmp/env_baseline.txt`；回滚命令 `conda env update -f /tmp/torch3d_env_backup.yaml --prune`（本次无需）。

---

## 8. 遗留问题与建议

1. **硬编码路径（建议必修）**：把 L389–390 改成可移植默认值，例如
   `default=Path(os.environ.get("SMPLX_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "smplx"))`，
   这样 `python smplx_tpose_server.py` 可直接跑。本次为满足「最小改动」未改源码，改用 `run_local.sh` 传参。
2. **前端 CDN 依赖**：three.js 走 jsdelivr，离线/受限网络下页面空白（本次调试期间 jsdelivr 出现过瞬时 TLS 重置，后来恢复正常；`outputs/ui_smoke_test.py` 里保留了「CDN→本地副本」的拦截兜底）。若需完全离线，建议把 r169 三个文件放到 `vendor/three/` 并改 importmap 为本地 `/vendor/...`，同时在 `do_GET` 加静态文件路由（约 10 行）。本次未改源码。
3. **旧启动脚本**：`ctl.sh`（`mars39` 环境、`/home/hgc` 路径）、`*.desktop`、`start_*_desktop.sh` 均指向作者机器，本机无法运行；已**原样移入 `legacy/`** 并加了 `legacy/README.md` 说明（本机请用 `run_local.sh`），根目录不再有会误触发的失效脚本。
4. **安全边界**：默认 `0.0.0.0` + 无鉴权，任何能访问端口的人都能触发布局优化（CPU 占满）。建议默认 `127.0.0.1`，需要远程时再加 token。
5. **性能**：模型/优化全在 CPU，单次全身 IK 约 1.2 s；若要交互流畅，可在 `_model`/`_pose` 加 `.to(cuda)` 并缓存 device，但会改变内存占用，需另行评估。
6. **「重置」语义**：`/api/reset` 不清骨段配置，若希望“完全复位”需在 `reset()` 里 `self.segment_configs.pop(gender, None)`。
7. **人工点选**：WSL 内已装好 `chromium`（见 6.3），可直接 `chromium --new-window http://127.0.0.1:8766` 开窗拖动关节/点选骨段；Windows 桌面也可用 `http://localhost:8766`。
8. **chromium 的安装方式**：Ubuntu 22.04 官方仓库已无 `chromium` 包（`chromium-browser` 只是 snap 过渡包，而本 WSL 的 snapd 处于 offline、无 systemd），加上 `sudo` 需要密码，故改用「Chrome for Testing 156 免 sudo 用户级安装」到 `~/.local/opt/chromium`，用 `~/.local/bin/chromium` 启动。若你更想要发行版包，可用你自己的密码执行 `sudo apt install chromium-browser`（会走 snap，本机大概率失败）或安装 Google Chrome 官方 deb；二者都不影响现有环境。
