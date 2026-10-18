# legacy/ —— 原作者机器的启动脚本（仅存档，本机不可用）

这里存放的是工程原作者在自己机器上使用的启动链路，**内容一字未改**，仅为保留可追溯性而移入本目录：

| 文件 | 原用途 | 为什么在本机跑不起来 |
|---|---|---|
| `smplx_tpose_viewer_ctl.sh` | `start/stop/restart/status` 管理服务进程 | 硬编码 `PROJECT=/home/hgc/project/AnimeAvatarGen`、`source /home/hgc/miniconda3/...`、`conda activate mars39`（本机无此环境，实际为 `torch3d`）、PID/日志写 `/tmp`、对外地址写死 Tailscale IP `100.95.255.117` |
| `start_smplx_tpose_viewer_desktop.sh` | 双击用的包装脚本，调用上面的 ctl.sh 并提示浏览器地址 | 同样 `cd /home/hgc/project/AnimeAvatarGen`，并调用 `./tools/smplx_tpose_viewer_ctl.sh` |
| `启动 SMPL-X T-pose 预览.desktop` | 桌面启动器 | `Exec` / `Path` 均指向 `/home/hgc/project/AnimeAvatarGen` |

三者还假设服务代码位于 `AnimeAvatarGen/tools/smplx_tpose_server.py`，而本包中它是同级的 `smplx_tpose_server.py`。

## 本机请改用

```bash
cd /home/lomerson/MyProjects/smplx_tpose_viewer_package
./run_local.sh          # 自动激活 torch3d、定位 ../models/smplx、缓存到 ./outputs
# 浏览器打开 http://127.0.0.1:8766
```

可用环境变量覆盖：`SMPLX_MODEL_DIR`、`SMPLX_CACHE_DIR`、`SMPLX_BIND_HOST`、`SMPLX_PORT`。

停止服务：`pkill -f smplx_tpose_server.py`。

## 需要恢复原状时

```bash
cd /home/lomerson/MyProjects/smplx_tpose_viewer_package
mv legacy/smplx_tpose_viewer_ctl.sh legacy/start_smplx_tpose_viewer_desktop.sh \
   "legacy/启动 SMPL-X T-pose 预览.desktop" .
```

（注意：原作者机器的 `mars39` 环境与 `/home/hgc/...` 路径在本机并不存在，恢复后依然无法直接运行。）
