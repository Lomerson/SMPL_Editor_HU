#!/usr/bin/env bash
set -eu
PROJECT=/home/hgc/project/AnimeAvatarGen
cd "$PROJECT"
./tools/smplx_tpose_viewer_ctl.sh restart
echo
echo "SMPL-X T-pose 预览服务已启动。"
echo "请在浏览器打开: http://127.0.0.1:8766"
echo "远程访问: http://100.95.255.117:8766"
echo
echo "按 Enter 键关闭此窗口..."
read -r _
