SMPL-X 一周实验托管（Tailscale Funnel）

1. 参与者清单（在 WSL 内，不要提交到 git）
   mkdir -p ~/.config/smplx_experiment
   cp experiment/participants.example.json ~/.config/smplx_experiment/participants.json
   chmod 600 ~/.config/smplx_experiment/participants.json
   为每位受试者生成随机 token（至少 32 字符），填 username/token/label。
   可选：写入 ~/.config/smplx_experiment/admin_token（用于管理员强制释放会话）。

2. 启动
   WSL 内手动：SMPLX_EXPERIMENT=1 SMPLX_ACCESS_FILE=... ./run_local.sh
   或从 Windows 运行（PowerShell）：
     powershell -ExecutionPolicy Bypass -File ops\start_viewer.ps1

3. 开启 Funnel（首次）
   & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel --bg 8766
   & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel status
   若提示未启用，按输出的管理台链接启用 HTTPS Certificates/Funnel。
   公网地址：https://lom-cybernugget.tail17e361.ts.net

4. 常驻（管理员 PowerShell）
   powershell -ExecutionPolicy Bypass -File ops\install_tasks.ps1
   需要 Windows 自动登录，或使用 ONLOGON 任务；主机重启后验证 URL。

5. 电源/稳定性
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   关闭 Windows Update 自动重启（暂停一周）；Tailscale 管理台对该节点 Disable key expiry。

6. 验证
   手机 4G（关闭 Tailscale）打开公网 URL，登录，跑 Workflow 1/2，下载四视图 zip。
   未登录 /api/* 应为 401；实验模式 /api/autofit/path 应为 403。
   第二个用户名同时登录应得到 423。
