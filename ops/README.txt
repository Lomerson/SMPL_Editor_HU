SMPL-X 一周实验托管（Tailscale Funnel）

1. 参与者清单（在 WSL 内，不要提交到 git）
   推荐用生成器（会同时生成 admin token，权限 600）：
     python3 ops/make_participants.py 3 --labels P01,P02,P03 --force
   它会写入：
     ~/.config/smplx_experiment/participants.json
     ~/.config/smplx_experiment/admin_token
   登录只接受这里精确列出的 username/token，不会“随机输入都放行”，
   也不会把输入保存成新账号。生成器会打印每位受试者的用户名和 token。

2. 启动
   WSL 内手动：SMPLX_EXPERIMENT=1 SMPLX_ACCESS_FILE=... ./run_local.sh
   或从 Windows 运行（PowerShell）：
     powershell -ExecutionPolicy Bypass -File ops\start_viewer.ps1

3. 开启 Funnel（首次）
   & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel --bg 8766
   & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel status
   若提示 "Funnel is not enabled on your tailnet"，CLI 会打印类似：
     https://login.tailscale.com/f/funnel?node=nnz6RExFCt11CNTRL
   用 Owner 账号打开该链接并 Enable Funnel；HTTPS Certificates 只是前置条件，
   Funnel 是单独开关（管理台 DNS 页面）。
   公网地址：https://lom-cybernugget.taile0e0a9.ts.net

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

6.1 强制踢掉当前测试者（管理员）
   WSL:  ./ops/kick_tester.sh
   Windows: powershell -ExecutionPolicy Bypass -File ops\kick_tester.ps1
   它会读取 ~/.config/smplx_experiment/admin_token，调用本机
   POST /api/admin/reset，释放当前会话并清空所有 gender 的编辑历史/状态。
   也可以手动：
     curl -X POST http://127.0.0.1:8766/api/admin/reset \
       -H 'Content-Type: application/json' \
       -d "{\"token\":\"$(cat ~/.config/smplx_experiment/admin_token)\"}"

7. 如果 Funnel 无法启用（当前 tailnet 的 HTTPS/Funnel 未开启，且当前账号不是 Owner）：
   a) 在 https://console.tailscale.com/admin/dns 页面启用 Funnel
      （MagicDNS/HTTPS Certificates 只是前置条件；Funnel 是同一页面的单独开关）。
      若当前 tailnet 是 lomlomerson@gmail.com，你应当是 Owner，可以直接开启。
      然后运行：
      & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel --bg 8766
   b) 临时私网方案（受试者需安装 Tailscale 并被共享该节点）：
      & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" serve --bg --http=80 8766
      地址：http://lom-cybernugget.taile0e0a9.ts.net/
      实验登录 cookie 会根据浏览器 Origin 自动选择 Secure/非 Secure，HTTP Serve 可正常登录。
   c) 不要用 Quick Tunnel 作为一周实验主入口（URL 重启会变、无 SLA）。
