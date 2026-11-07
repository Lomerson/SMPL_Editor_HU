SMPL-X 一周实验托管（Tailscale Funnel）

1. 参与者清单（在 WSL 内，不要提交到 git）
   推荐用生成器（会同时生成 admin token，权限 600）：
     python3 ops/make_participants.py 3 --labels P01,P02,P03 --force
   它会写入：
     ~/.config/smplx_experiment/participants.json
     ~/.config/smplx_experiment/admin_token
   登录只接受这里精确列出的 username/token，不会“随机输入都放行”，
   也不会把输入保存成新账号。生成器会打印每位受试者的用户名和 token。

2. 启动（推荐）
   WSL:
     ./ops/start_experiment.sh
   Windows:
     powershell -ExecutionPolicy Bypass -File ops\start_experiment.ps1
   启动脚本会先执行 stop_experiment.sh 初始化，再启动 WSL tmux supervisor、
   检查旧 Windows 计划任务是否已删除、检查 Funnel，最后开启公网广播。
   不再使用会弹出 PowerShell 窗口的 Windows 计划任务。
   默认 5 个账号并发；可用环境变量覆盖：
     SMPLX_MAX_ACTIVE=5
     SMPLX_TORCH_THREADS=4
     SMPLX_HTTP_KEEPALIVE=1
     SMPLX_GZIP=1
     SMPLX_MAX_CONCURRENT_AUTOFIT=1

3. 开启 Funnel（首次）
   & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel --bg 8766
   & "D:\ForProgrammingOnly\Tailscale\tailscale.exe" funnel status
   若提示 "Funnel is not enabled on your tailnet"，CLI 会打印类似：
     https://login.tailscale.com/f/funnel?node=nnz6RExFCt11CNTRL
   用 Owner 账号打开该链接并 Enable Funnel；HTTPS Certificates 只是前置条件，
   Funnel 是单独开关（管理台 DNS 页面）。
   公网地址：https://lom-cybernugget.taile0e0a9.ts.net
   公网入口首页应返回登录页；/admin 是管理员页；/api/pics.zip 未登录应 401。

4. 常驻/停止
   实验模式由 WSL 内的 tmux supervisor 保活（server 崩溃后自动重启），
   不创建 Windows 计划任务，因此不会出现周期性 PowerShell 窗口。
   停止：
     ./ops/stop_experiment.sh
   它会删除旧计划任务、结束 tmux/server、重置 Serve/Funnel，但保留参与者数据。

5. 电源/稳定性
   powercfg /change standby-timeout-ac 0
   powercfg /change hibernate-timeout-ac 0
   关闭 Windows Update 自动重启（暂停一周）；Tailscale 管理台对该节点 Disable key expiry。

6. 验证
   手机 4G（关闭 Tailscale）打开公网 URL，登录，跑 Workflow 1/2，下载四视图 zip。
   未登录 /api/* 应为 401；实验模式 /api/autofit/path 应为 403。
   5 个不同账号可以同时登录；第 6 个账号应得到 423。
   每个账号的编辑状态、历史、导出目录互相独立。
   停止后再次检查：funnel/serve 都应为 No serve config，且 8766 不应有监听。

6.1 手机管理员（生成账号/踢人/重启）
   公网（Funnel）:
     https://lom-cybernugget.taile0e0a9.ts.net/admin
   或 tailnet HTTPS:
     https://lom-cybernugget.taile0e0a9.ts.net/admin
   在手机上输入 admin token（WSL 内）：
     cat ~/.config/smplx_experiment/admin_token
   管理页可以：
     - 查看当前最多 5 位测试者和参与者列表
     - 生成新账号（立即写入 participants.json，无需重启）
     - 单独踢掉某个测试者，或踢掉全部测试者
     - 重启服务（原地 re-exec，所有测试者会掉线）
   注意：admin token 权限很高，不要发到公开群；建议用密码管理器/私聊传递。

6.2 测试素材 pics.zip
   文件路径固定为项目根目录：
     /home/lomerson/MyProjects/smplx_tpose_viewer_package/pics.zip
   把图片打包成 zip 放在这里；不需要重启服务，路由每次请求都读当前文件。
   实验模式登录后会尝试自动下载一次到浏览器默认下载目录；侧栏“测试素材”
   也有按钮，Chromium 下可以弹出“另存为”选择位置。空 zip 或缺失时页面会提示。
   未登录直接访问 /api/pics.zip 返回 401。

6.3 登出即重置
   实验模式下 POST /api/logout 会立即清空当前账号所有 gender 的编辑状态和历史；
   其他 4 个账号不受影响；同一个账号重新登录也不会恢复上次的改动。

6.4 强制踢人（命令行）
   WSL 踢单个：./ops/kick_tester.sh u_ab12
   WSL 踢全部：./ops/kick_tester.sh
   Windows：powershell -ExecutionPolicy Bypass -File ops\kick_tester.ps1 [u_ab12]
   踢单个调用 POST /api/admin/kick，踢全部调用 POST /api/admin/reset。
   两者都读取 ~/.config/smplx_experiment/admin_token。

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
