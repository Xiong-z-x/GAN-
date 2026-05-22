# AutoDL Remote-SSH 与 Codex 远程连接经验

本文记录旧 AutoDL 实例 `connect.westc.seetacloud.com:34067` 的排障经验。旧实例的本机 SSH 配置、专用密钥、反向代理守护脚本和启动项已经清理；本文只保留可复用方法。

## 1. 推荐连接架构

新 AutoDL 实例不要把普通工作区连接和代理隧道混在同一个 SSH host 里。建议拆成两个角色：

- `autodl-gan`：VS Code / Cursor Remote-SSH 打开的工作区连接，不配置 `RemoteForward`。
- `autodl-gan-proxy`：后台反向代理隧道，只用于把 AutoDL 远端端口转发回本机代理。

示例：

```sshconfig
Host autodl-gan
    HostName <AutoDL连接域名>
    User root
    Port <AutoDL端口>
    IdentityFile C:/Users/熊振兴/.ssh/<新实例私钥>
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 6
    TCPKeepAlive yes
    ConnectionAttempts 3
    StrictHostKeyChecking accept-new

Host autodl-gan-proxy
    HostName <AutoDL连接域名>
    User root
    Port <AutoDL端口>
    IdentityFile C:/Users/熊振兴/.ssh/<新实例私钥>
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
    TCPKeepAlive yes
    ConnectionAttempts 3
    StrictHostKeyChecking accept-new
    RemoteForward 127.0.0.1:17892 127.0.0.1:7897
    ExitOnForwardFailure yes
```

这里 `7897` 是本机代理监听端口；如果以后本机代理端口变化，必须先确认再改，不要凭记忆替换。

## 2. 本机代理隧道启动

先确认本机代理端口可用：

```powershell
Test-NetConnection 127.0.0.1 -Port 7897
```

后台启动反向隧道：

```powershell
Start-Process -FilePath "$env:WINDIR\System32\OpenSSH\ssh.exe" -ArgumentList @("-N", "autodl-gan-proxy") -WindowStyle Hidden
```

检查：

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'ssh.exe'" |
  Where-Object { $_.CommandLine -match 'autodl-gan-proxy' } |
  Select-Object ProcessId,CommandLine
```

## 3. 远端 VS Code / Cursor 代理设置

在 AutoDL 远端执行：

```bash
mkdir -p ~/.vscode-server/data/User ~/.vscode-server/data/Machine
mkdir -p ~/.cursor-server/data/User ~/.cursor-server/data/Machine

cat > ~/.vscode-server/server-env-setup <<'EOF'
export HTTP_PROXY=http://127.0.0.1:17892
export HTTPS_PROXY=http://127.0.0.1:17892
export ALL_PROXY=http://127.0.0.1:17892
export http_proxy=http://127.0.0.1:17892
export https_proxy=http://127.0.0.1:17892
export all_proxy=http://127.0.0.1:17892
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
EOF

cp ~/.vscode-server/server-env-setup ~/.cursor-server/server-env-setup
chmod +x ~/.vscode-server/server-env-setup ~/.cursor-server/server-env-setup
```

然后写入 VS Code / Cursor server settings：

```bash
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path

files = [
    Path('/root/.vscode-server/data/User/settings.json'),
    Path('/root/.vscode-server/data/Machine/settings.json'),
    Path('/root/.cursor-server/data/User/settings.json'),
    Path('/root/.cursor-server/data/Machine/settings.json'),
]

for path in files:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
    else:
        data = {}

    data['http.proxy'] = 'http://127.0.0.1:17892'
    data['http.proxySupport'] = 'override'
    data['http.noProxy'] = ['127.0.0.1', 'localhost']
    data['extensions.supportNodeGlobalNavigator'] = True

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'updated {path}')
PY
```

`extensions.supportNodeGlobalNavigator = true` 很重要。旧实例里 Codex 扩展曾因为远端 extensionHost 缺少 `--supportGlobalNavigator` 触发：

```text
PendingMigrationError: navigator is now a global in nodejs
```

## 4. 健康检查

普通工作区连接必须不带 `RemoteForward`：

```powershell
ssh -G autodl-gan | Select-String -Pattern '^(hostname|user|port|identityfile|remoteforward|serveraliveinterval|serveralivecountmax)'
```

代理隧道连接必须带 `RemoteForward`：

```powershell
ssh -G autodl-gan-proxy | Select-String -Pattern '^(hostname|user|port|identityfile|remoteforward|exitonforwardfailure)'
```

远端检查：

```bash
pgrep -af "codex app-server|extensionHost"
env | grep -i proxy | sort
curl -I --max-time 20 https://chatgpt.com/backend-api/codex/responses
```

健康信号：

```text
extensionHost ... --supportGlobalNavigator
HTTP/1.1 200 Connection established
HTTP/2 405
allow: POST
```

`HTTP/2 405` 在这里不是失败；它说明代理链路到达了 Codex 后端，只是 `curl -I` 用了 HEAD 方法，而该接口要求 POST。

设备码接口也可以验证：

```bash
curl -i --max-time 20 \
  -X POST \
  https://auth.openai.com/api/accounts/deviceauth/usercode \
  -H 'content-type: application/json' \
  --data '{}'
```

健康信号是 `HTTP/2 200` 并返回 `device_auth_id` 和 `user_code`。

## 5. 旧实例清理清单

旧实例不用时，本机应清理：

- `~/.ssh/config` 里的旧 `Host autodl-gan`、`Host autodl-gan-base`、`Host autodl-gan-proxy`。
- 旧实例专用密钥，例如 `autodl_gan_ed25519` 和 `.pub`。
- 旧反向代理脚本，例如 `autodl-gan-proxy-watchdog.ps1`、`.vbs`、`.log`、`autodl-gan-proxy-keepalive.ps1`。
- Windows 启动目录里的 `AutoDL-GAN-ProxyTunnel.vbs`。
- 旧计划任务 `AutoDL-GAN-ProxyTunnel`。
- `known_hosts` 里旧实例端口记录，例如 `[connect.westc.seetacloud.com]:34067`。

清理后检查：

```powershell
$sshDir = Join-Path $env:USERPROFILE '.ssh'

Get-CimInstance Win32_Process |
  Where-Object {
    ($_.Name -in @('ssh.exe','powershell.exe','wscript.exe')) -and
    ($_.CommandLine -match 'autodl-gan|autodl_gan|AutoDL-GAN-ProxyTunnel')
  } |
  Select-Object ProcessId,Name,CommandLine

Get-ChildItem -LiteralPath $sshDir -Force |
  Where-Object { $_.Name -match 'autodl_gan|autodl-gan-proxy|AutoDL-GAN-ProxyTunnel' }

Select-String -Path "$sshDir\config" -Pattern 'autodl-gan|autodl_gan|17892|34067'
```

这些检查应无残留。

## 6. 易错点

1. 不要把 `RemoteForward` 放到 `Host` 块外面。否则它会变成全局配置，普通 `autodl-gan` 连接也会抢占远端端口。
2. 不要用 `autodl-gan-proxy` 打开 VS Code 工作区。它只负责后台隧道。
3. 不要只看 `env | grep proxy`。Codex app-server 的进程环境也要看：

```bash
for pid in $(pgrep -f "codex app-server" 2>/dev/null); do
  echo "PID=$pid"
  tr '\0' '\n' < /proc/$pid/environ |
    grep -Ei 'HTTP_PROXY|HTTPS_PROXY|ALL_PROXY|NO_PROXY|http_proxy|https_proxy|all_proxy|no_proxy' |
    sort
done
```

4. 如果聊天中出现 `Connection refused (os error 111)`，优先查远端 `127.0.0.1:17892` 是否打开，以及本机反向隧道是否还活着。
5. 如果登录阶段失败，分别检查 `auth.openai.com/oauth/token` 和 `auth.openai.com/api/accounts/deviceauth/usercode`，不要只看 ChatGPT 主站首页。
