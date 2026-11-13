# setup_windows.ps1 - 一键环境准备(幂等): venv / pip / mkcert / 证书 / 防火墙 / 二维码
# 用法: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
#       [-TorchIndexUrl <url>]           # CUDA 版 PyTorch 安装源(可选;见 README"CUDA >= 12.8"要求)
#       [-SkipVenv] [-SkipFirewall]

param(
    [string]$TorchIndexUrl = "",
    [switch]$SkipVenv,
    [switch]$SkipFirewall
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== 胶囊计数系统 一键环境准备 ===" -ForegroundColor Cyan

# ---- 1. 检查 Python >= 3.10 ----
$pyVer = $null
try {
    $out = & py -3 --version 2>&1
    if ($out -match "(\d+)\.(\d+)") { $pyVer = [version]"$($Matches[1]).$($Matches[2])" }
} catch {}
if (-not $pyVer -or $pyVer.Major -lt 3 -or ($pyVer.Major -eq 3 -and $pyVer.Minor -lt 10)) {
    Write-Host "[ERROR] 未找到 Python >= 3.10 (py -3 --version: $out)" -ForegroundColor Red
    Write-Host "请从 https://www.python.org/downloads/ 安装 Python 3.10+,勾选 'Add to PATH',"
    Write-Host "或从 Microsoft Store 安装。安装完成后重新运行本脚本。"
    exit 1
}
Write-Host "[OK] Python $pyVer"

# ---- 2. 创建 venv 并安装依赖 ----
if (-not $SkipVenv) {
    if (-not (Test-Path ".venv")) {
        Write-Host "创建虚拟环境 .venv ..."
        & py -3 -m venv .venv
    }
    $pip = ".\.venv\Scripts\python.exe" -replace "\.\\", ".\"
    Write-Host "安装基础依赖 ..."
    & .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check --quiet -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] pip 安装失败" -ForegroundColor Red; exit 1 }
}

# ---- 2.5 NVIDIA 显卡检查 (v1.2 §2.3) ----
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "[OK] 检测到 NVIDIA 显卡:"
    & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    Write-Host "提示: RTX 50 系需要较新驱动,请从 NVIDIA 官网更新到最新 Game Ready / Studio 驱动。"

    # CUDA 版 PyTorch(可选,不硬编码版本号)
    if ($TorchIndexUrl -ne "") {
        Write-Host "安装 CUDA 版 PyTorch: $TorchIndexUrl"
        & .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url $TorchIndexUrl
        if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] PyTorch 安装失败" -ForegroundColor Red; exit 1 }
        & .\.venv\Scripts\python.exe -m pip install --quiet ultralytics
    } else {
        Write-Host "提示: 未指定 -TorchIndexUrl,未安装 CUDA 版 PyTorch。"
        Write-Host "      GPU 加速请到 PyTorch 官网 'Get Started' 选 Windows/Pip/CUDA(版本须 >= 12.8,"
        Write-Host "      RTX 50 系(Blackwell/sm_120)必须 >= 12.8),复制安装命令后重新运行本脚本并传入:"
        Write-Host "      -TorchIndexUrl <复制的 --index-url 地址>"
    }
} else {
    Write-Host "[INFO] 未检测到 nvidia-smi(无 NVIDIA 显卡或驱动未装),将使用 CPU 档位"
}

# ---- 3. mkcert ----
if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 mkcert,尝试 winget 安装 ..."
    winget install FiloSottile.mkcert
    if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
        Write-Host "[WARN] winget 安装 mkcert 失败。请手动安装:" -ForegroundColor Yellow
        Write-Host "  https://github.com/FiloSottile/mkcert/releases 下载 mkcert.exe 放入 PATH"
        Write-Host "然后运行: scripts\gen_cert.ps1"
        exit 1
    }
}

# ---- 4/5. 证书 ----
& powershell -ExecutionPolicy Bypass -File "scripts\gen_cert.ps1"
if ($LASTEXITCODE -ne 0) { exit 1 }

# ---- 6. 防火墙规则 ----
if (-not $SkipFirewall) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
              ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if ($admin) {
        netsh advfirewall firewall delete rule name="capsule-tracker-8443" | Out-Null
        netsh advfirewall firewall add rule name="capsule-tracker-8443" dir=in action=allow protocol=TCP localport=8443 | Out-Null
        netsh advfirewall firewall add rule name="capsule-tracker-8080" dir=in action=allow protocol=TCP localport=8080 | Out-Null
        Write-Host "[OK] 防火墙规则已添加 (8443/8080)"
    } else {
        Write-Host "[WARN] 无管理员权限,请手动执行以下命令(或以管理员重跑本脚本):" -ForegroundColor Yellow
        Write-Host '  netsh advfirewall firewall add rule name="capsule-tracker-8443" dir=in action=allow protocol=TCP localport=8443'
        Write-Host '  netsh advfirewall firewall add rule name="capsule-tracker-8080" dir=in action=allow protocol=TCP localport=8080'
    }
}

# ---- 7. 打印访问地址与二维码 ----
$ip = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "127.0.0.1" -and
                   $_.InterfaceAlias -notmatch "vEthernet|Loopback|Virtual|WSL|Docker" } |
    Select-Object -First 1).IPAddress
if ($ip) {
    $url = "https://$ip`:8443"
    Write-Host ""
    Write-Host "手机(Safari)访问: $url" -ForegroundColor Green
    Write-Host "证书下载:         http://${ip}:8080/cert/rootCA.pem"
    Write-Host ""
    try {
        & .\.venv\Scripts\python.exe -c "import qrcode; qrcode.print_ascii('$url', small=True)"
    } catch {
        Write-Host "(二维码生成失败,不影响使用;请手动输入地址)"
    }
}
Write-Host ""
Write-Host "启动服务: .\.venv\Scripts\python.exe server.py"
