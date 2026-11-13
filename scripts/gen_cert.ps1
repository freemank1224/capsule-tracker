# gen_cert.ps1 - 生成/更新 mkcert 证书(自动获取本机局域网 IP)
# 用法: powershell -ExecutionPolicy Bypass -File scripts\gen_cert.ps1

$ErrorActionPreference = "Stop"

if (-not (Get-Command mkcert -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 未找到 mkcert。请先运行 setup_windows.ps1 或:" -ForegroundColor Red
    Write-Host "  winget install FiloSottile.mkcert"
    exit 1
}

mkcert -install

# 自动获取局域网 IPv4(过滤虚拟网卡 / 169.254.x.x / 回环)
$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "169.254.*" -and
        $_.IPAddress -ne "127.0.0.1" -and
        $_.IPAddress -notlike "127.*" -and
        $_.InterfaceAlias -notmatch "vEthernet|Loopback|Virtual|WSL|Docker"
    } |
    Select-Object -ExpandProperty IPAddress

if (-not $ips) {
    Write-Host "[ERROR] 未检测到局域网 IPv4,请检查网络连接" -ForegroundColor Red
    exit 1
}

$ip = $ips | Select-Object -First 1
$hostname = $env:COMPUTERNAME

New-Item -ItemType Directory -Force -Path "certs" | Out-Null
Write-Host "使用 IP: $ip (主机名: $hostname)"
mkcert -cert-file certs\cert.pem -key-file certs\key.pem $ip $hostname localhost 127.0.0.1

# 复制根证书供手机下载 (/cert/rootCA.pem)
$caroot = (mkcert -CAROOT).Trim()
if (Test-Path "$caroot\rootCA.pem") {
    Copy-Item "$caroot\rootCA.pem" "certs\rootCA.pem" -Force
    Write-Host "[OK] 根证书已复制到 certs\rootCA.pem"
}

Write-Host ""
Write-Host "[OK] 证书生成完成。手机访问: https://$ip`:8443"
