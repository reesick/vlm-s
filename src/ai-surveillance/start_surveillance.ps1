# start_surveillance.ps1
# One-shot launcher for the KURUKSHETRA surveillance stack.
#
#   1. Sets up the Python virtualenv (created if missing).
#   2. Downloads model weights: YOLOv8n (detector), YOLOv8n-pose, Qwen2.5-VL-3B-Instruct (VLM).
#   3. Checks whether the Android Bitchat app is reachable (VLM API on port 8765).
#   4. Starts the pipeline: full detector stack by default, VLM+Bitchat mode with -VlmOnly.
#
# Usage:
#   .\start_surveillance.ps1                     # full stack
#   .\start_surveillance.ps1 -VlmOnly            # VLM scene descriptions to Bitchat only
#   .\start_surveillance.ps1 -CheckOnly          # model check + Bitchat connectivity only
#   .\start_surveillance.ps1 -AndroidIp 10.0.0.5 # override Bitchat IP (else read from pipeline.yaml)
#   .\start_surveillance.ps1 -SkipModels         # don't touch weight files

param(
    [string]$AndroidIp = "",
    [int]$BitchatPort = 8765,
    [switch]$VlmOnly,
    [switch]$CheckOnly,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

# --- helpers -----------------------------------------------------------------
function Read-BitchatConfig {
    # Pulls ip/port from the bitchat block of configs/pipeline.yaml (regex, no yaml dep).
    $yaml = Join-Path $PSScriptRoot "configs\pipeline.yaml"
    if (-not (Test-Path $yaml)) { return $null }
    $text = Get-Content -LiteralPath $yaml -Raw
    if ($text -match 'bitchat:\s*\n(?:.*\n)*?\s*ip:\s*"([^"]+)"') { $ip = $Matches[1] }
    if ($portMatch = [regex]::Match($text, 'bitchat:\s*\n(?:.*\n)*?\s*port:\s*(\d+)')) { $port = [int]$portMatch.Groups[1].Value }
    if ($ip) { return [pscustomobject]@{ ip = $ip; port = if ($port) { $port } else { 8765 } } }
    return $null
}

function Test-TcpPort {
    param([string]$Ip, [int]$Port, [int]$TimeoutMs = 2000)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($Ip, $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne($TimeoutMs) -and $client.Connected) { return $true } else { return $false }
    } catch { return $false } finally { $client.Close() }
}

# --- 1. virtualenv + dependencies --------------------------------------------
Write-Host "`n[1/4] Virtual environment" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $PSScriptRoot ".venv\Scripts\python.exe"))) {
    Write-Host "  creating .venv ..."
    python -m venv (Join-Path $PSScriptRoot ".venv")
}
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --quiet -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed - see output above" }
Write-Host "  environment ready." -ForegroundColor Green

# --- 2. model weights --------------------------------------------------------
Write-Host "`n[2/4] Model weights" -ForegroundColor Cyan
$modelDir = Join-Path $PSScriptRoot "models"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

if (-not $SkipModels) {
    $yoloSmall  = Join-Path $modelDir "yolov8n.pt"
    $yoloPose   = Join-Path $modelDir "yolov8n-pose.pt"

    if (-not (Test-Path $yoloSmall)) {
        Write-Host "  downloading YOLOv8n detector (~6 MB) ..."
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" -OutFile $yoloSmall
    } else { Write-Host "  yolov8n.pt present." }

    if (-not (Test-Path $yoloPose)) {
        Write-Host "  downloading YOLOv8n-pose (~7 MB) ..."
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt" -OutFile $yoloPose
    } else { Write-Host "  yolov8n-pose.pt present." }

    foreach ($optional in @("fire_smoke_yolov8n.pt", "smoking_yolov8n.pt")) {
        if (-not (Test-Path (Join-Path $modelDir $optional))) {
            Write-Host "  note: $optional not found (fine-tuned weights) - falls back to COCO/HSV heuristic." -ForegroundColor Yellow
        }
    }

    & $venvPython (Join-Path $PSScriptRoot "download_vlm.py")
    if ($LASTEXITCODE -ne 0) { throw "VLM download failed - see output above" }
} else {
    Write-Host "  skipped (-SkipModels)." -ForegroundColor Yellow
}

# --- 3. Bitchat connectivity -------------------------------------------------
Write-Host "`n[3/4] Bitchat connectivity check" -ForegroundColor Cyan
$target = $null
if ($AndroidIp) {
    $target = [pscustomobject]@{ ip = $AndroidIp; port = $BitchatPort }
} else {
    $cfg = Read-BitchatConfig
    if ($cfg) { $target = $cfg }
}

if ($target) {
    Write-Host ("  testing {0}:{1} (from pipeline.yaml) ..." -f $target.ip, $target.port)
    $reachable = Test-TcpPort -Ip $target.ip -Port $target.port
    if ($reachable) {
        Write-Host "  TCP: OPEN - Bitchat VLM API is reachable." -ForegroundColor Green
        try {
            $resp = Invoke-WebRequest -Uri ("http://{0}:{1}/status" -f $target.ip, $target.port) -TimeoutSec 3 -UseBasicParsing
            Write-Host ("  HTTP /status: {0}" -f $resp.StatusCode) -ForegroundColor Green
        } catch {
            Write-Host "  HTTP /status: no response (TCP open is enough for alerts)." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  TCP: CLOSED - phone NOT reachable." -ForegroundColor Red
        Write-Host "  Context: the Bitchat app may show 127.0.0.1:8765 - that is the phone's"
        Write-Host "  own localhost. Do ONE of:"
        Write-Host "    1. Phone + PC on the SAME Wi-Fi, and use the phone's Wi-Fi IP in pipeline.yaml."
        Write-Host "    2. USB: adb forward tcp:8765 tcp:8765, then set ip to 127.0.0.1."
        Write-Host "    3. Bitchat Settings -> VLM API -> bind to Wi-Fi IP, not localhost."
    }
} else {
    Write-Host "  no Bitchat IP configured - scanning local subnet for port 8765 ..." -ForegroundColor Yellow
    & $venvPython -c "from core.network_scanner import scan_subnet; print(scan_subnet() or 'no phone found')"
}

# --- 4. start the layers -----------------------------------------------------
if ($CheckOnly) {
    Write-Host "`n[-CheckOnly] stopping before the pipeline starts.`n" -ForegroundColor Cyan
    exit 0
}

Write-Host "`n[4/4] Starting pipeline" -ForegroundColor Cyan
try {
    if ($VlmOnly) {
        & $venvPython (Join-Path $PSScriptRoot "run_vlm_to_bitchat.py")
    } else {
        & $venvPython -m pipeline.main_loop
    }
} finally {
    Write-Host "`nPipeline stopped (press Q / Esc in the video window to exit cleanly).`n"
}
