# run.ps1 - KURUKSHETRA one-script launcher (run this from anywhere)
#
#   cd D:\kurukshetra_hackathon
#   .\run.ps1                # full stack: env + model downloads + Bitchat check + pipeline
#   .\run.ps1 -VlmOnly       # only the VLM scene-description layer -> Bitchat
#   .\run.ps1 -CheckOnly     # set everything up + check Bitchat, start nothing
#   .\run.ps1 -AndroidIp 10.0.0.5   # override the Bitchat phone IP
#   .\run.ps1 -SkipModels    # don't touch weight files
#
# Writes a timestamped log file to src\ai-surveillance\logs\run_<timestamp>.log
# and prints elapsed time per stage plus a final summary.

param(
    [string]$AndroidIp = "",
    [int]$BitchatPort = 8765,
    [switch]$VlmOnly,
    [switch]$CheckOnly,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$script:t0 = Get-Date    # run start

# ---- structured logging ------------------------------------------------------
$logsDir = Join-Path $PSScriptRoot "src\ai-surveillance\logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$script:logFile = Join-Path $logsDir ("run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log {
    param([string]$Stage, [string]$Message, [string]$Color = "Gray")
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format "HH:mm:ss.fff"), $Stage, $Message
    Write-Host $line -ForegroundColor $Color
    Add-Content -LiteralPath $script:logFile -Value $line
}

function Write-Stage {
    param([string]$Stage, [string]$Message)
    Write-Log $Stage "" Cyan
    Write-Log $Stage ("========== {0} ==========" -f $Message) Cyan
}

function Start-Timer { param([string]$Name) $script:tmr[$Name] = Get-Date }

function Stop-Timer {
    param([string]$Name, [string]$Label = "")
    if (-not $script:tmr.ContainsKey($Name)) { return }
    $span = (Get-Date) - $script:tmr[$Name]
    $secs = [math]::Round($span.TotalSeconds, 1)
    $readable = if ($secs -ge 60) { "{0}m {1:0}s" -f [int][math]::Floor($secs / 60), ($secs % 60) } else { "{0:0.#}s" -f $secs }
    if ($Label) { Write-Log $Name ("{0} -> DONE in {1}" -f $Label, $readable) Green }
    $script:timings[$Name] = $readable
}

$script:tmr = @{}
$script:timings = @{}

# Noise filter: progress bars (tqdm), PS RemoteException artifacts, repeated
# pip "already satisfied"/notice lines are dropped from console+log. Warnings,
# errors, tracebacks and anything else still pass through untouched.
$script:noisePattern = (
    'System\.Management\.Automation\.RemoteException',
    '\d+%[|]',                                # tqdm progress bars
    '[\d.]+(?:KB|MB|GB)/s\]',
    'Requirement already satisfied',
    '^\[notice\]\s*A new release of pip',
    '^\[notice\]\s*To update'
) -join '|'

function Invoke-NativeLogged {
    # Runs a native command with stderr merged for logging, but WITHOUT the
    # PS 5.1 NativeCommandError trap (EAP=Stop turns merged stderr into
    # terminating errors). Fail/success is judged by $LASTEXITCODE afterwards.
    param([scriptblock]$Cmd, [string]$Stage)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Cmd 2>&1 | ForEach-Object {
            $s = "$_"
            if ($s -notmatch $noisePattern) { Write-Log $Stage $s }
        }
    }
    finally { $ErrorActionPreference = $prev }
}

Write-Stage "run" "KURUKSHETRA launcher - log: $script:logFile"

# Everything lives in src\ai-surveillance
$App = Join-Path $PSScriptRoot "src\ai-surveillance"

function Read-BitchatConfig {
    # Pulls ip/port out of the bitchat block of configs\pipeline.yaml (regex, no yaml dep)
    $yaml = Join-Path $App "configs\pipeline.yaml"
    if (-not (Test-Path $yaml)) { return $null }
    $text = Get-Content -LiteralPath $yaml -Raw
    $ip = $null; $port = 0
    if ($m = [regex]::Match($text, 'bitchat:\s*\n(?:.*\n)*?\s*ip:\s*"([^"]+)"')) { $ip = $m.Groups[1].Value }
    if ($m = [regex]::Match($text, 'bitchat:\s*\n(?:.*\n)*?\s*port:\s*(\d+)')) { $port = [int]$m.Groups[1].Value }
    if ($ip) { return [pscustomobject]@{ ip = $ip; port = if ($port) { $port } else { 8765 } } }
    return $null
}

function Test-TcpPort {
    param([string]$Ip, [int]$Port, [int]$TimeoutMs = 2000)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($Ip, $Port, $null, $null)
        return ($async.AsyncWaitHandle.WaitOne($TimeoutMs) -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

$overallFailed = $false
try {
    # == 1. virtualenv + dependencies ============================================
    Start-Timer "deps"
    Write-Stage "deps" "[1/4] Virtual environment"
    if (-not (Test-Path (Join-Path $App ".venv\Scripts\python.exe"))) {
        Write-Log "deps" "creating .venv ..."
        python -m venv (Join-Path $App ".venv")
    } else {
        Write-Log "deps" ".venv already exists - reusing."
    }
    $py = Join-Path $App ".venv\Scripts\python.exe"
    Write-Log "deps" ("python: {0}" -f $py)

    $deps = Join-Path $App "requirements.txt"
    if (-not $SkipModels -or -not $CheckOnly) {
        Write-Log "deps" "verifying key dependencies ..."
        # PS 5.1 trap: with EAP=Stop, stderr lines from the import check (e.g.
        # torch's "triton not found" warning) become terminating errors even
        # when redirected. Judge success by $LASTEXITCODE only.
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $py -c "import torch, torchvision, cv2, ultralytics, numpy, yaml, scipy, lap, onnx, onnxruntime, faiss, insightface, transformers, accelerate, bitsandbytes" > $null 2>&1
        $ErrorActionPreference = $prevEap
        if ($LASTEXITCODE -eq 0) {
            Write-Log "deps" "all key dependencies already installed - skipping pip" Green
        } else {
            Write-Log "deps" "some dependencies missing - running pip install (FIRST run: torch cu128 ~3 GB -> 20-45 min)" Yellow
            Invoke-NativeLogged { & $py -m pip install -r $deps --progress-bar off } "pip"
            if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
        }
    }
    Stop-Timer "deps" "deps install"

if ($SkipModels) {
        Write-Log "models" "weight stage skipped (-SkipModels)" Yellow
    } else {
        # == 2. model weights: detection layers + VLM ================================
    Start-Timer "models"
    Write-Stage "models" "[2/4] Model weights (detection + VLM layers)"
    $modelDir = Join-Path $App "models"
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

    $yoloSmall = Join-Path $modelDir "yolov8n.pt"
    $yoloPose  = Join-Path $modelDir "yolov8n-pose.pt"

    $dlStart = [System.Diagnostics.Stopwatch]::StartNew()
    if (-not (Test-Path $yoloSmall)) {
        $dlStart.Restart()
        Write-Log "models" "downloading YOLOv8n detector (~6 MB) ..."
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" -OutFile $yoloSmall
        Write-Log "models" ("yolov8n.pt: {0:N1} MB in {1:0.#}s" -f ((Get-Item $yoloSmall).Length / 1MB), $dlStart.Elapsed.TotalSeconds)
    } else {
        Write-Log "models" ("yolov8n.pt present ({0:N1} MB)" -f ((Get-Item $yoloSmall).Length / 1MB))
    }
    if (-not (Test-Path $yoloPose)) {
        $dlStart.Restart()
        Write-Log "models" "downloading YOLOv8n-pose (~7 MB) ..."
        Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt" -OutFile $yoloPose
        Write-Log "models" ("yolov8n-pose.pt: {0:N1} MB in {1:0.#}s" -f ((Get-Item $yoloPose).Length / 1MB), $dlStart.Elapsed.TotalSeconds)
    } else {
        Write-Log "models" ("yolov8n-pose.pt present ({0:N1} MB)" -f ((Get-Item $yoloPose).Length / 1MB))
    }

    foreach ($optional in @("fire_smoke_yolov8n.pt", "smoking_yolov8n.pt")) {
        if (-not (Test-Path (Join-Path $modelDir $optional))) {
            Write-Log "models" "$optional missing (fine-tuned) - pipeline falls back to heuristic" Yellow
        } else {
            Write-Log "models" "$optional present"
        }
    }

    # ONNX exports: pose runs runtime=onnx_direct and REQUIRES models/yolov8n-pose.onnx
    $needOx = -not (Test-Path (Join-Path $modelDir "yolov8n-pose.onnx"))
    if ($needOx) {
        Write-Log "models" "ONNX exports missing - running one-time export (~2 min) ..."
        Push-Location $App
        Invoke-NativeLogged { & $py "$App\tools\export_onnx.py" } "onnx"
        Pop-Location
        if ($LASTEXITCODE -ne 0) { throw "ONNX export failed" }
        Write-Log "models" ("yolov8n-pose.onnx: {0:N1} MB" -f ((Get-Item (Join-Path $modelDir "yolov8n-pose.onnx")).Length / 1MB))
    } else {
        Write-Log "models" "yolov8n-pose.onnx present"
    }

    # VLM: Qwen2.5-VL-3B-Instruct (~7.1 GB first run, cached afterwards).
    # We do NOT override HF_HOME/HF_HUB_CACHE - your machine already points the
    # HF cache at D:\vlm\.cache, and overriding would force a second 7 GB copy.
    # We just report where the cache resolves, download if needed, and the
    # pipeline inherits the same env automatically.
    Start-Timer "vlm"
    Invoke-NativeLogged { & $py -m pip install --quiet huggingface_hub } "vlm"
    Invoke-NativeLogged { & $py -c "from huggingface_hub import scan_cache_dir; reps = scan_cache_dir().repos; q = [r for r in reps if 'Qwen2.5-VL-3B' in r.repo_id]; print('[cache] {0}: {1:.2f} GB at {2}'.format(q[0].repo_id, q[0].size_on_disk/1e9, q[0].repo_path) if q else '[cache] Qwen2.5-VL-3B not present yet')" } "vlm"
    Write-Log "vlm" "ensuring Qwen2.5-VL-3B-Instruct (~7.1 GB first run, instant when cached)" Yellow
    $dlStart = [System.Diagnostics.Stopwatch]::StartNew()
    Invoke-NativeLogged { & $py -c "from huggingface_hub import snapshot_download; p = snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct'); print('[vlm] ready at:', p)" } "vlm"
    if ($LASTEXITCODE -ne 0) { throw "VLM download failed" }
    Write-Log "vlm" ("VLM ready. Total elapsed: {0:0.#}s" -f $dlStart.Elapsed.TotalSeconds)
    Stop-Timer "models" "all weights"
}
if (-not $SkipModels) { Stop-Timer "vlm" "VLM weights" }

# == 3. Bitchat connectivity ===================================================
Start-Timer "bitchat"
Write-Stage "bitchat" "[3/4] Bitchat connectivity check"
if ($AndroidIp) {
    $target = [pscustomobject]@{ ip = $AndroidIp; port = $BitchatPort }
} else {
    $target = Read-BitchatConfig
}

if ($target) {
    Write-Log "bitchat" ("target from config: {0}:{1}" -f $target.ip, $target.port)
    if (Test-TcpPort -Ip $target.ip -Port $target.port) {
        Write-Log "bitchat" "TCP: OPEN - Bitchat app IS reachable" Green
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $resp = Invoke-WebRequest -Uri ("http://{0}:{1}/status" -f $target.ip, $target.port) -TimeoutSec 3 -UseBasicParsing
            Write-Log "bitchat" ("HTTP /status -> {0} ({1:0} ms)" -f $resp.StatusCode, $sw.Elapsed.TotalMilliseconds)
        } catch {
            Write-Log "bitchat" "HTTP /status: no reply (TCP open is enough for alerts)" Yellow
        }
    } else {
        Write-Log "bitchat" "TCP: CLOSED - phone NOT reachable" Red
        Write-Log "bitchat" "the app often shows 127.0.0.1:8765 = the PHONE's own localhost. Fixes:" Yellow
        Write-Log "bitchat" "  1. phone + PC on the SAME Wi-Fi -> set its Wi-Fi IP in configs\pipeline.yaml" Yellow
        Write-Log "bitchat" "  2. USB: adb forward tcp:8765 tcp:8765, then set ip to 127.0.0.1" Yellow
        Write-Log "bitchat" "  3. Bitchat Settings -> VLM API -> bind to Wi-Fi IP, not localhost" Yellow
    }
} else {
    Write-Log "bitchat" "no Bitchat IP configured - scanning subnet for port 8765 ..." Yellow
    Push-Location $App
    Invoke-NativeLogged { & $py -c "from core.network_scanner import scan_for_bitchat_api; r = scan_for_bitchat_api(); print('[scanner]', r or 'no Bitchat device found on the subnet')" } "scanner"
    Pop-Location
}
Stop-Timer "bitchat" "connectivity check"

# == 4. start the layers =======================================================
Write-Stage "pipeline" "[4/4] Starting pipeline"
if ($CheckOnly) {
    Write-Log "pipeline" "-CheckOnly: stopping before the pipeline starts" Cyan
} else {
    Start-Timer "pipeline"
    Push-Location $App
    try {
        if ($VlmOnly) {
            Write-Log "pipeline" "mode: VLM layer -> Bitchat only"
            Invoke-NativeLogged { & $py "$App\run_vlm_to_bitchat.py" } "vlm"
        } else {
            Write-Log "pipeline" "mode: full detection + pose + VLM + alerts"
            Invoke-NativeLogged { & $py -m pipeline.main_loop } "pipe"
        }
    } finally {
        Pop-Location
        Stop-Timer "pipeline" ""
        $code = $LASTEXITCODE
        if ($code -eq 0) {
            Stop-Timer "pipeline" "pipeline ran"
            Write-Log "pipeline" "stopped. (Press Q / Esc in the video window to exit cleanly.)" Yellow
        } else {
            Write-Log "pipeline" ("pipeline exited with code {0}" -f $code) Red
            $overallFailed = $true
        }
    }
}
} catch {
    $overallFailed = $true
    Write-Log "run" ("FAILED: {0}" -f $_.Exception.Message) Red
}

# == summary (always prints, even after a failure) =============================
$total = [math]::Round(((Get-Date) - $script:t0).TotalSeconds, 1)
Write-Log "run" "" Cyan
Write-Log "run" "========== TIMING SUMMARY ==========" Cyan
foreach ($k in @("deps", "models", "vlm", "bitchat", "pipeline")) {
    if ($script:timings.ContainsKey($k)) { Write-Log "run" ("  {0,-10} {1}" -f $k, $script:timings[$k]) }
}
Write-Log "run" ("  TOTAL      {0}s" -f $total) Cyan
if ($overallFailed) {
    Write-Log "run" "RUN FAILED - see red lines above and the log file" Red
} else {
    Write-Log "run" "RUN OK" Green
}
Write-Log "run" ("full log file: {0}" -f $script:logFile) Cyan
if ($overallFailed) { exit 1 } else { exit 0 }
