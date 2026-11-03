# start.ps1 - one file to set up and run everything.
#
#   .\start.ps1                       # live camera -> VLM description on screen + to the Bitchat mesh
#   .\start.ps1 -Camera 0             # use the laptop webcam (default 2 = phone camera, e.g. Samsung A36)
#   .\start.ps1 -PhoneIp 192.168.1.5  # phone IP (default: find it on this Wi-Fi)
#   .\start.ps1 -NoShare              # describe on screen only, send nothing
#   .\start.ps1 -Mode images          # send data\disaster_images\* + VLM descriptions to the phone
#   .\start.ps1 -Mode pipeline        # full stack: detection, pose, events, VLM, alerts
#   .\start.ps1 -Mode setup           # only install / download, run nothing
#   .\start.ps1 -Api https://disruptionops-jq8g.onrender.com -GatewayKey <MESH_GATEWAY_KEY> -Lat 17.9583 -Lon 74.4127
#                                     # also post every frame + VLM description (+ where the camera is)
#                                     # straight to the command centre over the internet
#   .\start.ps1 -Api ... -GatewayKey ... -DemoPin
#                                     # camera placed inside the PCCOE drone map, labelled "demo position"
#
# First run: creates src\ai-surveillance\.venv, installs requirements (torch is ~3 GB)
# and downloads Qwen2.5-VL-3B (~7 GB). Later runs start in seconds.
# If PowerShell blocks the script:  powershell -ExecutionPolicy Bypass -File .\start.ps1
param(
    [ValidateSet("live", "images", "pipeline", "setup")] [string]$Mode = "live",
    [int]$Camera = 2,
    [string]$PhoneIp = "",
    [int]$Port = 8765,
    [switch]$NoShare,
    [string]$CcUrl = "",
    [string]$CcKey = "",
    [string]$Api = $env:INDRADHANU_API,
    [string]$GatewayKey = $env:MESH_GATEWAY_KEY,
    [string]$Lat = "",
    [string]$Lon = "",
    [switch]$DemoPin,
    [string]$Node = "vlm-cam-01"
)
    # Load private defaults (key, URL, real lat/lon) from start.local.ps1 (git-ignored).
    $local = Join-Path $PSScriptRoot "start.local.ps1"
    if (Test-Path $local) { . $local }
    if (-not $Api -and $LocalApi) { $Api = $LocalApi }
    if (-not $GatewayKey -and $LocalGatewayKey) { $GatewayKey = $LocalGatewayKey }
    if (-not $Lat -and $LocalLat) { $Lat = "$LocalLat" }
    if (-not $Lon -and $LocalLon) { $Lon = "$LocalLon" }
if (-not $Api -and $CcUrl) { $Api = $CcUrl }
if (-not $GatewayKey -and $CcKey) { $GatewayKey = $CcKey }
$ErrorActionPreference = "Stop"
$App = Join-Path $PSScriptRoot "src\ai-surveillance"
$py = Join-Path $App ".venv\Scripts\python.exe"

function Say($msg, $color = "Cyan") { Write-Host "[start] $msg" -ForegroundColor $color }

function Invoke-Py {
    # Run python without PowerShell 5.1 turning its stderr warnings into errors.
    $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    try { & $py @args | Out-Host } finally { $ErrorActionPreference = $prev }
    return $LASTEXITCODE
}

# ---- 1. Python environment ---------------------------------------------------
if (-not (Test-Path $py)) {
    Say "creating virtual environment (Python 3.11) ..."
    $made = $false
    foreach ($cmd in @(@("py", "-3.11"), @("python"))) {
        if (Get-Command $cmd[0] -ErrorAction SilentlyContinue) {
            $ErrorActionPreference = "Continue"
            & $cmd[0] @($cmd[1..9] | Where-Object { $_ }) -m venv (Join-Path $App ".venv")
            $ErrorActionPreference = "Stop"
            if (Test-Path $py) { $made = $true; break }
        }
    }
    if (-not $made) { Say "Python 3.11 not found. Install it from https://www.python.org/downloads/ and run again." Red; exit 1 }
}

$check = "import torch, cv2, yaml, requests, transformers, accelerate, bitsandbytes, peft, huggingface_hub, ultralytics"
if ((Invoke-Py -c $check 2>$null) -ne 0) {
    Say "installing Python packages (first run: 20-45 min) ..." Yellow
    if ((Invoke-Py -m pip install --upgrade pip --quiet) -ne 0) { Say "pip upgrade failed" Red; exit 1 }
    if ((Invoke-Py -m pip install -r (Join-Path $App "requirements.txt") --progress-bar off) -ne 0) {
        Say "pip install failed - see the error above" Red; exit 1
    }
} else {
    Say "Python packages OK" Green
}

if ((Invoke-Py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null) -ne 0) {
    Say "WARNING: no CUDA GPU visible to torch. The VLM needs an NVIDIA GPU (6 GB+)." Yellow
}

# ---- 2. VLM weights (Hugging Face cache, downloaded once) ---------------------
Say "checking Qwen2.5-VL-3B-Instruct weights (~7 GB on first run) ..."
if ((Invoke-Py -c "from huggingface_hub import snapshot_download; print('[start] VLM at', snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct'))") -ne 0) {
    Say "VLM download failed (check internet / disk space)" Red; exit 1
}

# ---- 3. run -------------------------------------------------------------------
Push-Location $App
try {
    switch ($Mode) {
        "setup" { Say "setup done. Run .\start.ps1 to start." Green; $code = 0 }
        "live" {
            $a = @("run_vlm_to_bitchat.py", "--camera", "$Camera", "--port", "$Port")
            if ($PhoneIp) { $a += @("--ip", $PhoneIp) }
            if ($NoShare) { $a += "--no-share" }
            if ($CcUrl)   { $a += @("--cc-url", $CcUrl) }
            if ($CcKey)   { $a += @("--cc-key", $CcKey) }
            if ($Api)     { $a += @("--api", $Api, "--key", $GatewayKey, "--node", $Node) }
            if ($Lat -and $Lon) { $a += @("--lat", $Lat, "--lon", $Lon) }
            if ($DemoPin) { $a += "--demo-pin" }
            Say "live: camera $Camera. Keys in the video window: S / Space = send now, Q = quit" Green
            $code = Invoke-Py @a
        }
        "images" {
            if (-not (Get-ChildItem "data\disaster_images\*.jpg" -ErrorAction SilentlyContinue)) {
                Say "fetching sample disaster images ..."
                Invoke-Py tools\fetch_disaster_images.py | Out-Host
            }
            $a = @("send_vlm_alerts.py", "--port", "$Port")
            if ($PhoneIp) { $a += @("--ip", $PhoneIp) }
            if ($Api)     { $a += @("--api", $Api, "--key", $GatewayKey, "--node", $Node) }
            if ($Lat -and $Lon) { $a += @("--lat", $Lat, "--lon", $Lon) }
            if ($DemoPin) { $a += "--demo-pin" }
            $code = Invoke-Py @a
        }
        "pipeline" {
            Pop-Location; Push-Location $PSScriptRoot
            $a = @()
            if ($PhoneIp) { $a += @("-AndroidIp", $PhoneIp) }
            & (Join-Path $PSScriptRoot "run.ps1") @a
            $code = $LASTEXITCODE
        }
    }
} finally { Pop-Location }
exit $code
