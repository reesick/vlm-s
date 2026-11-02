# One command to run everything on the live camera: .\go.ps1
#   .\go.ps1                  # camera from configs/pipeline.yaml (A36 = index 2, DSHOW), generic profile
#   .\go.ps1 -Profile flood   # flood-monitoring prompts
#   .\go.ps1 -Camera 0        # another camera index
#   .\go.ps1 -WaitSec 300     # wait longer for the phone camera to come up
# The video window shows the camera with boxes, and the VLM's latest
# description of what it sees in the panel at the bottom. Q / Esc quits.
param(
    [string]$Profile = "generic",
    [int]$Camera = -1,
    [int]$WaitSec = 180
)
Set-Location "$PSScriptRoot\src\ai-surveillance"
$env:CAMERA_OPEN_TIMEOUT_S = "$WaitSec"
$py = ".\.venv\Scripts\python.exe"

# Wait until the camera actually delivers a non-black frame (phone camera
# apps take a while to start streaming). Model loading takes ~30 s on top.
$cam = if ($Camera -ge 0) { $Camera } else {
    [int]([regex]::Match((Get-Content configs\pipeline.yaml -Raw), '(?m)^source:[\s\S]*?^\s*path:\s*(\d+)').Groups[1].Value)
}
Write-Host "[go] waiting up to $WaitSec s for camera $cam to send frames (open the camera app on the phone now)..." -ForegroundColor Cyan
& $py -c @"
import cv2, sys, time
idx, timeout = $cam, $WaitSec
t0 = time.time(); last = -10
while time.time() - t0 < timeout:
    for be in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        c = cv2.VideoCapture(idx, be)
        if c.isOpened():
            t1 = time.time()
            while time.time() - t1 < 3:
                ok, f = c.read()
                if ok and f is not None and f.mean() > 5:
                    c.release()
                    print(f'[go] camera {idx} live: {f.shape[1]}x{f.shape[0]} after {time.time()-t0:.0f}s')
                    sys.exit(0)
        c.release()
    if time.time() - last >= 10:
        last = time.time(); print(f'[go] still waiting for camera {idx} ... {time.time()-t0:.0f}s')
    time.sleep(1)
print(f'[go] camera {idx} sent no frames in {timeout}s'); sys.exit(1)
"@ 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "[go] camera not live; check the phone camera app / USB, or try -Camera 0" -ForegroundColor Red; exit 1 }

$args2 = @("-m", "pipeline.main_loop", "--profile", $Profile)
if ($Camera -ge 0) {
    $cfg = Join-Path $env:TEMP "pipeline_go.yaml"
    (Get-Content configs\pipeline.yaml -Raw) -replace '(?m)^(source:[\s\S]*?^\s*path:\s*)\d+', "`${1}$Camera" | Set-Content $cfg -Encoding UTF8
    $args2 += @("--config", $cfg)
}
& $py @args2
