# Send test images (and optionally VLM descriptions) to the Bitchat phone.
#   .\send_to_phone.ps1                      # check + 3 plain images
#   .\send_to_phone.ps1 -Mode vlm            # every disaster image + its VLM description
#   .\send_to_phone.ps1 -Ip 172.31.148.5 -Port 8765
param(
    [string]$Ip = "172.31.148.5",
    [int]$Port = 8765,
    [ValidateSet("plain", "vlm")] [string]$Mode = "plain"
)
$App = Join-Path $PSScriptRoot "src\ai-surveillance"
$py = Join-Path $App ".venv\Scripts\python.exe"
$base = "http://${Ip}:${Port}"

Write-Host "[phone] checking $base/status ..." -ForegroundColor Cyan
try {
    $st = Invoke-RestMethod "$base/status" -TimeoutSec 6
    Write-Host ("[phone] OK  api_enabled={0}  mesh={1}  peers={2}  rate_limit_ms={3}" -f $st.api_enabled, $st.mesh_running, $st.peers_count, $st.rate_limit_ms) -ForegroundColor Green
    if (-not $st.api_enabled) { Write-Host "[phone] VLM API is OFF in the app (Settings -> VLM API)" -ForegroundColor Red; exit 1 }
} catch {
    Write-Host "[phone] cannot reach $base ; scanning this Wi-Fi for the Bitchat API ..." -ForegroundColor Yellow
    $myIp = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias Wi-Fi -ErrorAction SilentlyContinue | Select -First 1).IPAddress
    $found = $null
    if ($myIp) {
        $net = ($myIp -split '\.')[0..2] -join '.'
        $jobs = foreach ($k in 0..7) { Start-Job -ArgumentList $net, $k, $Port -ScriptBlock { param($net, $k, $port) foreach ($h in ($k*32+1)..([math]::Min(254, $k*32+32))) { $c = New-Object Net.Sockets.TcpClient; try { $x = $c.BeginConnect("$net.$h", $port, $null, $null); if ($x.AsyncWaitHandle.WaitOne(350) -and $c.Connected) { "$net.$h" } } catch {} finally { $c.Close() } } } }
        $found = $jobs | Wait-Job | Receive-Job | Select -First 1
        $jobs | Remove-Job
    }
    if (-not $found) {
        Write-Host "[phone] no Bitchat API found on port $Port. Is 'Accept camera alerts' on and the app open?" -ForegroundColor Red
        exit 1
    }
    $Ip = $found; $base = "http://${Ip}:${Port}"
    Write-Host "[phone] found the phone at $base" -ForegroundColor Green
    $st = Invoke-RestMethod "$base/status" -TimeoutSec 6
    Write-Host ("[phone] OK  api_enabled={0}  mesh={1}  peers={2}" -f $st.api_enabled, $st.mesh_running, $st.peers_count) -ForegroundColor Green
}

if ($Mode -eq "vlm") {
    Push-Location $App
    & $py send_disaster_images.py --ip $Ip --port $Port
    Pop-Location
    exit $LASTEXITCODE
}

# plain mode: 3 images, no text, 6 s apart (phone drops anything inside its 5 s window)
$code = @"
import cv2, glob, requests, sys, time
base = '$base'
fs = sorted(glob.glob(r'$App\data\disaster_images\*.jpg'))[:3]
if not fs: sys.exit('no images in data/disaster_images')
for i, f in enumerate(fs):
    img = cv2.imread(f); h, w = img.shape[:2]; s = min(1.0, 640 / max(h, w))
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    name = f.split('\\')[-1]
    r = requests.post(base + '/send/image', files={'image': (name, jpg, 'image/jpeg')}, timeout=15)
    print(f'[phone] {i + 1}/{len(fs)} {name} {len(jpg) // 1024} KB -> {r.status_code} {r.text[:80]}')
    if i < len(fs) - 1: time.sleep(6)
"@
& $py -c $code
