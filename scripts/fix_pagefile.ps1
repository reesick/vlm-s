# fix_pagefile.ps1 - move/size the Windows pagefile onto D: so the VLM can load
#
#   right-click -> Run with PowerShell, or:
#   powershell -ExecutionPolicy Bypass -File .\fix_pagefile.ps1
#
# Fixes error 1455 ("The paging file is too small") when Qwen2.5-VL mmaps ~7 GB.
# Sets D:\pagefile.sys  Initial 16384 MB / Maximum 32768 MB, disables Windows'
# automatic management, and REMOVES any pagefile on other drives (frees C:).
# A REBOOT is required - it asks before restarting.

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "not admin - relaunching elevated..." -ForegroundColor Yellow
    Start-Process powershell.exe -Verb RunAs -ArgumentList @(
        "-ExecutionPolicy", "Bypass", "-NoExit", "-File", "`"$PSCommandPath`""
    )
    exit
}

$ErrorActionPreference = "Stop"
Write-Host "`n=== pagefile fixer (KURUKSHETRA) ===" -ForegroundColor Cyan

$key = "HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"
$prop = "PagingFiles"
$old = (Get-ItemProperty -Path $key -Name $prop -ErrorAction SilentlyContinue).$prop
Write-Host "current PagingFiles: $($old -join ' | ')"
$backup = Join-Path $env:TEMP "pagefile_backup.txt"
Set-Content -LiteralPath $backup -Value ($old -join "`n")
Write-Host "(backed up to $backup)" -ForegroundColor Gray

# 1. turn off automatic management
$cs = Get-CimInstance Win32_ComputerSystem
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false }
    Write-Host "automatic management: OFF" -ForegroundColor Green
} else {
    Write-Host "automatic management: already off"
}

# 2. write the explicit setting - D: only, 16 GB initial / 32 GB max
$target = @("D:\pagefile.sys 16384 32768")
Set-ItemProperty -Path $key -Name $prop -Value $target -Type MultiString
Write-Host "new PagingFiles:     D:\pagefile.sys 16384 32768" -ForegroundColor Green

# 3. verify
$now = (Get-ItemProperty -Path $key -Name $prop).$prop
if ($now -eq $target) { Write-Host "verified. takes effect after a REBOOT." -ForegroundColor Green }

$ans = Read-Host "`nreboot NOW? (y/N)"
if ($ans -match '^[Yy]') { Restart-Computer -Force } else { Write-Host "reboot manually, then run .\run.ps1" }
