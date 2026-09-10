# Restores the Cosys-AirSim 5.8-v3.4.1 plugin into sim/SightlineSim/Plugins/AirSim (it is not committed: 3.3 GB).
# Resumable, size-verified. Usage:  powershell -ExecutionPolicy Bypass -File tools\setup\fetch_airsim.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dl = Join-Path $repo "_downloads"
$base = "https://github.com/Cosys-Lab/Cosys-AirSim/releases/download/5.8-v3.4.1"
$files = @{
    "AirSim_plugin_Windows_58_341.zip"  = 739790152
    "python_api_client_341.whl"         = 4520622
    "Blocks_editor_project_58_341.zip"  = 9870311
}
New-Item -ItemType Directory -Force $dl | Out-Null
foreach ($f in $files.Keys) {
    $out = Join-Path $dl $f
    for ($i = 1; $i -le 20 -and -not ((Test-Path $out) -and (Get-Item $out).Length -eq $files[$f]); $i++) {
        & curl.exe -L -C - --retry 10 --retry-all-errors --retry-delay 5 -s -S -o $out "$base/$f"
    }
    if ((Get-Item $out).Length -ne $files[$f]) { throw "download incomplete: $f" }
    Write-Host "ok $f"
}
$plugins = Join-Path $repo "sim\SightlineSim\Plugins"
if (-not (Test-Path (Join-Path $plugins "AirSim\AirSim.uplugin"))) {
    $tmp = Join-Path $dl "airsim_plugin_extract"
    Expand-Archive (Join-Path $dl "AirSim_plugin_Windows_58_341.zip") -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force $plugins | Out-Null
    Move-Item (Join-Path $tmp "AirSim") (Join-Path $plugins "AirSim")
}
New-Item -ItemType Directory -Force (Join-Path $repo "vendor\wheels") | Out-Null
Copy-Item (Join-Path $dl "python_api_client_341.whl") (Join-Path $repo "vendor\wheels\cosysairsim-3.4.1-py3-none-any.whl") -Force
Write-Host "Cosys-AirSim plugin installed at $plugins\AirSim"
