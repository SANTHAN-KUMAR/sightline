# Restores the non-Python external tools used by Sightline (see docs/verification/external_tools.md).
# Idempotent and resumable: every download is size- and SHA-256-checked; finished steps are skipped.
# Everything lands on D: (repo _downloads/_staging/data/app, D:\Tools). Nothing is installed system-wide.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup\fetch_external.ps1              # everything
#   powershell -ExecutionPolicy Bypass -File tools\setup\fetch_external.ps1 -Only map    # subset: cesium, xal, pmtiles, map, drone
#   ... -RefreshBasemap   re-extract wayanad.pmtiles from the newest Protomaps daily build
param(
    [string[]]$Only = @("cesium", "xal", "pmtiles", "map", "drone"),
    [string]$ToolsRoot = "D:\Tools",
    [switch]$RefreshBasemap
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$dl = Join-Path $repo "_downloads"
New-Item -ItemType Directory -Force $dl | Out-Null

function Get-Verified([string]$Url, [string]$Out, [long]$Size, [string]$Sha256) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
    for ($i = 1; $i -le 20 -and -not ((Test-Path $Out) -and (Get-Item $Out).Length -eq $Size); $i++) {
        if ((Test-Path $Out) -and (Get-Item $Out).Length -gt $Size) { Remove-Item $Out -Force }
        & curl.exe -L -C - --retry 10 --retry-all-errors --retry-delay 5 -s -S -o $Out $Url
    }
    if ((Get-Item $Out).Length -ne $Size) { throw "size mismatch: $Out" }
    if ($Sha256) {
        $h = (Get-FileHash $Out -Algorithm SHA256).Hash
        if ($h -ne $Sha256.ToUpper()) { throw "sha256 mismatch: $Out ($h)" }
    }
    Write-Host "ok  $(Split-Path -Leaf $Out)"
}

# 1. Cesium for Unreal v2.29.1, UE 5.8 package -> _staging/plugins/CesiumForUnreal (NOT copied into the project)
if ($Only -contains "cesium") {
    $zip = Join-Path $dl "CesiumForUnreal-58-v2.29.1.zip"
    Get-Verified "https://github.com/CesiumGS/cesium-unreal/releases/download/v2.29.1/CesiumForUnreal-58-v2.29.1.zip" `
        $zip 1226845642 "4c45422d638b3f8814727b1424364ce8d6108f867d5e49fab514b266eae68b0b"
    $dst = Join-Path $repo "_staging\plugins\CesiumForUnreal"
    if (-not (Test-Path (Join-Path $dst "CesiumForUnreal.uplugin"))) {
        $tmp = Join-Path $repo "_staging\cesium_extract"
        if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
        New-Item -ItemType Directory -Force $tmp | Out-Null
        & tar.exe -xf $zip -C $tmp
        if ($LASTEXITCODE -ne 0) { throw "tar failed on $zip" }
        New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
        Move-Item (Join-Path $tmp "CesiumForUnreal") $dst
        Remove-Item $tmp -Recurse -Force
    }
    $mods = Get-Content (Join-Path $dst "Binaries\Win64\UnrealEditor.modules") -Raw | ConvertFrom-Json
    $eng = Get-Content "D:\UE_5.8\Engine\Build\Build.version" -Raw -EA SilentlyContinue | ConvertFrom-Json
    Write-Host "Cesium staged at $dst (BuildId $($mods.BuildId); engine CompatibleChangelist $($eng.CompatibleChangelist))"
}

# 2. X-AnyLabeling v4.0.6 Windows CUDA 12 (one-file PyInstaller exe) + launcher that keeps data on D:
if ($Only -contains "xal") {
    $xd = Join-Path $ToolsRoot "X-AnyLabeling"
    Get-Verified "https://github.com/CVHub520/X-AnyLabeling/releases/download/v4.0.6/X-AnyLabeling-v4.0.6-Windows-CUDA12.exe" `
        (Join-Path $xd "X-AnyLabeling-v4.0.6-Windows-CUDA12.exe") 583328533 "d2256b42ea6c4bffad86b7d995b4dbd9d1f857e8b0e5c7e30b4ef55e02cd9b02"
    $cmd = Join-Path $xd "X-AnyLabeling.cmd"
    if (-not (Test-Path $cmd)) {
        @'
@echo off
rem Sightline launcher for X-AnyLabeling v4.0.6: config + models under .\work, PyInstaller unpack under .\tmp (both on D:)
setlocal
set "XAL_HOME=%~dp0"
if not exist "%XAL_HOME%work" mkdir "%XAL_HOME%work"
if not exist "%XAL_HOME%tmp" mkdir "%XAL_HOME%tmp"
set "TEMP=%XAL_HOME%tmp"
set "TMP=%XAL_HOME%tmp"
if not defined HF_HOME set "HF_HOME=D:\Tools\cache\hf"
if not defined TORCH_HOME set "TORCH_HOME=D:\Tools\cache\torch"
if exist "D:\Tools\cudnn\bin\cudnn64_9.dll" set "PATH=D:\Tools\cudnn\bin;%PATH%"
if not defined XANYLABELING_MODEL_HUB set "XANYLABELING_MODEL_HUB=github"
"%XAL_HOME%X-AnyLabeling-v4.0.6-Windows-CUDA12.exe" --work-dir "%XAL_HOME%work" %*
endlocal
'@ | Set-Content -Path $cmd -Encoding ascii
    }
    Write-Host "X-AnyLabeling ready: $cmd"
}

# 3. go-pmtiles CLI v1.31.2 -> D:\Tools\pmtiles\pmtiles.exe
$pmtilesExe = Join-Path $ToolsRoot "pmtiles\pmtiles.exe"
if ($Only -contains "pmtiles" -or $Only -contains "map") {
    if (-not (Test-Path $pmtilesExe)) {
        $zip = Join-Path $dl "go-pmtiles_1.31.2_Windows_x86_64.zip"
        Get-Verified "https://github.com/protomaps/go-pmtiles/releases/download/v1.31.2/go-pmtiles_1.31.2_Windows_x86_64.zip" `
            $zip 17842310 "a658baa4d7e55020aef6ca17bd9ff9faa1582671266b36f58c52db0ac8e785a1"
        Expand-Archive $zip -DestinationPath (Split-Path -Parent $pmtilesExe) -Force
    }
    & $pmtilesExe version
}

# 4. Offline basemap + MapLibre/pmtiles/basemaps vendoring for app/map
if ($Only -contains "map") {
    # 4a. Wayanad area-of-operations extract from the Protomaps daily build (daily builds are deleted after
    #     some weeks, so the exact build cannot be pinned; the file is kept in the repo tree instead).
    $pm = Join-Path $repo "data\basemap\wayanad.pmtiles"
    if ($RefreshBasemap -or -not (Test-Path $pm)) {
        New-Item -ItemType Directory -Force (Split-Path -Parent $pm) | Out-Null
        $build = $null
        foreach ($d in 0..10) {
            $s = (Get-Date).AddDays(-$d).ToString("yyyyMMdd")
            $head = & curl.exe -s -I --max-time 20 "https://build.protomaps.com/$s.pmtiles"
            if ($head -match "HTTP/\S+ 200") { $build = $s; break }
        }
        if (-not $build) { throw "no Protomaps daily build found in the last 10 days" }
        Write-Host "extracting from Protomaps build $build"
        $tmpPm = "$pm.part"
        if (Test-Path $tmpPm) { Remove-Item $tmpPm -Force }
        & $pmtilesExe extract "https://build.protomaps.com/$build.pmtiles" $tmpPm --bbox=76.05,11.42,76.25,11.58 --maxzoom=15
        if ($LASTEXITCODE -ne 0) { throw "pmtiles extract failed" }
        Move-Item $tmpPm $pm -Force
    }
    & $pmtilesExe verify $pm
    if ($LASTEXITCODE -ne 0) { throw "pmtiles verify failed: $pm" }

    # 4b. npm tarballs (registry, integrity-checked by npm) -> app/map/vendor
    $npmDir = Join-Path $dl "npm"
    New-Item -ItemType Directory -Force $npmDir | Out-Null
    $pkgs = [ordered]@{
        "maplibre-gl-6.9.0.tgz"        = @("maplibre-gl@6.9.0", 4249537)
        "pmtiles-4.5.0.tgz"            = @("pmtiles@4.5.0", 96124)
        "protomaps-basemaps-5.7.2.tgz" = @("@protomaps/basemaps@5.7.2", 59103)
    }
    foreach ($k in $pkgs.Keys) {
        $t = Join-Path $npmDir $k
        if (-not ((Test-Path $t) -and (Get-Item $t).Length -eq $pkgs[$k][1])) {
            Push-Location $npmDir
            try { & npm pack $pkgs[$k][0] --cache (Join-Path $ToolsRoot "cache\npm") --silent | Out-Null } finally { Pop-Location }
        }
        if ((Get-Item $t).Length -ne $pkgs[$k][1]) { throw "size mismatch: $t" }
        $x = Join-Path $npmDir ($k -replace "\.tgz$", "")
        if (-not (Test-Path (Join-Path $x "package\package.json"))) {
            New-Item -ItemType Directory -Force $x | Out-Null
            & tar.exe -xzf $t -C $x
        }
        Write-Host "ok  $k"
    }
    $v = Join-Path $repo "app\map\vendor"
    $ml = Join-Path $v "maplibre-gl"; $pt = Join-Path $v "pmtiles"; $bm = Join-Path $v "protomaps-basemaps"
    New-Item -ItemType Directory -Force $ml, $pt, $bm | Out-Null
    foreach ($f in "maplibre-gl.mjs", "maplibre-gl-shared.mjs", "maplibre-gl-worker.mjs", "maplibre-gl.css") {
        Copy-Item (Join-Path $npmDir "maplibre-gl-6.9.0\package\dist\$f") $ml -Force
    }
    Copy-Item (Join-Path $npmDir "maplibre-gl-6.9.0\package\LICENSE.txt") $ml -Force
    Copy-Item (Join-Path $npmDir "pmtiles-4.5.0\package\dist\pmtiles.js") $pt -Force
    Copy-Item (Join-Path $npmDir "protomaps-basemaps-5.7.2\package\dist\basemaps.js") $bm -Force

    # 4c. Protomaps fonts + sprites (protomaps/basemaps-assets, pinned commit). The Devanagari font folder is
    #     skipped: its name is not a valid Windows path and the basemap's Latin labels do not need it.
    $commit = "028c18f713baecad011301ff7a69acc39bcc2ae7"
    $az = Join-Path $dl "basemaps-assets-028c18f.zip"
    Get-Verified "https://codeload.github.com/protomaps/basemaps-assets/zip/$commit" $az 6731115 "e942a417d94a12596842a20b53d6b785cbf6d47f2545e458538191ba6d74b305"
    $ba = Join-Path $v "basemaps-assets"
    if (-not (Test-Path (Join-Path $ba "sprites\v4\light.json"))) {
        $ax = Join-Path $dl "basemaps-assets"
        if (Test-Path $ax) { Remove-Item $ax -Recurse -Force }
        New-Item -ItemType Directory -Force $ax | Out-Null
        & tar.exe -xf $az -C $ax --exclude "*Devanagari*"
        $root = Join-Path $ax "basemaps-assets-$commit"
        New-Item -ItemType Directory -Force (Join-Path $ba "fonts"), (Join-Path $ba "sprites") | Out-Null
        foreach ($f in "Noto Sans Regular", "Noto Sans Medium", "Noto Sans Italic") {
            Copy-Item (Join-Path $root "fonts\$f") (Join-Path $ba "fonts\") -Recurse -Force
        }
        Copy-Item (Join-Path $root "fonts\OFL.txt") (Join-Path $ba "fonts\") -Force
        Copy-Item (Join-Path $root "sprites\v4") (Join-Path $ba "sprites\") -Recurse -Force
    }
    Write-Host "map vendor ready: $v  (check: node app\map\serve.mjs  then  node app\map\headless_check.mjs)"
}

# 5. Theta-Limited DroneModels (Apache-2.0), pinned commit
if ($Only -contains "drone") {
    $sha = "edae83415913edbf972cf25365c18e5f6a701dfb"
    $ref = Join-Path $repo "data\reference"
    Get-Verified "https://raw.githubusercontent.com/Theta-Limited/DroneModels/$sha/droneModels.json" `
        (Join-Path $ref "DroneModels.json") 60822 "0155d855529e09b6adb71bd7f8f28737a4808d74516622338450c00691499639"
    Get-Verified "https://raw.githubusercontent.com/Theta-Limited/DroneModels/$sha/LICENSE" `
        (Join-Path $ref "DroneModels.LICENSE") 11350 $null
}
Write-Host "fetch_external: done ($($Only -join ', '))"
