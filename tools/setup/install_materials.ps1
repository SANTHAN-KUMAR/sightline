# Installs sim/settings/materials.csv next to every executable that loads Cosys-AirSim.
# Cosys-AirSim looks for materials.csv in the executable's directory first, then in Documents\AirSim (on C:).
# Without it, material stencil initialisation is skipped (SimModeBase.cpp InitializeMaterialStencils) and the
# on-screen error "Cannot start stencil initialization. Material list was not found" appears.
# Re-run after editing materials.csv or after packaging a new build.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$src = Join-Path $repo "sim\settings\materials.csv"
$targets = @(
    "D:\UE_5.8\Engine\Binaries\Win64",                                          # UnrealEditor.exe (editor, PIE, -game)
    (Join-Path $repo "_downloads\blocks_packaged\Windows\Blocks\Binaries\Win64"), # packaged Blocks reference build
    (Join-Path $repo "_build\Development\Windows\SightlineSim\Binaries\Win64")  # our packaged build (ue_package)
)
foreach ($t in $targets) {
    if (Test-Path $t) { Copy-Item $src (Join-Path $t "materials.csv") -Force; Write-Host "installed -> $t" }
    else { Write-Host "skip (not present) -> $t" }
}
