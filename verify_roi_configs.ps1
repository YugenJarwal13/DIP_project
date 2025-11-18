#!/usr/bin/env powershell
# Verify all ROI configs - Quick visual check of all 6 sequences

Write-Host "=== ROI Config Verification ===" -ForegroundColor Cyan
Write-Host "This will display each config overlaid on its reference frame."
Write-Host "Press any key in the image window to proceed to next sequence.`n"

$sequences = @(
    @{Name="MVI_20011"; Frame="data/subset/MVI_20011/img00563.jpg"},
    @{Name="MVI_20032"; Frame="data/subset/MVI_20032/img00435.jpg"},
    @{Name="MVI_39031"; Frame="data/subset/MVI_39031/img00363.jpg"},
    @{Name="MVI_39311"; Frame="data/subset/MVI_39311/img00409.jpg"},
    @{Name="MVI_39851"; Frame="data/subset/MVI_39851/img00597.jpg"},
    @{Name="MVI_40711"; Frame="data/subset/MVI_40711/img00524.jpg"}
)

$found = 0
$missing = 0

foreach ($seq in $sequences) {
    $seqName = $seq.Name
    $framePath = $seq.Frame
    $configPath = "configs/$seqName`_config.json"
    
    if (Test-Path $configPath) {
        Write-Host "✓ $seqName" -ForegroundColor Green -NoNewline
        Write-Host " - Visualizing..."
        .\venv\Scripts\python.exe src/roi_tools/visualize_zones.py --config $configPath --frame $framePath
        $found++
    } else {
        Write-Host "✗ $seqName" -ForegroundColor Red -NoNewline
        Write-Host " - Config not found at $configPath"
        $missing++
    }
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "Configs found: $found / $($sequences.Count)"
Write-Host "Configs missing: $missing / $($sequences.Count)"

if ($missing -gt 0) {
    Write-Host "`nTo create missing configs, run:" -ForegroundColor Yellow
    Write-Host "  .\run_roi_config.ps1"
}
