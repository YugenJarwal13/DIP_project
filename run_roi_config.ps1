#!/usr/bin/env powershell
# ROI Configuration Batch Script
# Run each GUI command sequentially for all 6 sequences

Write-Host "=== ROI Configuration Workflow ===" -ForegroundColor Cyan
Write-Host ""

# Array of sequences with their representative frames
$sequences = @(
    @{Name="MVI_20011"; Frame="data/subset/MVI_20011/img00563.jpg"},
    @{Name="MVI_20032"; Frame="data/subset/MVI_20032/img00435.jpg"},
    @{Name="MVI_39031"; Frame="data/subset/MVI_39031/img00363.jpg"},
    @{Name="MVI_39311"; Frame="data/subset/MVI_39311/img00409.jpg"},
    @{Name="MVI_39851"; Frame="data/subset/MVI_39851/img00597.jpg"},
    @{Name="MVI_40711"; Frame="data/subset/MVI_40711/img00524.jpg"}
)

foreach ($seq in $sequences) {
    $seqName = $seq.Name
    $framePath = $seq.Frame
    $configPath = "configs/$seqName`_config.json"
    
    Write-Host "`n--- Processing $seqName ---" -ForegroundColor Yellow
    Write-Host "Frame: $framePath"
    Write-Host "Output: $configPath"
    Write-Host ""
    
    # Check if config already exists
    if (Test-Path $configPath) {
        Write-Host "Config already exists. Options:" -ForegroundColor Green
        Write-Host "  1. Skip (press S)"
        Write-Host "  2. Edit existing (press E)"
        Write-Host "  3. Create new (press N)"
        $choice = Read-Host "Choose"
        
        if ($choice -eq "S" -or $choice -eq "s") {
            Write-Host "Skipping $seqName" -ForegroundColor Gray
            continue
        }
    }
    
    Write-Host "Starting ROI GUI..." -ForegroundColor Cyan
    Write-Host "Controls: Left-click=point | n=finish polygon | l=line mode | s=save | q=quit"
    Write-Host ""
    
    # Run ROI GUI
    .\venv\Scripts\python.exe src/roi_tools/roi_gui.py --frame $framePath --out $configPath
    
    # Verify if config was saved
    if (Test-Path $configPath) {
        Write-Host "`n✓ Config saved successfully!" -ForegroundColor Green
        
        # Ask if user wants to verify
        $verify = Read-Host "Verify visualization? (Y/n)"
        if ($verify -ne "n" -and $verify -ne "N") {
            Write-Host "Opening visualization..." -ForegroundColor Cyan
            .\venv\Scripts\python.exe src/roi_tools/visualize_zones.py --config $configPath --frame $framePath
        }
    } else {
        Write-Host "`n✗ Config not saved (user may have quit)" -ForegroundColor Red
    }
    
    # Pause between sequences
    Write-Host ""
    $continue = Read-Host "Continue to next sequence? (Y/n)"
    if ($continue -eq "n" -or $continue -eq "N") {
        Write-Host "Stopping batch processing." -ForegroundColor Yellow
        break
    }
}

Write-Host "`n=== Batch Processing Complete ===" -ForegroundColor Cyan
Write-Host "Check configs/ directory for all generated configuration files."
