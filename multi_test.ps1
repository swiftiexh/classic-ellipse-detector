$env:PATH += ";D:\OpenCV\Build\install\x64\vc17\bin"

$outputDir = "multi_test_results"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

Write-Host "=== Testing Synthetic Overlap Ellipses ==="
Write-Host ""

$testImages = @(
    "aamed_ellipse_datasets/Synthetic Images - Overlap Ellipses/images/synth_overlap_4ellipses_img1.jpg",
    "aamed_ellipse_datasets/Synthetic Images - Overlap Ellipses/images/synth_overlap_4ellipses_img2.jpg",
    "aamed_ellipse_datasets/Synthetic Images - Overlap Ellipses/images/synth_overlap_8ellipses_img1.jpg",
    "aamed_ellipse_datasets/Synthetic Images - Overlap Ellipses/images/synth_overlap_8ellipses_img2.jpg",
    "aamed_ellipse_datasets/Synthetic Images - Overlap Ellipses/images/synth_overlap_12ellipses_img1.jpg",
    "aamed_ellipse_datasets/Synthetic Images - Overlap Ellipses/images/synth_overlap_12ellipses_img2.jpg",
    "aamed_ellipse_datasets/Random Images - Dataset #1/images/im1.jpg",
    "aamed_ellipse_datasets/Random Images - Dataset #1/images/im5.jpg",
    "aamed_ellipse_datasets/Random Images - Dataset #1/images/im9.jpg",
    "aamed_ellipse_datasets/Random Images - Dataset #1/images/im15.jpg"
)

$results = @()

foreach ($imgPath in $testImages) {
    if (-not (Test-Path $imgPath)) {
        Write-Host "Skip: $imgPath (not found)"
        continue
    }
    
    $imgName = [System.IO.Path]::GetFileName($imgPath)
    $imgOutputDir = Join-Path $outputDir $imgName
    New-Item -ItemType Directory -Path $imgOutputDir -Force | Out-Null
    
    $output = & .\build-vs\bin\Debug\aamed_demo.exe --input $imgPath --output-dir $imgOutputDir 2>&1
    $detections = 0
    
    foreach ($line in $output) {
        if ($line -match "Detections:\s*(\d+)") {
            $detections = [int]$matches[1]
            break
        }
    }
    
    $results += [PSCustomObject]@{
        Image = $imgName
        Detections = $detections
    }
    
    Write-Host "$imgName : $detections ellipses"
}

Write-Host ""
Write-Host "=== Test Completed ==="
$results | Format-Table -AutoSize