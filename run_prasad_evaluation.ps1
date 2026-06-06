$env:PATH += ";D:\OpenCV\Build\install\x64\vc17\bin"

$datasetPath = "aamed_ellipse_datasets/Prasad Images - Dataset Prasad"
$outputPath = "prasad_eval_output"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

Write-Host "=== 运行优化后的椭圆检测 ==="
$imageNames = Get-Content "$datasetPath/imagenames.txt"

$count = 0
$successCount = 0
foreach ($imgName in $imageNames) {
    $imgPath = Join-Path "$datasetPath/images" $imgName
    if (-not (Test-Path $imgPath)) {
        continue
    }
    
    & .\build-vs\bin\Debug\aamed_demo.exe --input $imgPath --output-dir $outputPath --quiet 2>&1 | Out-Null
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($imgName)
    $fledFile = Join-Path $outputPath "$baseName.fled.txt"
    
    if (Test-Path $fledFile) {
        $successCount++
    }
    $count++
    
    if ($count % 20 -eq 0) {
        Write-Host "已处理: $count / $($imageNames.Count)"
    }
}

Write-Host ""
Write-Host "=== 检测完成 ==="
Write-Host "成功处理: $successCount / $count"

Write-Host ""
Write-Host "=== 运行评估 ==="

& .\build-vs\bin\Debug\aamed_eval.exe `
    --imagenames "$datasetPath/imagenames.txt" `
    --gt-dir "$datasetPath/gt" `
    --results-dir $outputPath `
    --gt-format prasad `
    --result-format aamed_fled `
    --gt-prefix "gt_" `
    --overlap 0.8 `
    --report enhanced_prasad.txt

Write-Host ""
Write-Host "=== 评估完成 ==="
Write-Host ""
Get-Content enhanced_prasad.txt