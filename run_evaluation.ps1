$env:PATH += ";D:\OpenCV\Build\install\x64\vc17\bin"

$datasetPath = "aamed_ellipse_datasets/Random Images - Dataset #1"
$outputPath = "eval_output"
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

Write-Host "=== 1. 运行椭圆检测 ==="
$imageNames = Get-Content "$datasetPath/imagenames.txt" | Select-Object -First 10

foreach ($imgName in $imageNames) {
    $imgPath = Join-Path "$datasetPath/images" $imgName
    if (-not (Test-Path $imgPath)) {
        Write-Host "跳过: $imgName (文件不存在)"
        continue
    }
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($imgName)
    $fledFile = Join-Path $outputPath "$baseName.fled.txt"
    
    & .\build-vs\bin\Debug\aamed_demo.exe --input $imgPath --output-dir $outputPath --quiet 2>&1 | Out-Null
    
    if (Test-Path $fledFile) {
        Write-Host "处理成功: $imgName"
    } else {
        Write-Host "处理失败: $imgName"
    }
}

Write-Host ""
Write-Host "=== 2. 运行评估 ==="

& .\build-vs\bin\Debug\aamed_eval.exe `
    --dataset-root $datasetPath `
    --results-dir $outputPath `
    --gt-format plain_rad `
    --result-format aamed_fled `
    --overlap 0.8 `
    --report eval_result.txt

Write-Host ""
Write-Host "=== 评估完成 ==="
Write-Host ""
Get-Content eval_result.txt