$env:PATH += ";D:\OpenCV\Build\install\x64\vc17\bin"

$outputDir = "batch_results_final"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$resultsFile = Join-Path $outputDir "test_results.txt"
if (Test-Path $resultsFile) {
    Remove-Item $resultsFile -Force
}

$allImages = Get-ChildItem -Path "aamed_ellipse_datasets" -Recurse -Filter "*.jpg"

$totalImages = $allImages.Count
$processed = 0
$totalDetections = 0
$imagesWithDetections = 0

Add-Content $resultsFile "=== 椭圆检测批量测试报告 ==="
Add-Content $resultsFile "测试时间: $(Get-Date)"
Add-Content $resultsFile "测试图片数量: $totalImages"
Add-Content $resultsFile ""
Add-Content $resultsFile "=============================="
Add-Content $resultsFile ""

foreach ($img in $allImages) {
    $processed++
    Write-Progress -Activity "批量测试" -Status "处理中: $processed/$totalImages" -CurrentOperation $img.Name
    
    $imgOutputDir = Join-Path $outputDir $img.BaseName
    New-Item -ItemType Directory -Path $imgOutputDir -Force | Out-Null
    
    try {
        $result = & .\build-vs\bin\Debug\aamed_demo.exe --input $img.FullName --output-dir $imgOutputDir 2>&1
        
        $detections = 0
        foreach ($line in $result) {
            if ($line -match "Detections:\s*(\d+)") {
                $detections = [int]$matches[1]
                break
            }
        }
        
        $totalDetections += $detections
        if ($detections -gt 0) {
            $imagesWithDetections++
        }
        
        Add-Content $resultsFile "[$processed/$totalImages] 图片: $($img.Name)"
        Add-Content $resultsFile "    检测数量: $detections"
        Add-Content $resultsFile ""
    }
    catch {
        Add-Content $resultsFile "[$processed/$totalImages] 图片: $($img.Name)"
        Add-Content $resultsFile "    错误: $($_.Exception.Message)"
        Add-Content $resultsFile ""
    }
}

Add-Content $resultsFile "=============================="
Add-Content $resultsFile "=== 测试统计 ==="
Add-Content $resultsFile "总检测椭圆数: $totalDetections"
Add-Content $resultsFile "有检测结果的图片数: $imagesWithDetections"
Add-Content $resultsFile "平均每张图检测: $([math]::Round($totalDetections / $totalImages, 2))"

Write-Progress -Activity "批量测试" -Completed
Write-Host "批量测试完成！结果已保存到 $resultsFile"