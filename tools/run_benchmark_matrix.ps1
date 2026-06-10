param(
    [switch]$ListOnly,
    [switch]$ValidateOnly,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build\cmake_build"
$OutputRoot = Join-Path $ProjectRoot "output"
$SummaryDir = Join-Path $OutputRoot "benchmark_matrix"

$Methods = @(
    [pscustomobject]@{ Key = "baseline"; Env = "baseline"; Display = "Baseline (isCombValid fixed)" },
    [pscustomobject]@{ Key = "weighted_arc"; Env = "weighted_arc"; Display = "weighted-arc only" },
    [pscustomobject]@{ Key = "multi_scale_fpn"; Env = "multi_scale_fpn"; Display = "multi_scale_fpn only" },
    [pscustomobject]@{ Key = "small_ellipse_guard"; Env = "small_ellipse_guard"; Display = "small_ellipse_guard only" },
    [pscustomobject]@{ Key = "weighted_arc__small_ellipse_guard"; Env = "weighted_arc,small_ellipse_guard"; Display = "weighted-arc + small_ellipse_guard" },
    [pscustomobject]@{ Key = "weighted_arc__multi_scale_fpn"; Env = "weighted_arc,multi_scale_fpn"; Display = "weighted-arc + multi_scale_fpn" },
    [pscustomobject]@{ Key = "multi_scale_fpn__small_ellipse_guard"; Env = "multi_scale_fpn,small_ellipse_guard"; Display = "small_ellipse_guard + multi_scale_fpn" },
    [pscustomobject]@{ Key = "weighted_arc__multi_scale_fpn__small_ellipse_guard"; Env = "weighted_arc,multi_scale_fpn,small_ellipse_guard"; Display = "weighted-arc + small_ellipse_guard + multi_scale_fpn" }
)

$MissingRuns = @(
    [pscustomobject]@{ Dataset = "prasad"; ExpectedImages = 198; Method = $Methods[4] },
    [pscustomobject]@{ Dataset = "prasad"; ExpectedImages = 198; Method = $Methods[5] },
    [pscustomobject]@{ Dataset = "prasad"; ExpectedImages = 198; Method = $Methods[6] },
    [pscustomobject]@{ Dataset = "prasad"; ExpectedImages = 198; Method = $Methods[7] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[1] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[2] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[3] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[4] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[5] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[6] },
    [pscustomobject]@{ Dataset = "random"; ExpectedImages = 400; Method = $Methods[7] }
)

$ProvidedRows = @(
    [pscustomobject]@{ Dataset = "Prasad"; Configuration = "Official"; Precision = "77.13"; Recall = "39.66"; FMeasure = "52.38"; AverageDetectedTimeMs = "4.21"; Source = "provided" },
    [pscustomobject]@{ Dataset = "Prasad"; Configuration = "Baseline (isCombValid fixed)"; Precision = "77.78"; Recall = "40.26"; FMeasure = "53.05"; AverageDetectedTimeMs = "3.67"; Source = "provided" },
    [pscustomobject]@{ Dataset = "Prasad"; Configuration = "weighted-arc only"; Precision = "77.72"; Recall = "40.43"; FMeasure = "53.19"; AverageDetectedTimeMs = "3.73"; Source = "provided" },
    [pscustomobject]@{ Dataset = "Prasad"; Configuration = "multi_scale_fpn only"; Precision = "64.93"; Recall = "46.09"; FMeasure = "53.92"; AverageDetectedTimeMs = "70.74"; Source = "provided" },
    [pscustomobject]@{ Dataset = "Prasad"; Configuration = "small_ellipse_guard only"; Precision = "78.17"; Recall = "40.26"; FMeasure = "53.14"; AverageDetectedTimeMs = "3.55"; Source = "provided" },
    [pscustomobject]@{ Dataset = "Random"; Configuration = "Official"; Precision = "65.52"; Recall = "50.65"; FMeasure = "57.13"; AverageDetectedTimeMs = "11.07"; Source = "provided" },
    [pscustomobject]@{ Dataset = "Random"; Configuration = "Baseline (isCombValid fixed)"; Precision = "65.50"; Recall = "51.08"; FMeasure = "57.40"; AverageDetectedTimeMs = "9.83"; Source = "provided" }
)

function Set-ExperimentEnvironment($Dataset, $MethodEnv) {
    $env:AAMED_DATASET = $Dataset
    $env:AAMED_METHODS = $MethodEnv
}

function Find-Executable($Name) {
    $candidate = Get-ChildItem -LiteralPath $BuildDir -Recurse -File -Filter $Name |
        Where-Object { $_.FullName -match "[\\/]bin[\\/]" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "Could not find $Name under $BuildDir"
    }
    return $candidate.FullName
}

function Assert-Contains($Text, $Expected, $Context) {
    if ($Text -notmatch [regex]::Escape($Expected)) {
        throw "$Context did not contain expected text: $Expected"
    }
}

function Validate-Configurations($DemoExe, $EvalExe) {
    Write-Host "Validating all 16 dataset/method mappings..."
    foreach ($dataset in @("prasad", "random")) {
        foreach ($method in $Methods) {
            Set-ExperimentEnvironment $dataset $method.Env
            $env:AAMED_PRINT_CONFIG = "1"
            $demoConfig = & $DemoExe 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) { throw "Detector config validation failed: $dataset/$($method.Key)" }
            $evalConfig = & $EvalExe 2>&1 | Out-String
            if ($LASTEXITCODE -ne 0) { throw "Evaluator config validation failed: $dataset/$($method.Key)" }

            foreach ($text in @($demoConfig, $evalConfig)) {
                Assert-Contains $text "Dataset: $dataset" "$dataset/$($method.Key)"
                Assert-Contains $text "Experiment: $($method.Key)" "$dataset/$($method.Key)"
                Assert-Contains $text "output\$dataset\$($method.Key)" "$dataset/$($method.Key)"
                Assert-Contains $text "GtPrefix: gt_" "$dataset/$($method.Key)"
                Assert-Contains $text "OverlapThreshold: 0.8" "$dataset/$($method.Key)"
            }
            $expectedConvention = if ($dataset -eq "prasad") { "GroundTruthConvention: 0" } else { "GroundTruthConvention: 1" }
            Assert-Contains $demoConfig $expectedConvention "$dataset/$($method.Key)"
            Assert-Contains $evalConfig $expectedConvention "$dataset/$($method.Key)"
        }
    }
    Remove-Item Env:AAMED_PRINT_CONFIG -ErrorAction SilentlyContinue
    Write-Host "Configuration validation passed."
}

function Remove-ExperimentOutput($Dataset, $MethodKey) {
    $target = Join-Path $OutputRoot (Join-Path $Dataset $MethodKey)
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot).TrimEnd('\') + '\'
    $resolvedTarget = [System.IO.Path]::GetFullPath($target)
    if (-not $resolvedTarget.StartsWith($resolvedOutput, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside output: $resolvedTarget"
    }
    if (Test-Path -LiteralPath $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
    }
}

function Read-Report($Path, $Dataset, $Method) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing evaluation report: $Path"
    }
    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^([^:]+):\s*(.+)$") {
            $values[$matches[1]] = $matches[2]
        }
    }
    foreach ($key in @("Images", "Precision", "Recall", "FMeasure", "AverageDetectedTimeMs")) {
        if (-not $values.ContainsKey($key)) { throw "Report $Path is missing $key" }
    }
    return [pscustomobject]@{
        Dataset = (Get-Culture).TextInfo.ToTitleCase($Dataset)
        Configuration = $Method.Display
        Precision = ([double]$values.Precision * 100.0).ToString("F2", [cultureinfo]::InvariantCulture)
        Recall = ([double]$values.Recall * 100.0).ToString("F2", [cultureinfo]::InvariantCulture)
        FMeasure = ([double]$values.FMeasure * 100.0).ToString("F2", [cultureinfo]::InvariantCulture)
        AverageDetectedTimeMs = ([double]$values.AverageDetectedTimeMs).ToString("F2", [cultureinfo]::InvariantCulture)
        Source = "measured"
    }
}

function Write-Summaries($Rows) {
    New-Item -ItemType Directory -Path $SummaryDir -Force | Out-Null
    $csvPath = Join-Path $SummaryDir "results.csv"
    $mdPath = Join-Path $SummaryDir "results.md"
    $Rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

    $lines = @(
        "| Dataset | Configuration | Precision | Recall | FMeasure | AverageDetectedTimeMs | Source |",
        "|---|---|---:|---:|---:|---:|---|"
    )
    foreach ($row in $Rows) {
        $lines += "| $($row.Dataset) | $($row.Configuration) | $($row.Precision) | $($row.Recall) | $($row.FMeasure) | $($row.AverageDetectedTimeMs) | $($row.Source) |"
    }
    Set-Content -LiteralPath $mdPath -Value $lines -Encoding UTF8
    Write-Host "Wrote $csvPath"
    Write-Host "Wrote $mdPath"
}

Write-Host "Missing experiment matrix:"
$MissingRuns | ForEach-Object { Write-Host ("  {0}/{1}" -f $_.Dataset, $_.Method.Key) }
if ($MissingRuns.Count -ne 11) { throw "Expected exactly 11 missing experiments." }
if ($ListOnly) { exit 0 }

if (-not $SkipBuild) {
    cmake --build $BuildDir --config Release
    if ($LASTEXITCODE -ne 0) { throw "Build failed." }
}

$DemoExe = Find-Executable "aamed_demo.exe"
$EvalExe = Find-Executable "aamed_eval.exe"
Validate-Configurations $DemoExe $EvalExe
if ($ValidateOnly) { exit 0 }

$MeasuredRows = @()
foreach ($run in $MissingRuns) {
    $dataset = $run.Dataset
    $method = $run.Method
    Write-Host "Running $dataset/$($method.Key)..."
    Remove-ExperimentOutput $dataset $method.Key
    Set-ExperimentEnvironment $dataset $method.Env

    & $DemoExe
    if ($LASTEXITCODE -ne 0) { throw "Detection failed: $dataset/$($method.Key)" }

    $resultDir = Join-Path $OutputRoot (Join-Path $dataset $method.Key)
    $summaryPath = Join-Path $resultDir "batch_summary.txt"
    if (-not (Test-Path -LiteralPath $summaryPath)) { throw "Missing batch summary: $summaryPath" }
    $summaryLine = Get-Content -LiteralPath $summaryPath -Tail 1
    if ($summaryLine -notmatch "^Summary\s+(\d+)\s+") { throw "Invalid batch summary: $summaryLine" }
    if ([int]$matches[1] -ne $run.ExpectedImages) {
        throw "Expected $($run.ExpectedImages) successful images, got $($matches[1]): $dataset/$($method.Key)"
    }

    & $EvalExe
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $dataset/$($method.Key)" }
    $MeasuredRows += Read-Report (Join-Path $resultDir "eval_report.txt") $dataset $method
}

$allRows = @()
foreach ($dataset in @("Prasad", "Random")) {
    $allRows += $ProvidedRows | Where-Object Dataset -eq $dataset
    $allRows += $MeasuredRows | Where-Object Dataset -eq $dataset
}
Write-Summaries $allRows
Write-Host "Completed all 11 missing experiments."
