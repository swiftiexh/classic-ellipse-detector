param(
    [string]$ProjectRoot = (Resolve-Path ".").Path,
    [string[]]$ImageNames = @("031_0035.jpg", "031_0051.jpg")
)

Add-Type -AssemblyName System.Drawing

$imageDir = Join-Path $ProjectRoot "datasets\prasad\images"
$baselineDir = Join-Path $ProjectRoot "datasets\prasad\AAMED"
$sccDir = Join-Path $ProjectRoot "output\scc_geofilter_prasad_strict"
$figDir = Join-Path $ProjectRoot "scc_doc\figures"
New-Item -ItemType Directory -Force -Path $figDir | Out-Null

function Read-FledEllipses {
    param([string]$Path)
    $ellipses = @()
    if (-not (Test-Path -LiteralPath $Path)) {
        return $ellipses
    }
    $lines = Get-Content -LiteralPath $Path
    foreach ($line in $lines | Select-Object -Skip 1) {
        $parts = $line.Trim() -split "\s+"
        if ($parts.Count -lt 6) {
            continue
        }
        try {
            $values = $parts | ForEach-Object { [double]$_ }
        } catch {
            continue
        }
        if ([int]$values[0] -eq 2) {
            continue
        }
        $ellipses += [pscustomobject]@{
            X = $values[2] + 1.0
            Y = $values[1] + 1.0
            W = $values[3]
            H = $values[4]
            Angle = -$values[5]
        }
    }
    return $ellipses
}

function Draw-EllipseSet {
    param(
        [string]$ImagePath,
        [array]$Ellipses,
        [string]$OutPath,
        [System.Drawing.Color]$Color
    )
    $bitmap = [System.Drawing.Bitmap]::FromFile($ImagePath)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $pen = New-Object System.Drawing.Pen($Color, 2)
    foreach ($ellipse in $Ellipses) {
        $state = $graphics.Save()
        $graphics.TranslateTransform([single]$ellipse.X, [single]$ellipse.Y)
        $graphics.RotateTransform([single]$ellipse.Angle)
        $rect = New-Object System.Drawing.RectangleF(
            [single](-$ellipse.W / 2.0),
            [single](-$ellipse.H / 2.0),
            [single]$ellipse.W,
            [single]$ellipse.H
        )
        $graphics.DrawEllipse($pen, $rect)
        $graphics.Restore($state)
    }
    $bitmap.Save($OutPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $pen.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}

function Draw-Comparison {
    param(
        [string]$ImagePath,
        [array]$BaselineEllipses,
        [array]$SccEllipses,
        [string]$OutPath
    )
    $bitmap = [System.Drawing.Bitmap]::FromFile($ImagePath)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $baselinePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(220, 255, 80, 80), 2)
    $sccPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(220, 60, 220, 100), 2)

    foreach ($set in @(@($BaselineEllipses, $baselinePen), @($SccEllipses, $sccPen))) {
        foreach ($ellipse in $set[0]) {
            $state = $graphics.Save()
            $graphics.TranslateTransform([single]$ellipse.X, [single]$ellipse.Y)
            $graphics.RotateTransform([single]$ellipse.Angle)
            $rect = New-Object System.Drawing.RectangleF(
                [single](-$ellipse.W / 2.0),
                [single](-$ellipse.H / 2.0),
                [single]$ellipse.W,
                [single]$ellipse.H
            )
            $graphics.DrawEllipse($set[1], $rect)
            $graphics.Restore($state)
        }
    }

    $bitmap.Save($OutPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $baselinePen.Dispose()
    $sccPen.Dispose()
    $graphics.Dispose()
    $bitmap.Dispose()
}

$primary = $ImageNames[0]
$secondary = if ($ImageNames.Count -gt 1) { $ImageNames[1] } else { $ImageNames[0] }

$primaryImage = Join-Path $imageDir $primary
$primaryBaseline = Read-FledEllipses (Join-Path $baselineDir "$primary.fled.txt")
$primaryScc = Read-FledEllipses (Join-Path $sccDir "$primary.fled.txt")

Draw-EllipseSet $primaryImage $primaryBaseline (Join-Path $figDir "baseline_case_001.png") ([System.Drawing.Color]::Red)
Draw-EllipseSet $primaryImage $primaryScc (Join-Path $figDir "scc_case_001.png") ([System.Drawing.Color]::LimeGreen)
Draw-Comparison $primaryImage $primaryBaseline $primaryScc (Join-Path $figDir "success_case_001.png")

$secondaryImage = Join-Path $imageDir $secondary
$secondaryBaseline = Read-FledEllipses (Join-Path $baselineDir "$secondary.fled.txt")
$secondaryScc = Read-FledEllipses (Join-Path $sccDir "$secondary.fled.txt")
Draw-Comparison $secondaryImage $secondaryBaseline $secondaryScc (Join-Path $figDir "failure_case_001.png")

Write-Output "Generated figures in $figDir"
