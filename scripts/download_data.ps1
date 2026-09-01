# Downloads the real MNIST and California Housing datasets into data/.
# Windows / PowerShell. Uses only built-in .NET facilities: no curl, no
# gzip, no extra tooling required.
#
# The project runs fine without this script (it falls back to a
# deterministic synthetic dataset), but running it once gives you the
# real data.

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$mnistDir = Join-Path $root "data\mnist"
$housingDir = Join-Path $root "data\housing"

New-Item -ItemType Directory -Force -Path $mnistDir | Out-Null
New-Item -ItemType Directory -Force -Path $housingDir | Out-Null

function Expand-GzipFile($sourcePath, $destinationPath) {
    $input = New-Object System.IO.FileStream $sourcePath, ([IO.FileMode]::Open), ([IO.FileAccess]::Read)
    $output = New-Object System.IO.FileStream $destinationPath, ([IO.FileMode]::Create), ([IO.FileAccess]::Write)
    $gzip = New-Object System.IO.Compression.GZipStream $input, ([IO.Compression.CompressionMode]::Decompress)
    try {
        $gzip.CopyTo($output)
    } finally {
        $gzip.Dispose(); $output.Dispose(); $input.Dispose()
    }
}

$mnistBase = "https://storage.googleapis.com/cvdf-datasets/mnist"
$mnistFiles = @(
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
)

foreach ($name in $mnistFiles) {
    $target = Join-Path $mnistDir $name
    if (Test-Path $target) {
        Write-Host "already present: $name"
        continue
    }
    $archive = "$target.gz"
    Write-Host "downloading $name ..."
    Invoke-WebRequest -Uri "$mnistBase/$name.gz" -OutFile $archive
    Expand-GzipFile $archive $target
    Remove-Item $archive
}

$housingTarget = Join-Path $housingDir "housing.csv"
if (Test-Path $housingTarget) {
    Write-Host "already present: housing.csv"
} else {
    Write-Host "downloading housing.csv ..."
    Invoke-WebRequest `
        -Uri "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv" `
        -OutFile $housingTarget
}

Write-Host ""
Write-Host "Datasets ready in data/."
