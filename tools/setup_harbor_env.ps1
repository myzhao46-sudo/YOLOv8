[CmdletBinding()]
param([string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $repo '.local-runtime'
$venv = Join-Path $repo '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$logs = Join-Path $runtime 'logs'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
$env:PIP_CACHE_DIR = Join-Path $runtime 'pip-cache'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PYTHONUTF8 = '1'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Start-Transcript -Path (Join-Path $logs "setup_$stamp.log")
try {
    if (-not (Test-Path -LiteralPath $python)) {
        if (-not $PythonPath) { $PythonPath = Join-Path $runtime 'python310\python.exe' }
        if (-not (Test-Path -LiteralPath $PythonPath)) {
            throw 'Python not found. Install Python 3.10 x64, then pass -PythonPath <python.exe>.'
        }
        Write-Host '[1/5] Creating the repository virtual environment'
        & $PythonPath -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
    }
    Write-Host '[2/5] Installing packaging tools'
    & $python -m pip install --progress-bar on --log (Join-Path $logs "packaging_$stamp.log") pip==26.0.1 setuptools==80.10.2 wheel==0.46.3
    if ($LASTEXITCODE -ne 0) { throw 'Packaging tools installation failed' }
    Write-Host '[3/5] Downloading/installing PyTorch 2.7.1 + CUDA 12.8 (large download)'
    $torchSource = 'torch==2.7.1'
    $cachedWheel = Join-Path $runtime 'wheels\torch-2.7.1+cu128-cp310-cp310-win_amd64.whl'
    if (Test-Path -LiteralPath $cachedWheel) {
        Write-Host 'Verifying the cached official PyTorch wheel SHA256'
        $actualHash = (Get-FileHash -LiteralPath $cachedWheel -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne '5174f02de8ca14df87c8e333c4c39cf3ce93a323c9d470d690301d110a053b3c') {
            throw 'Cached wheel is incomplete or damaged. Complete its download before retrying setup.'
        }
        $torchSource = $cachedWheel
    }
    & $python -m pip install --progress-bar raw --log (Join-Path $logs "torch_$stamp.log") $torchSource torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 -c (Join-Path $repo 'requirements\harbor-runtime.txt')
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    Write-Host '[4/5] Installing pinned project dependencies'
    & $python -m pip install --progress-bar raw --log (Join-Path $logs "dependencies_$stamp.log") -r (Join-Path $repo 'requirements\harbor-runtime.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed' }
    Write-Host '[5/5] Binding this checkout of Ultralytics and checking dependencies'
    & $python -m pip install --no-deps --no-build-isolation -e (Join-Path $repo 'ultralytics')
    if ($LASTEXITCODE -ne 0) { throw 'Local Ultralytics installation failed' }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
    Write-Host "Environment ready: $python"
} finally { Stop-Transcript }
