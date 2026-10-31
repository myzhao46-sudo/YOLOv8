[CmdletBinding()]
param(
    [ValidateSet('DryRun', 'OneEpoch', 'Train', 'Resume', 'Status')]
    [string]$Mode = 'DryRun',
    [string]$DataRoot = 'D:\RSdataset',
    [string]$ExperimentDir = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run tools\setup_harbor_env.ps1 first.' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $candidates = @(
        (Join-Path $env:ProgramFiles 'Git\cmd\git.exe'),
        (Join-Path $env:LOCALAPPDATA 'GitHubDesktop\app-*\resources\app\git\cmd\git.exe')
    )
    $gitExe = Get-ChildItem -Path $candidates -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $gitExe) { throw 'Git is required by the training source snapshot.' }
    $env:PATH = (Split-Path -Parent $gitExe.FullName) + ';' + $env:PATH
}
$env:WANDB_MODE = 'disabled'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$arguments = @('-B', '-u', '-X', 'utf8', (Join-Path $repo 'tools\train_harbor_expert.py'))
if ($Mode -in @('Resume', 'Status')) {
    if (-not $ExperimentDir) { throw 'Resume/Status requires -ExperimentDir <existing CLI run>.' }
    $arguments += @('--experiment_dir', $ExperimentDir)
    if ($Mode -eq 'Resume') { $arguments += '--resume' }
    else { $arguments += @('--status', '--tail', '12') }
} else {
    $arguments += @(
        '--base_weights', (Join-Path $repo 'ultralytics\best.pt'),
        '--data_root_dir', (Join-Path $DataRoot '_derived\harbor_poc_20261007_124509'),
        '--reference_experiment', (Join-Path $DataRoot '_experiments\harbor_poc_20261007_124509'),
        '--run_root_dir', (Join-Path $DataRoot 'experiments'),
        '--run_id_note', 'harbor_rgb_sar', '--device', '0',
        '--epochs', '150', '--patience', '30', '--seed', '43',
        '--image_size', '640', '--batch_size', '4', '--grad_accumulation_steps', '4',
        '--val_batch_size', '4', '--num_workers', '4', '--val_num_workers', '0',
        '--num_threads', '8', '--num_interop_threads', '1',
        '--prefetch_factor', '2', '--persistent_workers', 'True',
        '--pin_memory', 'True', '--cache_images', 'shared',
        '--optimizer', 'AdamW', '--learning_rate', '1e-4', '--weight_decay', '5e-4',
        '--warmup_epochs', '3', '--lrf', '0.1', '--gradient_clip', '10',
        '--rgb_repeat', '4', '--sar_repeat', '1',
        '--horizontal_flip', '0.5', '--vertical_flip', '0.5',
        '--box_loss_gain', '7.5', '--cls_loss_gain', '0.5', '--dfl_loss_gain', '1.5',
        '--amp', 'False'
    )
    if ($ExperimentDir) { $arguments += @('--experiment_dir', $ExperimentDir) }
    if ($Mode -eq 'OneEpoch') { $arguments += @('--pause_after_epoch', '1') }
    if ($Mode -eq 'DryRun') { $arguments += '--dry_run' }
}
Push-Location $repo
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Harbor command failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
