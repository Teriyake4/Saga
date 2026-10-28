[CmdletBinding()]
param(
    [string]$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$BuildRoot = '',
    [string]$ConfigurePreset = 'cuda-windows',
    [ValidateRange(0, 2147483647)][int]$Device = 0
)
$ErrorActionPreference = 'Stop'
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
if (-not $BuildRoot) { $BuildRoot = Join-Path $SourceRoot 'build/cuda-windows-evidence' }
$evidence = Join-Path ([IO.Path]::GetFullPath($BuildRoot)) (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Force -Path $evidence | Out-Null
$log = Join-Path $evidence 'cuda-smoke.log'
$transcript = Join-Path $evidence 'transcript.txt'
Start-Transcript -Path $transcript -Force | Out-Null
function Invoke-Checked([string]$Name, [string]$Executable, [string[]]$Arguments, [string]$OutputFile = '') {
    "`n=== $Name ===`nCOMMAND: $Executable $($Arguments -join ' ')" | Tee-Object -FilePath $log -Append
    if ($OutputFile) { & $Executable @Arguments 2>&1 | Tee-Object -FilePath $OutputFile | Tee-Object -FilePath $log -Append }
    else { & $Executable @Arguments 2>&1 | Tee-Object -FilePath $log -Append }
    $code = $LASTEXITCODE
    "EXIT_CODE: $code" | Tee-Object -FilePath $log -Append
    if ($code -ne 0) { throw "$Name failed with native exit code $code" }
}
$pushedLocation = $false
try {
    Push-Location $SourceRoot
    $pushedLocation = $true
    "UTC: $([DateTime]::UtcNow.ToString('o'))`nSource: $SourceRoot`nEvidence: $evidence`nDevice ordinal: $Device`nPreset: $ConfigurePreset`nOS: $([Environment]::OSVersion)`nPowerShell: $($PSVersionTable.PSVersion)" | Tee-Object -FilePath $log -Append
    foreach ($tool in 'nvcc', 'cmake', 'cl') {
        $found = Get-Command $tool -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { "TOOL $tool : $($found.Source)" | Tee-Object -FilePath $log -Append }
        else { "TOOL $tool : not directly available on PATH" | Tee-Object -FilePath $log -Append }
    }
    Invoke-Checked 'NVCC version' 'nvcc' @('--version')
    Invoke-Checked 'CMake version' 'cmake' @('--version')
    $build = Join-Path $evidence 'build'
    Invoke-Checked 'Configure CUDA' 'cmake' @('-S', $SourceRoot, '-B', $build, '--preset', $ConfigurePreset)
    Get-Content (Join-Path $build 'CMakeCache.txt') | Select-String '^(CMAKE_(CUDA_COMPILER|CUDA_COMPILER_VERSION|CXX_COMPILER|CXX_COMPILER_VERSION|GENERATOR)|SAGA_ENABLE_CUDA)' | Tee-Object -FilePath $log -Append
    Invoke-Checked 'Verbose build and device link' 'cmake' @('--build', $build, '--target', 'saga_cuda_portability_smoke', '--verbose', '--parallel', '2')
    $exe = Join-Path $build 'saga_cuda_portability_smoke.exe'
    if (-not (Test-Path $exe)) { throw "CUDA smoke executable missing: $exe" }
    $runtimeFile = Join-Path $evidence 'runtime.json'
    $runtimeErrors = Join-Path $evidence 'runtime.stderr.log'
    & $exe --device "$Device" 1> $runtimeFile 2> $runtimeErrors
    $runtimeCode = $LASTEXITCODE
    Get-Content $runtimeFile | Tee-Object -FilePath $log -Append
    Get-Content $runtimeErrors | Tee-Object -FilePath $log -Append
    "RUNTIME_EXIT_CODE: $runtimeCode" | Tee-Object -FilePath $log -Append
    if ($runtimeCode -ne 0) { throw "CUDA runtime probe failed with native exit code $runtimeCode" }
    $runtime = Get-Content -Raw $runtimeFile | ConvertFrom-Json -ErrorAction Stop
    if ($runtime.passed -ne $true) { throw 'CUDA runtime JSON did not report passed=true' }
    'RESULT: PASS — inspect retained JSON and build log; this probe does not qualify serving.' | Tee-Object -FilePath $log -Append
} catch {
    "RESULT: FAIL`nERROR: $_" | Tee-Object -FilePath $log -Append
    throw
} finally {
    if ($pushedLocation) { Pop-Location }
    Stop-Transcript | Out-Null
}
