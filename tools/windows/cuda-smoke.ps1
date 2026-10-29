[CmdletBinding()]
param(
    [string]$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$BuildRoot = '',
    [string]$ConfigurePreset = 'cuda-windows',
    [ValidateRange(0, 2147483647)][int]$Device = 0
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion -lt [version]'7.4') { throw 'PowerShell 7.4 or newer is required for byte-preserving native output capture' }
if (-not [OperatingSystem]::IsWindows()) { throw 'cuda-smoke.ps1 requires native Windows' }
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) { throw 'MSVC compiler is unavailable; enter an x64 Visual Studio developer shell first' }
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
if (-not $BuildRoot) { $BuildRoot = Join-Path $SourceRoot 'build/cuda-windows-evidence' }
$evidence = Join-Path ([IO.Path]::GetFullPath($BuildRoot)) ((Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8))
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
    $evidenceTool = Join-Path $SourceRoot 'tools/m1_evidence.py'
    Invoke-Checked 'Source snapshot before build' 'python' @('-B', $evidenceTool, 'snapshot', $SourceRoot, (Join-Path $evidence 'source-before.json'))
    Invoke-Checked 'NVCC version' 'nvcc' @('--version')
    Invoke-Checked 'CMake version' 'cmake' @('--version')
    $versionSource = Join-Path $evidence 'compiler-version.cpp'
    'int main() { return 0; }' | Set-Content -Path $versionSource -Encoding utf8NoBOM
    Invoke-Checked 'MSVC compiler version' 'cl' @('/nologo', '/Bv', '/c', $versionSource, "/Fo$(Join-Path $evidence 'compiler-version.obj')")
    $nvidiaSmi = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nvidiaSmi) {
        & $nvidiaSmi.Source '--query-gpu=name,uuid,driver_version,power.draw,display_active' '--format=csv,noheader' 2>&1 | Tee-Object -FilePath (Join-Path $evidence 'nvidia-smi.log')
        $smiCode = $LASTEXITCODE
        "NVIDIA_SMI_EXIT_CODE: $smiCode" | Tee-Object -FilePath $log -Append
        if ($smiCode -ne 0) { 'NVIDIA_SMI_STATUS: blocked (query unavailable)' | Tee-Object -FilePath $log -Append }
    } else {
        'NVIDIA_SMI_STATUS: blocked (nvidia-smi unavailable)' | Tee-Object -FilePath $log -Append
    }
    $build = Join-Path $evidence 'build'
    Invoke-Checked 'Configure CUDA' 'cmake' @('-S', $SourceRoot, '-B', $build, '--preset', $ConfigurePreset)
    $cache = Join-Path $build 'CMakeCache.txt'
    if (-not (Test-Path $cache)) { throw 'Configure succeeded but CMakeCache.txt is absent' }
    Copy-Item $cache (Join-Path $evidence 'CMakeCache.txt')
    $compilerFacts = Join-Path $evidence 'compiler-facts'
    New-Item -ItemType Directory -Path $compilerFacts | Out-Null
    $compilerFiles = @(Get-ChildItem (Join-Path $build 'CMakeFiles') -Recurse -File -Include 'CMakeCXXCompiler.cmake', 'CMakeCUDACompiler.cmake')
    if ($compilerFiles.Count -lt 2) { throw 'CMake C++/CUDA compiler facts are missing' }
    $compilerFiles | Copy-Item -Destination $compilerFacts
    Get-Content $cache | Select-String '^(CMAKE_(CUDA_COMPILER|CUDA_COMPILER_VERSION|CXX_COMPILER|CXX_COMPILER_VERSION|GENERATOR)|SAGA_ENABLE_CUDA)' | Tee-Object -FilePath $log -Append
    Invoke-Checked 'Verbose build and device link' 'cmake' @('--build', $build, '--target', 'saga_cuda_portability_smoke', '--verbose', '--parallel', '2')
    $exe = Join-Path $build 'saga_cuda_portability_smoke.exe'
    if (-not (Test-Path $exe)) { throw "CUDA smoke executable missing: $exe" }
    $runtimeFile = Join-Path $evidence 'runtime.json'
    $runtimeErrors = Join-Path $evidence 'runtime.stderr.log'
    & $exe --device "$Device" 1> $runtimeFile 2> $runtimeErrors
    $runtimeCode = $LASTEXITCODE
    Get-Content $runtimeFile -Encoding utf8 | Tee-Object -FilePath $log -Append
    Get-Content $runtimeErrors -Encoding utf8 | Tee-Object -FilePath $log -Append
    "RUNTIME_EXIT_CODE: $runtimeCode" | Tee-Object -FilePath $log -Append
    if ($runtimeCode -ne 0) { throw "CUDA runtime probe failed with native exit code $runtimeCode" }

    Invoke-Checked 'Validate complete CUDA runtime report' 'python' @('-B', $evidenceTool, 'validate-cuda', $runtimeFile, '--device', "$Device")
    Invoke-Checked 'Source snapshot after build and comparison' 'python' @('-B', $evidenceTool, 'snapshot', $SourceRoot, (Join-Path $evidence 'source-after.json'))
    Invoke-Checked 'Source snapshot comparison' 'python' @('-B', $evidenceTool, 'compare', (Join-Path $evidence 'source-before.json'), (Join-Path $evidence 'source-after.json'))
    'RESULT: PASS — inspect retained JSON and build log; this probe does not qualify serving.' | Tee-Object -FilePath $log -Append
} catch {
    "RESULT: FAIL`nERROR: $_" | Tee-Object -FilePath $log -Append
    throw
} finally {
    try {
        if ($pushedLocation) { Pop-Location }
    } finally {
        Stop-Transcript | Out-Null
    }
}
