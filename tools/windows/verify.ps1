[CmdletBinding()]
param(
    [string]$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path,
    [string]$BuildRoot = '',
    [ValidateSet('Debug', 'Release')][string]$Configuration = 'Debug',
    [string]$ConfigurePreset = ''
)
$ErrorActionPreference = 'Stop'
if (-not $ConfigurePreset) { $ConfigurePreset = 'windows-msvc-' + $Configuration.ToLowerInvariant() }
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
if (-not $BuildRoot) { $BuildRoot = Join-Path $SourceRoot 'build/windows-evidence' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$evidence = Join-Path $BuildRoot "$Configuration-$stamp"
$binary = Join-Path $evidence 'binary'
$install = Join-Path $evidence 'installed application with spaces'
New-Item -ItemType Directory -Force -Path $binary, $install | Out-Null
$log = Join-Path $evidence 'verify.log'
$transcript = Join-Path $evidence 'transcript.txt'
Start-Transcript -Path $transcript -Force | Out-Null
function Invoke-Checked([string]$Name, [string]$Executable, [string[]]$Arguments) {
    "`n=== $Name ===" | Tee-Object -FilePath $log -Append
    "COMMAND: $Executable $($Arguments -join ' ')" | Tee-Object -FilePath $log -Append
    & $Executable @Arguments 2>&1 | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    "EXIT_CODE: $code" | Tee-Object -FilePath $log -Append
    if ($code -ne 0) { throw "$Name failed with native exit code $code" }
}
$pushedLocation = $false
try {
    Push-Location $SourceRoot
    $pushedLocation = $true
    "UTC: $([DateTime]::UtcNow.ToString('o'))`nSource: $SourceRoot`nBuild: $binary`nInstall: $install`nConfiguration: $Configuration`nPreset: $ConfigurePreset`nOS: $([Environment]::OSVersion)`nPowerShell: $($PSVersionTable.PSVersion)" | Tee-Object -FilePath $log -Append
    foreach ($tool in 'cmake', 'ctest', 'python', 'cl') {
        $found = Get-Command $tool -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { "TOOL $tool : $($found.Source)" | Tee-Object -FilePath $log -Append }
        else { "TOOL $tool : not directly available on PATH" | Tee-Object -FilePath $log -Append }
    }
    foreach ($name in 'VSINSTALLDIR', 'VCToolsInstallDir', 'VCToolsVersion', 'WindowsSDKVersion') {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { "ENV $name=$value" | Tee-Object -FilePath $log -Append }
    }
    Invoke-Checked 'CMake version' 'cmake' @('--version')
    Invoke-Checked 'Configure' 'cmake' @('-S', $SourceRoot, '-B', $binary, '--preset', $ConfigurePreset, "-DCMAKE_INSTALL_PREFIX=$install")
    $cache = Join-Path $binary 'CMakeCache.txt'
    if (-not (Test-Path $cache)) { throw 'Configure succeeded but CMakeCache.txt is absent' }
    Get-Content $cache | Select-String '^(CMAKE_(CXX_COMPILER|CXX_COMPILER_ID|CXX_COMPILER_VERSION|MSVC_RUNTIME_LIBRARY|GENERATOR|INSTALL_PREFIX)|SAGA_)' | Tee-Object -FilePath $log -Append
    Invoke-Checked 'Build' 'cmake' @('--build', $binary, '--config', $Configuration, '--parallel', '2', '--verbose')
    Invoke-Checked 'CTest' 'ctest' @('--test-dir', $binary, '-C', $Configuration, '--output-on-failure')
    Invoke-Checked 'Install' 'cmake' @('--install', $binary, '--config', $Configuration)
    $server = Join-Path $install 'bin/saga-server.exe'
    $installedConfig = Join-Path $install 'share/saga/server.example.toml'
    $installedModelConfig = Join-Path $install 'share/saga/model.example.ini'
    if (-not (Test-Path $server)) { throw "Installed executable missing: $server" }
    if (-not (Test-Path $installedConfig)) { throw "Installed server config missing: $installedConfig" }
    Invoke-Checked 'Installed server config self-check' $server @('--config', $installedConfig, '--check-config')
    if (-not (Test-Path $installedModelConfig)) { throw "Installed model config missing: $installedModelConfig" }
    Invoke-Checked 'Installed model config self-check (single-model INI)' $server @('--config', $installedConfig, '--model-config', $installedModelConfig, '--check-config')
    Invoke-Checked 'Dependency hashes' 'python' @('-B', (Join-Path $SourceRoot 'tools/verify_dependencies.py'))
    Invoke-Checked 'Source verifier tests and hashes retained by CTest' 'ctest' @('--test-dir', $binary, '-C', $Configuration, '--output-on-failure', '-R', 'dependency_hashes|source_manifest_structure|source_verifier_tests')
    'RESULT: PASS' | Tee-Object -FilePath $log -Append
} catch {
    "RESULT: FAIL`nERROR: $_" | Tee-Object -FilePath $log -Append
    throw
} finally {
    if ($pushedLocation) { Pop-Location }
    Stop-Transcript | Out-Null
}
