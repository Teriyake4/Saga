import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = ROOT / "tools" / "windows" / "verify.ps1"
POWERSHELL = shutil.which("pwsh")


@unittest.skipUnless(POWERSHELL, "PowerShell 7 is not available")
class WindowsVerifyCacheTests(unittest.TestCase):
    HARNESS = r'''
param([string]$Source, [string]$Cache, [string]$Configuration)
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -gt 0) { throw $parseErrors[0] }
$definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Assert-ConfiguredBuildCache' }, $true)
if (-not $definition) { throw 'Cache validation function not found' }
Invoke-Expression $definition.Extent.Text
Assert-ConfiguredBuildCache $Cache $Configuration
'''

    def assert_cache(self, generator, configuration="Debug", build_type="Debug",
                     configurations=None, tests="ON", should_pass=True):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cache = folder / "CMakeCache.txt"
            entries = [
                f"CMAKE_GENERATOR:INTERNAL={generator}",
                f"CMAKE_BUILD_TYPE:STRING={build_type}",
            ]
            if tests is not None:
                entries.append(f"SAGA_BUILD_TESTS:BOOL={tests}")
            if configurations is not None:
                entries.append(f"CMAKE_CONFIGURATION_TYPES:STRING={configurations}")
            cache.write_text("\n".join(entries) + "\n", encoding="utf-8")
            harness = folder / "invoke-function.ps1"
            harness.write_text(self.HARNESS, encoding="utf-8")
            result = subprocess.run(
                [POWERSHELL, "-NoProfile", "-File", str(harness), str(VERIFY_SCRIPT), str(cache), configuration],
                capture_output=True, text=True, check=False, timeout=15,
            )
            if should_pass:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            else:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ninja_requires_matching_single_config_build_type(self):
        self.assert_cache("Ninja", configuration="Debug", build_type="Debug")
        self.assert_cache("Ninja", configuration="Release", build_type="Release")
        self.assert_cache("Ninja", configuration="Debug", build_type="Release", should_pass=False)

    def test_configuration_types_cannot_spoof_ninja_as_multi_config(self):
        self.assert_cache("Ninja", configuration="Debug", build_type="Release",
                          configurations="Debug;Release", should_pass=False)

    def test_visual_studio_uses_available_multi_configurations(self):
        self.assert_cache("Visual Studio 17 2022", configuration="Release",
                          configurations="Debug;Release")
        self.assert_cache("Visual Studio 17 2022", configuration="Release",
                          configurations="Debug", should_pass=False)

    def test_ninja_multi_config_uses_available_configurations(self):
        self.assert_cache("Ninja Multi-Config", configuration="Release",
                          configurations="Debug;Release")

    def test_tests_must_be_explicitly_enabled(self):
        self.assert_cache("Ninja", tests="OFF", should_pass=False)
        self.assert_cache("Ninja", tests=None, should_pass=False)

    def test_unknown_generator_fails_closed(self):
        self.assert_cache("Unknown Generator", should_pass=False)


if __name__ == "__main__":
    unittest.main()
