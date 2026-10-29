"""Contract tests for the native Linux CUDA evidence runner (no GPU required)."""
from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/linux/cuda_smoke.py"
spec = importlib.util.spec_from_file_location("linux_cuda_smoke", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load Linux CUDA runner")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class LinuxCudaSmokeTests(unittest.TestCase):
    def test_native_linux_identity_is_required(self):
        with self.assertRaisesRegex(ValueError, "native Linux"):
            smoke.validate_linux_report({"toolchain": {"os": "Windows", "host_compiler": "MSVC"}}, 0, (12, 8, 93))

    def test_nvcc_version_parser_is_strict(self):
        self.assertEqual(smoke.parse_nvcc_version("Cuda compilation tools, release 12.8, V12.8.93"), (12, 8, 93))
        with self.assertRaisesRegex(ValueError, "parse"):
            smoke.parse_nvcc_version("not an nvcc version")
        with self.assertRaisesRegex(ValueError, "does not match"):
            smoke.parse_nvcc_version("Cuda compilation tools, release 12.8, V99.99.93")

    def test_runtime_nvcc_build_mismatch_is_rejected(self):
        report = {"toolchain": {"os": "Linux", "host_compiler": "GCC", "host_compiler_version": "13.2",
                  "nvcc_major": 12, "nvcc_minor": 8, "nvcc_build": 92,
                  "cuda_headers_version": 12080, "cplusplus": 201703}, "expected_compile_arch": 890}
        with self.assertRaisesRegex(ValueError, "NVCC version disagrees"):
            smoke.validate_linux_report(report, 0, (12, 8, 93))

    def test_runtime_timeout_is_structured_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime.json"
            stderr = root / "runtime.stderr.log"
            data = {"commands": []}
            log = root / "cuda-smoke.log"
            def timeout(argv, cwd, limit, stdout_path, stderr_path):
                stdout_path.write_bytes(b"partial-json")
                stderr_path.write_bytes(b"partial-error")
                raise subprocess.TimeoutExpired(argv, limit)
            def save():
                (root / "saved.json").write_text(__import__("json").dumps(data), encoding="utf-8")
            with patch.object(smoke, "run_command", side_effect=timeout):
                with self.assertRaisesRegex(RuntimeError, "runtime-probe timed out"):
                    smoke.record_command(data, root, log, "runtime-probe", ["/probe", "--device", "0"],
                                         root, 5, save, stdout_path=runtime, stderr_path=stderr)
            rec = data["commands"][0]
            self.assertEqual(rec["argv"], ["/probe", "--device", "0"])
            self.assertEqual(rec["stdout_log"], str(runtime))
            self.assertEqual(rec["stderr_log"], str(stderr))
            self.assertEqual(rec["timeout_seconds"], 5)
            self.assertIsNone(rec["exit_code"])
            self.assertTrue(rec["timeout"])
            self.assertEqual(runtime.read_bytes(), b"partial-json")
            self.assertEqual(stderr.read_bytes(), b"partial-error")

    def test_architecture_gate_rejects_wrong_target_before_probe_claim(self):
        report = {"toolchain": {"os": "Linux", "host_compiler": "GCC", "host_compiler_version": "13.2",
                  "nvcc_major": 12, "nvcc_minor": 8, "nvcc_build": 93,
                  "cuda_headers_version": 12080, "cplusplus": 201703}, "expected_compile_arch": 860}
        with self.assertRaisesRegex(ValueError, "architecture"):
            smoke.validate_linux_report(report, 0, (12, 8, 93))


if __name__ == "__main__":
    unittest.main()
