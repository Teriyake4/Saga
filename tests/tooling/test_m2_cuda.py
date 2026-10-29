"""GPU-free rejection tests for the bounded M2 CUDA evidence runner."""
from __future__ import annotations

import gc
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/m2_cuda.py"
spec = importlib.util.spec_from_file_location("m2_cuda", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load M2 CUDA runner")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)


def report_fixture():
    return {
        "status": "physical_projection_test_passed", "qualification": "not_claimed",
        "readiness_approval": False, "ledger_reconciliation": "passed",
        "device": {"name": "NVIDIA GeForce RTX 4080", "architecture": "sm89",
                   "architecture_version": 89, "identity": "device-uuid", "ordinal": 0,
                   "driver": 12000, "runtime": 12080},
        "iterations_per_batch": 30, "timing_batches_per_variant": 5,
        "variants": [
            {"variant": "bf16_projection_baseline", "max_abs_error": 0.001,
             "event_ms_per_launch": [0.1, 0.2, 0.3, 0.4, 0.5],
             "kernel_attributes": {"registers_per_thread": 16, "static_shared_bytes": 0,
                 "local_bytes_per_thread": 0, "max_threads_per_block": 1024, "ptx_version": 89,
                 "binary_version": 89, "active_blocks_per_multiprocessor": 2}},
            {"variant": "bf16_projection_wmma", "max_abs_error": 0.002,
             "event_ms_per_launch": [0.1, 0.2, 0.3, 0.4, 0.5],
             "kernel_attributes": {"registers_per_thread": 32, "static_shared_bytes": 0,
                 "local_bytes_per_thread": 0, "max_threads_per_block": 1024, "ptx_version": 89,
                 "binary_version": 89, "active_blocks_per_multiprocessor": 1}},
        ],
        "ledger": {"profile": "cuda-projection-test", "live_bytes": [0, 0, 0],
                   "allocations": [{"id": "weights", "cap_bytes": 4096, "live_bytes": 0,
                                    "measured_peak_bytes": 4096}]},
    }


class M2CudaTests(unittest.TestCase):
    def test_synthetic_or_incomplete_reports_are_rejected(self):
        mutations = (
            ("status", "pass"), ("qualification", "qualified"),
            ("readiness_approval", True), ("ledger_reconciliation", "failed"),
            ("iterations_per_batch", 29),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                report = report_fixture()
                report[key] = value
                with self.assertRaises(ValueError):
                    m2.validate_report(report, 30)

    def test_mock_device_missing_variant_bad_measurement_or_over_cap_is_rejected(self):
        mutations = (
            ("device", {"name": "mock RTX 4080", "architecture": "sm89", "architecture_version": 89,
                        "identity": "mock", "ordinal": 0, "driver": 1, "runtime": 1}),
            ("variants", [report_fixture()["variants"][0]]),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                report = report_fixture()
                report[key] = value
                with self.assertRaises(ValueError):
                    m2.validate_report(report, 30)
        report = report_fixture()
        report["variants"][1]["event_ms_per_launch"][2] = float("nan")
        with self.assertRaisesRegex(ValueError, "timing"):
            m2.validate_report(report, 30)
        report = report_fixture()
        report["variants"][0]["max_abs_error"] = 0.020001
        with self.assertRaisesRegex(ValueError, "0.02"):
            m2.validate_report(report, 30)
        report = report_fixture()
        report["variants"][1]["kernel_attributes"]["binary_version"] = 86
        with self.assertRaisesRegex(ValueError, "SM89"):
            m2.validate_report(report, 30)
        report = report_fixture()
        report["ledger"]["allocations"][0]["measured_peak_bytes"] = 4097
        with self.assertRaisesRegex(ValueError, "exceeds cap"):
            m2.validate_report(report, 30)

    def test_windows_taskkill_failure_uses_bounded_fallback_and_reports_unresolved(self):
        class StuckProcess:
            pid = 12345
            killed = False
            waits = []
            def poll(self):
                return None
            def kill(self):
                self.killed = True
            def wait(self, timeout=None):
                self.waits.append(timeout)
                raise subprocess.TimeoutExpired(["fake-child"], timeout)

        process = StuckProcess()
        with (patch.object(m2.shutil, "which", return_value="taskkill"),
              patch.object(m2.subprocess, "run", return_value=type("Result", (), {"returncode": 1})())):
            self.assertFalse(m2.terminate_tree(process, windows=True))
        self.assertTrue(process.killed)
        self.assertEqual(process.waits, [10, 10])

    def test_noisy_command_is_capped_and_fails_with_cleanup_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = root / "stdout.log"
            stderr = root / "stderr.log"
            script = "import sys; sys.stdout.write('x'*100000); sys.stderr.write('y'*100000)"
            real_popen = subprocess.Popen
            launched = []
            def track_process(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                launched.append(process)
                return process
            with warnings.catch_warnings():
                warnings.simplefilter("error", ResourceWarning)
                with patch.object(m2.subprocess, "Popen", side_effect=track_process):
                    with self.assertRaises(m2.LogLimitExceeded) as raised:
                        m2.run_bounded([sys.executable, "-c", script], root, 10, stdout, stderr,
                                       windows=False, max_log_bytes=1024)
                gc.collect()
            self.assertTrue(raised.exception.cleanup_resolved)
            self.assertEqual(len(launched), 1)
            self.assertTrue(launched[0].stdout.closed)
            self.assertTrue(launched[0].stderr.closed)
            self.assertLessEqual(stdout.stat().st_size, 1024)
            self.assertLessEqual(stderr.stat().st_size, 1024)

    def test_timeout_closes_both_pipes_and_reaps_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_popen = subprocess.Popen
            launched = []
            def track_process(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                launched.append(process)
                return process
            with patch.object(m2.subprocess, "Popen", side_effect=track_process):
                with self.assertRaises(m2.CommandTimeout):
                    m2.run_bounded([sys.executable, "-c", "import time; time.sleep(30)"],
                                   root, 0.1, root / "out.log", root / "err.log", windows=False)
            self.assertEqual(len(launched), 1)
            self.assertIsNotNone(launched[0].poll())
            self.assertTrue(launched[0].stdout.closed)
            self.assertTrue(launched[0].stderr.closed)

    def test_platform_rejection_creates_no_evidence(self):
        with patch.object(m2.sys, "platform", "darwin"), patch.object(sys, "argv", [str(SCRIPT)]):
            self.assertEqual(m2.main(), 1)


if __name__ == "__main__":
    unittest.main()
