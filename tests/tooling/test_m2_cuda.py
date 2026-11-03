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
                   "architecture_version": 89, "identity": "0123456789abcdef0123456789abcdef", "ordinal": 0,
                   "driver": 12000, "runtime": 12080},
        "iterations_per_batch": 30, "timing_batches_per_variant": 5,
        "variants": [
            {"variant": "sm89.bf16_projection.baseline.decode", "max_abs_error": 0.001,
             "request": {"operation": "projection", "weights": "bf16", "activation": "bf16",
                 "output": "fp32", "numerics": "fp32_accumulate", "layout": "contiguous",
                 "phase": "decode", "outputs": 19, "inputs": 23, "tokens": 3,
                 "weight_row_stride_bytes": 48, "activation_row_stride_bytes": 50,
                 "output_row_stride_bytes": 80, "alignment": 2},
             "event_ms_per_launch": [0.1, 0.2, 0.3, 0.4, 0.5],
             "kernel_attributes": {"registers_per_thread": 16, "static_shared_bytes": 0,
                 "local_bytes_per_thread": 0, "max_threads_per_block": 1024, "ptx_version": 89,
                 "binary_version": 89, "active_blocks_per_multiprocessor": 2}},
            {"variant": "sm89.bf16_projection.wmma.decode", "max_abs_error": 0.002,
             "request": {"operation": "projection", "weights": "bf16", "activation": "bf16",
                 "output": "fp32", "numerics": "fp32_accumulate", "layout": "contiguous",
                 "phase": "decode", "outputs": 32, "inputs": 32, "tokens": 32,
                 "weight_row_stride_bytes": 96, "activation_row_stride_bytes": 96,
                 "output_row_stride_bytes": 192, "alignment": 32},
             "event_ms_per_launch": [0.1, 0.2, 0.3, 0.4, 0.5],
             "kernel_attributes": {"registers_per_thread": 32, "static_shared_bytes": 0,
                 "local_bytes_per_thread": 0, "max_threads_per_block": 1024, "ptx_version": 89,
                 "binary_version": 89, "active_blocks_per_multiprocessor": 1}},
        ],
        "ledger": {"profile": "cuda-projection-test", "live_bytes": {"host": 0, "pinned": 0, "device": 16384},
                   "allocations": [
                       {"id": "weights", "location": "device", "lifetime": "persistent", "source": "project",
                        "cap_bytes": 3072, "live_bytes": 3072, "measured_peak_bytes": 3072, "measured": True},
                       {"id": "input", "location": "device", "lifetime": "persistent", "source": "project",
                        "cap_bytes": 3072, "live_bytes": 3072, "measured_peak_bytes": 3072, "measured": True},
                       {"id": "output", "location": "device", "lifetime": "persistent", "source": "project",
                        "cap_bytes": 6144, "live_bytes": 6144, "measured_peak_bytes": 6144, "measured": True},
                       {"id": "staging-host", "location": "host", "lifetime": "bringup", "source": "project",
                        "cap_bytes": 3072, "live_bytes": 0, "measured_peak_bytes": 3072, "measured": True},
                       {"id": "staging-pinned", "location": "pinned", "lifetime": "bringup", "source": "project",
                        "cap_bytes": 4096, "live_bytes": 0, "measured_peak_bytes": 4096, "measured": True},
                       {"id": "runtime", "location": "device", "lifetime": "bringup", "source": "runtime_overhead",
                        "cap_bytes": 1073741824, "live_bytes": 4096, "measured_peak_bytes": 4096, "measured": True}]},
    }


def session_report_fixture():
    expected = {
        "weights": ("device", "persistent", "project", 1024),
        "activations": ("device", "persistent", "project", 256),
        "output": ("device", "persistent", "project", 256),
        "host-output": ("host", "persistent", "project", 256),
        "staging-host": ("host", "bringup", "project", 256),
        "staging-pinned": ("pinned", "bringup", "project", 4096),
        "runtime": ("device", "bringup", "runtime_overhead", 1 << 30),
        "main-kv": ("device", "persistent", "project", 64),
        "warmup-scratch": ("device", "bringup", "project", 256),
        "zero-1": ("device", "persistent", "project", 0),
        "zero-2": ("device", "persistent", "project", 0),
        "zero-3": ("device", "persistent", "project", 0),
        "zero-4": ("device", "persistent", "project", 0),
        "zero-5": ("device", "persistent", "project", 0),
        "zero-7": ("host", "persistent", "project", 0),
        "zero-9": ("device", "persistent", "project", 0),
        "zero-10": ("device", "persistent", "project", 0),
        "zero-13": ("host", "persistent", "project", 0),
        "zero-15": ("pinned", "persistent", "project", 0),
    }
    allocations = []
    for allocation_id, (location, lifetime, source, cap) in expected.items():
        if lifetime == "persistent" and cap:
            current = peak = cap
        elif allocation_id == "runtime":
            current = peak = 4096
        elif allocation_id in ("staging-host", "staging-pinned", "warmup-scratch"):
            current, peak = 0, cap
        else:
            current = peak = 0
        allocations.append({"id": allocation_id, "location": location, "lifetime": lifetime,
            "source": source, "cap_bytes": cap, "live_bytes": current,
            "measured_peak_bytes": peak, "measured": True})
    checks = sorted(m2.SESSION_CHECKS)
    return {
        "status": "physical_session_test_passed", "qualification": "not_claimed",
        "serving_readiness": False, "readiness_approval": False,
        "ledger_reconciliation": "passed", "all_checks_passed": True,
        "device_loss_injection": "not_run", "checks": checks, "expected_checks": checks.copy(),
        "projection_scope": "synthetic_fixture_only", "tuning_claim": False, "iterations": 30,
        "device": {"name": "NVIDIA GeForce RTX 4080", "architecture": "sm89",
            "architecture_version": 89, "identity": "0123456789abcdef0123456789abcdef",
            "ordinal": 0, "driver": 12000, "runtime": 12080},
        "ledger": {"profile": "cuda-physical-session-fixture-v1",
            "live_bytes": {"host": 256, "pinned": 0, "device": 5696}, "allocations": allocations},
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
        m2.validate_report(report, 30)
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

    def test_strict_projection_contract_and_ledger_reject_tampering(self):
        mutations = (
            ("variant ID", lambda r: r["variants"][0].update(variant="alias.baseline.decode")),
            ("shape", lambda r: r["variants"][0]["request"].update(outputs=20)),
            ("stride", lambda r: r["variants"][1]["request"].update(output_row_stride_bytes=193)),
            ("operation", lambda r: r["variants"][0]["request"].update(operation="other")),
            ("weight encoding", lambda r: r["variants"][0]["request"].update(weights="fp16")),
            ("activation encoding", lambda r: r["variants"][0]["request"].update(activation="fp16")),
            ("output encoding", lambda r: r["variants"][0]["request"].update(output="bf16")),
            ("numerics", lambda r: r["variants"][0]["request"].update(numerics="tf32")),
            ("layout", lambda r: r["variants"][0]["request"].update(layout="strided")),
            ("phase", lambda r: r["variants"][0]["request"].update(phase="prefill")),
            ("input stride", lambda r: r["variants"][0]["request"].update(activation_row_stride_bytes=51)),
            ("alignment", lambda r: r["variants"][1]["request"].update(alignment=16)),
            ("measured", lambda r: r["ledger"]["allocations"][0].update(measured=False)),
            ("boolean cap", lambda r: r["ledger"]["allocations"][0].update(cap_bytes=True)),
            ("weights cap", lambda r: r["ledger"]["allocations"][0].update(cap_bytes=3073)),
            ("input cap", lambda r: r["ledger"]["allocations"][1].update(cap_bytes=3073)),
            ("output cap", lambda r: r["ledger"]["allocations"][2].update(cap_bytes=6145)),
            ("host cap", lambda r: r["ledger"]["allocations"][3].update(cap_bytes=3073)),
            ("pinned cap", lambda r: r["ledger"]["allocations"][4].update(cap_bytes=4097)),
            ("runtime cap", lambda r: r["ledger"]["allocations"][5].update(cap_bytes=1073741823)),
            ("boolean peak", lambda r: r["ledger"]["allocations"][0].update(measured_peak_bytes=True)),
            ("lifetime", lambda r: r["ledger"]["allocations"][0].update(lifetime="bringup")),
            ("source", lambda r: r["ledger"]["allocations"][0].update(source="runtime_overhead")),
            ("runtime unmeasured", lambda r: r["ledger"]["allocations"][-1].update(measured=False)),
            ("runtime above cap", lambda r: r["ledger"]["allocations"][-1].update(measured_peak_bytes=1073741825)),
            ("runtime live/peak mismatch", lambda r: (r["ledger"]["allocations"][-1].update(live_bytes=4095),
                r["ledger"]["live_bytes"].update(device=16383))),
            ("current above peak", lambda r: r["ledger"]["allocations"][0].update(live_bytes=3073)),
            ("domain total", lambda r: r["ledger"]["live_bytes"].update(device=1)),
            ("location", lambda r: r["ledger"]["allocations"][0].update(location="host")),
            ("missing allocation", lambda r: r["ledger"]["allocations"].pop()),
            ("UUID", lambda r: r["device"].update(identity="device-uuid")),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                report = report_fixture()
                mutate(report)
                with self.assertRaises(ValueError):
                    m2.validate_report(report, 30)

    def test_nonfinite_projection_errors_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                report = report_fixture()
                report["variants"][0]["max_abs_error"] = value
                with self.assertRaisesRegex(ValueError, "projection error"):
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

    def test_session_report_accepts_exact_physical_fixture(self):
        m2.validate_session_report(session_report_fixture(), 30)

    def test_session_report_rejects_missing_checks_readiness_device_and_ledger_tampering(self):
        mutations = (
            ("status", lambda r: r.update(status="physical_projection_test_passed")),
            ("qualification", lambda r: r.update(qualification="qualified")),
            ("serving readiness", lambda r: r.update(serving_readiness=True)),
            ("approval", lambda r: r.update(readiness_approval=True)),
            ("checks", lambda r: r["checks"].pop()),
            ("check substituted", lambda r: r["checks"].__setitem__(0, "invented_check")),
            ("expected checks", lambda r: r["expected_checks"].pop()),
            ("device loss", lambda r: r.update(device_loss_injection="passed")),
            ("scope", lambda r: r.update(projection_scope="production_model")),
            ("tuning claim", lambda r: r.update(tuning_claim=True)),
            ("iteration", lambda r: r.update(iterations=29)),
            ("device name", lambda r: r["device"].update(name="GeForce RTX 4080 Super")),
            ("device identity", lambda r: r["device"].update(identity="uuid")),
            ("device architecture", lambda r: r["device"].update(architecture="sm90")),
            ("ledger domain totals", lambda r: r["ledger"]["live_bytes"].update(device=5695)),
            ("ledger record", lambda r: r["ledger"]["allocations"].pop()),
            ("cap", lambda r: r["ledger"]["allocations"][0].update(cap_bytes=1025)),
            ("unmeasured", lambda r: r["ledger"]["allocations"][0].update(measured=False)),
            ("metadata", lambda r: r["ledger"]["allocations"][0].update(location="host")),
            ("runtime mismatch with consistent total", lambda r: (
                r["ledger"]["allocations"][6].update(live_bytes=4095),
                r["ledger"]["live_bytes"].update(device=5695))),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                report = session_report_fixture()
                mutate(report)
                with self.assertRaises(ValueError):
                    m2.validate_session_report(report, 30)

    def test_combined_reports_require_same_device_uuid_ordinal_and_toolchain(self):
        projection = report_fixture()
        session = session_report_fixture()
        session["device"]["name"] = "GeForce RTX 4080"
        m2.validate_same_device(projection, session)
        for field, value in (("identity", "fedcba9876543210fedcba9876543210"),
                             ("ordinal", 1), ("driver", 12001), ("runtime", 12081)):
            with self.subTest(field=field):
                mutated = session_report_fixture()
                mutated["device"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    m2.validate_same_device(projection, mutated)

    def test_platform_rejection_creates_no_evidence(self):
        with patch.object(m2.sys, "platform", "darwin"), patch.object(sys, "argv", [str(SCRIPT)]):
            self.assertEqual(m2.main(), 1)

    def test_non_x86_64_host_rejection_precedes_evidence_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence_root = Path(directory) / "cuda-evidence"
            argv = [str(SCRIPT), "--source-root", directory, "--build-root", str(evidence_root)]
            with (patch.object(m2.sys, "platform", "linux"),
                  patch.object(m2.platform, "machine", return_value="aarch64"),
                  patch.object(sys, "argv", argv)):
                self.assertEqual(m2.main(), 1)
            self.assertFalse(evidence_root.exists())


if __name__ == "__main__":
    unittest.main()
