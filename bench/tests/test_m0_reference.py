import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("m0_reference", ROOT / "tools/m0_reference.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TraceParserTests(unittest.TestCase):
    def record(self, **overrides):
        value = {"site": "q3_linear", "phase": "prefill", "dispatch": "A8",
                 "m": 32, "n": 4096, "k": 4096, "executed": True, "event": "launch"}
        value.update(overrides)
        return value

    def test_multiple_routes_by_geometry_are_valid(self):
        records = [self.record(), self.record(dispatch="A16", m=1)]
        result = MODULE.parse_trace_records(records, {("q3_linear", "prefill")})
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["dispatch_completeness"].startswith("not established"), True)
        with self.assertRaisesRegex(MODULE.HarnessError, "contradictory"):
            MODULE.parse_trace_records([self.record(), self.record(dispatch="A16")],
                                       {("q3_linear", "prefill")})

    def test_q3_selection_trace_preserves_unknown_phase_and_capture_event(self):
        record = {"schema": 1, "record_id": 4, "site": "q3_linear_projection_dispatch",
                  "phase": "unknown", "dispatch": "A8", "route": "a8_small_t_r32_c8",
                  "kernel": "launch_q3_mma_small_t_a8_r32_c8", "m": 4, "n": 4096,
                  "k": 4096, "selected": True, "event": "graph-record-selection",
                  "capture_state": "active"}
        result = MODULE.parse_trace_records([record],
                                            {("q3_linear_projection_dispatch", "unknown")})
        self.assertEqual(result["records"][0]["phase"], "unknown")
        with self.assertRaisesRegex(MODULE.HarnessError, "incomplete dispatch coverage"):
            MODULE.parse_trace_records([record],
                                       {("q3_linear_projection_dispatch", "prefill")})
        record["event"] = "graph-replay-execution"
        with self.assertRaisesRegex(MODULE.HarnessError, "unsupported dispatch event"):
            MODULE.parse_trace_records([record],
                                       {("q3_linear_projection_dispatch", "unknown")})

    def test_invalid_trace_types_and_inventory(self):
        with self.assertRaisesRegex(MODULE.HarnessError, "absent"):
            MODULE.parse_trace_records([], {("q3", "prefill")})
        with self.assertRaisesRegex(MODULE.HarnessError, "incomplete"):
            MODULE.parse_trace_records([{"site": "q3"}], {("q3", "prefill")})
        with self.assertRaisesRegex(MODULE.HarnessError, "geometry"):
            MODULE.parse_trace_records([self.record(m=True)], {("q3_linear", "prefill")})
        with self.assertRaisesRegex(MODULE.HarnessError, "inventory is empty"):
            MODULE.parse_trace_records([self.record()], set())


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.source.mkdir()
        (self.source / "main.cu").write_text("pinned source\n")
        self.outside_build = self.base / "build"
        self.outside_build.mkdir()
        self.output = self.base / "evidence"
        self.binary = Path(sys.executable).resolve()
        self.binary_hash = MODULE.sha256_file(self.binary)

    def tearDown(self):
        self.temp.cleanup()

    def recipe(self, run_code=None):
        trace = MODULE.TRACE_PREFIX.decode() + json.dumps({
            "site": "site", "phase": "decode", "dispatch": "A8", "m": 1,
            "n": 32, "k": 64, "executed": True, "event": "launch"}) + chr(10)
        run_code = run_code if run_code is not None else f"import sys; sys.stdout.write({trace!r})"
        return {
            "repository": MODULE.REPOSITORY, "revision": MODULE.REVISION,
            "mode": "reviewed-prebuilt",
            "source_root": str(self.source), "source_sha256": MODULE.source_digest(self.source),
            "source_review": {"reviewer": "reviewer-id", "reviewed_at": "2026-10-29", "basis": "archive provenance"},
            "trace_inventory_review": {"reviewer": "reviewer-id", "reviewed_at": "2026-10-29", "basis": "reviewed source sites"},
            "build_dir": str(self.outside_build), "binary": str(self.binary),
            "binary_sha256": self.binary_hash,
            "prebuilt_provenance": {"reviewer": "reviewer-id", "reviewed_at": "2026-10-29",
                                    "basis": "reviewed external build record", "source_revision": MODULE.REVISION,
                                    "binary_sha256": self.binary_hash},
            "patches": [], "environment": {},
            "required_site_phases": [{"site": "site", "phase": "decode"}],
            "commands": [
                {"role": "run", "argv": [str(self.binary), "-c", run_code], "cwd": str(self.source), "timeout_seconds": 5}
            ]
        }

    def invoke(self, config):
        config_path = self.base / "recipe.json"
        config_path.write_text(json.dumps(config))
        return MODULE.execute(config_path, self.output)

    def test_dispatch_patch_dry_runs_against_exact_pinned_sources(self):
        source = ROOT / ".cache/m0-reference/ninfer-4080-2469b5752672ad597055af61fe28e8e7ced20056"
        patch = ROOT / "bench/reference/dispatch-audit.patch"
        if not source.is_dir():
            self.skipTest("pinned source archive is not present")
        patch_executable = shutil.which("patch")
        if patch_executable is None:
            self.skipTest("patch executable is not available; applicability is untested in this environment")
        before = {
            "q3": MODULE.sha256_file(source / "src/ops/linear/q3/q3_dispatch.cpp"),
            "swiglu": MODULE.sha256_file(source / "src/ops/linear_swiglu/q3/q3_linear_swiglu.cu")}
        result = subprocess.run([patch_executable, "--dry-run", "--batch", "-p1", "-i", str(patch)],
                                cwd=source, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("q3_dispatch_audit.h", result.stdout)
        self.assertEqual(before["q3"], MODULE.sha256_file(source / "src/ops/linear/q3/q3_dispatch.cpp"))
        self.assertEqual(before["swiglu"], MODULE.sha256_file(source / "src/ops/linear_swiglu/q3/q3_linear_swiglu.cu"))
        patch_text = patch.read_text()
        for expected in ("launch_q3_mma_tall_a8_r128_c128", "launch_q3_mma_small_t_a8_r32_c8",
                         "launch_q3_mma_small_t_a8_r32_c16", "launch_q3_mma_tall_a8_swiglu_r64_c128",
                         "launch_q3_mma_small_t_a8_swiglu_r16_c8", "launch_q3_mma_small_t_a8_swiglu_r16_c16",
                         "NINFER_M0_Q3_DISPATCH_TRACE", "cudaStreamIsCapturing"):
            self.assertIn(expected, patch_text)

    def test_patch_test_skips_explicitly_when_patch_command_is_missing(self):
        with (mock.patch.object(shutil, "which", return_value=None),
              self.assertRaisesRegex(unittest.SkipTest, "applicability is untested")):
            self.test_dispatch_patch_dry_runs_against_exact_pinned_sources()

    def test_valid_run_trace_is_stdout_linked_and_never_claims_completion(self):
        self.assertEqual(self.invoke(self.recipe()), 0)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertEqual(result["trace"]["origin"], "captured-run-stdout")
        self.assertEqual(result["status"], "prebuilt-run-captured-incomplete-dispatch")
        self.assertTrue(result["prebuilt_provenance"]["is_review_assertion_not_independent_build_verification"])

    @unittest.skipIf(os.name == "nt", "POSIX executable shim used for CMake test")
    def test_build_mode_requires_absent_then_fresh_binary_snapshots(self):
        cmake = self.base / "cmake"
        binary = self.outside_build / "ninfer-test"
        generated_code = self.recipe()["commands"][0]["argv"][2]
        cmake.write_text("#!/usr/bin/env python3\nimport pathlib,sys\n"
                         f"dst={str(binary)!r}\n"
                         f"payload={('#!' + sys.executable + chr(10) + generated_code + chr(10))!r}\n"
                         "if '--build' in sys.argv: pathlib.Path(dst).write_text(payload); pathlib.Path(dst).chmod(0o755)\n")
        cmake.chmod(0o755)
        config = self.recipe()
        config["mode"] = "build"
        config.pop("prebuilt_provenance")
        config["binary"] = str(binary)
        config.pop("binary_sha256")
        config["commands"] = [
            {"role": "configure", "argv": [str(cmake), "-S", str(self.source), "-B", str(self.outside_build)], "cwd": str(self.source), "timeout_seconds": 5},
            {"role": "build", "argv": [str(cmake), "--build", str(self.outside_build)], "cwd": str(self.source), "timeout_seconds": 5},
            {"role": "run", "argv": [str(binary)], "cwd": str(self.source), "timeout_seconds": 5}
        ]
        self.assertEqual(self.invoke(config), 0)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertEqual(result["status"], "fresh-build-run-trace-validated-incomplete-dispatch")
        self.assertFalse(result["build_binary_snapshots"]["before"]["exists"])
        self.assertEqual(result["build_binary_snapshots"]["after"]["sha256"], result["binary"]["sha256"])

    def test_build_mode_rejects_missing_stage_and_preexisting_artifact(self):
        config = self.recipe()
        config["mode"] = "build"
        config.pop("prebuilt_provenance")
        config["binary"] = str(self.outside_build / "existing-bin")
        config.pop("binary_sha256")
        config["commands"] = [config["commands"][-1]]
        self.assertEqual(self.invoke(config), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("configure then build", result["failure"])

        self.output = self.base / "preexisting-evidence"
        binary = Path(config["binary"])
        binary.write_bytes(Path(sys.executable).read_bytes())
        cmake = self.base / "cmake"
        cmake.write_text("#!/bin/sh\\nexit 0\\n")
        cmake.chmod(0o755)
        config["commands"] = [
            {"role": "configure", "argv": [str(cmake), "-S", str(self.source), "-B", str(self.outside_build)], "cwd": str(self.source), "timeout_seconds": 5},
            {"role": "build", "argv": [str(cmake), "--build", str(self.outside_build)], "cwd": str(self.source), "timeout_seconds": 5},
            {"role": "run", "argv": [str(binary)], "cwd": str(self.source), "timeout_seconds": 5}
        ]
        self.assertEqual(self.invoke(config), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("absent before configure", result["failure"])
        self.assertTrue(result["build_binary_snapshots"]["before"]["exists"])
        self.assertEqual(result["commands"], [])

    def test_missing_source_pin_and_invalid_recipe_preserve_capture(self):
        config = self.recipe()
        config["source_sha256"] = None
        self.assertEqual(self.invoke(config), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("source_sha256", result["failure"])
        self.assertEqual(result["commands"], [])

    def test_rejects_source_symlinks(self):
        config = self.recipe()
        linked = self.source / "linked"
        try:
            linked.symlink_to(self.source / "main.cu")
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        self.assertEqual(self.invoke(config), 2)
        self.assertIn("symlinks", json.loads((self.output / "capture.json").read_text())["failure"])

    def test_rejects_nested_output(self):
        nested_output = self.source / "evidence"
        config_path = self.base / "nested-recipe.json"
        config_path.write_text(json.dumps(self.recipe()))
        with self.assertRaisesRegex(MODULE.HarnessError, "outside the source"):
            MODULE.execute(config_path, nested_output)
        self.assertFalse(nested_output.exists())

    def test_rejects_existing_output(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("untouched")
        with self.assertRaisesRegex(MODULE.HarnessError, "already exists"):
            MODULE.execute(self.base / "missing.json", self.output)
        self.assertEqual(marker.read_text(), "untouched")

    def test_stale_external_trace_not_accepted_and_run_binary_bound(self):
        config = self.recipe("pass")
        config["trace"] = str(self.base / "stale.jsonl")
        (self.base / "stale.jsonl").write_text("fake")
        config["commands"][-1]["argv"] = [sys.executable, "-c", "pass"]
        self.assertEqual(self.invoke(config), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("unsupported fields", result["failure"])
        self.assertIsNone(result["trace"])

    def test_timeout_and_overflow_are_bounded_and_fail(self):
        result = MODULE.run_command([sys.executable, "-c", "import time; print('start', flush=True); time.sleep(10)"],
                                    self.base, os.environ.copy(), 0.2, self.base / "timeout.log", max_log=512)
        self.assertTrue(result["timed_out"])
        result = MODULE.run_command([sys.executable, "-c", "import sys,time; sys.stdout.write('x'*200000); sys.stdout.flush(); time.sleep(10)"],
                                    self.base, os.environ.copy(), 5, self.base / "overflow.log", max_log=1024)
        self.assertTrue(result["log_truncated"])
        self.assertLessEqual(result["log_bytes"], 1024)
        self.assertIsNotNone(result["exit_code"])

    @unittest.skipIf(os.name == "nt", "POSIX process groups only")
    def test_timeout_kills_descendant_after_group_leader_exits(self):
        pid_file = self.base / "child.pid"
        child_code = "import time; time.sleep(30)"
        parent_code = ("import subprocess,sys; "
                       f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                       f"open({str(pid_file)!r},'w').write(str(p.pid))")
        result = MODULE.run_command([sys.executable, "-c", parent_code], self.base,
                                    os.environ.copy(), 0.3, self.base / "descendant.log", max_log=1024)
        self.assertTrue(result["timed_out"])
        pid = int(pid_file.read_text())
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            self.fail("descendant survived group timeout after leader exit")

    def test_final_unterminated_trace_line_is_parsed_or_reported(self):
        trace = MODULE.TRACE_PREFIX.decode() + json.dumps({"site": "site", "phase": "decode",
                "dispatch": "A8", "m": 1, "n": 32, "k": 64, "executed": True,
                "event": "launch"})
        code = f"import sys; sys.stdout.write({trace!r})"
        self.assertEqual(self.invoke(self.recipe(code)), 0)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertEqual(len(result["trace"]["records"]), 1)

        self.output = self.base / "malformed-evidence"
        code = f"import sys; sys.stdout.write({(MODULE.TRACE_PREFIX.decode() + '{bad')!r})"
        self.assertEqual(self.invoke(self.recipe(code)), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("malformed final runtime trace", result["failure"])

    def test_incomplete_run_trace_persists_failure_and_is_not_success(self):
        code = "import sys; print('ordinary output')"
        self.assertEqual(self.invoke(self.recipe(code)), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("no explicit dispatch records", result["failure"])
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue((self.output / "command-00.log").is_file())

    def test_nonfinite_timeout_rejected_before_any_command(self):
        config = self.recipe()
        config["commands"][0]["timeout_seconds"] = float("inf")
        self.assertEqual(self.invoke(config), 2)
        result = json.loads((self.output / "capture.json").read_text())
        self.assertIn("finite positive", result["failure"])
        self.assertEqual(result["commands"], [])


if __name__ == "__main__":
    unittest.main()
