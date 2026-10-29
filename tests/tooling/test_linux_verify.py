"""Isolated full-flow tests for the Linux verification runner."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/linux/verify.py"

import importlib.util
spec = importlib.util.spec_from_file_location("linux_verify", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load Linux verifier for tests")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class RunnerHarness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source tree with spaces"
        self.source.mkdir()
        (self.source / "tools").mkdir()
        (self.source / "tools/m1_evidence.py").write_text("", encoding="utf-8")
        self.output = self.root / "evidence output"
        self.args = argparse.Namespace(
            preset="linux-gcc-debug", parallel=2, enable_image_decoder=False,
            build_root=self.output, source_root=self.source, timeout_seconds=15,
        )
        self.stage = None
        self.cache_overrides = {}
        self.missing_install: str | None = None
        self.missing_tool: str | None = None
        self.junit_mode: str | None = None
        self.snapshot_count = 0
        self.snapshot_raises_after = False
        self.fact_overrides = {}
        self.calls = []

    def snapshot(self, _source):
        self.snapshot_count += 1
        if self.snapshot_raises_after and self.snapshot_count == 2:
            raise RuntimeError("simulated after-snapshot failure")
        digest = "changed" if self.stage == "source-change" and self.snapshot_count > 1 else "stable"
        return {"files": [{"path": "src/a.cpp", "sha256": digest}]}

    def fake_run(self, argv, **kwargs):
        argv = [str(part) for part in argv]
        label = argv[0]
        self.calls.append((argv, kwargs.get("cwd")))
        if self.stage == "start-error" and label.endswith("cmake") and "--build" in argv:
            raise OSError("spawn denied")
        if self.stage == "timeout" and "--build" in argv:
            raise verify.subprocess.TimeoutExpired(argv, 15, output=b"partial-out", stderr=b"partial-err")
        failing = {
            "configure": "--preset" in argv,
            "build": "--build" in argv,
            "test": "--output-junit" in argv,
            "install": "--install" in argv,
            "toml": "--check-config" in argv and "--model-config" not in argv,
            "ini": "--model-config" in argv,
            "dependency": argv[-1].endswith("verify_dependencies.py"),
        }
        if any(failing.get(name, False) for name in [self.stage]):
            return type("Result", (), {"returncode": 7})()
        if "--preset" in argv:
            build = Path(argv[argv.index("-B") + 1])
            contract = verify.PRESETS[self.args.preset]
            selected = Path("/fake/bin") / contract["compiler"]
            cache = {
                "SAGA_BUILD_TESTS": "ON", "SAGA_ENABLE_CUDA": "OFF", "SAGA_ENABLE_IMAGE_DECODER": "OFF",
                "CMAKE_BUILD_TYPE": contract["build_type"], "CMAKE_CXX_COMPILER": str(selected),
                "SAGA_SANITIZERS": "ON" if contract["sanitizers"] else "OFF",
                "SAGA_THREAD_SANITIZER": "ON" if contract["thread_sanitizer"] else "OFF",
            }
            cache.update(self.cache_overrides)
            (build / "CMakeCache.txt").write_text("\n".join(f"{key}:STRING={value}" for key, value in cache.items()), encoding="utf-8")
            files = build / "CMakeFiles/3.30.0"
            files.mkdir(parents=True)
            compiler_id = "GNU" if contract["compiler"] == "g++" else "Clang"
            compiler_facts = (
                f'set(CMAKE_CXX_COMPILER_ID "{compiler_id}")\n'
                'set(CMAKE_CXX_COMPILER_VERSION "14.1")\n'
                f'set(CMAKE_CXX_COMPILER "{selected}")\n')
            for old, new in self.fact_overrides.items():
                compiler_facts = compiler_facts.replace(old, new)
            system_facts = 'set(CMAKE_SYSTEM_NAME "Linux")\n'
            if "system" in self.fact_overrides:
                system_facts = system_facts.replace("Linux", self.fact_overrides["system"])
            (files / "CMakeSystem.cmake").write_text(system_facts, encoding="utf-8")
            (files / "CMakeCXXCompiler.cmake").write_text(compiler_facts, encoding="utf-8")
        if "--show-only=json-v1" in argv:
            kwargs["stdout"].write(b'{"tests":[{"name":"alpha"},{"name":"beta"}]}')
        if "--output-junit" in argv:
            junit = Path(argv[argv.index("--output-junit") + 1])
            report = '<testsuite><testcase name="alpha"/><testcase name="beta"/></testsuite>'
            if self.junit_mode == "skipped":
                report = '<testsuite><testcase name="alpha"/><testcase name="beta"><skipped/></testcase></testsuite>'
            elif self.junit_mode == "incomplete":
                report = '<testsuite><testcase name="alpha"/></testsuite>'
            elif self.junit_mode == "missing":
                report = None
            if report is not None:
                junit.write_text(report, encoding="utf-8")
        if "--install" in argv:
            prefix = Path(argv[argv.index("--install-prefix") + 1]) if "--install-prefix" in argv else self.output
            # Install prefix is conveyed through cache by the runner; resolve its evidence path.
            prefix = self.output_dir / "installed application"
            paths = {"server": prefix / "bin/saga-server", "toml": prefix / "share/saga/server.example.toml",
                     "ini": prefix / "share/saga/model.example.ini"}
            for key, path in paths.items():
                if key != self.missing_install:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")
        return type("Result", (), {"returncode": 0})()

    def fake_bounded(self, argv, cwd, stdout, stderr, timeout):
        result = self.fake_run(argv, cwd=cwd, stdout=stdout, stderr=stderr, timeout=timeout)
        return result.returncode

    @property
    def output_dir(self):
        return next(self.output.iterdir())

    def execute(self, stage=None):
        self.stage = stage
        self.snapshot_count = 0
        compiler = verify.PRESETS[self.args.preset]["compiler"]
        fake_tools = {name: f"/fake/bin/{name}" for name in ("cmake", "ctest", "ninja", "python3", compiler)}
        if self.missing_tool:
            fake_tools.pop(self.missing_tool, None)
        with patch.object(verify.sys, "platform", "linux"), \
             patch.object(verify.shutil, "which", side_effect=lambda name: fake_tools.get(name)), \
             patch.object(verify, "load_snapshotter", return_value=self.snapshot), \
             patch.object(verify.platform, "platform", return_value="Linux-test"), \
             patch.object(verify.platform, "machine", return_value="aarch64"), \
             patch.object(verify.platform, "release", return_value="test-kernel"), \
             patch.object(verify, "run_bounded", side_effect=self.fake_bounded):
            return verify.run(self.args)

    def evidence(self):
        return json.loads((self.output_dir / "evidence.json").read_text(encoding="utf-8"))


class LinuxVerifyTests(unittest.TestCase):
    def test_success_sequence_cache_facts_and_space_paths(self):
        harness = RunnerHarness()
        harness.setUp()
        self.addCleanup(harness.temp.cleanup)
        self.assertEqual(harness.execute(), 0)
        evidence = harness.evidence()
        self.assertEqual(evidence["result"], "PASS")
        self.assertEqual(evidence["compiler_facts"]["compiler_id"], "GNU")
        self.assertEqual(evidence["test_report"]["executed_count"], 2)
        labels = [entry["label"] for entry in evidence["commands"]]
        self.assertLess(labels.index("configure"), labels.index("build"))
        self.assertLess(labels.index("build"), labels.index("ctest-list"))
        self.assertLess(labels.index("test"), labels.index("install"))
        self.assertTrue(any("source tree with spaces" in part for entry, _ in harness.calls for part in entry))
        self.assertTrue(all(cwd == harness.source for argv, cwd in harness.calls if "--check-config" not in argv))
        configure = next(entry["argv"] for entry in evidence["commands"] if entry["label"] == "configure")
        self.assertIn("-DSAGA_ENABLE_CUDA=OFF", configure)
        self.assertIn("-DSAGA_BUILD_TESTS=ON", configure)
        self.assertIn("-DSAGA_ENABLE_IMAGE_DECODER=OFF", configure)

    def test_all_test_presets_match_compiler_and_sanitizer_contracts(self):
        for preset, contract in verify.PRESETS.items():
            with self.subTest(preset=preset):
                harness = RunnerHarness()
                harness.setUp()
                harness.args.preset = preset
                self.addCleanup(harness.temp.cleanup)
                self.assertEqual(harness.execute(), 0)
                cache = harness.evidence()["effective_cache"]
                self.assertEqual(Path(cache["CMAKE_CXX_COMPILER"]).name, contract["compiler"])
                self.assertEqual(cache["SAGA_SANITIZERS"] == "ON", contract["sanitizers"])
                self.assertEqual(cache["SAGA_THREAD_SANITIZER"] == "ON", contract["thread_sanitizer"])

    def test_platform_rejection_creates_no_evidence(self):
        harness = RunnerHarness()
        harness.setUp()
        self.addCleanup(harness.temp.cleanup)
        with patch.object(verify.sys, "platform", "darwin"):
            self.assertEqual(verify.run(harness.args), 1)
        self.assertFalse(harness.output.exists())

    def test_directory_creation_failure_is_clean(self):
        harness = RunnerHarness()
        harness.setUp()
        self.addCleanup(harness.temp.cleanup)
        harness.output.write_text("not a directory", encoding="utf-8")
        with patch.object(verify.sys, "platform", "linux"):
            self.assertEqual(verify.run(harness.args), 1)
        self.assertTrue(harness.output.is_file())

    def test_in_source_roots_outside_build_are_rejected_without_artifacts(self):
        for relative in ("tools/output", "docs/output", ".git/output", "other-child/output", "."):
            with self.subTest(relative=relative):
                harness = RunnerHarness()
                harness.setUp()
                self.addCleanup(harness.temp.cleanup)
                harness.args.build_root = harness.source / relative
                with patch.object(verify.sys, "platform", "linux"):
                    self.assertEqual(verify.run(harness.args), 1)
                if relative == ".":
                    self.assertEqual(sorted(path.name for path in harness.source.iterdir()), ["tools"])
                else:
                    self.assertFalse((harness.source / relative).exists())

    def test_default_and_external_evidence_roots_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            default_root = source / "build/linux-evidence"
            verify.safe_build_root(source, default_root)
            verify.safe_build_root(source, source / "build")
            verify.safe_build_root(source, source / "build/custom")
            verify.safe_build_root(source, Path(directory) / "external evidence")

    def test_unsafe_root_and_product_preset_rejected_without_artifacts(self):
        harness = RunnerHarness()
        harness.setUp()
        self.addCleanup(harness.temp.cleanup)
        harness.args.build_root = harness.source / "tools" / "output"
        with patch.object(verify.sys, "platform", "linux"):
            self.assertEqual(verify.run(harness.args), 1)
        self.assertFalse((harness.source / "tools/output").exists())
        self.assertNotIn("linux-product-release", verify.PRESETS)

    def test_failure_stages_are_recorded(self):
        for stage in ("configure", "build", "test", "install", "toml", "ini", "dependency"):
            with self.subTest(stage=stage):
                harness = RunnerHarness()
                harness.setUp()
                self.addCleanup(harness.temp.cleanup)
                self.assertNotEqual(harness.execute(stage), 0)
                evidence = harness.evidence()
                self.assertEqual(evidence["result"], "FAIL")
                self.assertTrue(evidence["failure"])

    def test_timeout_and_command_start_error_preserve_failure(self):
        for stage in ("timeout", "start-error"):
            with self.subTest(stage=stage):
                harness = RunnerHarness()
                harness.setUp()
                self.addCleanup(harness.temp.cleanup)
                self.assertNotEqual(harness.execute(stage), 0)
                evidence = harness.evidence()
                self.assertEqual(evidence["result"], "FAIL")
                build_record = next(item for item in evidence["commands"] if item["label"] == "build")
                self.assertTrue(build_record.get("timeout") or build_record.get("start_error"))

    def test_cache_mismatch_and_missing_install_artifacts_fail(self):
        for key, value in (("CMAKE_BUILD_TYPE", "Release"), ("CMAKE_CXX_COMPILER", "/fake/bin/clang++"),
                           ("SAGA_BUILD_TESTS", "OFF"), ("SAGA_SANITIZERS", "ON")):
            with self.subTest(key=key):
                harness = RunnerHarness()
                harness.setUp()
                harness.cache_overrides[key] = value
                self.addCleanup(harness.temp.cleanup)
                self.assertNotEqual(harness.execute(), 0)
                self.assertIn("cache", harness.evidence()["failure"])
        for missing in ("server", "toml", "ini"):
            harness = RunnerHarness()
            harness.setUp()
            harness.missing_install = missing
            self.addCleanup(harness.temp.cleanup)
            self.assertNotEqual(harness.execute(), 0)
            self.assertIn("missing", harness.evidence()["failure"])

    def test_after_snapshot_error_and_final_save_error_return_failure(self):
        harness = RunnerHarness()
        harness.setUp()
        harness.snapshot_raises_after = True
        self.addCleanup(harness.temp.cleanup)
        self.assertNotEqual(harness.execute(), 0)
        self.assertIn("simulated after-snapshot failure", harness.evidence()["failure"])

        harness = RunnerHarness()
        harness.setUp()
        self.addCleanup(harness.temp.cleanup)
        original_write_text = Path.write_text

        def fail_final_save(path, data, *args, **kwargs):
            if path.name == "evidence.json" and '"result": "PASS"' in data:
                raise OSError("simulated final evidence write failure")
            return original_write_text(path, data, *args, **kwargs)

        with patch.object(Path, "write_text", new=fail_final_save):
            self.assertNotEqual(harness.execute(), 0)

    def test_strict_booleans_and_generated_facts_reject_malformed_values(self):
        for key, value in (("SAGA_ENABLE_CUDA", "garbage"), ("SAGA_BUILD_TESTS", "maybe"),
                           ("SAGA_SANITIZERS", "perhaps")):
            with self.subTest(key=key):
                harness = RunnerHarness()
                harness.setUp()
                harness.cache_overrides[key] = value
                self.addCleanup(harness.temp.cleanup)
                self.assertNotEqual(harness.execute(), 0)
                self.assertIn("invalid effective CMake boolean", harness.evidence()["failure"])
        for old, new in (("GNU", "Unknown"), ('"14.1"', '"not-a-version"')):
            with self.subTest(fact=old):
                harness = RunnerHarness()
                harness.setUp()
                harness.fact_overrides[old] = new
                self.addCleanup(harness.temp.cleanup)
                self.assertNotEqual(harness.execute(), 0)
                self.assertTrue(harness.evidence().get("failure"))
        harness = RunnerHarness()
        harness.setUp()
        harness.fact_overrides["system"] = "Darwin"
        self.addCleanup(harness.temp.cleanup)
        self.assertNotEqual(harness.execute(), 0)
        self.assertIn("system=Darwin", harness.evidence()["failure"])

    def test_missing_tools_and_changed_source_fail(self):
        harness = RunnerHarness()
        harness.setUp()
        harness.missing_tool = "ninja"
        self.addCleanup(harness.temp.cleanup)
        self.assertNotEqual(harness.execute(), 0)
        self.assertIn("missing required tool", harness.evidence()["failure"])
        harness = RunnerHarness()
        harness.setUp()
        harness.stage = "source-change"
        self.addCleanup(harness.temp.cleanup)
        self.assertNotEqual(harness.execute("source-change"), 0)
        self.assertIn("source files changed", harness.evidence()["failure"])

    def test_ctest_incomplete_and_skipped_reports_fail_full_flow(self):
        for mode in ("skipped", "incomplete", "missing"):
            with self.subTest(mode=mode):
                harness = RunnerHarness()
                harness.setUp()
                harness.junit_mode = mode
                self.addCleanup(harness.temp.cleanup)
                self.assertNotEqual(harness.execute(), 0)
                self.assertEqual(harness.evidence()["result"], "FAIL")

    def test_skipped_or_incomplete_junit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "expected.json"
            report = Path(directory) / "junit.xml"
            expected.write_text('{"tests":[{"name":"alpha"},{"name":"beta"}]}', encoding="utf-8")
            report.write_text('<testsuite><testcase name="alpha"/><testcase name="beta"><skipped/></testcase></testsuite>', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "skipped"):
                verify.validate_ctest(expected, report)
            report.write_text('<testsuite><testcase name="alpha"/></testsuite>', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify.validate_ctest(expected, report)

    @unittest.skipUnless(verify.sys.platform.startswith("linux"), "Linux process-group regression uses /proc")
    def test_timed_out_process_group_kills_descendant(self):
        import time

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_file = root / "descendant.pid"
            script = root / "spawn.py"
            script.write_text(
                "import subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c', "
                "'import time; time.sleep(60)'])\n"
                "open(sys.argv[1], 'w').write(str(child.pid))\n"
                "time.sleep(60)\n", encoding="utf-8")
            with (root / "out.log").open("wb") as stdout, (root / "err.log").open("wb") as stderr:
                with self.assertRaises(verify.subprocess.TimeoutExpired):
                    verify.run_bounded([verify.sys.executable, str(script), str(pid_file)], root,
                                       stdout, stderr, 0.2)
            descendant = int(pid_file.read_text(encoding="utf-8"))
            proc_stat = Path(f"/proc/{descendant}/stat")
            for _ in range(50):
                try:
                    stat = proc_stat.read_text(encoding="utf-8")
                    state = stat[stat.rfind(")") + 2:].split()[0]
                except (FileNotFoundError, ProcessLookupError):
                    break
                if state in ("Z", "X"):
                    break
                time.sleep(0.02)
            else:
                self.fail("timed-out descendant remained executing after process-group cleanup")

    def test_argument_validation(self):
        self.assertEqual(verify.positive("2"), 2)
        with self.assertRaises(argparse.ArgumentTypeError):
            verify.positive("0")


if __name__ == "__main__":
    unittest.main()
