#!/usr/bin/env python3
"""Run bounded Linux host build/test checks; this is not production qualification."""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

PRESETS = {
    "linux-gcc-debug": {"compiler": "g++", "build_type": "Debug", "sanitizers": False, "thread_sanitizer": False},
    "linux-gcc-release": {"compiler": "g++", "build_type": "Release", "sanitizers": False, "thread_sanitizer": False},
    "linux-clang-debug": {"compiler": "clang++", "build_type": "Debug", "sanitizers": False, "thread_sanitizer": False},
    "linux-clang-release": {"compiler": "clang++", "build_type": "Release", "sanitizers": False, "thread_sanitizer": False},
    "linux-clang-sanitized": {"compiler": "clang++", "build_type": "Debug", "sanitizers": True, "thread_sanitizer": False},
    "linux-clang-thread-sanitized": {"compiler": "clang++", "build_type": "Debug", "sanitizers": False, "thread_sanitizer": True},
}
ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOTS = ("include", "src", "apps", "tests", "cmake", "tools", "third_party")


def positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--parallel", required=True, type=positive)
    parser.add_argument("--enable-image-decoder", action="store_true")
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--timeout-seconds", type=positive, default=1200)
    return parser


def load_snapshotter(source: Path):
    spec = importlib.util.spec_from_file_location("m1_evidence", source / "tools/m1_evidence.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tools/m1_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.snapshot


def cache_value(cache: Path, key: str) -> str:
    prefix = key + ":"
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    raise RuntimeError(f"missing effective CMake cache value: {key}")


def safe_build_root(source: Path, build_root: Path) -> None:
    source = source.resolve()
    build_root = build_root.resolve()
    try:
        relative = build_root.relative_to(source)
    except ValueError:
        return
    parts = relative.parts
    if not parts:
        raise ValueError("evidence root cannot be the source root")
    if parts[0] == "build":
        return
    raise ValueError(f"evidence root must be outside the source tree or under source/build: {build_root}")


def compiler_facts(binary: Path, selected: Path, compiler_name: str) -> dict[str, Any]:
    system_files = list((binary / "CMakeFiles").glob("*/CMakeSystem.cmake"))
    compiler_files = list((binary / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(system_files) != 1 or len(compiler_files) != 1:
        raise RuntimeError("expected exactly one generated CMake system/compiler facts file")
    system_text = system_files[0].read_text(encoding="utf-8", errors="replace")
    compiler_text = compiler_files[0].read_text(encoding="utf-8", errors="replace")
    system = _cmake_set(system_text, "CMAKE_SYSTEM_NAME")
    compiler_id = _cmake_set(compiler_text, "CMAKE_CXX_COMPILER_ID")
    compiler_version = _cmake_set(compiler_text, "CMAKE_CXX_COMPILER_VERSION")
    compiler_path = _cmake_set(compiler_text, "CMAKE_CXX_COMPILER")
    expected_id = "GNU" if compiler_name == "g++" else "Clang"
    if system != "Linux" or compiler_id != expected_id:
        raise RuntimeError(f"generated CMake facts mismatch: system={system}, compiler={compiler_id}")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", compiler_version) is None:
        raise RuntimeError(f"invalid generated compiler version: {compiler_version!r}")
    if Path(compiler_path).resolve() != selected.resolve():
        raise RuntimeError(f"CMake selected unexpected compiler path: {compiler_path}")
    return {"system_name": system, "compiler_id": compiler_id, "compiler_version": compiler_version,
            "compiler_path": compiler_path, "system_file": str(system_files[0]),
            "compiler_file": str(compiler_files[0])}


def strict_bool(value: str, key: str) -> bool:
    normalized = value.strip().upper()
    true_values = {"ON", "TRUE", "YES", "1"}
    false_values = {"OFF", "FALSE", "NO", "0", "", "IGNORE"}
    if normalized in true_values:
        return True
    if normalized in false_values or normalized.endswith("-NOTFOUND"):
        return False
    raise RuntimeError(f"invalid effective CMake boolean for {key}: {value!r}")


def _cmake_set(text: str, key: str) -> str:
    prefix = f'set({key} '
    for line in text.splitlines():
        if line.startswith(prefix) and line.endswith(")"):
            return line[len(prefix):-1].strip().strip('"')
    raise RuntimeError(f"missing generated CMake field: {key}")


def run_bounded(argv: list[str], cwd: Path, stdout, stderr, timeout: int) -> int:
    process = subprocess.Popen(argv, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def validate_ctest(expected_json: Path, junit_path: Path) -> dict[str, Any]:
    expected_data = json.loads(expected_json.read_text(encoding="utf-8"))
    expected = [test["name"] for test in expected_data.get("tests", [])]
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("CTest expected-test list is empty or contains duplicates")
    if not junit_path.is_file() or junit_path.stat().st_size == 0:
        raise RuntimeError("CTest JUnit report is missing or empty")
    root = ET.parse(junit_path).getroot()
    cases = root.findall(".//testcase")
    actual = [case.attrib.get("name", "") for case in cases]
    if not all(actual) or len(actual) != len(set(actual)):
        raise RuntimeError("CTest JUnit report has missing or duplicate test names")
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise RuntimeError(f"CTest test list mismatch: expected {len(expected)}, reported {len(actual)}")
    skipped = [case.attrib.get("name", "") for case in cases if case.find("skipped") is not None]
    failed = [case.attrib.get("name", "") for case in cases
              if case.find("failure") is not None or case.find("error") is not None]
    if skipped:
        raise RuntimeError("CTest reported skipped tests: " + ", ".join(skipped))
    if failed:
        raise RuntimeError("CTest JUnit reports failed tests: " + ", ".join(failed))
    return {"expected_count": len(expected), "executed_count": len(actual), "skipped": skipped, "failed": failed}


def run(args: argparse.Namespace) -> int:
    source = args.source_root.resolve()
    build_root = (args.build_root or source / "build/linux-evidence").resolve()
    if sys.platform != "linux":
        print("FAIL: Linux host required", file=sys.stderr)
        return 1
    try:
        safe_build_root(source, build_root)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = build_root / f"{stamp}-{uuid.uuid4().hex[:12]}"
    binary = run_dir / "binary"
    install = run_dir / "installed application"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
        binary.mkdir()
    except OSError as exc:
        print(f"FAIL: cannot create evidence directory {run_dir}: {exc}", file=sys.stderr)
        return 1

    evidence: dict[str, Any] = {
        "schema_version": 1, "kind": "linux-host-development-test", "qualification": False,
        "preset": args.preset, "source_root": str(source), "run_root": str(run_dir),
        "os": platform.platform(), "architecture": platform.machine(), "kernel": platform.release(),
        "distro": Path("/etc/os-release").read_text(errors="replace") if Path("/etc/os-release").is_file() else None,
        "commands": [], "result": "FAIL",
    }
    log_path = run_dir / "evidence.json"

    def save() -> None:
        log_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    def command(label: str, argv: list[str], cwd: Path) -> None:
        index = len(evidence["commands"])
        stdout_path = run_dir / f"{index:02d}-{label}.stdout.log"
        stderr_path = run_dir / f"{index:02d}-{label}.stderr.log"
        record = {"label": label, "argv": argv, "cwd": str(cwd), "timeout_seconds": args.timeout_seconds,
                  "exit_code": None, "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
        evidence["commands"].append(record)
        save()
        print(f"RUN {label}: {' '.join(argv)}", flush=True)
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                record["exit_code"] = run_bounded(argv, cwd, stdout, stderr, args.timeout_seconds)
        except subprocess.TimeoutExpired:
            record["timeout"] = True
            save()
            raise RuntimeError(f"{label} timed out; partial output retained")
        except OSError as exc:
            record["start_error"] = str(exc)
            save()
            raise RuntimeError(f"{label} could not start: {exc}") from exc
        save()
        if record["exit_code"]:
            raise RuntimeError(f"{label} failed with exit code {record['exit_code']}")

    snapshot = None
    before = None
    after_snapshot_attempted = False
    final_status = 1
    try:
        snapshot = load_snapshotter(source)
        before = snapshot(source)
        evidence["source_snapshot_before"] = before
        compiler_name = PRESETS[args.preset]["compiler"]
        tool_names = ("cmake", "ctest", "ninja", "python3", compiler_name)
        tools = {name: shutil.which(name) for name in tool_names}
        evidence["tools"] = tools
        missing = [name for name, path in tools.items() if not path]
        if missing:
            raise RuntimeError("missing required tool(s): " + ", ".join(missing))
        tools = cast(dict[str, str], tools)
        selected = Path(tools[compiler_name]).resolve()
        for name, path in tools.items():
            command("version-" + name, [path, "--version"], source)

        image = "ON" if args.enable_image_decoder else "OFF"
        command("configure", [tools["cmake"], "-S", str(source), "-B", str(binary), "--preset", args.preset,
                               "-DSAGA_ENABLE_CUDA=OFF", "-DSAGA_BUILD_TESTS=ON",
                               f"-DCMAKE_INSTALL_PREFIX={install}", "-DCMAKE_INSTALL_BINDIR=bin",
                               "-DCMAKE_INSTALL_DATADIR=share", f"-DSAGA_ENABLE_IMAGE_DECODER={image}",
                               f"-DPython3_EXECUTABLE={sys.executable}"], source)
        cache_path = binary / "CMakeCache.txt"
        cache = {key: cache_value(cache_path, key) for key in (
            "SAGA_BUILD_TESTS", "SAGA_ENABLE_CUDA", "SAGA_ENABLE_IMAGE_DECODER", "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_COMPILER", "SAGA_SANITIZERS", "SAGA_THREAD_SANITIZER")}
        evidence["effective_cache"] = cache
        expected = PRESETS[args.preset]
        tests_enabled = strict_bool(cache["SAGA_BUILD_TESTS"], "SAGA_BUILD_TESTS")
        cuda_enabled = strict_bool(cache["SAGA_ENABLE_CUDA"], "SAGA_ENABLE_CUDA")
        image_enabled = strict_bool(cache["SAGA_ENABLE_IMAGE_DECODER"], "SAGA_ENABLE_IMAGE_DECODER")
        sanitizers = strict_bool(cache["SAGA_SANITIZERS"], "SAGA_SANITIZERS")
        thread_sanitizer = strict_bool(cache["SAGA_THREAD_SANITIZER"], "SAGA_THREAD_SANITIZER")
        compiler_cache = Path(cache["CMAKE_CXX_COMPILER"]).resolve()
        if (not tests_enabled or cuda_enabled or image_enabled != (image == "ON")
                or cache["CMAKE_BUILD_TYPE"] != expected["build_type"] or compiler_cache != selected
                or sanitizers != expected["sanitizers"] or thread_sanitizer != expected["thread_sanitizer"]
                or (sanitizers and thread_sanitizer)):
            raise RuntimeError("effective CMake cache violates preset contract")
        evidence["cache_copy"] = str(run_dir / "CMakeCache.txt")
        shutil.copy2(cache_path, run_dir / "CMakeCache.txt")
        facts = compiler_facts(binary, selected, compiler_name)
        evidence["compiler_facts"] = facts
        for key, path in (("system", facts["system_file"]), ("compiler", facts["compiler_file"])):
            evidence[key + "_facts_copy"] = str(run_dir / (key + "-facts.cmake"))
            shutil.copy2(path, run_dir / (key + "-facts.cmake"))

        command("build", [tools["cmake"], "--build", str(binary), "--parallel", str(args.parallel)], source)
        expected_json = run_dir / "ctest-expected.json"
        junit = run_dir / "ctest-results.xml"
        command("ctest-list", [tools["ctest"], "--test-dir", str(binary), "--show-only=json-v1"], source)
        # Retain CTest's expected list directly from a bounded command log.
        list_record = evidence["commands"][-1]
        expected_json.write_text(Path(list_record["stdout_log"]).read_text(encoding="utf-8"), encoding="utf-8")
        command("test", [tools["ctest"], "--test-dir", str(binary), "--output-on-failure", "--no-tests=error",
                          "--parallel", str(args.parallel), "--output-junit", str(junit)], source)
        evidence["test_report"] = validate_ctest(expected_json, junit)
        command("install", [tools["cmake"], "--install", str(binary)], source)
        server = install / "bin/saga-server"
        toml = install / "share/saga/server.example.toml"
        ini = install / "share/saga/model.example.ini"
        for path in (server, toml, ini):
            if not path.is_file():
                raise RuntimeError(f"installed artifact missing: {path}")
        command("check-config-toml", [str(server), "--config", str(toml), "--check-config"], run_dir)
        command("check-config-toml-ini", [str(server), "--config", str(toml), "--model-config", str(ini), "--check-config"], run_dir)
        command("dependency-hashes", [sys.executable, "-B", str(source / "tools/verify_dependencies.py")], source)
        after_snapshot_attempted = True
        after = snapshot(source)
        evidence["source_snapshot_after"] = after
        if before["files"] != after["files"]:
            raise RuntimeError("relevant source files changed during run")
        evidence["result"] = "PASS"
        evidence["scope_note"] = "Linux host development/testing only; no CUDA, RTX 4080, model-fit, production, or performance qualification."
        final_status = 0
    except Exception as exc:
        evidence["failure"] = str(exc)
        evidence["result"] = "FAIL"
        final_status = 1
    finally:
        if snapshot is not None and before is not None and not after_snapshot_attempted:
            after_snapshot_attempted = True
            try:
                after = snapshot(source)
                evidence["source_snapshot_after"] = after
                if before["files"] != after["files"]:
                    evidence["failure"] = evidence.get("failure", "relevant source files changed during run")
                    evidence["result"] = "FAIL"
                    final_status = 1
            except Exception as exc:
                evidence["snapshot_failure"] = str(exc)
                evidence["result"] = "FAIL"
                evidence.setdefault("failure", f"source after-snapshot failed: {exc}")
                final_status = 1
        try:
            save()
            print(f"{evidence['result']}: evidence at {log_path}", flush=True)
            if evidence.get("failure"):
                print(f"FAIL: {evidence['failure']}", file=sys.stderr, flush=True)
        except OSError as exc:
            print(f"FAIL: evidence output unavailable at {log_path}: {exc}", file=sys.stderr, flush=True)
            final_status = 1
    return final_status


def main() -> int:
    args = make_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
