#!/usr/bin/env python3
"""Collect bounded native-Linux CUDA smoke evidence; not product qualification."""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def run_command(argv: list[str], cwd: Path, timeout: int, stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
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


def record_command(data: dict[str, Any], evidence: Path, log: Path, label: str, argv: list[str], cwd: Path,
                   timeout: int, save, optional: bool = False, stdout_path: Path | None = None,
                   stderr_path: Path | None = None) -> int:
    idx = len(data["commands"])
    out = stdout_path or evidence / f"{idx:02d}-{label}.stdout.log"
    err = stderr_path or evidence / f"{idx:02d}-{label}.stderr.log"
    rec = {"label": label, "argv": argv, "cwd": str(cwd), "timeout_seconds": timeout,
           "stdout_log": str(out), "stderr_log": str(err), "exit_code": None}
    data["commands"].append(rec)
    save()
    with log.open("a") as stream:
        stream.write(f"\n=== {label} ===\nCOMMAND: {' '.join(argv)}\n")
    try:
        code = run_command(argv, cwd, timeout, out, err)
    except subprocess.TimeoutExpired as exc:
        rec["timeout"] = True
        save()
        raise RuntimeError(f"{label} timed out; partial logs retained") from exc
    rec["exit_code"] = code
    with log.open("a") as stream:
        stream.write(out.read_text(errors="replace")); stream.write(err.read_text(errors="replace"))
        stream.write(f"EXIT_CODE: {code}\n")
    save()
    if code and not optional:
        raise RuntimeError(f"{label} failed with exit code {code}")
    return code


def validate_linux_report(report: Any, device: int, nvcc_version: tuple[int, int, int]) -> None:
    if type(report) is not dict:
        raise ValueError("invalid CUDA report")
    tc = report.get("toolchain")
    if type(tc) is not dict or tc.get("os") != "Linux" or tc.get("host_compiler") not in ("GCC", "Clang"):
        raise ValueError("report does not identify a native Linux GCC/Clang build")
    if type(tc.get("host_compiler_version")) is not str or not tc["host_compiler_version"]:
        raise ValueError("missing native host compiler version")
    for name, value in (("nvcc_major", tc.get("nvcc_major")), ("nvcc_minor", tc.get("nvcc_minor")),
                        ("nvcc_build", tc.get("nvcc_build")), ("cuda_headers_version", tc.get("cuda_headers_version")),
                        ("cplusplus", tc.get("cplusplus"))):
        if type(value) is not int or value < (1 if name in ("nvcc_major", "cuda_headers_version", "cplusplus") else 0):
            raise ValueError(f"invalid toolchain {name}")
    if (tc["nvcc_major"], tc["nvcc_minor"], tc["nvcc_build"]) != nvcc_version or nvcc_version[:2] < (12, 8):
        raise ValueError("runtime report NVCC version disagrees with collected NVCC version or is below 12.8")
    if report.get("expected_compile_arch") != 890:
        raise ValueError("expected compile architecture mismatch")
    # Reuse the established strict device/resource/check validator after validating Linux-native identity.
    evidence = importlib.util.spec_from_file_location("m1_evidence", ROOT / "tools/m1_evidence.py")
    if evidence is None or evidence.loader is None:
        raise RuntimeError("cannot load tools/m1_evidence.py")
    module = importlib.util.module_from_spec(evidence)
    evidence.loader.exec_module(module)
    compatible = json.loads(json.dumps(report))
    compatible["toolchain"]["os"] = "Windows"
    compatible["toolchain"]["host_compiler"] = "MSVC"
    compatible["toolchain"]["host_compiler_version"] = 1
    module.validate_cuda(compatible, device)


def required_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required tool unavailable: {name}")
    return path


def parse_nvcc_version(text: str) -> tuple[int, int, int]:
    import re
    match = re.search(r"release (\d+)\.(\d+),\s*V(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        raise ValueError("could not parse full nvcc release and Vmajor.minor.build version")
    release = (int(match.group(1)), int(match.group(2)))
    version = (int(match.group(3)), int(match.group(4)))
    if release != version:
        raise ValueError("nvcc release version does not match Vmajor.minor")
    return release[0], release[1], int(match.group(5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--preset", default="cuda-linux", choices=("cuda-linux",))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if sys.platform != "linux":
        print("FAIL: native Linux required", file=sys.stderr); return 1
    if args.device < 0 or args.timeout_seconds <= 0:
        parser.error("device must be nonnegative and timeout must be positive")
    build_root = (args.build_root or source / "build/cuda-linux-evidence").resolve()
    try:
        build_root.relative_to(source)
        if not build_root.is_relative_to(source / "build"):
            raise ValueError("evidence root must be outside source or under source/build")
    except ValueError as exc:
        if str(exc).startswith("evidence root"):
            print(f"FAIL: {exc}", file=sys.stderr); return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence = build_root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    evidence.mkdir(parents=True, exist_ok=False)
    log = evidence / "cuda-smoke.log"
    data: dict[str, Any] = {"schema_version": 1, "kind": "linux-original-rtx4080-cuda-smoke",
        "qualification": False, "source_root": str(source), "evidence_root": str(evidence),
        "preset": args.preset, "device": args.device, "timeout_seconds": args.timeout_seconds,
        "platform": platform.platform(), "architecture": platform.machine(), "kernel": platform.release(),
        "commands": [], "result": "FAIL"}
    def save():
        (evidence / "evidence.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    def command(label: str, argv: list[str], cwd: Path = source, optional: bool = False) -> int:
        return record_command(data, evidence, log, label, argv, cwd, args.timeout_seconds, save, optional)
    try:
        for required in ("nvcc", "cmake", "ninja", "g++"):
            required_tool(required)
        evidence_tool = str(source / "tools/m1_evidence.py")
        command("source-before", [sys.executable, "-B", evidence_tool, "snapshot", str(source), str(evidence / "source-before.json")])
        for tool in ("nvcc", "cmake", "ninja", "g++"):
            path = shutil.which(tool)
            data.setdefault("tools", {})[tool] = path
            if path: command("version-" + tool, [path, "--version"])
        nvcc_log = next(evidence.glob("*-version-nvcc.stdout.log"))
        nvcc_ver = parse_nvcc_version(nvcc_log.read_text(errors="replace"))
        data["nvcc_version"] = {"major": nvcc_ver[0], "minor": nvcc_ver[1], "build": nvcc_ver[2]}
        smi = shutil.which("nvidia-smi")
        data["nvidia_smi"] = smi
        if smi:
            command("nvidia-smi", [smi, "--query-gpu=name,uuid,driver_version,power.draw,display_active", "--format=csv,noheader"], optional=True)
        binary = evidence / "build"
        command("configure", [required_tool("cmake"), "-S", str(source), "-B", str(binary), "--preset", args.preset])
        cache = (binary / "CMakeCache.txt").read_text(errors="replace")
        (evidence / "CMakeCache.txt").write_text(cache, encoding="utf-8")
        for k, expected in (("SAGA_ENABLE_CUDA", "ON"), ("CMAKE_BUILD_TYPE", "Release"), ("CMAKE_CUDA_ARCHITECTURES", "89")):
            if not any(line.startswith(k + ":") and line.split("=", 1)[-1] == expected for line in cache.splitlines()):
                raise RuntimeError(f"effective CMake cache violates {k}={expected}")
        fact_files = list((binary / "CMakeFiles").glob("*/CMakeCUDACompiler.cmake")) + list((binary / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
        if len(fact_files) < 2: raise RuntimeError("generated CMake C++/CUDA compiler facts missing")
        (evidence / "compiler-facts").mkdir()
        for fact in fact_files: shutil.copy2(fact, evidence / "compiler-facts" / fact.name)
        command("verbose-build-and-device-link", [required_tool("cmake"), "--build", str(binary), "--target", "saga_cuda_portability_smoke", "--verbose", "--parallel", "2"])
        exe = binary / "saga_cuda_portability_smoke"
        if not exe.is_file(): raise RuntimeError(f"CUDA smoke executable missing: {exe}")
        runtime = evidence / "runtime.json"; runtime_err = evidence / "runtime.stderr.log"
        runtime_argv = [str(exe), "--device", str(args.device)]
        code = record_command(data, evidence, log, "runtime-probe", runtime_argv, source,
                              args.timeout_seconds, save, stdout_path=runtime, stderr_path=runtime_err)
        data["runtime_exit_code"] = code
        report = json.loads(runtime.read_text(encoding="utf-8"))
        validate_linux_report(report, args.device, nvcc_ver)
        data["runtime_validated"] = True
        command("source-after", [sys.executable, "-B", evidence_tool, "snapshot", str(source), str(evidence / "source-after.json")])
        command("source-compare", [sys.executable, "-B", evidence_tool, "compare", str(evidence / "source-before.json"), str(evidence / "source-after.json")])
        data["result"] = "PASS"
        data["scope_note"] = "Native Linux original RTX 4080 CUDA smoke only; not serving, performance, or production qualification."
    except Exception as exc:
        data["failure"] = str(exc)
    finally:
        before_snapshot = evidence / "source-before.json"
        after_snapshot = evidence / "source-after.json"
        if before_snapshot.is_file() and not after_snapshot.exists():
            try:
                command("source-after", [sys.executable, "-B", str(source / "tools/m1_evidence.py"),
                                          "snapshot", str(source), str(after_snapshot)])
                command("source-compare", [sys.executable, "-B", str(source / "tools/m1_evidence.py"),
                                            "compare", str(before_snapshot), str(after_snapshot)])
            except Exception as snapshot_error:
                data["snapshot_failure"] = str(snapshot_error)
                data.setdefault("failure", str(snapshot_error))
        if data.get("failure"):
            data["result"] = "FAIL"
            with log.open("a") as stream: stream.write(f"RESULT: FAIL\nERROR: {data['failure']}\n")
    save()
    print(f"{data['result']}: evidence at {evidence}")
    return 0 if data["result"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
