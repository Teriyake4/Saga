#!/usr/bin/env python3
"""Bounded native CUDA projection-device evidence runner; not qualification."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]


class CommandTimeout(subprocess.TimeoutExpired):
    def __init__(self, argv, timeout: int, cleanup_resolved: bool):
        super().__init__(argv, timeout)
        self.cleanup_resolved = cleanup_resolved


class TreeCleanupUnresolved(RuntimeError):
    pass


class LogLimitExceeded(RuntimeError):
    def __init__(self, cleanup_resolved: bool):
        super().__init__("stdout or stderr exceeded the configured byte limit")
        self.cleanup_resolved = cleanup_resolved


def terminate_tree(process: subprocess.Popen, windows: bool) -> bool:
    tree_stopped = False
    if windows:
        taskkill = shutil.which("taskkill")
        if taskkill:
            try:
                result = subprocess.run([taskkill, "/PID", str(process.pid), "/T", "/F"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        check=False, timeout=10)
                tree_stopped = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                tree_stopped = False
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            tree_stopped = True
        except ProcessLookupError:
            tree_stopped = process.poll() is not None
        except OSError:
            tree_stopped = False
    if process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return False
    return tree_stopped


def run_bounded(argv: list[str], cwd: Path, timeout: int, stdout_path: Path,
                stderr_path: Path, windows: bool, max_log_bytes: int = 16 * 1024 * 1024) -> int:
    overflow = threading.Event()
    kwargs: dict[str, Any] = {}
    if windows:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               bufsize=0, **kwargs)

    captures = [{"lock": threading.Lock(), "file": None, "ready": threading.Event()},
                {"lock": threading.Lock(), "file": None, "ready": threading.Event()}]

    def close_captures() -> None:
        for capture in captures:
            with capture["lock"]:
                output = capture["file"]
                if output is not None and not output.closed:
                    output.flush()
                    output.close()

    def pump(source, destination: Path, capture) -> None:
        written = 0
        output = destination.open("wb")
        with capture["lock"]:
            capture["file"] = output
            capture["ready"].set()
        try:
            while True:
                chunk = source.read(65536)
                if not chunk:
                    break
                remaining = max_log_bytes - written
                with capture["lock"]:
                    if remaining > 0 and not output.closed:
                        piece = chunk[:remaining]
                        output.write(piece)
                        written += len(piece)
                    if len(chunk) > remaining:
                        if not output.closed:
                            output.flush()
                            output.close()
                        overflow.set()
        finally:
            with capture["lock"]:
                if not output.closed:
                    output.close()

    readers = [threading.Thread(target=pump, args=(process.stdout, stdout_path, captures[0]), daemon=True),
               threading.Thread(target=pump, args=(process.stderr, stderr_path, captures[1]), daemon=True)]
    for reader in readers:
        reader.start()
    for capture in captures:
        capture["ready"].wait()
    cleanup_resolved = True
    termination_attempted = False
    try:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if overflow.is_set():
                cleanup_resolved = terminate_tree(process, windows)
                termination_attempted = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cleanup_resolved = terminate_tree(process, windows)
                termination_attempted = True
                raise CommandTimeout(argv, timeout, cleanup_resolved)
            time.sleep(min(0.02, remaining))
        for reader in readers:
            reader.join(timeout=2)
        if overflow.is_set() and not termination_attempted:
            cleanup_resolved = terminate_tree(process, windows)
            termination_attempted = True
        if any(reader.is_alive() for reader in readers):
            cleanup_resolved = terminate_tree(process, windows) and cleanup_resolved
        if overflow.is_set():
            raise LogLimitExceeded(cleanup_resolved)
        if any(reader.is_alive() for reader in readers):
            raise TreeCleanupUnresolved("output pipe remained open after process-tree termination")
        if not cleanup_resolved:
            raise TreeCleanupUnresolved("descendant cleanup unresolved after process-tree termination")
        return process.wait(timeout=10)
    finally:
        if process.poll() is None:
            cleanup_resolved = terminate_tree(process, windows) and cleanup_resolved
        close_captures()
        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        for reader in readers:
            reader.join(timeout=5)
        if any(reader.is_alive() for reader in readers):
            raise TreeCleanupUnresolved("stdout/stderr reader thread did not stop after pipe closure")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_nonnegative(value: Any) -> bool:
    if type(value) is int:
        return value >= 0
    return type(value) is float and math.isfinite(value) and value >= 0


def validate_report(report: Any, expected_iterations: int) -> None:
    require(type(report) is dict, "runtime JSON must be an object")
    report = cast(dict[str, Any], report)
    require(report.get("status") == "physical_projection_test_passed", "physical projection test did not pass")
    require(report.get("qualification") == "not_claimed", "qualification must remain not_claimed")
    require(report.get("readiness_approval") is False, "readiness approval must remain false")
    require(report.get("ledger_reconciliation") == "passed", "ledger reconciliation did not pass")
    device = report.get("device")
    require(type(device) is dict, "device identity is missing")
    device = cast(dict[str, Any], device)
    require(device.get("name") in ("NVIDIA GeForce RTX 4080", "GeForce RTX 4080"),
            "device is not an original GeForce RTX 4080")
    require(device.get("architecture") == "sm89" and device.get("architecture_version") == 89,
            "device architecture is not SM89")
    identity = device.get("identity")
    require(type(identity) is str and len(identity) == 32
            and all(character in "0123456789abcdef" for character in identity),
            "device identity must be a 32-character lowercase hexadecimal UUID")
    for key in ("ordinal", "driver", "runtime"):
        require(type(device.get(key)) is int and device[key] >= (0 if key == "ordinal" else 1),
                f"invalid device {key}")
    iterations = report.get("iterations_per_batch")
    require(type(iterations) is int and iterations == expected_iterations,
            "runtime iteration count does not match the requested count")
    require(report.get("timing_batches_per_variant") == 5, "expected five timing batches per variant")
    variants = report.get("variants")
    require(type(variants) is list and len(variants) == 2, "expected baseline and WMMA variants")
    variants = cast(list[Any], variants)
    ids: list[str] = []
    for variant in variants:
        require(type(variant) is dict, "malformed projection variant")
        variant = cast(dict[str, Any], variant)
        identity = variant.get("variant")
        require(type(identity) is str, "variant ID is missing")
        ids.append(identity)
        request = variant.get("request")
        require(type(request) is dict, "missing projection request")
        request = cast(dict[str, Any], request)
        expected_request = {
            "operation": "projection", "weights": "bf16", "activation": "bf16",
            "output": "fp32", "numerics": "fp32_accumulate", "layout": "contiguous",
            "phase": "decode",
        }
        require(all(request.get(key) == value for key, value in expected_request.items()),
                "projection request contract mismatch")
        base = identity == "sm89.bf16_projection.baseline.decode"
        wmma = identity == "sm89.bf16_projection.wmma.decode"
        require(base or wmma, "unexpected projection variant ID")
        shape = (19, 23, 3) if base else (32, 32, 32)
        strides = (48, 50, 80) if base else (96, 96, 192)
        alignment = 2 if base else 32
        integer_fields = ("outputs", "inputs", "tokens", "weight_row_stride_bytes",
                          "activation_row_stride_bytes", "output_row_stride_bytes", "alignment")
        require(all(type(request.get(key)) is int for key in integer_fields),
                "projection request dimensions/strides/alignment must be integers")
        require((request.get("outputs"), request.get("inputs"), request.get("tokens")) == shape,
                "projection request shape mismatch")
        require((request.get("weight_row_stride_bytes"), request.get("activation_row_stride_bytes"),
                 request.get("output_row_stride_bytes")) == strides,
                "projection request strides mismatch")
        require(request.get("alignment") == alignment, "projection request alignment mismatch")
        error = variant.get("max_abs_error")
        require(finite_nonnegative(error), "invalid projection error")
        require(cast(Any, error) <= 0.02, "projection error exceeds 0.02")
        attributes = variant.get("kernel_attributes")
        require(type(attributes) is dict, "missing queried kernel resource attributes")
        attributes = cast(dict[str, Any], attributes)
        for key in ("registers_per_thread", "static_shared_bytes", "local_bytes_per_thread", "ptx_version"):
            require(type(attributes.get(key)) is int and attributes[key] >= 0,
                    f"invalid kernel resource {key}")
        require(type(attributes.get("max_threads_per_block")) is int
                and attributes["max_threads_per_block"] >= 1, "invalid kernel maximum thread count")
        require(attributes.get("binary_version") == 89, "projection kernel binary is not SM89")
        require(type(attributes.get("active_blocks_per_multiprocessor")) is int
                and attributes["active_blocks_per_multiprocessor"] >= 1,
                "kernel has no queried active block occupancy")
        timings = variant.get("event_ms_per_launch")
        require(type(timings) is list and len(timings) >= 5
                and all(finite_nonnegative(item) for item in cast(list[Any], timings)), "invalid timing samples")
    require(set(ids) == {"sm89.bf16_projection.baseline.decode", "sm89.bf16_projection.wmma.decode"},
            "unexpected baseline/WMMA variant IDs")
    ledger = report.get("ledger")
    require(type(ledger) is dict, "missing ledger profile")
    ledger = cast(dict[str, Any], ledger)
    require(type(ledger.get("profile")) is str and bool(ledger["profile"]), "missing ledger profile")
    live = ledger.get("live_bytes")
    require(type(live) is dict and set(live) == {"host", "pinned", "device"}
            and all(type(value) is int and value >= 0 for value in cast(dict[str, Any], live).values()),
            "invalid reconciled ledger live-byte totals")
    allocations = ledger.get("allocations")
    require(type(allocations) is list and bool(allocations), "missing ledger allocations")
    expected_allocations = {
        "weights": ("device", "persistent", "project"),
        "input": ("device", "persistent", "project"),
        "output": ("device", "persistent", "project"),
        "staging-host": ("host", "bringup", "project"),
        "staging-pinned": ("pinned", "bringup", "project"),
        "runtime": ("device", "bringup", "runtime_overhead"),
    }
    observed: dict[str, dict[str, Any]] = {}
    for allocation in cast(list[Any], allocations):
        require(type(allocation) is dict, "malformed ledger allocation")
        allocation = cast(dict[str, Any], allocation)
        allocation_id = allocation.get("id")
        require(type(allocation_id) is str and allocation_id in expected_allocations
                and allocation_id not in observed, "unexpected or duplicate ledger allocation ID")
        observed[allocation_id] = allocation
        require((allocation.get("location"), allocation.get("lifetime"), allocation.get("source"))
                == expected_allocations[allocation_id], "allocation location/lifetime/source mismatch")
        require(allocation.get("measured") is True, "allocation is not measured")
        cap, current, peak = (allocation.get("cap_bytes"), allocation.get("live_bytes"),
                              allocation.get("measured_peak_bytes"))
        require(type(cap) is int and cap >= 0, "invalid allocation cap")
        cap_value = cast(int, cap)
        require(type(current) is int and current >= 0, "invalid allocation live bytes")
        require(cast(int, current) <= cap_value, "allocation live bytes exceed cap")
        require(type(peak) is int and peak >= 0, "invalid allocation measured peak")
        require(cast(int, current) <= cast(int, peak), "allocation live bytes exceed measured peak")
        require(cast(int, peak) <= cap_value, "allocation measured peak exceeds cap")
    require(set(observed) == set(expected_allocations), "missing required ledger allocation")
    expected_caps = {"weights": 3072, "input": 3072, "output": 6144,
                     "staging-host": 3072, "staging-pinned": 4096, "runtime": 1 << 30}
    for allocation_id, expected_cap in expected_caps.items():
        require(observed[allocation_id]["cap_bytes"] == expected_cap,
                f"allocation {allocation_id} cap does not match its declared reservation")
    expected_peaks = {"weights": 3072, "input": 3072, "output": 6144,
                      "staging-host": 3072, "staging-pinned": 4096}
    for allocation_id, expected_peak in expected_peaks.items():
        allocation = observed[allocation_id]
        require(allocation["measured_peak_bytes"] == expected_peak,
                f"allocation {allocation_id} measured peak does not match its padded request")
        if allocation_id in ("weights", "input", "output"):
            require(allocation["live_bytes"] == expected_peak,
                    "persistent allocation must remain live at its exact padded size")
        else:
            require(allocation["live_bytes"] == 0,
                    "staging allocation must be released after its measured peak")
    runtime = observed["runtime"]
    require(runtime["live_bytes"] == runtime["measured_peak_bytes"],
            "runtime overhead live bytes must match its observed peak")
    totals = {location: sum(item["live_bytes"] for item in observed.values()
                            if item["location"] == location) for location in ("host", "pinned", "device")}
    require(live == totals, "domain live-byte totals do not match allocation records")


SESSION_CHECKS = {
    "startup_selected_real_registered_bf16_decode_kernel",
    "independent_scalar_reference_matches_all_logits",
    "decode_returns_owned_bounded_logits",
    "caller_lease_retained_until_pending_consumer_released",
    "warmup_callback_failure_after_submit_drains_and_releases",
    "decode_callback_failure_after_submit_drains_without_result",
    "wrong_thread_rejected_before_work",
    "post_submit_cancellation_physically_fenced",
    "unselected_preparation_rejected_before_cuda_launch",
    "binder_failure_before_submit",
    "repeated_startup_after_healthy_teardown",
}


def validate_session_report(report: Any, expected_iterations: int) -> None:
    require(type(report) is dict, "session runtime JSON must be an object")
    report = cast(dict[str, Any], report)
    require(report.get("status") == "physical_session_test_passed", "physical session test did not pass")
    require(report.get("qualification") == "not_claimed", "session qualification must remain not_claimed")
    require(report.get("serving_readiness") is False and report.get("readiness_approval") is False,
            "session test must not claim serving readiness or approval")
    require(report.get("ledger_reconciliation") == "passed", "session ledger reconciliation did not pass")
    require(report.get("all_checks_passed") is True, "not all physical session checks passed")
    require(report.get("device_loss_injection") == "not_run",
            "device-loss injection must remain explicitly not_run")
    require(report.get("projection_scope") == "synthetic_fixture_only"
            and report.get("tuning_claim") is False,
            "session report must identify fixture scope and make no tuning claim")
    checks = report.get("checks")
    expected_checks = report.get("expected_checks")
    require(type(checks) is list and type(expected_checks) is list,
            "physical session check sets are missing")
    require(all(type(item) is str for item in checks + expected_checks), "malformed physical session check set")
    require(set(checks) == SESSION_CHECKS and len(checks) == len(SESSION_CHECKS)
            and set(expected_checks) == SESSION_CHECKS and len(expected_checks) == len(SESSION_CHECKS),
            "physical session check set does not match required exercised coverage")
    require(type(report.get("iterations")) is int and report["iterations"] == expected_iterations,
            "session iteration count does not match the request")
    device = report.get("device")
    require(type(device) is dict, "session device identity is missing")
    device = cast(dict[str, Any], device)
    require(device.get("name") in ("NVIDIA GeForce RTX 4080", "GeForce RTX 4080"),
            "session device is not an original GeForce RTX 4080")
    require(device.get("architecture") == "sm89" and type(device.get("architecture_version")) is int
            and device["architecture_version"] == 89, "session device architecture is not SM89")
    identity = device.get("identity")
    require(type(identity) is str and len(identity) == 32
            and all(character in "0123456789abcdef" for character in identity),
            "session device identity must be a 32-character lowercase hexadecimal UUID")
    for key in ("ordinal", "driver", "runtime"):
        require(type(device.get(key)) is int and device[key] >= (0 if key == "ordinal" else 1),
                f"invalid session device {key}")
    ledger = report.get("ledger")
    require(type(ledger) is dict, "session ledger report is missing")
    ledger = cast(dict[str, Any], ledger)
    require(type(ledger.get("profile")) is str and bool(ledger["profile"]),
            "session ledger profile is missing")
    totals = ledger.get("live_bytes")
    require(type(totals) is dict and set(totals) == {"host", "pinned", "device"}
            and all(type(value) is int and value >= 0 for value in cast(dict[str, Any], totals).values()),
            "invalid session domain live-byte totals")
    allocations = ledger.get("allocations")
    require(type(allocations) is list, "session allocation report is missing")
    expected_meta: dict[str, tuple[str, str, str, int]] = {
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
    observed: dict[str, dict[str, Any]] = {}
    for value in allocations:
        require(type(value) is dict, "malformed session allocation report")
        allocation = cast(dict[str, Any], value)
        allocation_id = allocation.get("id")
        require(type(allocation_id) is str and allocation_id in expected_meta
                and allocation_id not in observed, "unexpected or duplicate session allocation ID")
        observed[allocation_id] = allocation
        location, lifetime, source, expected_cap = expected_meta[allocation_id]
        require((allocation.get("location"), allocation.get("lifetime"), allocation.get("source"))
                == (location, lifetime, source), "session allocation metadata mismatch")
        require(allocation.get("measured") is True, "session allocation is not measured")
        cap, current, peak = (allocation.get("cap_bytes"), allocation.get("live_bytes"),
                              allocation.get("measured_peak_bytes"))
        require(type(cap) is int and cap == expected_cap,
                f"session allocation {allocation_id} cap mismatch")
        require(type(current) is int and current >= 0 and type(peak) is int and peak >= 0
                and current <= peak <= cap, "invalid session allocation live/peak/cap bounds")
    require(set(observed) == set(expected_meta), "session allocation set is incomplete")
    for allocation_id, (_, lifetime, _, cap) in expected_meta.items():
        allocation = observed[allocation_id]
        if lifetime == "persistent" and cap:
            require(allocation["live_bytes"] == cap and allocation["measured_peak_bytes"] == cap,
                    f"persistent session allocation {allocation_id} is not live at its exact reservation")
        elif allocation_id in ("staging-host", "staging-pinned", "warmup-scratch"):
            require(allocation["live_bytes"] == 0 and allocation["measured_peak_bytes"] == cap,
                    f"bringup allocation {allocation_id} was not released after its exact peak")
    runtime = observed["runtime"]
    require(runtime["live_bytes"] == runtime["measured_peak_bytes"],
            "session runtime live bytes must equal the observed overhead")
    domain_totals = {location: sum(item["live_bytes"] for item in observed.values()
                                   if item["location"] == location)
                     for location in ("host", "pinned", "device")}
    require(totals == domain_totals, "session domain live-byte totals do not match allocation records")


def validate_same_device(projection_report: Any, session_report: Any) -> None:
    require(type(projection_report) is dict and type(session_report) is dict,
            "combined evidence reports must both be objects")
    projection_device = cast(dict[str, Any], projection_report).get("device")
    session_device = cast(dict[str, Any], session_report).get("device")
    require(type(projection_device) is dict and type(session_device) is dict,
            "combined evidence device identity is missing")
    projection_device = cast(dict[str, Any], projection_device)
    session_device = cast(dict[str, Any], session_device)
    for key in ("identity", "ordinal", "driver", "runtime"):
        require(projection_device.get(key) == session_device.get(key),
                f"projection and session evidence differ in device {key}")


def safe_evidence_root(source: Path, build_root: Path) -> None:
    try:
        relative = build_root.relative_to(source)
    except ValueError:
        return
    if not relative.parts or relative.parts[0] != "build":
        raise ValueError("evidence must be outside source or under source/build")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--max-log-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    if sys.platform not in ("win32", "linux"):
        print("FAIL: native Windows or Linux required", file=sys.stderr)
        return 1
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        print("FAIL: native x86_64/AMD64 required for CUDA session evidence", file=sys.stderr)
        return 1
    if args.iterations < 1 or args.iterations > 10000 or args.timeout_seconds < 1 or args.max_log_bytes < 1:
        parser.error("iterations must be in 1..10000; timeout and max-log-bytes must be positive")
    windows = sys.platform == "win32"
    source = args.source_root.resolve()
    preset = "cuda-windows" if windows else "cuda-linux"
    base = (args.build_root or source / "build/m2-cuda-evidence").resolve()
    try:
        safe_evidence_root(source, base)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = base / f"{stamp}-{uuid.uuid4().hex[:10]}"
    binary = run_root / "binary"
    binary.mkdir(parents=True, exist_ok=False)
    data: dict[str, Any] = {
        "schema_version": 1, "kind": "m2-native-cuda-projection-and-session-device-tests",
        "qualification": False, "readiness_approval": False, "source_root": str(source),
        "run_root": str(run_root), "binary_root": str(binary), "preset": preset,
        "iterations": args.iterations, "timeout_seconds": args.timeout_seconds,
        "max_log_bytes_per_stream": args.max_log_bytes,
        "os": platform.platform(), "architecture": platform.machine(), "kernel": platform.release(),
        "commands": [], "result": "FAIL",
    }
    evidence_file = run_root / "evidence.json"
    log = run_root / "m2-cuda.log"
    def save() -> None:
        evidence_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    def command(label: str, argv: list[str], cwd: Path = source,
                stdout_path: Path | None = None, stderr_path: Path | None = None) -> int:
        index = len(data["commands"])
        stdout = stdout_path or run_root / f"{index:02d}-{label}.stdout.log"
        stderr = stderr_path or run_root / f"{index:02d}-{label}.stderr.log"
        record = {"label": label, "argv": argv, "cwd": str(cwd), "timeout_seconds": args.timeout_seconds,
                  "stdout_log": str(stdout), "stderr_log": str(stderr), "exit_code": None}
        data["commands"].append(record)
        save()
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {label} ===\nARGV: {json.dumps(argv)}\n")
        try:
            result = run_bounded(argv, cwd, args.timeout_seconds, stdout, stderr, windows,
                                 args.max_log_bytes)
        except CommandTimeout as exc:
            record["timeout"] = True
            record["descendant_cleanup_resolved"] = exc.cleanup_resolved
            save()
            raise RuntimeError(f"{label} timed out; logs retained") from exc
        except LogLimitExceeded as exc:
            record["log_limit_exceeded"] = True
            record["descendant_cleanup_resolved"] = exc.cleanup_resolved
            save()
            raise RuntimeError(f"{label} log_limit_exceeded; output capped and logs retained") from exc
        except TreeCleanupUnresolved as exc:
            record["descendant_cleanup_resolved"] = False
            save()
            raise RuntimeError(f"{label}: {exc}") from exc
        record["exit_code"] = result
        with log.open("a", encoding="utf-8", errors="replace") as stream:
            stream.write(stdout.read_text(encoding="utf-8", errors="replace"))
            stream.write(stderr.read_text(encoding="utf-8", errors="replace"))
            stream.write(f"EXIT_CODE: {result}\n")
        save()
        if result:
            raise RuntimeError(f"{label} failed with exit code {result}")
        return result
    before = run_root / "source-before.json"
    after = run_root / "source-after.json"
    compare_attempted = False
    try:
        tools = {name: shutil.which(name) for name in ("cmake", "ninja", "nvcc")}
        if any(value is None for value in tools.values()):
            raise RuntimeError("cmake, ninja, and nvcc must be available on PATH")
        tools = cast(dict[str, str], tools)
        data["tools"] = tools
        python = sys.executable
        evidence_tool = source / "tools/m1_evidence.py"
        command("source-before", [python, "-B", str(evidence_tool), "snapshot", str(source), str(before)])
        for name in ("cmake", "ninja", "nvcc"):
            command("version-" + name, [tools[name], "--version"])
        if windows and not shutil.which("cl"):
            raise RuntimeError("MSVC cl.exe is unavailable; use an x64 Visual Studio developer shell")
        configure = [tools["cmake"], "-S", str(source), "-B", str(binary), "--preset", preset,
                     "-DSAGA_BUILD_CUDA_TESTS=ON", "-DSAGA_BUILD_TESTS=OFF", "-DSAGA_ENABLE_CUDA=ON"]
        command("configure", configure)
        cache_file = binary / "CMakeCache.txt"
        if not cache_file.is_file():
            raise RuntimeError("configure succeeded but CMakeCache.txt is missing")
        cache_text = cache_file.read_text(encoding="utf-8", errors="replace")
        shutil.copy2(cache_file, run_root / "CMakeCache.txt")
        for key, expected in (("SAGA_BUILD_CUDA_TESTS", "ON"), ("SAGA_BUILD_TESTS", "OFF"),
                              ("SAGA_ENABLE_CUDA", "ON"), ("CMAKE_CUDA_ARCHITECTURES", "89")):
            if not any(line.startswith(key + ":") and line.split("=", 1)[-1] == expected
                       for line in cache_text.splitlines()):
                raise RuntimeError(f"effective CMake cache violates {key}={expected}")
        facts = list((binary / "CMakeFiles").glob("*/CMakeCUDACompiler.cmake"))
        facts += list((binary / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
        if len(facts) < 2:
            raise RuntimeError("generated CMake C++/CUDA compiler facts missing")
        facts_dir = run_root / "compiler-facts"
        facts_dir.mkdir()
        for fact in facts:
            shutil.copy2(fact, facts_dir / fact.name)
        build = [tools["cmake"], "--build", str(binary), "--target",
                 "saga_cuda_projection_test", "saga_cuda_session_test_device",
                 "--verbose", "--parallel", "2"]
        command("verbose-build", build)
        suffix = ".exe" if windows else ""
        projection_executable = binary / f"saga_cuda_projection_test{suffix}"
        session_executable = binary / f"saga_cuda_session_test_device{suffix}"
        for label, executable in (("projection", projection_executable), ("session", session_executable)):
            if not executable.is_file():
                raise RuntimeError(f"CUDA {label} test executable missing: {executable}")
        data["binaries"] = {}
        for label, executable in (("projection", projection_executable), ("session", session_executable)):
            digest = hashlib.sha256()
            with executable.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            data["binaries"][label] = {"path": str(executable), "size_bytes": executable.stat().st_size,
                                       "sha256": digest.hexdigest()}
        runtime_json = run_root / "runtime.json"
        runtime_stderr = run_root / "runtime.stderr.log"
        command("runtime-projection", [str(projection_executable), str(args.iterations)],
                source, runtime_json, runtime_stderr)
        report = json.loads(runtime_json.read_text(encoding="utf-8"))
        validate_report(report, args.iterations)
        data["projection_runtime_validated"] = True
        session_json = run_root / "session-runtime.json"
        session_stderr = run_root / "session-runtime.stderr.log"
        command("runtime-session", [str(session_executable), str(args.iterations)],
                source, session_json, session_stderr)
        session_report = json.loads(session_json.read_text(encoding="utf-8"))
        validate_session_report(session_report, args.iterations)
        validate_same_device(report, session_report)
        data["session_runtime_validated"] = True
        data["combined_evidence_scope"] = "same-original-rtx4080-projection-and-synthetic-session-lifecycle"
        command("source-after", [python, "-B", str(evidence_tool), "snapshot", str(source), str(after)])
        command("source-compare", [python, "-B", str(evidence_tool), "compare", str(before), str(after)])
        compare_attempted = True
        data["result"] = "PASS"
        data["scope_note"] = ("Physical CUDA projection and synthetic BF16 session lifecycle tests on the same "
                               "GPU UUID, ordinal, driver, and runtime; no serving readiness or product "
                               "qualification claimed. Device-loss injection was not_run.")
    except Exception as exc:
        data["failure"] = str(exc)
    finally:
        if before.is_file() and not after.exists():
            try:
                command("source-after", [sys.executable, "-B", str(source / "tools/m1_evidence.py"),
                                          "snapshot", str(source), str(after)])
                command("source-compare", [sys.executable, "-B", str(source / "tools/m1_evidence.py"),
                                            "compare", str(before), str(after)])
                compare_attempted = True
            except Exception as exc:
                data["snapshot_failure"] = str(exc)
                data.setdefault("failure", str(exc))
        if before.is_file() and not compare_attempted:
            data["result"] = "FAIL"
        if data.get("failure"):
            data["result"] = "FAIL"
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"RESULT: FAIL\nERROR: {data['failure']}\n")
        save()
    print(f"{data['result']}: evidence at {evidence_file}")
    return 0 if data["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
