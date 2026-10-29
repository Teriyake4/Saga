"""Small, local evidence helpers for M1 Windows/CUDA verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOTS = ("include", "src", "apps", "tests", "cmake", "tools", "third_party")
FILES = ("CMakeLists.txt", "CMakePresets.json", "config/server.example.toml", "config/model.example.ini")
SCOPE = list(ROOTS) + list(FILES)
SKIP_DIRS = {"build", ".git", "__pycache__", ".pytest_cache", "node_modules", ".cache"}
SKIP_SUFFIXES = {".weights", ".gguf", ".safetensors", ".pem", ".key", ".p12", ".pfx"}
SENSITIVE_NAMES = {".env", ".env.local", "credentials", "credentials.json", "secrets.json"}
MAX_FILE_SIZE = 32 * 1024 * 1024


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"symlink is outside the evidence scope: {path}")


def snapshot(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    files: dict[str, Path] = {}
    for relative in FILES:
        path = root
        for component in Path(relative).parts:
            path = path / component
            _reject_symlink(path)
        if path.is_file():
            files[relative] = path
    for name in ROOTS:
        base = root / name
        _reject_symlink(base)
        if not base.is_dir():
            continue
        for current, dirs, names in os.walk(base, followlinks=False):
            current_path = Path(current)
            for dirname in dirs:
                child = current_path / dirname
                _reject_symlink(child)
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for filename in sorted(names):
                path = current_path / filename
                _reject_symlink(path)
                lowered = filename.lower()
                if (lowered in SENSITIVE_NAMES or lowered.startswith((".env", "credential", "secret"))
                        or lowered in {"id_rsa", "id_ed25519"} or path.suffix.lower() in SKIP_SUFFIXES):
                    continue
                if not path.is_file():
                    continue
                size = path.stat().st_size
                if size > MAX_FILE_SIZE:
                    continue
                files[path.relative_to(root).as_posix()] = path
    entries = []
    for key in sorted(files):
        content = files[key].read_bytes()
        entries.append({"path": key, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    try:
        revision = subprocess.run(["git", "--no-pager", "--no-optional-locks", "-C", str(root),
                                   "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        dirty = subprocess.run(["git", "--no-pager", "--no-optional-locks", "-C", str(root),
                                "status", "--porcelain", "--untracked-files=all"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"schema_version": 1, "scope": SCOPE, "revision": revision, "dirty": dirty, "files": entries}


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _validate_snapshot(data: Any) -> dict[str, Any]:
    if type(data) is not dict or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("invalid source snapshot schema")
    if data.get("scope") != SCOPE or type(data.get("files")) is not list:
        raise ValueError("invalid source snapshot scope or file list")
    seen: set[str] = set()
    for entry in data["files"]:
        if type(entry) is not dict:
            raise ValueError("invalid source snapshot entry")
        path, size, digest = entry.get("path"), entry.get("size"), entry.get("sha256")
        if type(path) is not str or path in seen or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("invalid or duplicate source snapshot path")
        if not any(path == f or path.startswith(f + "/") for f in ROOTS) and path not in FILES:
            raise ValueError("source snapshot path is outside declared scope")
        if not _integer(size) or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("invalid source snapshot size or digest")
        seen.add(path)
    if data.get("revision") is not None and type(data["revision"]) is not str:
        raise ValueError("invalid source snapshot revision")
    if data.get("dirty") is not None and type(data["dirty"]) is not bool:
        raise ValueError("invalid source snapshot dirty indicator")
    return data


def validate_cuda(data: Any, device: int) -> None:
    expected_checks = {
        "arguments", "driver_runtime_versions", "device_selection", "device_properties", "original_rtx_4080",
        "compute_capability_sm89", "sm_count_76", "memory_query", "cooperative_support_query",
        "cp_async_resources", "bf16_wmma_resources", "resource_initialization", "allocations_alignment",
        "cp_async_launch_and_sync", "cp_async_result", "bf16_wmma_launch_and_sync", "bf16_wmma_result",
        "resource_cleanup", "allocation_cleanup",
    }
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)
    require(_integer(device), "requested device ordinal must be nonnegative")
    require(type(data) is dict, "invalid CUDA report schema")
    data = cast(dict[str, Any], data)
    require(_integer(data.get("schema_version")) and data["schema_version"] == 1, "invalid CUDA report schema")
    require(data.get("probe") == "saga_cuda_portability_smoke", "unexpected probe")
    require(data.get("passed") is True and data.get("error") == "", "probe did not pass")
    toolchain = data.get("toolchain")
    require(type(toolchain) is dict and toolchain.get("os") == "Windows", "expected Windows toolchain")
    require(_integer(toolchain.get("nvcc_major"), 1), "invalid toolchain nvcc_major")
    for key in ("nvcc_minor", "nvcc_build"):
        require(_integer(toolchain.get(key)), f"invalid toolchain {key}")
    require((toolchain["nvcc_major"], toolchain["nvcc_minor"]) >= (12, 8), "CUDA toolkit must be at least 12.8")
    for key in ("cuda_headers_version", "cplusplus"):
        require(_integer(toolchain.get(key), 1), f"invalid toolchain {key}")
    require(toolchain.get("host_compiler") == "MSVC", "expected MSVC host compiler")
    require(_integer(toolchain.get("host_compiler_version"), 1), "invalid host compiler version")
    require(_integer(data.get("driver_version"), 1) and _integer(data.get("runtime_version"), 1),
            "invalid driver/runtime version")
    require(_integer(data.get("device_ordinal")) and data["device_ordinal"] == device,
            "device ordinal mismatch")
    require(_integer(data.get("device_count"), 1) and device < data["device_count"], "device ordinal unavailable")
    gpu = data.get("device")
    require(type(gpu) is dict, "missing device properties")
    gpu = cast(dict[str, Any], gpu)
    require(gpu.get("name") in ("NVIDIA GeForce RTX 4080", "GeForce RTX 4080"), "not original RTX 4080")
    for key, expected in (("compute_major", 8), ("compute_minor", 9), ("sm_count", 76),
                          ("expected_sm_count", 76), ("warp_size", 32)):
        require(_integer(gpu.get(key)) and gpu[key] == expected, f"device {key} mismatch")
    require(_integer(gpu.get("total_global_mem_bytes"), 1), "invalid device memory capacity")
    require(type(gpu.get("cooperative_launch_property")) is bool, "invalid cooperative launch property")
    cooperative = gpu.get("cooperative_launch_attribute")
    require(type(cooperative) is int and cooperative in (0, 1), "invalid cooperative launch attribute")
    require(gpu["cooperative_launch_property"] is (cooperative == 1), "cooperative support facts disagree")
    require(data.get("cooperative_launch_exercised") is False, "unexpected cooperative launch claim")
    require(_integer(data.get("expected_compile_arch")) and data["expected_compile_arch"] == 890,
            "expected compile architecture mismatch")
    for key in ("cp_async_compiled_arch", "bf16_wmma_compiled_arch"):
        require(_integer(data.get(key)) and data[key] == 890, f"{key} must report __CUDA_ARCH__ 890")
    memory = data.get("memory_before_allocations")
    require(type(memory) is dict, "missing pre-allocation memory facts")
    memory = cast(dict[str, Any], memory)
    free_bytes, total_bytes = memory.get("free_bytes"), memory.get("total_bytes")
    require(_integer(free_bytes) and _integer(total_bytes, 1), "invalid pre-allocation memory facts")
    free_bytes, total_bytes = cast(int, free_bytes), cast(int, total_bytes)
    require(free_bytes <= total_bytes, "invalid pre-allocation memory facts")
    kernels = data.get("kernels")
    require(type(kernels) is list and len(kernels) == 2, "expected exactly two kernel resource records")
    kernels = cast(list[Any], kernels)
    expected_kernels = {"cp_async_smoke", "bf16_wmma_smoke"}
    kernel_names: list[Any] = []
    for kernel in kernels:
        require(type(kernel) is dict, "invalid kernel resource record")
        kernel = cast(dict[str, Any], kernel)
        name = kernel.get("name")
        require(type(name) is str, "invalid kernel name")
        kernel_names.append(name)
        require(kernel.get("queried") is True, f"kernel resources were not queried: {name}")
        require(_integer(kernel.get("binary_version")) and kernel["binary_version"] == 89,
                f"kernel binary is not SM89: {name}")
        for key in ("ptx_version", "registers_per_thread", "static_shared_bytes", "local_bytes_per_thread"):
            require(_integer(kernel.get(key)), f"invalid {key}: {name}")
        require(_integer(kernel.get("max_threads_per_block"), 32), f"invalid max thread count: {name}")
        require(_integer(kernel.get("launch_blocks")) and kernel["launch_blocks"] == 1
                and _integer(kernel.get("threads_per_block")) and kernel["threads_per_block"] == 32
                and _integer(kernel.get("dynamic_shared_bytes")) and kernel["dynamic_shared_bytes"] == 0,
                f"invalid queried launch geometry: {name}")
        active = kernel.get("active_blocks_per_sm")
        grid_bound = kernel.get("resident_grid_block_bound")
        require(_integer(active, 1) and _integer(grid_bound, 1), f"invalid occupancy/resource bound: {name}")
        active, grid_bound = cast(int, active), cast(int, grid_bound)
        require(grid_bound == active * 76, f"invalid occupancy/resource bound: {name}")
        require(_integer(kernel.get("resident_warps_per_sm")) and kernel["resident_warps_per_sm"] == active,
                f"invalid resident warp count: {name}")
        coop_bound = kernel.get("cooperative_grid_block_bound")
        if cooperative == 1:
            require(_integer(coop_bound, 1) and coop_bound == grid_bound,
                    f"invalid cooperative grid bound: {name}")
        else:
            require(coop_bound is None, f"invalid cooperative grid bound: {name}")
    require(len(kernel_names) == len(set(kernel_names)) and set(kernel_names) == expected_kernels,
            "missing, duplicate, or unknown kernel resource records")
    checks = data.get("checks")
    require(type(checks) is list, "checks must be an array")
    checks = cast(list[Any], checks)
    names: list[Any] = []
    for check in checks:
        require(type(check) is dict, "malformed check record")
        check = cast(dict[str, Any], check)
        name, status = check.get("name"), check.get("status")
        require(type(name) is str and type(status) is str, "check name/status must be strings")
        names.append(name)
        require(status == "pass", "a required check did not pass")
    require(len(names) == len(set(names)), "duplicate checks")
    require(set(names) == expected_checks, "missing or unknown checks")


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    snap = commands.add_parser("snapshot"); snap.add_argument("root"); snap.add_argument("output")
    compare = commands.add_parser("compare"); compare.add_argument("before"); compare.add_argument("after")
    cuda = commands.add_parser("validate-cuda"); cuda.add_argument("report"); cuda.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "snapshot":
            Path(args.output).write_text(json.dumps(snapshot(args.root), indent=2) + "\n", encoding="utf-8")
        elif args.command == "compare":
            before = _validate_snapshot(json.loads(Path(args.before).read_text(encoding="utf-8")))
            after = _validate_snapshot(json.loads(Path(args.after).read_text(encoding="utf-8")))
            if before["files"] != after["files"]:
                raise ValueError("relevant source files changed during build")
        else:
            validate_cuda(json.loads(Path(args.report).read_text(encoding="utf-8")), args.device)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError) as exc:
        print(f"m1_evidence: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
