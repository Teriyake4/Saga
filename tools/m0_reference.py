"""Reproducible, bounded harness for the pinned NInfer-4080 M0 reference.

Commands are argv arrays passed directly to subprocess; no shell is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

REVISION = "2469b5752672ad597055af61fe28e8e7ced20056"
REPOSITORY = "yalding/ninfer-4080"
MAX_LOG_BYTES = 1_000_000
MAX_TRACE_BYTES = 10_000_000
MAX_TRACE_LINE_BYTES = 65_536
TRACE_PREFIX = b"M0DISPATCH "
STREAM_END = object()


class HarnessError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contains_symlink(root: Path) -> bool:
    return root.is_symlink() or any(path.is_symlink() for path in root.rglob("*"))


def source_digest(root: Path) -> str:
    """Hash regular source-tree files; symlinks are prohibited to prevent escapes."""
    if _contains_symlink(root):
        raise HarnessError("source tree contains symlinks")
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _required_pairs(value: Any) -> set[tuple[str, str]]:
    if not isinstance(value, list) or not value:
        raise HarnessError("required_site_phases must be a nonempty reviewed array")
    pairs: set[tuple[str, str]] = set()
    for pair in value:
        if (not isinstance(pair, dict) or set(pair) != {"site", "phase"} or
                not all(isinstance(pair[key], str) and pair[key].strip()
                        for key in ("site", "phase"))):
            raise HarnessError("each required site/phase entry must contain nonempty strings")
        item = (pair["site"], pair["phase"])
        if item in pairs:
            raise HarnessError(f"duplicate required site/phase entry: {item}")
        pairs.add(item)
    return pairs


def validate_config(config: Any, output: Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise HarnessError("recipe must be a JSON object")
    allowed = {"repository", "revision", "source_root", "source_sha256", "source_review",
               "trace_inventory_review", "mode", "build_dir", "commands", "binary", "binary_sha256",
               "prebuilt_provenance", "patches", "environment", "required_site_phases"}
    if set(config) - allowed:
        raise HarnessError("recipe contains unsupported fields; trace files are accepted only from run stdout")
    if config.get("revision") != REVISION or config.get("repository") != REPOSITORY:
        raise HarnessError("recipe must identify the exact pinned repository and revision")
    source_pin = config.get("source_sha256")
    if not isinstance(source_pin, str) or len(source_pin) != 64 or any(c not in "0123456789abcdef" for c in source_pin):
        raise HarnessError("an explicit reviewed lowercase source_sha256 is required")
    for field in ("source_review", "trace_inventory_review"):
        review = config.get(field)
        if (not isinstance(review, dict) or set(review) != {"reviewer", "reviewed_at", "basis"} or
                not all(isinstance(review[k], str) and review[k].strip()
                        for k in ("reviewer", "reviewed_at", "basis"))):
            raise HarnessError(f"{field} must record reviewer, reviewed_at, and basis")
    mode = config.get("mode")
    if mode not in ("build", "reviewed-prebuilt"):
        raise HarnessError("mode must be build or reviewed-prebuilt")
    required = _required_pairs(config.get("required_site_phases"))
    root_input = Path(config.get("source_root", ""))
    if not str(config.get("source_root", "")):
        raise HarnessError("source_root is required")
    if root_input.is_symlink():
        raise HarnessError("source_root itself must not be a symlink")
    root = root_input.resolve()
    if not root.is_dir():
        raise HarnessError(f"source_root is not a directory: {root}")
    if _contains_symlink(root):
        raise HarnessError("source tree contains symlinks")
    output_abs = output.absolute()
    if _is_within(output_abs.resolve(strict=False), root):
        raise HarnessError("evidence output must be outside the source tree")
    build_name = config.get("build_dir")
    if mode == "build" and (not isinstance(build_name, str) or not build_name):
        raise HarnessError("build mode requires an explicit external build_dir")
    build_dir = Path(build_name).resolve() if build_name else None
    if build_dir is not None and _is_within(build_dir, root):
        raise HarnessError("build_dir must be outside the source tree")
    commands = config.get("commands")
    if not isinstance(commands, list) or not commands:
        raise HarnessError("commands must be a nonempty array")
    roles: list[str] = []
    normalized_commands = []
    for item in commands:
        if not isinstance(item, dict) or set(item) - {"role", "argv", "cwd", "timeout_seconds"}:
            raise HarnessError("each command requires only role, argv, cwd, timeout_seconds")
        role = item.get("role")
        argv = item.get("argv")
        if role not in ("configure", "build", "run"):
            raise HarnessError("command role must be configure, build, or run")
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise HarnessError("argv must be a nonempty array of nonempty strings")
        timeout = item.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise HarnessError("timeout_seconds must be a finite positive number")
        cwd = Path(item.get("cwd", str(root))).resolve()
        if not cwd.is_dir():
            raise HarnessError(f"command cwd does not exist: {cwd}")
        roles.append(role)
        normalized_commands.append({"role": role, "argv": argv, "cwd": cwd,
                                    "timeout_seconds": float(timeout)})
    if roles.count("run") != 1 or roles[-1] != "run":
        raise HarnessError("exactly one run command is required and it must be last")
    if mode == "build":
        if roles[0] != "configure" or "build" not in roles[1:-1]:
            raise HarnessError("build mode requires configure then build stages before run")
    elif roles != ["run"]:
        raise HarnessError("reviewed-prebuilt mode accepts only a run command")
    for item in normalized_commands:
        argv = item["argv"]
        build_args = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "-B"]
        build_args.extend(arg[2:] for arg in argv if arg.startswith("-B") and arg != "-B")
        if item["role"] == "build":
            if not Path(argv[0]).name.lower().startswith("cmake") or argv.count("--build") != 1:
                raise HarnessError("build stage must invoke CMake --build exactly once")
            build_index = argv.index("--build")
            if build_dir is None or build_index + 1 >= len(argv) or (item["cwd"] / argv[build_index + 1]).resolve() != build_dir:
                raise HarnessError("CMake --build must use the declared external build_dir")
        if item["role"] == "configure" and (not Path(argv[0]).name.lower().startswith("cmake") or "-S" not in argv):
            raise HarnessError("configure stage must invoke CMake with -S and -B")
        if item["role"] == "configure" and Path(argv[0]).name.lower().startswith("cmake") and "-S" in argv:
            if argv.count("-S") != 1 or len(build_args) != 1:
                raise HarnessError("CMake configure stage must declare exactly one -S and -B")
            source_index = argv.index("-S")
            source_arg = (item["cwd"] / argv[source_index + 1]).resolve() if source_index + 1 < len(argv) else None
            if source_arg != root:
                raise HarnessError("CMake -S must identify the pinned source root")
        if item["role"] == "configure" and Path(argv[0]).name.lower().startswith("cmake"):
            if not build_args:
                raise HarnessError("CMake configure stage must declare -B")
            for build_arg in build_args:
                actual_build = (item["cwd"] / build_arg).resolve()
                if _is_within(actual_build, root) or actual_build != build_dir:
                    raise HarnessError("CMake -B must use the declared external build_dir")
    binary_name = config.get("binary")
    binary_hash = config.get("binary_sha256")
    if not isinstance(binary_name, str) or not binary_name:
        raise HarnessError("binary path is required")
    if mode == "reviewed-prebuilt":
        if not isinstance(binary_hash, str) or len(binary_hash) != 64 or any(c not in "0123456789abcdef" for c in binary_hash):
            raise HarnessError("reviewed-prebuilt requires an explicit expected binary_sha256")
    elif binary_hash is not None and (not isinstance(binary_hash, str) or len(binary_hash) != 64 or any(c not in "0123456789abcdef" for c in binary_hash)):
        raise HarnessError("optional build-mode binary_sha256 must be 64 lowercase hex characters")
    binary_input = Path(binary_name)
    if not binary_input.is_absolute():
        raise HarnessError("declared binary path must be absolute")
    binary = binary_input.resolve()
    if _is_within(binary, root):
        raise HarnessError("declared binary must be outside the source tree")
    if mode == "build" and (build_dir is None or not _is_within(binary, build_dir)):
        raise HarnessError("built binary must be inside the declared external build_dir")
    run_argv0_input = Path(normalized_commands[-1]["argv"][0])
    if not run_argv0_input.is_absolute():
        raise HarnessError("run argv[0] must be an absolute executable path")
    run_argv0 = run_argv0_input.resolve()
    if binary != run_argv0:
        raise HarnessError("run argv[0] must resolve to the declared binary")
    prebuilt = config.get("prebuilt_provenance")
    if mode == "reviewed-prebuilt":
        required_prebuilt = {"reviewer", "reviewed_at", "basis", "source_revision", "binary_sha256"}
        if (not isinstance(prebuilt, dict) or set(prebuilt) != required_prebuilt or
                not all(isinstance(prebuilt[key], str) and prebuilt[key].strip() for key in required_prebuilt) or
                prebuilt["source_revision"] != REVISION or prebuilt["binary_sha256"] != binary_hash):
            raise HarnessError("reviewed-prebuilt requires matching reviewed provenance and binary hash")
    elif prebuilt is not None:
        raise HarnessError("prebuilt_provenance is valid only in reviewed-prebuilt mode")
    patches = config.get("patches", [])
    if not isinstance(patches, list) or any(not isinstance(p, str) or not p for p in patches):
        raise HarnessError("patches must be an array of file paths")
    env = config.get("environment", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) for k, v in env.items()):
        raise HarnessError("environment must map nonempty variable names to string values")
    return {"mode": mode, "root": root, "source_sha256": source_pin, "build_dir": build_dir,
            "commands": normalized_commands, "binary": binary, "binary_sha256": binary_hash,
            "patches": [Path(p).resolve() for p in patches], "environment": env,
            "required": required, "source_review": config["source_review"],
            "trace_inventory_review": config["trace_inventory_review"], "prebuilt_provenance": prebuilt}


def _terminate_tree(proc: subprocess.Popen[Any]) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        return
    # Signal the process group even if its leader already exited: children may remain.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        except PermissionError:
            if proc.poll() is None:
                proc.terminate() if sig == signal.SIGTERM else proc.kill()
        if sig == signal.SIGTERM:
            time.sleep(0.15)
    if proc.poll() is None:
        proc.wait()


def _drain_output(stream: Any, events: queue.Queue[Any], stop: threading.Event) -> None:
    try:
        while not stop.is_set():
            chunk = stream.read(4096)
            if not chunk:
                break
            while not stop.is_set():
                try:
                    events.put(chunk, timeout=0.05)
                    break
                except queue.Full:
                    continue
    finally:
        while not stop.is_set():
            try:
                events.put(STREAM_END, timeout=0.05)
                break
            except queue.Full:
                continue


def run_command(command: list[str], cwd: Path, environment: dict[str, str], timeout: float,
                log_path: Path, max_log: int = MAX_LOG_BYTES,
                expected_binary: Path | None = None, expected_binary_hash: str | None = None,
                is_run: bool = False) -> dict[str, Any]:
    start = time.monotonic()
    if is_run:
        if expected_binary is None or expected_binary_hash is None:
            raise HarnessError("run command lacks declared binary identity")
        actual_hash = sha256_file(expected_binary)
        if actual_hash != expected_binary_hash:
            raise HarnessError("declared binary hash mismatch immediately before run")
        if Path(command[0]).resolve() != expected_binary:
            raise HarnessError("run argv[0] does not resolve to declared binary")
    kwargs: dict[str, Any] = {"cwd": cwd, "env": environment, "stdout": subprocess.PIPE,
                              "stderr": subprocess.STDOUT, "bufsize": 0}
    if os.name != "nt":
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(command, **kwargs)
    events: queue.Queue[Any] = queue.Queue(maxsize=8)
    stop = threading.Event()
    reader = threading.Thread(target=_drain_output, args=(proc.stdout, events, stop), daemon=True)
    reader.start()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_hash = hashlib.sha256()
    log_bytes = 0
    trace_bytes = 0
    trace_records: list[dict[str, Any]] = []
    line_buffer = bytearray()
    stream_error: str | None = None
    overflow = False
    timed_out = False
    stream_end = False
    deadline = start + timeout
    with log_path.open("wb") as log:
        while not stream_end:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                item = events.get(timeout=min(0.05, remaining))
            except queue.Empty:
                if proc.poll() is not None and not reader.is_alive():
                    break
                continue
            if item is STREAM_END:
                stream_end = True
                continue
            if log_bytes + len(item) > max_log:
                keep = max(0, max_log - log_bytes)
                chunk = item[:keep]
                overflow = True
            else:
                chunk = item
            if chunk:
                log.write(chunk)
                log_hash.update(chunk)
                log_bytes += len(chunk)
                if is_run:
                    line_buffer.extend(chunk)
                    if len(line_buffer) > MAX_TRACE_LINE_BYTES and b"\n" not in line_buffer:
                        overflow = True
                    while b"\n" in line_buffer:
                        raw, _, rest = line_buffer.partition(b"\n")
                        line_buffer = bytearray(rest)
                        if raw.startswith(TRACE_PREFIX):
                            trace_bytes += len(raw) + 1
                            if trace_bytes > MAX_TRACE_BYTES:
                                overflow = True
                            try:
                                trace_records.append(json.loads(raw[len(TRACE_PREFIX):]))
                            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                                stream_error = f"malformed runtime trace line: {exc}"
            if overflow:
                break
        if is_run and stream_end and line_buffer and line_buffer.startswith(TRACE_PREFIX):
            trace_bytes += len(line_buffer)
            if trace_bytes > MAX_TRACE_BYTES:
                overflow = True
            else:
                try:
                    trace_records.append(json.loads(line_buffer[len(TRACE_PREFIX):]))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    stream_error = f"malformed final runtime trace line: {exc}"
        if timed_out or overflow:
            _terminate_tree(proc)
        else:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_tree(proc)
        stop.set()
        reader.join(timeout=2)
        if proc.stdout is not None:
            proc.stdout.close()
    result = {"argv": command, "cwd": str(cwd), "exit_code": proc.returncode,
              "timed_out": timed_out, "log": str(log_path), "log_sha256": log_hash.hexdigest(),
              "log_bytes": log_bytes, "log_truncated": overflow,
              "elapsed_seconds": round(time.monotonic() - start, 3),
              "trace_records": trace_records if is_run else None,
              "trace_bytes": trace_bytes if is_run else None,
              "trace_error": stream_error if is_run else None}
    if overflow:
        result["failure"] = stream_error or "bounded output limit exceeded; process tree terminated"
    return result


def parse_trace_records(records: list[Any], required: set[tuple[str, str]]) -> dict[str, Any]:
    if not required:
        raise HarnessError("required site/phase inventory is empty")
    seen: set[tuple[str, str]] = set()
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    valid: list[dict[str, Any]] = []
    for number, item in enumerate(records, 1):
        fields = ("site", "phase", "dispatch", "m", "n", "k", "event")
        if not isinstance(item, dict) or any(field not in item for field in fields):
            raise HarnessError(f"incomplete dispatch record {number}")
        event = item["event"]
        if event in ("launch", "graph-record"):
            if item.get("executed") is not True:
                raise HarnessError(f"record is not an explicit executed selection {number}")
        elif event in ("launch-selection", "graph-record-selection"):
            if item.get("selected") is not True:
                raise HarnessError(f"record is not an explicit dispatch selection {number}")
            if not all(isinstance(item.get(key), str) and item[key].strip() for key in ("route", "kernel")):
                raise HarnessError(f"selection record lacks route/kernel identity {number}")
            if type(item.get("record_id")) is not int or item["record_id"] <= 0:
                raise HarnessError(f"selection record lacks a positive record_id {number}")
            if event == "graph-record-selection" and item.get("capture_state") != "active":
                raise HarnessError(f"graph-record selection lacks active capture state {number}")
            if event == "launch-selection" and item.get("capture_state") != "none":
                raise HarnessError(f"launch selection has inconsistent capture state {number}")
        else:
            raise HarnessError(f"unsupported dispatch event {event!r} in record {number}")
        if item["dispatch"] not in ("A16", "A8"):
            raise HarnessError(f"unknown dispatch in record {number}")
        if not all(type(item[key]) is int and item[key] > 0 for key in ("m", "n", "k")):
            raise HarnessError(f"invalid geometry in record {number}")
        if not all(isinstance(item[key], str) and item[key].strip() for key in ("site", "phase")):
            raise HarnessError(f"invalid site/phase in record {number}")
        identity = (("record", item["record_id"]) if "record_id" in item else
                    (item["site"], item["phase"], item["event"], item["m"], item["n"], item["k"]))
        prior = identities.get(identity)
        if prior is not None and prior["dispatch"] != item["dispatch"]:
            raise HarnessError(f"contradictory dispatch for event identity {identity}")
        identities[identity] = item
        seen.add((item["site"], item["phase"]))
        valid.append(item)
    missing = sorted(required - seen)
    if not valid:
        raise HarnessError("no explicit dispatch records; coverage is absent")
    if missing:
        raise HarnessError(f"incomplete dispatch coverage: {missing}")
    return {"records": valid, "covered_site_phases": [list(x) for x in sorted(seen)],
            "required_site_phases": [list(x) for x in sorted(required)],
            "coverage_complete_for_declared_inventory": True,
            "graph_capture_caveat": "graph-record indicates launch recording, not graph replay execution",
            "dispatch_completeness": "not established; instrumentation and complete inventory review required"}


def _load_config(path: Path) -> Any:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read recipe: {exc}") from exc
    return config


def _write_capture(output: Path, data: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    temp = output / "capture.json.tmp"
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(output / "capture.json")


def execute(config_path: Path, output: Path) -> int:
    output = output.absolute()
    if output.exists():
        raise HarnessError("evidence output already exists; refusing to overwrite")
    try:
        config = _load_config(config_path)
    except (HarnessError, OSError, ValueError, TypeError) as exc:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        _write_capture(output, {"format": 1, "repository": REPOSITORY, "revision": REVISION,
                                "status": "incomplete", "failure": f"{type(exc).__name__}: {exc}",
                                "commands": [], "trace": None})
        return 2
    source_name = config.get("source_root") if isinstance(config, dict) else None
    if isinstance(source_name, str) and source_name:
        source = Path(source_name).resolve()
        if _is_within(output.resolve(strict=False), source):
            raise HarnessError("evidence output must be outside the source tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    capture: dict[str, Any] = {"format": 1, "repository": REPOSITORY, "revision": REVISION,
                               "status": "incomplete", "failure": None, "commands": [],
                               "trace": None, "note": "Evidence is not M0 qualification."}
    try:
        validated = validate_config(config, output)
        root = validated["root"]
        actual_source_hash = source_digest(root)
        if actual_source_hash != validated["source_sha256"]:
            raise HarnessError(f"source tree hash mismatch: expected {validated['source_sha256']}, got {actual_source_hash}")
        for patch in validated["patches"]:
            if not patch.is_file():
                raise HarnessError(f"patch file does not exist: {patch}")
        capture.update(mode=validated["mode"], source_root=str(root), source_tree_sha256=actual_source_hash,
                       source_review=validated["source_review"],
                       trace_inventory_review=validated["trace_inventory_review"],
                       environment_overrides=validated["environment"],
                       inherited_environment_names=sorted(os.environ),
                       patches=[{"path": str(p), "sha256": sha256_file(p)} for p in validated["patches"]])
        environment = os.environ.copy()
        environment.update(validated["environment"])
        binary = validated["binary"]
        build_snapshot: dict[str, Any] | None = None
        run_binary_hash = validated["binary_sha256"]
        if validated["mode"] == "build":
            if binary.exists():
                build_snapshot = {"before": {"exists": True, "path": str(binary),
                                              "sha256": sha256_file(binary), "bytes": binary.stat().st_size}}
                capture["build_binary_snapshots"] = build_snapshot
                raise HarnessError("build mode requires the declared artifact to be absent before configure")
            build_snapshot = {"before": {"exists": False, "path": str(binary)}}
            capture["build_binary_snapshots"] = build_snapshot
        else:
            capture["prebuilt_provenance"] = {**validated["prebuilt_provenance"],
                                               "is_review_assertion_not_independent_build_verification": True}
        for index, spec in enumerate(validated["commands"]):
            is_run = spec["role"] == "run"
            if is_run and validated["mode"] == "build":
                if build_snapshot is None:
                    raise HarnessError("build snapshot was not initialized")
                if not binary.is_file():
                    raise HarnessError("build stage did not produce the declared binary")
                run_binary_hash = sha256_file(binary)
                if validated["binary_sha256"] is not None and run_binary_hash != validated["binary_sha256"]:
                    raise HarnessError("built binary hash differs from the optional expected hash")
                build_snapshot["after"] = {"exists": True, "path": str(binary),
                                            "sha256": run_binary_hash, "bytes": binary.stat().st_size}
            log = output / f"command-{index:02d}.log"
            record = run_command(spec["argv"], spec["cwd"], environment,
                                 spec["timeout_seconds"], log,
                                 expected_binary=validated["binary"] if is_run else None,
                                 expected_binary_hash=run_binary_hash if is_run else None,
                                 is_run=is_run)
            capture["commands"].append({key: value for key, value in record.items() if key != "trace_records"})
            if (record["timed_out"] or record["exit_code"] != 0 or record["log_truncated"] or
                    record["trace_error"]):
                raise HarnessError(record["trace_error"] or
                                   f"{spec['role']} command failed, timed out, or exceeded output cap")
            if is_run:
                capture["trace"] = parse_trace_records(record["trace_records"], validated["required"])
                capture["trace"].update(origin="captured-run-stdout", bytes=record["trace_bytes"])
        binary_hash = sha256_file(validated["binary"])
        if binary_hash != run_binary_hash:
            raise HarnessError("binary changed during capture")
        capture["binary"] = {"path": str(validated["binary"]), "sha256": binary_hash,
                              "bytes": validated["binary"].stat().st_size}
        if validated["mode"] == "build":
            capture["build_binary_snapshots"] = build_snapshot
            capture["status"] = "fresh-build-run-trace-validated-incomplete-dispatch"
        else:
            capture["status"] = "prebuilt-run-captured-incomplete-dispatch"
        return 0
    except (HarnessError, OSError, ValueError, KeyError, TypeError,
            subprocess.SubprocessError, OverflowError) as exc:
        capture["failure"] = f"{type(exc).__name__}: {exc}"
        return 2
    finally:
        try:
            _write_capture(output, capture)
        except (OSError, TypeError, ValueError) as exc:
            print(f"m0_reference: could not persist capture metadata: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON argv/build/run recipe")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    args = parser.parse_args()
    try:
        return execute(args.config, args.output)
    except (HarnessError, OSError, ValueError, KeyError) as exc:
        print(f"m0_reference: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
