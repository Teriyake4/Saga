#!/usr/bin/env python3
"""Offline inventory of explicitly supplied local M0 artifacts; never downloads or executes them."""
import argparse
import ctypes
import hashlib
import json
import os
import stat
import struct
import sys
from pathlib import Path, PurePosixPath

CHUNK = 1024 * 1024
MAX_MANIFEST = 4 * 1024 * 1024
MAX_JSON_DEPTH = 128
MAX_HEADER = 64 * 1024 * 1024
MAX_TENSORS = 1_000_000
MAX_RANK = 64
# Byte-aligned standard scalar dtype sizes only. Sub-byte layouts are intentionally unsupported.
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


def enforce_json_depth(raw):
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5c:
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in (0x7b, 0x5b):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("JSON nesting exceeds 128-level limit")
        elif byte in (0x7d, 0x5d) and depth:
            depth -= 1


def no_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _open_beneath(root, name):
    if not isinstance(name, str) or not name or '\\' in name or '\x00' in name or ':' in name:
        raise ValueError("invalid relative path")
    p = PurePosixPath(name)
    if p.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("path must be canonical and relative")
    if os.name == "nt":
        return _open_beneath_windows(Path(root).resolve(strict=True), p.parts)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory or os.open not in getattr(os, "supports_dir_fd", set()):
        raise ValueError("secure no-follow directory traversal is unavailable on this platform")
    root_fd = os.open(Path(root).resolve(strict=True), os.O_RDONLY | directory | nofollow)
    current_fd = root_fd
    try:
        for part in p.parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | directory | nofollow, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(p.parts[-1], os.O_RDONLY | nofollow, dir_fd=current_fd)
        info = os.fstat(file_fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(file_fd)
            raise ValueError("not a regular file")
        return file_fd
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _open_beneath_windows(root, parts):
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    get_info.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [("FileAttributes", wintypes.DWORD), ("ReparseTag", wintypes.DWORD)]

    class FileStandardInfo(ctypes.Structure):
        _fields_ = [("AllocationSize", ctypes.c_longlong), ("EndOfFile", ctypes.c_longlong),
                    ("NumberOfLinks", wintypes.DWORD), ("DeletePending", ctypes.c_ubyte),
                    ("Directory", ctypes.c_ubyte)]

    generic_read = 0x80000000
    share_read = 0x00000001
    open_existing = 3
    backup_semantics = 0x02000000
    open_reparse = 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value
    reparse_attribute = 0x00000400
    handles = []

    def open_handle(path, is_directory):
        handle = create_file(str(path), generic_read, share_read, None, open_existing,
                             backup_semantics | open_reparse, None)
        if handle == invalid_handle:
            error = ctypes.get_last_error()
            raise OSError(error, "CreateFileW failed", str(path))
        handles.append(handle)
        tag = FileAttributeTagInfo()
        if not get_info(handle, 9, ctypes.byref(tag), ctypes.sizeof(tag)):
            error = ctypes.get_last_error()
            raise OSError(error, "GetFileInformationByHandleEx failed", str(path))
        if tag.FileAttributes & reparse_attribute:
            raise ValueError("reparse-point path component rejected")
        standard = FileStandardInfo()
        if not get_info(handle, 1, ctypes.byref(standard), ctypes.sizeof(standard)):
            error = ctypes.get_last_error()
            raise OSError(error, "GetFileInformationByHandleEx failed", str(path))
        if bool(standard.Directory) != is_directory:
            raise ValueError("path component has wrong file type")
        return handle

    try:
        current = Path(root)
        open_handle(current, True)
        for part in parts[:-1]:
            current = current / part
            open_handle(current, True)
        final_handle = open_handle(current / parts[-1], False)
        fd = msvcrt.open_osfhandle(final_handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        handles.remove(final_handle)
        return fd
    finally:
        for handle in reversed(handles):
            close_handle(handle)


def _metadata(fd):
    info = os.fstat(fd)
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _hash_fd(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    count = 0
    while True:
        block = os.read(fd, CHUNK)
        if not block:
            break
        digest.update(block)
        count += len(block)
    return count, digest.hexdigest()


def parse_safetensors_fd(fd, size):
    result = {"status": "invalid", "tensors": [], "error": None}
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        prefix = os.read(fd, 8)
        if len(prefix) != 8:
            raise ValueError("truncated header length")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len < 2 or header_len > MAX_HEADER:
            raise ValueError("header length outside allowed limit")
        data_start = 8 + header_len
        if data_start > size:
            raise ValueError("truncated header")
        raw = bytearray()
        while len(raw) < header_len:
            block = os.read(fd, min(CHUNK, header_len - len(raw)))
            if not block:
                raise ValueError("truncated header")
            raw.extend(block)
        enforce_json_depth(raw)
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_pairs,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError("invalid JSON number")))
        if not isinstance(header, dict):
            raise ValueError("header must be an object")
        spans = []
        tensors = []
        for name, item in header.items():
            if name == "__metadata__":
                if not isinstance(item, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in item.items()):
                    raise ValueError("invalid __metadata__")
                continue
            if len(tensors) >= MAX_TENSORS:
                raise ValueError("tensor count exceeds limit")
            if not isinstance(name, str) or not name or not isinstance(item, dict):
                raise ValueError("invalid tensor entry")
            dtype, shape, offsets = item.get("dtype"), item.get("shape"), item.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
                raise ValueError("unsupported dtype for exact size validation: " + repr(dtype))
            if not isinstance(shape, list) or len(shape) > MAX_RANK or any(type(d) is not int or d < 0 for d in shape):
                raise ValueError("invalid tensor shape")
            if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
                raise ValueError("invalid data_offsets")
            start, end = offsets
            if start < 0 or end < start or data_start + end > size:
                raise ValueError("tensor range outside file")
            elements = 1
            for dim in shape:
                elements *= dim
            expected = elements * DTYPE_BYTES[dtype]
            if end - start != expected:
                raise ValueError("tensor byte range does not match exact dtype/shape size")
            spans.append((start, end, name))
            tensors.append({"name": name, "dtype": dtype, "shape": shape,
                            "data_offsets": [start, end], "byte_length": end - start})
        spans.sort()
        for left, right in zip(spans, spans[1:]):
            if left[1] > right[0]:
                raise ValueError("overlapping tensor ranges")
        if spans and (spans[0][0] != 0 or spans[-1][1] != size - data_start or
                      any(a[1] != b[0] for a, b in zip(spans, spans[1:]))):
            raise ValueError("gap or trailing/unclaimed data bytes")
        if not spans and size != data_start:
            raise ValueError("trailing data without tensors")
        result.update(status="structurally_valid", tensors=sorted(tensors, key=lambda t: t["name"]))
    except RecursionError:
        result["error"] = "safetensors JSON nesting exceeds parser limit"
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, OverflowError) as exc:
        result["error"] = str(exc)[:256]
    return result


def inventory_file(root, name, category):
    record = {"path": name, "category": category, "status": "unavailable", "size": None, "sha256": None}
    fd = None
    try:
        fd = _open_beneath(root, name)
        before = _metadata(fd)
        count, digest = _hash_fd(fd)
        structure = parse_safetensors_fd(fd, before[3]) if name.lower().endswith(".safetensors") else None
        count_after, digest_after = _hash_fd(fd)
        after = _metadata(fd)
        if before != after or count != count_after or digest != digest_after or count != before[3]:
            raise ValueError("file changed during inventory")
        record.update(status="hashed", size=count, sha256=digest)
        if structure is not None:
            record["safetensors"] = structure
    except (OSError, ValueError) as exc:
        record["error"] = str(exc)
        if name.lower().endswith(".safetensors"):
            record["safetensors"] = {"status": "invalid", "tensors": [], "error": str(exc)}
    finally:
        if fd is not None:
            os.close(fd)
    return record


def read_manifest(path):
    with open(path, "rb") as stream:
        raw = stream.read(MAX_MANIFEST + 1)
    if len(raw) > MAX_MANIFEST:
        raise ValueError("manifest exceeds 4 MiB limit")
    try:
        enforce_json_depth(raw)
        return json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicate_pairs)
    except RecursionError as exc:
        raise ValueError("manifest JSON nesting exceeds parser limit") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="explicit trusted directory containing supplied files")
    parser.add_argument("--manifest", help="JSON manifest with files:[{path,category?}]")
    parser.add_argument("--file", action="append", default=[], help="explicit root-relative file (repeatable)")
    parser.add_argument("--output", required=True, help="report JSON path (outside inputs recommended)")
    args = parser.parse_args(argv)
    try:
        entries = []
        if args.manifest:
            doc = read_manifest(args.manifest)
            if not isinstance(doc, dict) or not isinstance(doc.get("files"), list):
                raise ValueError("manifest must contain a files array")
            for entry in doc["files"]:
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    raise ValueError("each manifest file needs a path")
                entries.append((entry["path"], entry.get("category", "artifact")))
        entries.extend((name, "artifact") for name in args.file)
        if not entries:
            raise ValueError("supply --manifest or at least one --file")
        if any(not isinstance(category, str) for _, category in entries):
            raise ValueError("category must be a string")
        seen = set()
        records = []
        for name, category in entries:
            if name in seen:
                raise ValueError("duplicate input path: " + name)
            seen.add(name)
            records.append(inventory_file(args.root, name, category))
        report = {"schema": "m0-actual-local-artifact-inventory-v1", "root": str(Path(args.root).resolve()),
                  "files": records, "weights_present": any(r.get("safetensors", {}).get("status") == "structurally_valid" for r in records),
                  "compatibility": "not assessed", "qualification": "not assessed",
                  "expected_hashes_rewritten": False}
        output = json.dumps(report, indent=2, sort_keys=True) + "\n"
        Path(args.output).write_text(output, encoding="utf-8")
        print(output, end="")
        return 1 if any(r["status"] != "hashed" or r.get("safetensors", {}).get("status") == "invalid" for r in records) else 0
    except RecursionError:
        print("m0_artifacts: JSON nesting exceeds parser limit", file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("m0_artifacts: " + str(exc)[:256], file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
