#!/usr/bin/env python3
"""Offline integrity check of the vendored host headers and their licenses."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "third_party/lock.json").read_text(encoding="utf-8"))
    if lock["schema_version"] != 2:
        raise ValueError("unsupported dependency lock schema")
    entries = lock["files"]
    paths = {entry["path"] for entry in entries}
    required = {
        "third_party/cpp-httplib/httplib.h", "third_party/cpp-httplib/LICENSE",
        "third_party/json/json.hpp", "third_party/json/LICENSE",
        "third_party/tomlplusplus/toml.hpp", "third_party/tomlplusplus/LICENSE",
    }
    if len(paths) != len(entries) or not required.issubset(paths):
        raise ValueError("dependency lock must retain all original unique pinned files")
    if not {"third_party/stb-image/stb_image.h", "third_party/stb-image/LICENSE"}.issubset(paths):
        raise ValueError("optional decoder header and license must both be pinned")
    patches = lock["patches"]
    patch_paths = {entry["path"] for entry in patches}
    references = {path for entry in entries for path in entry.get("local_patches", [])}
    if len(patch_paths) != len(patches) or patch_paths != references:
        raise ValueError("local dependency patches must be unique and referenced")
    for entry in entries:
        if entry.get("local_patches"):
            if len(entry.get("upstream_sha256", "")) != 64 or entry.get("upstream_bytes", 0) <= 0:
                raise ValueError("patched dependency needs an upstream identity: " + entry["path"])
            for path in entry["local_patches"]:
                if not any(patch["path"] == path and patch["dependency"] == entry["dependency"]
                           for patch in patches):
                    raise ValueError("dependency/patch identity mismatch: " + path)
    for entry in entries + patches:
        path = (ROOT / entry["path"]).resolve()
        path.relative_to(ROOT / "third_party")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("vendored dependency mismatch: " + entry["path"])
    print(f"PASS: all {len(entries)} pinned dependency files and {len(patches)} local patches match their recorded SHA-256/size")


if __name__ == "__main__":
    main()
