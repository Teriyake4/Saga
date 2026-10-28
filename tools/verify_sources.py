
"""Audit M0 metadata only; never invoked by the build and never fetches weights.

Default: validate manifests and recorded retrieval evidence (no network).
--online: re-fetch the explicitly locked HTTPS metadata/source files.
--structure-only: check document structure, NOT M0 readiness or source availability.
Exit codes: 0 requested checks succeeded, 1 unresolved/failed input, 2 invalid input.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8 * 1024 * 1024
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
REPOSITORIES = {
    "Neroued/ninfer": "64d10fbf9f8e6bfc79ab0570b94674c5b9997929",
    "yalding/ninfer-4080": "2469b5752672ad597055af61fe28e8e7ced20056",
    "Niko1221/Strata": "5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d",
    "zero734kr/switchyard": "b4bd35679da0ec5de08027637e7b94b4578a2cf8",
    "ggml-org/llama.cpp": "e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf",
}
REQUIRED_FILES = {
    repo: {"README.md", "LICENSE", "CMakeLists.txt"} for repo in REPOSITORIES
}
REQUIRED_FILES["zero734kr/switchyard"] = {"README.md", "LICENSE", "pyproject.toml"}
REQUIRED_FILES["ggml-org/llama.cpp"].update({
    "common/arg.cpp", "ggml/src/ggml-common.h", "ggml/src/ggml-quants.c",
    "ggml/src/ggml-quants.h", "ggml/src/ggml-impl.h", "ggml/include/ggml.h",
})
REQUIRED_FILES["yalding/ninfer-4080"].update({
    "src/ops/linear/q3/q3_dispatch.cpp", "src/ops/linear/q3/q3_dispatch.h",
    "src/ops/linear/q3/q3_rowsplit_storage.cuh",
    "tools/convert/qwen3_8_27b/gsq3_source.py",
})
ARTIFACT_IDS = {"original-bf16-text", "gsq-text", "original-mtp", "original-vision",
                "tokenizer", "template", "reference-container"}
QUALITY_IDS = {"gsq-vs-original-bf16", "kv-vs-two-byte", "mtp-component-vs-source",
               "vision-component-vs-source", "exact-stochastic-mtp", "multimodal-vs-original-bf16"}
HF_FILES = {"config.json", "README.md", "tokenizer_config.json", "chat_template.jinja",
            "preprocessor_config.json", "processor_config.json", "generation_config.json",
                        "video_preprocessor_config.json", "LICENSE", "artifact-manifest.json"}
SOURCE_SUFFIXES = {".md", ".txt", ".h", ".hpp", ".c", ".cpp", ".cu", ".cuh", ".py", ".toml", ".json"}


def canonical(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def load(path):
    # Reject duplicate object keys rather than silently accepting the last value.
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def check_url(url, allow_cache=False):
    """Strict metadata/source allowlist, including redirects; no credentials/query."""
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port
            or (parsed.query and not allow_cache) or parsed.fragment):
        raise ValueError("only canonical HTTPS metadata URLs are allowed: " + url)
    parts = parsed.path.strip("/").split("/")
    if any(part in {".", ".."} or "%" in part for part in parts):
        raise ValueError("noncanonical URL path")
    if parsed.hostname == "raw.githubusercontent.com" and len(parts) >= 4:
        repo, revision = "/".join(parts[:2]), parts[2]
        name = parts[-1]
        if (REPOSITORIES.get(repo) == revision and
                (Path(name).suffix in SOURCE_SUFFIXES or name in {"LICENSE", "NOTICE", "COPYING"})):
            return
    if parsed.hostname == "api.github.com" and len(parts) >= 5 and parts[0] == "repos":
        repo = "/".join(parts[1:3])
        if repo in REPOSITORIES and parts[3:] == ["commits", REPOSITORIES[repo]]:
            return
    if parsed.hostname == "huggingface.co":
        if (allow_cache and len(parts) == 7 and parts[:3] == ["api", "resolve-cache", "models"]
                and REVISION.fullmatch(parts[5]) and parts[6] in HF_FILES):
            return
        if len(parts) == 4 and parts[:2] == ["api", "models"]:
            return
        if (len(parts) == 5 and parts[2] == "resolve"
                and REVISION.fullmatch(parts[3]) and parts[4] in HF_FILES):
            return
    raise ValueError("URL is not an allowed pinned source/metadata endpoint: " + url)


class MetadataRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, allow_cache=True)
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if new.path.startswith("/api/resolve-cache/models/"):
            original = old.path.strip("/").split("/")
            target = new.path.strip("/").split("/")[3:]
            if len(original) != 5 or original[2] != "resolve" or target != original[:2] + original[3:]:
                raise ValueError("metadata redirect changed repository/revision/file")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_input(entry, timeout=25):
    """Return byte evidence plus bytes for inspection; do not persist remote code."""
    url = entry["url"]
    check_url(url)
    result = {"id": entry["id"], "url": url, "http_status": None,
              "sha256": None, "bytes": None, "error": None}
    request = urllib.request.Request(url, headers={"User-Agent": "Saga-M0-source-verifier/1",
                                                  "Accept-Encoding": "identity"})
    try:
        with urllib.request.build_opener(MetadataRedirect()).open(request, timeout=timeout) as response:
            result["http_status"] = response.status
            result["final_url"] = response.url
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise ValueError("metadata response exceeds 8 MiB limit")
            if response.status != 200:
                raise ValueError("expected HTTP 200")
            result.update(sha256=hashlib.sha256(body).hexdigest(), bytes=len(body))
            return result, body
    except urllib.error.HTTPError as error:
        result.update(http_status=error.code, error=str(error),
                      error_body=error.read(4096).decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    return result, None


def validate_schema(value, schema, location="$"):
    """Validate the deliberately small JSON Schema subset used by bench schemas."""
    errors = []
    types = {"object": dict, "array": list, "string": str, "boolean": bool,
             "integer": int, "number": (int, float), "null": type(None)}
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(isinstance(value, types[name]) and
                   not (name in {"integer", "number"} and isinstance(value, bool)) for name in names):
            return [location + ": wrong type"]
    if "const" in schema and canonical(value) != canonical(schema["const"]):
        errors.append(location + ": wrong constant")
    if "enum" in schema and canonical(value) not in [canonical(item) for item in schema["enum"]]:
        errors.append(location + ": not an allowed value")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(location + ": missing " + name)
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                errors.extend(validate_schema(item, properties[name], location + "." + name))
            elif schema.get("additionalProperties") is False:
                errors.append(location + ": unexpected " + name)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(location + ": too few items")
        if schema.get("uniqueItems") and len({canonical(item) for item in value}) != len(value):
            errors.append(location + ": duplicate items")
        for index, item in enumerate(value):
            errors.extend(validate_schema(item, schema.get("items", {}), location + "[" + str(index) + "]"))
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        errors.append(location + ": pattern mismatch")
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and "minimum" in schema and value < schema["minimum"]):
        errors.append(location + ": below minimum")
    return errors


def semantic_errors(reference, quality):
    errors = []
    ids = set()
    for case in reference["candidate_cases"] + reference["mandatory_cases"]:
        if case["id"] in ids:
            errors.append("duplicate reference case ID: " + case["id"])
        ids.add(case["id"])
    for case in reference["candidate_cases"]:
        if case["status"] != "candidate" or case["frozen"] or case["mandatory"]:
            errors.append("candidate case is masquerading as frozen/mandatory: " + case["id"])
        if case["results"] is not None:
            errors.append("candidate cases cannot carry qualifying results: " + case["id"])
    artifact_ids = [item["id"] for item in reference["artifacts"]]
    if set(artifact_ids) != ARTIFACT_IDS or len(artifact_ids) != len(ARTIFACT_IDS):
        errors.append("required artifact IDs missing/duplicated or unknown")
    if not all(item["required"] for item in reference["artifacts"]):
        errors.append("required artifact cannot be made optional")
    frozen = reference["suite"]["status"] == "frozen"
    if not frozen and reference["mandatory_cases"]:
        errors.append("unfrozen suite must not claim frozen mandatory cases")
    if frozen:
        if not reference["mandatory_cases"] or reference["suite"]["freeze_record"] is None:
            errors.append("frozen suite needs mandatory cases and a freeze record")
        if reference["suite"]["os_scope"] not in {"linux-only", "linux-and-native-windows"}:
            errors.append("frozen suite needs an explicit native OS scope")
        frozen_definitions = [{key: value for key, value in case.items() if key != "results"}
                              for case in reference["mandatory_cases"]]
        digest = hashlib.sha256(canonical(frozen_definitions).encode("utf-8")).hexdigest()
        if reference["suite"]["mandatory_suite_digest"] != digest:
            errors.append("frozen mandatory suite digest mismatch")
        modes = {case["mode"] for case in reference["mandatory_cases"]}
        if not {"plain", "native-mtp"}.issubset(modes):
            errors.append("frozen suite must include plain and native-MTP cases")
    for case in reference["mandatory_cases"]:
        if not case["frozen"] or not case["mandatory"] or case["status"] != "frozen":
            errors.append("mandatory case must be explicitly frozen: " + case["id"])
        if case["unresolved"] or not case["configuration"]:
            errors.append("mandatory case has unresolved configuration: " + case["id"])
        config = case["configuration"] or {}
        for field in reference["freeze_required_fields"]:
            if config.get(field) is None or config.get(field) in ("", [], {}):
                errors.append(case["id"] + ": unresolved " + field)
        if config.get("prefix_reuse") is not False:
            errors.append(case["id"] + ": uncached comparison requires prefix_reuse=false")
        if case["mode"] == "native-mtp" and config.get("ngram") is not False:
            errors.append(case["id"] + ": native MTP requires ngram=false")
    availability = [item["pair"] for item in quality["kv_pair_availability"]]
    if set(availability) != set(quality["required_kv_pairs"]) or len(availability) != len(set(availability)):
        errors.append("required KV pair availability rows missing/duplicated")
    seen = set()
    for comparison in quality["comparisons"]:
        if comparison["id"] in seen:
            errors.append("duplicate quality comparison ID: " + comparison["id"])
        seen.add(comparison["id"])
        if comparison["status"] == "frozen":
            for field in quality["freeze_required_fields"]:
                value = comparison["preregistration"].get(field)
                if value is None or value in ("", [], {}):
                    errors.append(comparison["id"] + ": unresolved " + field)
            if comparison["unresolved"]:
                errors.append(comparison["id"] + ": frozen policy still has blockers")
        elif comparison["results"] is not None:
            errors.append(comparison["id"] + ": unfrozen policy cannot carry qualifying results")
    if seen != QUALITY_IDS:
        errors.append("required quality comparisons missing or unknown")
    return errors


def audit(root=ROOT, online=False, structure_only=False):
    reference = load(root / "bench/reference/manifest.json")
    quality = load(root / "bench/quality/manifest.json")
    lock = load(root / "bench/reference/source-lock.json")
    evidence = load(root / "bench/reference/retrieval-evidence.json")
    errors = []
    for value, schema_path in [(reference, "bench/reference/manifest.schema.json"),
                               (quality, "bench/quality/manifest.schema.json"),
                               (lock, "bench/reference/source-lock.schema.json")]:
        errors.extend(validate_schema(value, load(root / schema_path), schema_path))
    if not errors:
        errors.extend(semantic_errors(reference, quality))
    entries = lock["inputs"]
    if len({entry["id"] for entry in entries}) != len(entries):
        errors.append("duplicate source input IDs")
    for repo, paths in REQUIRED_FILES.items():
        actual = {entry["path"] for entry in entries if entry["repository"] == repo and entry["required"]}
        for path in sorted(paths - actual):
            errors.append("missing required source lock: " + repo + ":" + path)
    for entry in entries:
        try:
            check_url(entry["url"])
            if entry["kind"] == "model-discovery":
                expected_url = "https://huggingface.co/api/models/" + entry["repository"]
            elif entry["kind"] == "commit-api":
                expected_url = ("https://api.github.com/repos/" + entry["repository"] +
                                "/commits/" + (entry["revision"] or ""))
            elif entry["kind"] == "model-metadata":
                expected_url = ("https://huggingface.co/" + entry["repository"] + "/resolve/" +
                                (entry["revision"] or "") + "/" + (entry["path"] or ""))
            else:
                expected_url = ("https://raw.githubusercontent.com/" + entry["repository"] + "/" +
                                (entry["revision"] or "") + "/" + (entry["path"] or ""))
            if entry["url"] != expected_url:
                errors.append("source URL does not match declared repository/revision/path: " + entry["id"])
            if entry["kind"] != "model-discovery":
                if not REVISION.fullmatch(entry["revision"] or ""):
                    errors.append("missing source revision: " + entry["id"])
                if entry["repository"] in REPOSITORIES and entry["revision"] != REPOSITORIES[entry["repository"]]:
                    errors.append("unexpected pinned revision: " + entry["id"])
            if (entry["sha256"] is None) != (entry["bytes"] is None):
                errors.append("hash/byte-count must be resolved together: " + entry["id"])
        except ValueError as error:
            errors.append(str(error))
    report = {"format_version": 1, "mode": "structure-only" if structure_only else
              ("online" if online else "recorded-evidence-offline"),
              "qualification": "not-evaluated", "errors": errors, "blockers": [], "inputs": []}
    if errors or structure_only:
        report["check_status"] = "invalid" if errors else "structure-valid-not-qualified"
        return report, 2 if errors else 0
    recorded = {item["id"]: item for item in evidence["inputs"]}
    if online:
        with ThreadPoolExecutor(max_workers=4) as pool:
            observations = list(pool.map(lambda entry: fetch_input(entry)[0], entries))
    else:
        observations = [recorded.get(entry["id"], {}) for entry in entries]
    for entry, observed in zip(entries, observations):
        expected = entry["sha256"]
        local_error = None
        if entry["local_path"] is not None:
            local = (root / entry["local_path"]).resolve()
            licenses = (root / "bench/reference/licenses").resolve()
            if local.parent != licenses:
                local_error = "license path escaped bench/reference/licenses"
            else:
                try:
                    body = local.read_bytes()
                    if hashlib.sha256(body).hexdigest() != expected or len(body) != entry["bytes"]:
                        local_error = "preserved license bytes do not match source lock"
                except OSError as error:
                    local_error = str(error)
        verified = (expected is not None and observed.get("sha256") == expected
                    and observed.get("http_status") == 200 and observed.get("error") is None
                    and observed.get("url") == entry["url"]
                    and observed.get("bytes") == entry["bytes"] and local_error is None)
        item = {"id": entry["id"], "required": entry["required"], "expected_sha256": expected,
                "status": ("fetched-verified" if online else "recorded-verified") if verified else "unverified",
                "observation": observed, "local_error": local_error}
        report["inputs"].append(item)
        if entry["required"] and not verified:
            report["blockers"].append("required source unverified: " + entry["id"])
    for artifact in reference["artifacts"]:
        if artifact["required"] and (artifact["status"] != "verified" or not artifact["revision"]
                                      or not artifact["files"] or artifact["compatibility"] != "verified"
                                                                            or artifact["unresolved"]):
            report["blockers"].append("artifact bytes/compatibility unverified: " + artifact["id"])
    build = reference["reference_build"]
    if (build["status"] != "measured" or not SHA256.fullmatch(build["binary_sha256"] or "")
            or not build["build_commands"] or not build["run_commands"] or not build["runtime_dispatch_trace"]):
        report["blockers"].append("pinned reference build/commands/runtime dispatch evidence missing")
    if not quality["independent_fixture_inventory"]:
        report["blockers"].append("independent scalar/layer/state fixture evidence missing")
    if reference["suite"]["status"] != "frozen":
        report["blockers"].append("mandatory comparable suite is unfrozen")
    if reference["hardware"]["status"] != "measured-native-4080":
        report["blockers"].append("native original RTX 4080 hardware/runtime evidence missing")
    for comparison in quality["comparisons"]:
        if comparison["status"] != "frozen" or not comparison["results"]:
            report["blockers"].append("quality baseline/policy unresolved: " + comparison["id"])
    report["check_status"] = "blocked" if report["blockers"] else "inputs-verified-not-qualified"
    return report, 1 if report["blockers"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--online", action="store_true", help="explicit bounded HTTPS source/metadata checks")
    mode.add_argument("--structure-only", action="store_true", help="schemas/consistency only; not M0 readiness")
    parser.add_argument("--output", type=Path, help="write canonical JSON report (default stdout)")
    args = parser.parse_args()
    try:
        report, code = audit(online=args.online, structure_only=args.structure_only)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report, code = {"check_status": "invalid", "error": str(error), "qualification": "not-evaluated"}, 2
    text = canonical(report)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
