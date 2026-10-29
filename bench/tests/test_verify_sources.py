"""Host-only verifier tests. No network, remote execution, weights or GPU tests."""

import hashlib
import importlib.util
import io
import tempfile
import unittest
import urllib.error
import urllib.request
from copy import deepcopy
from email.message import Message
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("verify_sources", ROOT / "tools/verify_sources.py")
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class SourceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.reference = verifier.load(ROOT / "bench/reference/manifest.json")
        self.quality = verifier.load(ROOT / "bench/quality/manifest.json")

    def test_structure_is_valid_but_never_qualified(self):
        report, code = verifier.audit(structure_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(report["check_status"], "structure-valid-not-qualified")
        self.assertEqual(report["qualification"], "not-evaluated")

    def test_offline_m0_is_blocked(self):
        report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertEqual(report["check_status"], "blocked")
        self.assertIn("required source unverified: zero734kr/switchyard:LICENSE", report["blockers"])
        self.assertIn("mandatory comparable suite is unfrozen", report["blockers"])
        self.assertTrue(any("original-bf16" in item for item in report["blockers"]))

    def test_offline_report_is_byte_reproducible(self):
        first, _ = verifier.audit()
        second, _ = verifier.audit()
        self.assertEqual(verifier.canonical(first), verifier.canonical(second))

    def test_structure_mode_never_accesses_network(self):
        with mock.patch.object(verifier, "fetch_input", side_effect=AssertionError("network")):
            verifier.audit(structure_only=True)
            verifier.audit()

    def test_candidate_cannot_be_frozen_or_mandatory(self):
        for field in ("frozen", "mandatory"):
            reference = deepcopy(self.reference)
            reference["candidate_cases"][0][field] = True
            errors = verifier.semantic_errors(reference, self.quality)
            self.assertTrue(any("masquerading" in item for item in errors))

    def test_candidate_cannot_carry_qualifying_results(self):
        self.reference["candidate_cases"][0]["results"] = {"passed": True}
        self.assertTrue(any("qualifying results" in item for item in
                            verifier.semantic_errors(self.reference, self.quality)))

    def test_unfrozen_suite_cannot_claim_mandatory_rows(self):
        self.reference["mandatory_cases"] = [deepcopy(self.reference["candidate_cases"][0])]
        self.assertTrue(any("unfrozen suite" in item for item in
                            verifier.semantic_errors(self.reference, self.quality)))

    def test_empty_frozen_suite_does_not_pass_vacuously(self):
        self.reference["suite"]["status"] = "frozen"
        errors = verifier.semantic_errors(self.reference, self.quality)
        self.assertTrue(any("needs mandatory cases" in item for item in errors))
        self.assertTrue(any("OS scope" in item for item in errors))
        self.assertTrue(any("digest mismatch" in item for item in errors))

    def test_frozen_quality_rejects_missing_or_empty_budgets(self):
        comparison = self.quality["comparisons"][0]
        comparison["status"] = "frozen"
        for budget in (None, {}):
            comparison["preregistration"]["regression_budgets"] = budget
            errors = verifier.semantic_errors(self.reference, self.quality)
            self.assertTrue(any("unresolved regression_budgets" in item for item in errors))

    def test_unfrozen_quality_cannot_report_pass(self):
        self.quality["comparisons"][0]["results"] = {"passed": True}
        self.assertTrue(any("unfrozen policy" in item for item in
                            verifier.semantic_errors(self.reference, self.quality)))

    def test_required_kv_pair_cannot_be_removed(self):
        self.quality["kv_pair_availability"].pop()
        self.assertTrue(any("KV pair availability" in item for item in
                            verifier.semantic_errors(self.reference, self.quality)))

    def test_required_artifacts_cannot_be_removed_or_made_optional(self):
        self.reference["artifacts"][0]["required"] = False
        self.assertTrue(any("made optional" in item for item in
                            verifier.semantic_errors(self.reference, self.quality)))
        self.reference["artifacts"].pop()
        self.assertTrue(any("artifact IDs" in item for item in
                            verifier.semantic_errors(self.reference, self.quality)))

    def test_schema_rejects_boolean_version_and_unknown_field(self):
        schema = verifier.load(ROOT / "bench/reference/manifest.schema.json")
        self.reference["format_version"] = True
        self.reference["fake_pass"] = True
        errors = verifier.validate_schema(self.reference, schema)
        self.assertTrue(any("wrong constant" in item for item in errors))
        self.assertTrue(any("unexpected fake_pass" in item for item in errors))

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"pass": false, "pass": true}')
            with self.assertRaises(ValueError):
                verifier.load(path)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.json"
            path.write_text('{"value": NaN}')
            with self.assertRaises(ValueError):
                verifier.load(path)

    def test_weights_mutable_refs_and_non_https_urls_are_rejected(self):
        bad_urls = [
            "http://raw.githubusercontent.com/Neroued/ninfer/main/README.md",
            "https://raw.githubusercontent.com/Neroued/ninfer/main/README.md",
            "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/config.json",
            "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/" + "a" * 40 + "/model.safetensors",
            "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/" + "a" * 40 + "/model.ninfer",
            "https://cdn-lfs.huggingface.co/weights",
            "https://user:password@huggingface.co/api/models/Qwen/Qwen3.8-27B",
            "https://127.0.0.1/config.json",
        ]
        for url in bad_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                verifier.check_url(url)

    def test_hf_cache_redirect_must_preserve_repo_revision_and_file(self):
        revision = "a" * 40
        request = urllib.request.Request("https://huggingface.co/Qwen/Qwen3.8-27B/resolve/" +
                                         revision + "/config.json")
        target = "https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.8-27B/"
        valid = target + revision + "/config.json?etag=test"
        redirected = verifier.MetadataRedirect().redirect_request(request, None, 307, "", {}, valid)
        self.assertEqual(redirected.full_url, valid)
        verifier.check_response_identity(request.full_url, valid)
        for changed in (target + "b" * 40 + "/config.json", target + revision + "/model.safetensors"):
            with self.assertRaises(ValueError):
                verifier.MetadataRedirect().redirect_request(request, None, 307, "", {}, changed)

    def test_redirects_reject_other_allowed_raw_source_and_api_identities(self):
        revision = verifier.REPOSITORIES["Neroued/ninfer"]
        original = "https://raw.githubusercontent.com/Neroued/ninfer/" + revision + "/README.md"
        request = urllib.request.Request(original)
        changed_urls = [
            (request, "https://raw.githubusercontent.com/Neroued/ninfer/" + revision + "/LICENSE"),
            (request, "https://raw.githubusercontent.com/ggml-org/ggml/" +
             verifier.REPOSITORIES["ggml-org/ggml"] + "/README.md"),
        ]
        api_url = ("https://api.github.com/repos/Neroued/ninfer/commits/" + revision)
        api_request = urllib.request.Request(api_url)
        changed_urls.append((api_request,
                             "https://api.github.com/repos/yalding/ninfer-4080/commits/" +
                             verifier.REPOSITORIES["yalding/ninfer-4080"]))
        for source_request, changed in changed_urls:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                verifier.MetadataRedirect().redirect_request(source_request, None, 302, "", {}, changed)

    def test_fetch_rejects_changed_final_url_before_hashing_response(self):
        entry = {"id": "probe", "kind": "source",
                 "url": "https://raw.githubusercontent.com/Neroued/ninfer/" +
                       verifier.REPOSITORIES["Neroued/ninfer"] + "/README.md"}
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.url = entry["url"].replace("README.md", "LICENSE")
        response.read.return_value = b"same bytes do not establish URL identity"
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(verifier.urllib.request, "build_opener", return_value=opener):
            result, fetched = verifier.fetch_input(entry)
        self.assertIsNone(fetched)
        self.assertIsNone(result["sha256"])
        self.assertIn("response URL changed", result["error"])
        response.read.assert_not_called()

    def test_fetch_accepts_legitimate_hf_metadata_cache_final_url(self):
        revision = "a" * 40
        url = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/" + revision + "/config.json"
        entry = {"id": "probe", "kind": "model-metadata", "url": url}
        final_url = ("https://huggingface.co/api/resolve-cache/models/Qwen/Qwen3.8-27B/" +
                     revision + "/config.json?etag=test")
        body = b'{"model_type":"qwen3_5"}'
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.url = final_url
        response.read.return_value = body
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(verifier.urllib.request, "build_opener", return_value=opener):
            result, fetched = verifier.fetch_input(entry)
        self.assertEqual(fetched, body)
        self.assertEqual(result["final_url"], final_url)
        self.assertEqual(result["sha256"], hashlib.sha256(body).hexdigest())

    def test_http_errors_keep_status_body_and_null_hash(self):
        entry = {"id": "probe", "url": "https://huggingface.co/api/models/Qwen/Qwen3.8-27B"}
        error = urllib.error.HTTPError(entry["url"], 403, "rate limit exceeded", Message(), io.BytesIO(b"rate limited"))
        opener = mock.Mock()
        opener.open.side_effect = error
        with mock.patch.object(verifier.urllib.request, "build_opener", return_value=opener):
            result, body = verifier.fetch_input(entry)
        self.assertIsNone(body)
        self.assertEqual(result["http_status"], 403)
        self.assertEqual(result["error_body"], "rate limited")
        self.assertIsNone(result["sha256"])

    def test_success_hashes_bytes_without_executing_them(self):
        entry = {"id": "probe", "url": "https://huggingface.co/api/models/Qwen/Qwen3.8-27B"}
        body = b'__import__("sys").exit(99)'
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.url = entry["url"]
        response.read.return_value = body
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(verifier.urllib.request, "build_opener", return_value=opener):
            result, fetched = verifier.fetch_input(entry)
        self.assertEqual(fetched, body)
        self.assertEqual(result["sha256"], hashlib.sha256(body).hexdigest())
        self.assertEqual(result["bytes"], len(body))
        response.read.assert_called_once_with(verifier.MAX_BYTES + 1)

    def test_oversize_metadata_never_receives_a_verified_hash(self):
        entry = {"id": "probe", "url": "https://huggingface.co/api/models/Qwen/Qwen3.8-27B"}
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.url = entry["url"]
        response.read.return_value = b"x" * 17
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(verifier, "MAX_BYTES", 16), mock.patch.object(
                verifier.urllib.request, "build_opener", return_value=opener):
            result, body = verifier.fetch_input(entry)
        self.assertIsNone(body)
        self.assertIsNone(result["sha256"])
        self.assertIn("exceeds", result["error"])

    def test_recorded_changed_final_url_blocks_required_input_even_with_matching_hash(self):
        original_load = verifier.load
        def redirected(path):
            value = original_load(path)
            if path.name == "retrieval-evidence.json":
                for entry in value["inputs"]:
                    if entry["id"] == "Neroued/ninfer:README.md":
                        entry["final_url"] = entry["url"].replace("README.md", "LICENSE")
            return value
        with mock.patch.object(verifier, "load", side_effect=redirected):
            report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertIn("required source unverified: Neroued/ninfer:README.md", report["blockers"])

    def test_recorded_hash_mismatch_blocks_required_input(self):
        original_load = verifier.load
        def tampered(path):
            value = original_load(path)
            if path.name == "retrieval-evidence.json":
                for entry in value["inputs"]:
                    if entry["id"] == "yalding/ninfer-4080:README.md":
                        entry["sha256"] = "0" * 64
            return value
        with mock.patch.object(verifier, "load", side_effect=tampered):
            report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertIn("required source unverified: yalding/ninfer-4080:README.md", report["blockers"])

    def test_preserved_license_hash_mismatch_blocks_input(self):
        original_read = Path.read_bytes
        def altered(path):
            if path.name == "Neroued--ninfer.LICENSE":
                return b"changed license"
            return original_read(path)
        with mock.patch.object(Path, "read_bytes", altered):
            report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertIn("required source unverified: Neroued/ninfer:LICENSE", report["blockers"])
        item = next(item for item in report["inputs"] if item["id"] == "Neroued/ninfer:LICENSE")
        self.assertIn("license bytes do not match", item["local_error"])

    def test_lock_cannot_substitute_a_different_file_url(self):
        original_load = verifier.load
        def substituted(path):
            value = original_load(path)
            if path.name == "source-lock.json":
                for entry in value["inputs"]:
                    if entry["id"] == "Neroued/ninfer:README.md":
                        entry["url"] = entry["url"].replace("README.md", "LICENSE")
            return value
        with mock.patch.object(verifier, "load", side_effect=substituted):
            report, code = verifier.audit(structure_only=True)
        self.assertEqual(code, 2)
        self.assertTrue(any("does not match declared" in item for item in report["errors"]))

    def test_missing_required_lock_is_invalid_not_successful(self):
        original_load = verifier.load
        def missing(path):
            value = original_load(path)
            if path.name == "source-lock.json":
                value["inputs"] = [entry for entry in value["inputs"]
                                   if entry["id"] != "yalding/ninfer-4080:README.md"]
            return value
        with mock.patch.object(verifier, "load", side_effect=missing):
            report, code = verifier.audit(structure_only=True)
        self.assertEqual(code, 2)
        self.assertTrue(any("missing required source lock" in item for item in report["errors"]))

    def test_pinned_gguf_spec_and_selected_reader_are_required(self):
        repo = "ggml-org/ggml"
        revision = "facdc44a67da17df4433adcc468b424f48e9ded2"
        required = {"README.md", "LICENSE", "CMakeLists.txt", "docs/gguf.md", "include/gguf.h",
                    "src/gguf.cpp", "include/ggml.h", "include/ggml-backend.h", "src/ggml-impl.h"}
        entries = [item for item in verifier.load(ROOT / "bench/reference/source-lock.json")["inputs"]
                   if item["repository"] == repo]
        self.assertEqual({item["path"] for item in entries if item["required"]}, required)
        self.assertTrue(all(item["revision"] == revision for item in entries))
        verifier.check_url("https://raw.githubusercontent.com/ggml-org/ggml/" + revision + "/docs/gguf.md")

    def test_gguf_mutable_refs_and_wrong_repo_revision_or_filename_are_rejected(self):
        revision = "facdc44a67da17df4433adcc468b424f48e9ded2"
        urls = [
            "https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md",
            "https://raw.githubusercontent.com/other/ggml/" + revision + "/docs/gguf.md",
            "https://raw.githubusercontent.com/ggml-org/ggml/" + "a" * 40 + "/docs/gguf.md",
            "https://raw.githubusercontent.com/ggml-org/ggml/" + revision + "/docs/gguf.bin",
        ]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                verifier.check_url(url)
        original_load = verifier.load
        def renamed(path):
            value = original_load(path)
            if path.name == "source-lock.json":
                for entry in value["inputs"]:
                    if entry["id"] == "ggml-org/ggml:docs/gguf.md":
                        entry["url"] = entry["url"].replace("gguf.md", "other.md")
            return value
        with mock.patch.object(verifier, "load", side_effect=renamed):
            report, code = verifier.audit(structure_only=True)
        self.assertEqual(code, 2)
        self.assertTrue(any("does not match declared" in item for item in report["errors"]))

    def test_missing_required_gguf_lock_is_invalid(self):
        original_load = verifier.load
        def missing(path):
            value = original_load(path)
            if path.name == "source-lock.json":
                value["inputs"] = [entry for entry in value["inputs"]
                                   if entry["id"] != "ggml-org/ggml:docs/gguf.md"]
            return value
        with mock.patch.object(verifier, "load", side_effect=missing):
            report, code = verifier.audit(structure_only=True)
        self.assertEqual(code, 2)
        self.assertTrue(any("missing required source lock: ggml-org/ggml:docs/gguf.md" in item
                            for item in report["errors"]))

    def test_changed_gguf_evidence_hash_blocks_input(self):
        original_load = verifier.load
        def tampered(path):
            value = original_load(path)
            if path.name == "retrieval-evidence.json":
                for entry in value["inputs"]:
                    if entry["id"] == "ggml-org/ggml:docs/gguf.md":
                        entry["sha256"] = "0" * 64
            return value
        with mock.patch.object(verifier, "load", side_effect=tampered):
            report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertIn("required source unverified: ggml-org/ggml:docs/gguf.md", report["blockers"])

    def test_gguf_license_bytes_are_preserved_and_checked(self):
        entry = next(item for item in verifier.load(ROOT / "bench/reference/source-lock.json")["inputs"]
                     if item["id"] == "ggml-org/ggml:LICENSE")
        path = ROOT / entry["local_path"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
        original_read = Path.read_bytes
        def altered(target):
            if target == path:
                return b"changed license"
            return original_read(target)
        with mock.patch.object(Path, "read_bytes", altered):
            report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertIn("required source unverified: ggml-org/ggml:LICENSE", report["blockers"])

    def test_commit_api_response_must_identify_exact_revision(self):
        entry = {"id": "ggml-org/ggml:commit-api", "kind": "commit-api",
                 "revision": "facdc44a67da17df4433adcc468b424f48e9ded2",
                 "url": "https://api.github.com/repos/ggml-org/ggml/commits/facdc44a67da17df4433adcc468b424f48e9ded2"}
        body = b'{"sha":"' + entry["revision"].encode() + b'"}'
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.url = entry["url"]
        response.read.return_value = body
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(verifier.urllib.request, "build_opener", return_value=opener):
            result, fetched = verifier.fetch_input(entry)
        self.assertEqual(fetched, body)
        self.assertEqual(result["reported_revision"], entry["revision"])
        entry["revision"] = "b" * 40
        with mock.patch.object(verifier.urllib.request, "build_opener", return_value=opener):
            result, fetched = verifier.fetch_input(entry)
        self.assertIsNone(fetched)
        self.assertIn("does not match pin", result["error"])

    def test_source_provenance_does_not_make_offline_m0_pass(self):
        report, code = verifier.audit()
        self.assertEqual(code, 1)
        self.assertEqual(report["check_status"], "blocked")
        self.assertEqual(report["qualification"], "not-evaluated")
        self.assertTrue(any("mandatory comparable suite is unfrozen" in item for item in report["blockers"]))
        api = next(item for item in report["inputs"] if item["id"] == "ggml-org/ggml:commit-api")
        self.assertEqual(api["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
