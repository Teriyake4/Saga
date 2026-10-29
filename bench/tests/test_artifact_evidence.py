"""Opt-in artifact evidence tests using temporary local files only."""

import builtins
import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("verify_sources", ROOT / "tools/verify_sources.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ArtifactEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.artifact = {"id": "example"}
        self.body = b"local bytes"
        self.item = {"path": "nested/file with space-\u2603.bin", "bytes": len(self.body),
                     "sha256": hashlib.sha256(self.body).hexdigest()}
        self.path = self.root / "example" / self.item["path"]
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(self.body)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, path=None):
        item = dict(self.item)
        if path is not None:
            item["path"] = path
        return verifier.verify_artifact_file(self.root, self.artifact, item)

    def test_match_and_multiple_components_unicode_spaces(self):
        result = self.check()
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["bytes"], len(self.body))

    def test_hash_reads_at_most_declared_bytes_plus_one(self):
        self.path.write_bytes(b"x" * 100)
        item = dict(self.item, bytes=3)
        real_open = Path.open
        requested = []

        class TrackingStream:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def read(self, count=-1):
                requested.append(count)
                return self.stream.read(count)

        def tracked_open(path, *args, **kwargs):
            return TrackingStream(real_open(path, *args, **kwargs))

        with mock.patch.object(Path, "open", tracked_open):
            result = verifier.verify_artifact_file(self.root, self.artifact, item)
        self.assertEqual(result["status"], "unverified")
        self.assertLessEqual(sum(requested), 4)

    def test_file_changed_during_hash_is_unverified(self):
        real_open = Path.open

        class ReplacingStream:
            def __init__(self, stream, path):
                self.stream, self.path = stream, path
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def read(self, count=-1):
                data = self.stream.read(count)
                with builtins.open(self.path, "wb") as replacement:
                    replacement.write(b"changed file")
                return data

        def replacing_open(path, *args, **kwargs):
            return ReplacingStream(real_open(path, *args, **kwargs), path)

        with mock.patch.object(Path, "open", replacing_open):
            result = verifier.verify_artifact_file(self.root, self.artifact, self.item)
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["error"], "artifact size mismatch or file changed during hash")

    def test_tamper_missing_and_size_mismatch(self):
        self.path.write_bytes(b"tamperedbyt")
        self.assertEqual(self.check()["status"], "mismatch")
        self.path.unlink()
        missing = self.check()
        self.assertEqual(missing["status"], "unverified")
        self.assertEqual(missing["error"], "artifact file unavailable or unreadable")
        self.assertNotIn(str(self.root), missing["error"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(self.body + b"x")
        self.assertEqual(self.check()["status"], "unverified")

    def test_reject_noncanonical_paths(self):
        for path in ("/tmp/file", "C:/file", "//server/share", "a/../file", "a\\file",
                     "a//file", "a:stream", "a/./file", "a\x00b"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    verifier.safe_artifact_path(path)
                result = self.check(path)
                self.assertNotEqual(result["status"], "matched")
                self.assertIsNone(result["path"])

    def test_backslash_named_posix_file_rejected_before_open(self):
        odd = self.root / "example" / "nested\\file"
        odd.write_bytes(self.body)
        with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
            with self.assertRaises(ValueError):
                verifier.safe_artifact_path("nested\\file")

    def test_symlink_escape_and_nonregular(self):
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_bytes(self.body)
        try:
            self.path.unlink()
            self.path.symlink_to(outside)
            self.assertNotEqual(self.check()["status"], "matched")
        finally:
            outside.unlink(missing_ok=True)
        self.path.unlink()
        self.path.mkdir()
        self.assertNotEqual(self.check()["status"], "matched")

    def test_structure_only_does_not_consult_artifact_root(self):
        with mock.patch.object(verifier, "verify_artifact_file", side_effect=AssertionError("file read")):
            with mock.patch.object(verifier, "fetch_input", side_effect=AssertionError("network")):
                report, code = verifier.audit(structure_only=True, artifact_root=self.root)
        self.assertEqual(code, 0)
        self.assertNotIn("artifacts", report)

    def test_parent_symlink_escape_and_fifo(self):
        outside = self.root.parent / (self.root.name + "-parent")
        outside.mkdir()
        (outside / "file").write_bytes(self.body)
        link_parent = self.root / "example" / "linked"
        link_parent.symlink_to(outside, target_is_directory=True)
        escaped = dict(self.item, path="linked/file")
        self.assertNotEqual(verifier.verify_artifact_file(self.root, self.artifact, escaped)["status"], "matched")
        if hasattr(__import__("os"), "mkfifo"):
            import os
            fifo = self.root / "example" / "fifo"
            os.mkfifo(fifo)
            self.assertNotEqual(verifier.verify_artifact_file(
                self.root, self.artifact, dict(self.item, path="fifo"))["status"], "matched")
        shutil.rmtree(outside)

    def test_evidence_rows_require_shape_and_success_final_url(self):
        row = {"id": "one", "url": "https://example.invalid/source", "http_status": 200,
               "error": None, "bytes": 1, "sha256": "a" * 64, "final_url": "https://example.invalid/source"}
        self.assertEqual(verifier.evidence_shape_errors([row], ["one"]), [])
        for bad in (dict(row, final_url=3), dict(row, final_url=[]),
                    {key: value for key, value in row.items() if key != "error"},
                    {key: value for key, value in row.items() if key != "final_url"}):
            self.assertTrue(verifier.evidence_shape_errors([bad], ["one"]))
        failed = dict(row, http_status=403, final_url="", error="HTTPError", bytes=None, sha256=None)
        failed.pop("final_url")
        self.assertEqual(verifier.evidence_shape_errors([failed], ["one"]), [])

    def test_verified_claim_checks_each_artifact_gate_independently(self):
        temp = tempfile.TemporaryDirectory()
        try:
            root = Path(temp.name)
            for folder in ("bench/reference", "bench/quality"):
                shutil.copytree(ROOT / folder, root / folder)
            manifest_path = root / "bench/reference/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            artifact = manifest["artifacts"][0]
            artifact.update(status="verified", compatibility="verified", unresolved=[])
            artifact["files"] = [dict(self.item)]
            manifest_path.write_text(json.dumps(manifest))
            # The copied manifests and evidence form an isolated audit root.
            report, code = verifier.audit(root=root)
            self.assertEqual(code, 1)
            artifact_blocker = "artifact bytes/compatibility unverified: " + artifact["id"]
            self.assertIn(artifact_blocker, report["blockers"])
            local_file = root / "artifact-root" / artifact["id"] / self.item["path"]
            local_file.parent.mkdir(parents=True)
            local_file.write_bytes(self.body)
            report, code = verifier.audit(root=root, artifact_root=root / "artifact-root")
            evidence = next(row for row in report["artifacts"] if row["id"] == artifact["id"])
            self.assertEqual(evidence["files"][0]["status"], "matched")
            self.assertNotIn(artifact_blocker, report["blockers"])
            self.assertEqual(report["qualification"], "not-evaluated")
            self.assertEqual(code, 1)

            artifact["compatibility"] = "unverified"
            manifest_path.write_text(json.dumps(manifest))
            report, _ = verifier.audit(root=root, artifact_root=root / "artifact-root")
            self.assertIn(artifact_blocker, report["blockers"])

            artifact["compatibility"] = "verified"
            artifact["unresolved"] = ["compatibility not established"]
            manifest_path.write_text(json.dumps(manifest))
            report, _ = verifier.audit(root=root, artifact_root=root / "artifact-root")
            self.assertIn(artifact_blocker, report["blockers"])
        finally:
            temp.cleanup()

    def test_duplicate_artifact_paths_are_invalid_before_reads(self):
        temp = tempfile.TemporaryDirectory()
        try:
            root = Path(temp.name)
            shutil.copytree(ROOT / "bench/reference", root / "bench/reference")
            shutil.copytree(ROOT / "bench/quality", root / "bench/quality")
            path = root / "bench/reference/manifest.json"
            manifest = json.loads(path.read_text())
            item = {"path": "a/b", "bytes": 0, "sha256": "0" * 64}
            manifest["artifacts"][0]["files"] = [item, dict(item)]
            path.write_text(json.dumps(manifest))
            with mock.patch.object(verifier, "verify_artifact_file", side_effect=AssertionError("read")):
                report, code = verifier.audit(root=root)
            self.assertEqual(code, 2)
            self.assertTrue(any("duplicate artifact file paths" in error for error in report["errors"]))
        finally:
            temp.cleanup()

    def test_evidence_ids_and_observation_shape(self):
        row = {"id": "one", "url": "https://example.invalid/source", "http_status": 503,
               "error": "HTTPError", "bytes": None, "sha256": None}
        self.assertEqual(verifier.evidence_shape_errors([row], ["one"]), [])
        for rows in ([dict(row, id=[])], [row, dict(row)], [], [dict(row, id="unknown")], [dict(row, url=3)],
                     [dict(row, bytes=4)], [dict(row, http_status=True)]):
            self.assertTrue(verifier.evidence_shape_errors(rows, ["one"]))


if __name__ == "__main__":
    unittest.main()
