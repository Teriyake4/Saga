import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import m0_frontend

OPTIONS = {
    "add_generation_prompt": True, "tools": [], "documents": [], "date_string": "2025-01-15",
    "enable_thinking": True, "reasoning_effort": "xhigh", "preserve_thinking": True,
    "add_vision_id": False,
}


class FrontendInputValidationTests(unittest.TestCase):
    def _workload(self, directory, **overrides):
        spec = {"id": "safe-candidate", "class": "plain", "messages": [{"role": "user", "content": "hello"}],
                "max_new_tokens": 8, "target_prompt_tokens": 128, "template_options": dict(OPTIONS)}
        spec.update(overrides)
        path = Path(directory) / "candidate.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "file").write_bytes(b"altered")
            lock = root / "lock.json"
            lock.write_text(json.dumps({"schema": "m0-frontend-source-lock-v1", "models": {"m": {"repo": "repo"}}, "files": [
                {"model": "m", "name": "file", "local_path": "file", "size": 1, "sha256": "0" * 64}]}))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                m0_frontend.load_locked_files(lock, root)

    def test_lock_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lock = root / "lock.json"
            lock.write_text(json.dumps({"schema": "m0-frontend-source-lock-v1", "models": {"m": {"repo": "repo"}}, "files": [
                {"model": "m", "name": "file", "local_path": "../outside", "size": 4, "sha256": "0" * 64}]}))
            with self.assertRaises(ValueError):
                m0_frontend.load_locked_files(lock, root)

    def test_lock_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "actual").write_bytes(b"safe")
            try:
                (root / "link").symlink_to(root / "actual")
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink creation unavailable: {exc}")
            lock = root / "lock.json"
            lock.write_text(json.dumps({"schema": "m0-frontend-source-lock-v1", "models": {"m": {"repo": "repo"}}, "files": [
                {"model": "m", "name": "file", "local_path": "link", "size": 4, "sha256": "0" * 64}]}))
            with self.assertRaises(ValueError):
                m0_frontend.load_locked_files(lock, root)

    def test_candidate_id_and_numeric_bounds_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            for candidate in ("../evil", "nested/name", "x" * 81):
                with self.assertRaisesRegex(ValueError, "candidate id"):
                    m0_frontend._validate_candidate(json.loads(self._workload(temp, id=candidate).read_text()))
            for target in (True, 0, -1, m0_frontend.MAX_TARGET + 1, 1.5):
                with self.assertRaisesRegex(ValueError, "target_prompt_tokens"):
                    m0_frontend._validate_candidate(json.loads(self._workload(temp, target_prompt_tokens=target).read_text()))
            for reserve in (True, 0, m0_frontend.MAX_RESERVE + 1):
                with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                    m0_frontend._validate_candidate(json.loads(self._workload(temp, max_new_tokens=reserve).read_text()))

    def test_invalid_messages_missing_options_and_unsupported_effort(self):
        with tempfile.TemporaryDirectory() as temp:
            for messages in ([], [{"role": "user"}], "not-a-list"):
                with self.assertRaisesRegex(ValueError, "messages"):
                    m0_frontend._validate_candidate(json.loads(self._workload(temp, messages=messages).read_text()))
            options = dict(OPTIONS)
            options.pop("enable_thinking")
            with self.assertRaisesRegex(ValueError, "template options"):
                m0_frontend._validate_candidate(json.loads(self._workload(temp, template_options=options).read_text()))
            for effort in ("high", "invalid"):
                options = dict(OPTIONS, reasoning_effort=effort)
                with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                    m0_frontend._validate_candidate(json.loads(self._workload(temp, template_options=options).read_text()))

    def test_cli_reports_missing_optional_tokenizers_actionably(self):
        with mock.patch.object(m0_frontend, "_tokenizer_type", side_effect=m0_frontend.FrontendDependencyError("install pinned deps")):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = m0_frontend.main(["--compare"])
        self.assertEqual(code, 2)
        self.assertIn("install pinned deps", stderr.getvalue())

    def test_tracked_input_bundle_hashes_counts_and_specs(self):
        workloads = m0_frontend.WORKLOADS
        index = json.loads((workloads / "frontend-input-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(index["status"], "UNFROZEN")
        self.assertFalse(index["mandatory_comparison"])
        lock_bytes = m0_frontend.LOCK.read_bytes()
        self.assertEqual(hashlib.sha256(lock_bytes).hexdigest(), index["source_lock_sha256"])
        tracked_files = list((workloads / "inputs").glob("*.json"))
        self.assertLess(sum(path.stat().st_size for path in tracked_files), 1024 * 1024)
        comparison = workloads / "inputs/frontend-comparison.json"
        self.assertEqual(hashlib.sha256(comparison.read_bytes()).hexdigest(), index["comparison"]["repository_output_sha256"])
        comparison_data = json.loads(comparison.read_text(encoding="utf-8"))
        self.assertIn("generation_config", comparison_data)
        for item in index["outputs"]:
            output_path = workloads / "inputs" / (item["candidate"] + ".json")
            output_bytes = output_path.read_bytes()
            output = json.loads(output_bytes)
            spec = json.loads((workloads / (item["candidate"] + ".json")).read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(output_bytes).hexdigest(), item["repository_output_sha256"])
            self.assertEqual(hashlib.sha256(output_bytes).hexdigest(), item["output_sha256"])
            self.assertEqual(output["candidate"], spec["id"])
            self.assertEqual(output["target_prompt_tokens"], spec["target_prompt_tokens"])
            self.assertEqual(output["max_new_tokens"], 512)
            self.assertEqual(spec["max_new_tokens"], 512)
            self.assertEqual(output["actual_prompt_tokens"], item["prompt_tokens"])
            self.assertEqual(output["target_prompt_tokens"], output["actual_prompt_tokens"])
            self.assertEqual(len(output["token_ids"]), output["actual_prompt_tokens"])
            self.assertEqual(output["total_context_tokens"], output["actual_prompt_tokens"] + 512)
            self.assertEqual(hashlib.sha256(output["rendered_text"].encode("utf-8")).hexdigest(), item["rendered_utf8_sha256"])
            token_bytes = json.dumps(output["token_ids"], separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(token_bytes).hexdigest(), item["token_ids_sha256"])


def _optional_frontend_available():
    if importlib.util.find_spec("tokenizers") is None or importlib.util.find_spec("jinja2") is None:
        return False
    try:
        if importlib.metadata.version("tokenizers") != "0.22.1" or importlib.metadata.version("Jinja2") != "3.1.6":
            return False
    except importlib.metadata.PackageNotFoundError:
        return False
    if not m0_frontend.LOCK.is_file():
        return False
    try:
        m0_frontend.load_locked_files()
    except (OSError, ValueError, KeyError):
        return False
    return True


@unittest.skipUnless(_optional_frontend_available(), "pinned frontend dependencies and verified local source cache required")
class OptionalFrontendEvidenceTests(unittest.TestCase):
    def test_actual_render_options_prefix_suffix_and_empty_tools(self):
        result = m0_frontend.build_candidate(m0_frontend.WORKLOADS / "plain-128.json")
        self.assertEqual(result["actual_prompt_tokens"], 128)
        self.assertTrue(result["target_length_matches_actual"])
        self.assertTrue(result["rendered_text"].startswith("<|im_start|>system\nReasoning effort is set to xhigh."))
        self.assertTrue(result["rendered_text"].endswith("<|im_start|>assistant\n<think>\n"))
        self.assertEqual(result["template_options"]["tools"], [])
        self.assertNotIn("<tool_call>", result["rendered_text"])
        self.assertEqual(len(result["token_ids"]), 128)

    def test_original_and_gsq_render_same_actual_input(self):
        original = m0_frontend.build_candidate(m0_frontend.WORKLOADS / "plain-128.json")
        gsq = m0_frontend.build_candidate(m0_frontend.WORKLOADS / "plain-128.json", model="gsq")
        self.assertEqual(gsq["rendered_text"], original["rendered_text"])
        self.assertEqual(gsq["token_ids"], original["token_ids"])

    def test_exact_target_failure_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = json.loads((m0_frontend.WORKLOADS / "plain-128.json").read_text())
            spec["id"] = "target-too-short"
            spec["target_prompt_tokens"] = 1
            path = Path(temp) / "candidate.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "could not hit exact"):
                m0_frontend.build_candidate(path)

    def test_offline_source_comparison(self):
        comparison = m0_frontend.compare_frontends()
        self.assertTrue(comparison["file_comparison"]["tokenizer.json"]["byte_identical"])
        self.assertTrue(comparison["file_comparison"]["chat_template.jinja"]["byte_identical"])
        self.assertEqual(comparison["tokenizer_vocab_size"], {"original": 248077, "gsq": 248077})
        self.assertEqual(comparison["config_vocab_size"], {"original": 248320, "gsq": 248320})
        self.assertFalse(comparison["generation_config"]["byte_identical"])
        self.assertEqual(comparison["generation_config"]["fields"]["min_p"], {"original": None, "gsq": 0.0, "equal": False})


if __name__ == "__main__":
    unittest.main()
