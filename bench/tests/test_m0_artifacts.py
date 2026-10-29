import hashlib
import json
import os
import struct
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

from tools import m0_artifacts


def make_safe(path, header, data=b"abcd"):
    raw = header if isinstance(header, bytes) else json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


def parse_path(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        return m0_artifacts.parse_safetensors_fd(fd, os.fstat(fd).st_size)
    finally:
        os.close(fd)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_safetensors_and_stream_hash(self):
        make_safe(self.root / "w.safetensors", {"x": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}})
        result = m0_artifacts.inventory_file(self.root, "w.safetensors", "weights")
        self.assertEqual(result["sha256"], hashlib.sha256((self.root / "w.safetensors").read_bytes()).hexdigest())
        self.assertEqual(result["safetensors"]["status"], "structurally_valid")
        self.assertEqual(result["safetensors"]["tensors"][0]["name"], "x")

    def test_duplicate_json_keys_and_boolean_integer_rejected(self):
        raw = b'{"x":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}'
        make_safe(self.root / "dup.safetensors", raw)
        self.assertIn("duplicate JSON key", parse_path(self.root / "dup.safetensors")["error"])
        make_safe(self.root / "bool.safetensors", {"x": {"dtype": "I32", "shape": [True], "data_offsets": [0, 4]}})
        self.assertEqual(parse_path(self.root / "bool.safetensors")["status"], "invalid")

    def test_deeply_nested_header_is_invalid_report_not_exception(self):
        nested = b"[" * 2000 + b"0" + b"]" * 2000
        path = self.root / "nested.safetensors"
        path.write_bytes(struct.pack("<Q", len(nested)) + nested)
        result = m0_artifacts.inventory_file(self.root, path.name, "weights")
        self.assertEqual(result["status"], "hashed")
        self.assertEqual(result["safetensors"]["status"], "invalid")
        self.assertLessEqual(len(result["safetensors"]["error"]), 256)
        report_path = self.root / "nested-report.json"
        with redirect_stdout(StringIO()):
            code = m0_artifacts.main(["--root", str(self.root), "--file", path.name,
                                      "--output", str(report_path)])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(report_path.read_text())["files"][0]["safetensors"]["status"], "invalid")

    def test_overlap_truncation_and_trailing_rejected(self):
        cases = [
            ({"a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}, "b": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}, b"abcd"),
            ({"a": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]}}, b"abcd"),
            ({"a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}}, b"abcde"),
        ]
        for index, (header, data) in enumerate(cases):
            path = self.root / (str(index) + ".safetensors")
            make_safe(path, header, data)
            self.assertEqual(parse_path(path)["status"], "invalid")

    def test_bad_dtype_and_unsupported_subbyte_are_rejected_without_type_error(self):
        for index, dtype in enumerate(([], {}, "F4")):
            path = self.root / ("dtype" + str(index) + ".safetensors")
            make_safe(path, {"x": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4]}})
            result = m0_artifacts.inventory_file(self.root, path.name, "weights")
            self.assertEqual(result["status"], "hashed")
            self.assertEqual(result["safetensors"]["status"], "invalid")

    def test_zero_extent_and_odd_byte_extent(self):
        zero = self.root / "zero.safetensors"
        make_safe(zero, {"empty": {"dtype": "F16", "shape": [0, 3], "data_offsets": [0, 0]}}, b"")
        self.assertEqual(m0_artifacts.inventory_file(self.root, zero.name, "weights")["safetensors"]["status"], "structurally_valid")
        odd = self.root / "odd.safetensors"
        make_safe(odd, {"x": {"dtype": "F16", "shape": [1], "data_offsets": [0, 1]}}, b"x")
        self.assertEqual(m0_artifacts.inventory_file(self.root, odd.name, "weights")["safetensors"]["status"], "invalid")

    def test_path_replacement_during_inventory_fails_closed(self):
        path = self.root / "replace.safetensors"
        make_safe(path, {"x": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}})

        replacement = self.root / "replacement"
        replacement.write_bytes(b"not safetensors")
        real_parse = m0_artifacts.parse_safetensors_fd

        def replace_then_parse(fd, size):
            replacement.replace(path)
            return real_parse(fd, size)

        with mock.patch.object(m0_artifacts, "parse_safetensors_fd", replace_then_parse):
            result = m0_artifacts.inventory_file(self.root, path.name, "weights")
        self.assertNotEqual(result["status"], "hashed")
        self.assertIn("changed", result["error"])


    def test_mutation_during_parse_is_not_reported_as_hashed(self):
        path = self.root / "mutate.safetensors"
        make_safe(path, {"x": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}})
        real_parse = m0_artifacts.parse_safetensors_fd

        def mutate_after_parse(fd, size):
            result = real_parse(fd, size)
            with path.open("r+b") as stream:
                stream.write(b"z")
            return result

        with mock.patch.object(m0_artifacts, "parse_safetensors_fd", mutate_after_parse):
            result = m0_artifacts.inventory_file(self.root, path.name, "weights")
        self.assertNotEqual(result["status"], "hashed")
        self.assertIn("changed", result["error"])

    def test_json_depth_boundary_and_quoted_escaped_brackets(self):
        m0_artifacts.enforce_json_depth(b"[" * m0_artifacts.MAX_JSON_DEPTH + b"0" + b"]" * m0_artifacts.MAX_JSON_DEPTH)
        with self.assertRaisesRegex(ValueError, "128-level limit"):
            m0_artifacts.enforce_json_depth(b"[" * (m0_artifacts.MAX_JSON_DEPTH + 1) + b"0" + b"]" * (m0_artifacts.MAX_JSON_DEPTH + 1))
        quoted = json.dumps({"quoted": "[" * 400 + '\\"' + "{" * 400 + "]" * 400}).encode()
        m0_artifacts.enforce_json_depth(quoted)

    def test_deeply_nested_manifest_becomes_bounded_input_error(self):
        manifest = self.root / "nested-manifest.json"
        manifest.write_bytes(b'{"files":[],"nested":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}")
        with self.assertRaisesRegex(ValueError, "128-level limit"):
            m0_artifacts.read_manifest(manifest)

    def test_deeply_nested_safetensors_header_becomes_invalid(self):
        nested = b"[" * 2000 + b"0" + b"]" * 2000
        path = self.root / "nested-depth.safetensors"
        path.write_bytes(struct.pack("<Q", len(nested)) + nested)
        result = m0_artifacts.inventory_file(self.root, path.name, "weights")
        self.assertEqual(result["safetensors"]["status"], "invalid")
        self.assertIn("128-level limit", result["safetensors"]["error"])

    def test_safetensors_metadata_string_may_contain_escaped_brackets(self):
        path = self.root / "brackets.safetensors"
        make_safe(path, {"__metadata__": {"text": "[[[\\\"{{{]]]"}}, b"")
        result = m0_artifacts.inventory_file(self.root, path.name, "weights")
        self.assertEqual(result["safetensors"]["status"], "structurally_valid")

    def test_manifest_size_limit(self):
        manifest = self.root / "large.json"
        manifest.write_bytes(b" " * (m0_artifacts.MAX_MANIFEST + 1))
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            m0_artifacts.read_manifest(manifest)

    def test_backslash_and_nul_rejected_before_windows_open_dispatch(self):
        with mock.patch.object(m0_artifacts, "_open_beneath_windows") as windows_open:
            for name in ("nested\\..\\outside.txt", "nested/embedded\x00nul.txt"):
                with self.subTest(name=repr(name)):
                    with self.assertRaises(ValueError):
                        m0_artifacts._open_beneath(self.root, name)
            windows_open.assert_not_called()

    def test_symlink_and_parent_traversal_rejected(self):
        (self.root / "outside").write_text("x")
        for name in ("../outside", "C:outside", "folder/C:file"):
            result = m0_artifacts.inventory_file(self.root, name, "config")
            self.assertEqual(result["status"], "unavailable")
        try:
            (self.root / "link").symlink_to(self.root / "outside")
        except (OSError, NotImplementedError):
            return
        result = m0_artifacts.inventory_file(self.root, "link", "config")
        self.assertEqual(result["status"], "unavailable")

    @unittest.skipUnless(os.name == "nt", "native Windows handle test")
    def test_windows_native_secure_nested_inventory_and_reparse_rejection(self):
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "config.json").write_text('{"v":1}')
        result = m0_artifacts.inventory_file(self.root, "nested/config.json", "config")
        self.assertEqual(result["status"], "hashed")
        outside = self.root / "outside.txt"
        outside.write_text("x")
        link = self.root / "file-link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Windows symlink creation privilege is unavailable")
        result = m0_artifacts.inventory_file(self.root, "file-link.txt", "config")
        self.assertEqual(result["status"], "unavailable")

    def test_cli_output_and_exit_codes(self):
        (self.root / "config.json").write_text('{"v":1}')
        report_path = self.root / "report.json"
        output = StringIO()
        with redirect_stdout(output):
            code = m0_artifacts.main(["--root", str(self.root), "--file", "config.json", "--output", str(report_path)])
        self.assertEqual(code, 0)
        report = json.loads(report_path.read_text())
        self.assertEqual(report["compatibility"], "not assessed")
        self.assertEqual(report["qualification"], "not assessed")
        self.assertEqual(json.loads(output.getvalue()), report)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            code = m0_artifacts.main(["--root", str(self.root), "--file", "missing", "--output", str(report_path)])
        self.assertEqual(code, 1)
        with redirect_stderr(StringIO()):
            code = m0_artifacts.main(["--root", str(self.root), "--output", str(report_path)])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
