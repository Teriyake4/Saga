import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "new_model.py"
spec = importlib.util.spec_from_file_location("new_model", SCRIPT)
new_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(new_model)


class NewModelTests(unittest.TestCase):
    def test_all_files_deterministic_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = new_model.generate("AcmeText", "acme-text", root)
            contents = {p.name: p.read_bytes() for p in first.iterdir()}
            self.assertEqual(set(contents), {"acme-text.h", "acme-text.cpp", "CMakeLists.txt", "README.md"})
            second_root = root / "other"
            second = new_model.generate("AcmeText", "acme-text", second_root)
            self.assertEqual(contents, {p.name: p.read_bytes() for p in second.iterdir()})
            source = contents["acme-text.cpp"].decode()
            for method in ("inspect", "describe_programs", "describe_resources", "create_runtime", "prefill", "decode"):
                self.assertIn('unfinished("' + method + '")', source)
            self.assertIn("registry.add(\"acme-text\"", source)

    def test_rejects_invalid_names_and_existing_target_without_partial_output(self):
        for name, model_id in (("lowerCase", "valid-id"), ("Valid", "../escape"), ("Valid", "Bad_ID"), ("Valid;X", "valid")):
            with self.subTest(name=name, model_id=model_id), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaises(ValueError):
                    new_model.generate(name, model_id, root)
                self.assertEqual(list(root.iterdir()), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "family"
            target.mkdir()
            marker = target / "keep"
            marker.write_text("unchanged")
            with self.assertRaises(ValueError):
                new_model.generate("Family", "family", root)
            self.assertEqual(marker.read_text(), "unchanged")
            self.assertEqual(list(root.iterdir()), [target])

    def test_keyword_id_namespace_is_safe_and_racing_target_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = new_model.generate("KeywordProbe", "class", root)
            self.assertIn("namespace saga::models::model_class", (generated / "class.h").read_text())
            self.assertIn("saga_model_model_class", (generated / "CMakeLists.txt").read_text())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "race"
            competitor = root / "competitor"
            competitor.mkdir()
            (competitor / "keep").write_text("other writer")
            original_publish = new_model.publish

            def claim_then_publish(stage, destination):
                destination.mkdir()
                (destination / "keep").write_text("other writer")
                original_publish(stage, destination)

            with mock.patch.object(new_model, "publish", side_effect=claim_then_publish):
                with self.assertRaises(FileExistsError):
                    new_model.generate("RaceProbe", "race", root)
            self.assertEqual((target / "keep").read_text(), "other writer")
            self.assertEqual(sorted(p.name for p in target.iterdir()), ["keep"])
            self.assertFalse(any(p.name.startswith(".race.tmp-") for p in root.iterdir()))

    @unittest.skipUnless(shutil.which("c++") or shutil.which("clang++"), "C++ compiler unavailable")
    def test_generated_translation_unit_compiles_against_current_headers(self):
        compiler = shutil.which("c++") or shutil.which("clang++")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = new_model.generate("CompileProbe", "compile-probe", root)
            keyword = new_model.generate("KeywordProbe", "class", root)
            for translation_unit in (generated / "compile-probe.cpp", keyword / "class.cpp"):
                subprocess.run([compiler, "-std=c++20", "-fsyntax-only", "-I", str(ROOT / "include"), str(translation_unit)], check=True, capture_output=True, text=True)

    def test_cli_rejects_invalid_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(SCRIPT), "--name", "bad", "--id", "bad", "--output", tmp], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("PascalCase", result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
