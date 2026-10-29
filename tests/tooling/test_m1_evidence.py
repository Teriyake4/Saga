import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import m1_evidence

CHECKS = "arguments driver_runtime_versions device_selection device_properties original_rtx_4080 compute_capability_sm89 sm_count_76 memory_query cooperative_support_query cp_async_resources bf16_wmma_resources resource_initialization allocations_alignment cp_async_launch_and_sync cp_async_result bf16_wmma_launch_and_sync bf16_wmma_result resource_cleanup allocation_cleanup".split()


def report():
    return {"schema_version": 1, "probe": "saga_cuda_portability_smoke", "passed": True, "error": "",
            "toolchain": {"os": "Windows", "nvcc_major": 12, "nvcc_minor": 8, "nvcc_build": 90,
                          "cuda_headers_version": 12080, "cplusplus": 202002,
                          "host_compiler": "MSVC", "host_compiler_version": 193900000},
            "driver_version": 12080, "runtime_version": 12080, "device_ordinal": 0, "device_count": 1,
            "device": {"name": "NVIDIA GeForce RTX 4080", "compute_major": 8, "compute_minor": 9,
                      "sm_count": 76, "expected_sm_count": 76, "warp_size": 32,
                      "total_global_mem_bytes": 16000000000, "cooperative_launch_property": True,
                      "cooperative_launch_attribute": 1},
            "expected_compile_arch": 890, "cp_async_compiled_arch": 890, "bf16_wmma_compiled_arch": 890,
            "cooperative_launch_exercised": False,
            "memory_before_allocations": {"free_bytes": 1000, "total_bytes": 2000},
            "kernels": [kernel(name) for name in ("cp_async_smoke", "bf16_wmma_smoke")],
            "checks": [{"name": name, "status": "pass", "detail": ""} for name in CHECKS]}


def kernel(name):
    return {"name": name, "queried": True, "binary_version": 89, "ptx_version": 89,
            "registers_per_thread": 32, "static_shared_bytes": 0, "local_bytes_per_thread": 0,
            "max_threads_per_block": 1024, "launch_blocks": 1, "threads_per_block": 32,
            "dynamic_shared_bytes": 0, "active_blocks_per_sm": 2, "resident_warps_per_sm": 2,
            "resident_grid_block_bound": 152, "cooperative_grid_block_bound": 152}


class EvidenceTests(unittest.TestCase):
    def test_snapshot_tracks_tracked_and_untracked_source_but_not_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir(); (root / "build").mkdir()
            (root / "src/a.cpp").write_text("one")
            (root / "build/CMakeCache.txt").write_text("secret/cache")
            first = m1_evidence.snapshot(root)
            (root / "src/a.cpp").write_text("two")
            (root / "src/new.cpp").write_text("untracked")
            second = m1_evidence.snapshot(root)
            self.assertNotEqual(first["files"], second["files"])
            self.assertFalse(any("build/" in item["path"] for item in second["files"]))

    def test_valid_report(self):
        m1_evidence.validate_cuda(report(), 0)
        cuda_130 = report()
        cuda_130["toolchain"].update(nvcc_major=13, nvcc_minor=0, nvcc_build=0,
                                     cuda_headers_version=13000)
        cuda_130["driver_version"] = 13000
        cuda_130["runtime_version"] = 13000
        m1_evidence.validate_cuda(cuda_130, 0)
        cuda_127 = report()
        cuda_127["toolchain"].update(nvcc_major=12, nvcc_minor=7, cuda_headers_version=12070)
        with self.assertRaises(ValueError):
            m1_evidence.validate_cuda(cuda_127, 0)
        bad_arch = report()
        bad_arch["cp_async_compiled_arch"] = 89
        with self.assertRaises(ValueError):
            m1_evidence.validate_cuda(bad_arch, 0)
        with self.assertRaises(ValueError):
            m1_evidence.validate_cuda(report(), -1)

    def test_compare_rejects_invalid_snapshot_scope(self):
        valid = m1_evidence.snapshot(Path(__file__).resolve().parents[2])
        malformed = dict(valid, scope=["*"])
        with self.assertRaises(ValueError):
            m1_evidence._validate_snapshot(malformed)

    def test_snapshot_rejects_symlinked_config_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            external = Path(tmp) / "external-config"
            root.mkdir()
            external.mkdir()
            (external / "server.example.toml").write_text("outside")
            try:
                (root / "config").symlink_to(external, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks unavailable")
            with self.assertRaises(ValueError):
                m1_evidence.snapshot(root)

    def test_snapshot_omits_secrets_and_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tools").mkdir()
            (root / "tools/.env.local").write_text("not captured")
            link = root / "tools/escape.py"
            external = root.parent / f"{root.name}-external.py"
            external.write_text("outside")
            try:
                link.symlink_to(external)
            except OSError:
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                m1_evidence.snapshot(root)
            link.unlink()
            result = m1_evidence.snapshot(root)
            self.assertNotIn("tools/.env.local", {entry["path"] for entry in result["files"]})
            external.unlink()

    def test_rejects_malformed_reports(self):
        mutations = [
            lambda d: d.pop("checks"),
            lambda d: d["checks"].pop(),
            lambda d: d["checks"].append({"name": "arguments", "status": "pass"}),
            lambda d: d["checks"].append({"name": "surprise", "status": "pass"}),
            lambda d: d["checks"][0].update(status="notrun"),
            lambda d: d.pop("passed"),
            lambda d: d.update(passed="true"),
            lambda d: d["toolchain"].update(os="Linux"),
            lambda d: d.update(device_ordinal=1),
            lambda d: d["device"].update(compute_minor=8),
            lambda d: d.update(cp_async_compiled_arch=89),
            lambda d: d.update(cp_async_compiled_arch="890"),
            lambda d: d.update(cooperative_launch_exercised=True),
            lambda d: d["kernels"][0].update(binary_version=90),
            lambda d: d["kernels"][0].update(queried=False),
            lambda d: d["kernels"][0].update(resident_grid_block_bound=1),
            lambda d: d["kernels"][0].update(cooperative_grid_block_bound=152.0),
            lambda d: d["checks"][0].update(name=[]),
            lambda d: d["checks"][0].pop("status"),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(report())
            mutate(candidate)
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    m1_evidence.validate_cuda(candidate, 0)

if __name__ == "__main__":
    unittest.main()
