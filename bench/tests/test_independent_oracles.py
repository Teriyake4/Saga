import json
from fractions import Fraction
from pathlib import Path
import struct
import unittest

from bench.oracles.q3_scalar import (
    GROUP_SIZE,
    PACKED_BYTES,
    dot_exact,
    fp16_fraction,
    pack_signed3,
    represented_values,
    unpack_signed3,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "q3_scalar_known_answers.json"
LOCK = Path(__file__).resolve().parents[1] / "reference" / "source-lock.json"
LOCKED_ID = "yalding/ninfer-4080:src/ops/linear/q3/q3_rowsplit_storage.cuh"


def polynomial_pack(codes):
    """Independent positional-integer definition, not bytewise pack logic."""
    stream = sum((code % 8) * (8 ** index) for index, code in enumerate(codes))
    return stream.to_bytes(PACKED_BYTES, "little")


class IndependentQ3ScalarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.answers = cls.fixture["answers"]

    def test_fixture_provenance_matches_source_lock(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        entry = next(item for item in lock["inputs"] if item["id"] == LOCKED_ID)
        contract = self.fixture["contract"]
        self.assertEqual(contract["source_url"], entry["url"])
        self.assertEqual(contract["source_sha256"], entry["sha256"])
        self.assertEqual(contract["group_codes"], GROUP_SIZE)
        self.assertEqual(contract["packed_bytes"], PACKED_BYTES)

    def test_repeated_pattern_literal_and_independent_polynomial_answers(self):
        answer = self.answers["repeated_pattern"]
        codes = answer["pattern_codes"] * answer["repeat_count"]
        expected = bytes.fromhex(answer["expected_packed_hex"])
        self.assertEqual(len(codes), GROUP_SIZE)
        self.assertEqual(len(expected), PACKED_BYTES)
        self.assertEqual(expected[:3], bytes.fromhex("ac8f68"))
        self.assertEqual(polynomial_pack(codes), expected)
        self.assertEqual(pack_signed3(codes), expected)
        self.assertEqual(unpack_signed3(expected), tuple(codes))

    def test_polynomial_reference_exercises_every_code_position(self):
        for position in range(GROUP_SIZE):
            codes = [0] * GROUP_SIZE
            codes[position] = (-4, -1, 1, 3)[position % 4]
            expected = polynomial_pack(codes)
            with self.subTest(position=position):
                self.assertEqual(pack_signed3(codes), expected)
                self.assertEqual(unpack_signed3(expected), tuple(codes))

    def test_sparse_cross_byte_and_group_end_known_answer(self):
        answer = self.answers["sparse_boundary"]
        codes = [0] * GROUP_SIZE
        for index, code in answer["overrides"].items():
            codes[int(index)] = code
        expected = polynomial_pack(codes)
        packed = pack_signed3(codes)
        nonzero = {str(i): f"{byte:02x}" for i, byte in enumerate(expected) if byte}
        self.assertEqual(nonzero, answer["expected_nonzero_bytes"])
        self.assertEqual(packed, expected)
        self.assertEqual(unpack_signed3(expected), tuple(codes))
        self.assertEqual(expected[47], 0x80)

    def test_finite_fp16_exact_rationals(self):
        for name, answer in self.answers["scales"].items():
            with self.subTest(name=name):
                bits = int(answer["fp16_bits"], 16)
                numerator, denominator = map(int, answer["fraction"].split("/"))
                self.assertEqual(fp16_fraction(bits), Fraction(numerator, denominator))

    def test_every_finite_binary16_matches_stdlib_exact_value(self):
        for bits in range(0x10000):
            if ((bits >> 10) & 0x1F) == 0x1F:
                continue
            value = struct.unpack("<e", bits.to_bytes(2, "little"))[0]
            self.assertEqual(fp16_fraction(bits), Fraction.from_float(value), hex(bits))

    def test_represented_values_and_exact_full_group_projection(self):
        pattern = self.answers["repeated_pattern"]
        codes = pattern["pattern_codes"] * pattern["repeat_count"]
        packed = bytes.fromhex(pattern["expected_packed_hex"])
        decoded = unpack_signed3(packed)
        values = represented_values(decoded, 0x3C00)
        self.assertEqual(values, tuple(Fraction(code) for code in codes))
        self.assertEqual(dot_exact(values, [1] * GROUP_SIZE), Fraction(-64))

        answer = self.answers["projection"]
        scale = int(answer["scale_bits"], 16)
        expected = tuple(Fraction(*map(int, value.split("/"))) for value in answer["represented_values"])
        self.assertEqual(represented_values(answer["codes"], scale), expected)
        self.assertEqual(dot_exact(expected, answer["activation"]), Fraction(7))

    def test_rejects_malformed_groups_codes_buffers_and_fp16_inputs(self):
        with self.assertRaises(ValueError):
            pack_signed3([0] * 127)
        with self.assertRaises(ValueError):
            unpack_signed3(bytes(PACKED_BYTES - 1))
        with self.assertRaises(TypeError):
            unpack_signed3([0] * PACKED_BYTES)
        wide = memoryview(bytes(PACKED_BYTES * 2)).cast("H")
        with self.assertRaises(ValueError):
            unpack_signed3(wide)
        with self.assertRaises(TypeError):
            unpack_signed3(memoryview(bytes(PACKED_BYTES * 2))[::2])
        for value in (-5, 4):
            codes = [0] * GROUP_SIZE
            codes[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                pack_signed3(codes)
        for value in (True, 1.0, "1"):
            codes = [0] * GROUP_SIZE
            codes[0] = value
            with self.subTest(value=value), self.assertRaises(TypeError):
                pack_signed3(codes)
        with self.assertRaises(ValueError):
            represented_values([], 0x7C00)
        for codes in ([True], [1.0], [4]):
            with self.subTest(codes=codes), self.assertRaises((TypeError, ValueError)):
                represented_values(codes, 0x3C00)
        with self.assertRaises(TypeError):
            fp16_fraction(True)
        with self.assertRaises(ValueError):
            fp16_fraction(0x10000)
        for bits in (0x7C00, 0xFC00, 0x7E00):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                fp16_fraction(bits)

    def test_dot_input_validation(self):
        with self.assertRaises(ValueError):
            dot_exact([1, 2], [1])
        for left, right in (([True], [1]), ([1.0], [1]), ([1], [0.5])):
            with self.subTest(left=left, right=right), self.assertRaises(TypeError):
                dot_exact(left, right)


if __name__ == "__main__":
    unittest.main()
