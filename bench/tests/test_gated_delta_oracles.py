import json
import math
from fractions import Fraction
from pathlib import Path
import unittest

from bench.oracles.gated_delta import GatedDeltaState, transition


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "gated_delta_known_answers.json"


def rational(value):
    return float(Fraction(value)) if isinstance(value, str) else float(value)


class GatedDeltaOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.case = cls.fixture["case"]

    def assertClose(self, actual, expected):
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.assertClose(got, want)
        else:
            self.assertTrue(math.isclose(actual, rational(expected), rel_tol=1e-12, abs_tol=1e-12), (actual, expected))

    def test_two_dimensional_hand_derived_state_deltas_and_readout(self):
        state = self.case["initial_state"]
        for step in self.case["steps"]:
            state, delta, output = transition(
                state, step["key"], step["value"], step["decay"], step["beta"],
                step["query"], step["output_scale"])
            self.assertClose(delta, step["expected_delta"])
            self.assertClose(state, step["expected_state"])
            self.assertClose(output, step["expected_output"])

    def test_tokenwise_chunked_checkpoint_rollback_reset_and_clone(self):
        steps = self.case["steps"]
        tokenwise = GatedDeltaState(self.case["initial_state"])
        token_outputs = [tokenwise.step(s["key"], s["value"], s["decay"], s["beta"], s["query"], s["output_scale"]) for s in steps]
        self.assertClose(tokenwise.matrix, steps[-1]["expected_state"])

        chunked = GatedDeltaState(self.case["initial_state"])
        checkpoint = chunked.checkpoint()
        chunk_outputs = []
        for step in steps:
            chunk_outputs.append(chunked.step(step["key"], step["value"], step["decay"], step["beta"], step["query"], step["output_scale"]))
        self.assertEqual(token_outputs, chunk_outputs)
        self.assertEqual(tokenwise.checkpoint(), chunked.checkpoint())

        chunked.rollback(checkpoint)
        self.assertEqual(chunked.checkpoint(), tuple(tuple(row) for row in self.case["initial_state"]))
        left, right = chunked.clone(), chunked.clone()
        left.step([1, 0], [0, 0], 1, 1, [1, 0])
        right.step([0, 1], [0, 0], 1, 1, [0, 1])
        self.assertEqual(chunked.checkpoint(), checkpoint)
        self.assertNotEqual(left.checkpoint(), right.checkpoint())
        chunked.reset(self.case["initial_state"])
        self.assertEqual(chunked.checkpoint(), checkpoint)

    def test_rejects_bad_shapes_gates_and_nonfinite_or_overflow_results(self):
        bad_calls = (
            ([], [1], [1], 1, 0.5, [1]),
            ([[1, 2], [1]], [1, 1], [1, 1], 1, 0.5, [1, 1]),
            ([[1]], [], [1], 1, 0.5, [1]),
            ([[1]], [1], [1], -1, 0.5, [1]),
            ([[1]], [1], [1], 1, 2, [1]),
            ([[1]], [True], [1], 1, 0.5, [1]),
            ([[1e308]], [1e308], [1e308], 1, 1, [1]),
        )
        for state, key, value, decay, beta, query in bad_calls:
            with self.subTest(state=state, key=key), self.assertRaises((TypeError, ValueError)):
                transition(state, key, value, decay, beta, query)

    def test_metadata_limits_claim_to_locked_recurrent_transition(self):
        provenance = self.fixture["provenance"]
        self.assertEqual(self.fixture["revision"], "yalding/ninfer-4080@2469b5752672ad597055af61fe28e8e7ced20056")
        self.assertEqual(provenance["license"]["spdx"], "Apache-2.0")
        self.assertIn("normalization", self.fixture["contract"]["normalization_boundary"])
        self.assertIn("full model/state qualification", self.fixture["status"])


if __name__ == "__main__":
    unittest.main()
