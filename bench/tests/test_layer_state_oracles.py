import json
import math
import pathlib
import unittest

from bench.oracles import layer_state as oracle


FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "layer_state_known_answers.json"


class LayerStateOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.answers = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.cases = cls.answers["cases"]
        cls.atol = cls.answers["tolerances"]["absolute"]
        cls.rtol = cls.answers["tolerances"]["relative"]

    def assertClose(self, actual, expected):
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.assertClose(got, want)
        else:
            self.assertTrue(math.isclose(actual, expected, rel_tol=self.rtol, abs_tol=self.atol), (actual, expected))

    def test_known_answers(self):
        c = self.cases
        a = c["rms_norm"]
        self.assertClose(oracle.rms_norm(a["x"], a["weight"], a["epsilon"]), a["expected"])
        a = c["rope_nonzero"]
        self.assertClose(oracle.rope_pair(a["x"], a["position"], a["theta"]), a["expected"])
        a = c["swiglu_nonzero"]
        self.assertClose(oracle.swiglu(a["gate"], a["up"]), a["expected"])
        a = c["causal_attention"]
        self.assertClose(oracle.causal_attention(a["query"], a["keys"], a["values"]), a["expected"])
        a = c["gated_delta"]
        state = a["state"]
        for key, value, beta, expected_output, expected_state in a["steps"]:
            output, state = oracle.gated_delta_step(state, key, value, beta)
            self.assertClose(output, expected_output)
            self.assertClose(state, expected_state)
        a = c["depthwise_conv"]
        y, history = oracle.causal_depthwise_step(a["history"], a["sample"], a["kernel"])
        self.assertClose(y, a["expected_output"])
        self.assertClose(history, a["expected_history"])

    def test_state_object_continuation_checkpoint_rollback_reset_and_clone(self):
        state = oracle.ScalarRecurrenceState(2)
        first = state.step(1, 4, 0.5)
        checkpoint = state.checkpoint()
        second = state.step(2, 1, 0.25)
        self.assertEqual((first, second, state.value), (3.0, 3.5, 1.75))
        state.rollback(checkpoint)
        self.assertEqual(state.value, 3.0)
        left, right = state.clone(), state.clone()
        self.assertIsNot(left, right)
        self.assertEqual(left.step(1, 0, 1), 0.0)
        self.assertEqual(right.step(1, 8, 1), 8.0)
        self.assertEqual((state.value, left.value, right.value), (3.0, 0.0, 8.0))
        state.reset()
        self.assertEqual(state.step(1, 2, 1), 2.0)
        # Tokenwise and chunk-grouped calls preserve the explicit state threading.
        tokenwise = oracle.ScalarRecurrenceState(0)
        outs = (tokenwise.step(1, 2, 1), tokenwise.step(2, 1, 0.25))
        chunk = oracle.ScalarRecurrenceState(0)
        chunk_outs = []
        for k, v, beta in ((1, 2, 1), (2, 1, 0.25)):
            chunk_outs.append(chunk.step(k, v, beta))
        self.assertEqual((tuple(chunk_outs), chunk.value), (outs, tokenwise.value))

    def test_convolution_history_continues_and_resets(self):
        y1, h1 = oracle.causal_depthwise_step([], [3], [2, 1])
        y2, h2 = oracle.causal_depthwise_step(h1, [4], [2, 1])
        self.assertEqual((y1, y2, h2), ((6.0,), (11.0,), ((4.0,),)))
        # A fresh sequence has no prior-sample contribution.
        reset_y, reset_h = oracle.causal_depthwise_step([], [4], [2, 1])
        self.assertEqual((reset_y, reset_h), ((8.0,), ((4.0,),)))
        saved = tuple(tuple(row) for row in h1)
        branch_a = oracle.causal_depthwise_step(saved, [0], [2, 1])
        branch_b = oracle.causal_depthwise_step(saved, [8], [2, 1])
        self.assertEqual(saved, ((3.0,),))
        self.assertEqual(branch_a[0], (3.0,))
        self.assertEqual(branch_b[0], (19.0,))

    def test_rejects_invalid_shapes_nonfinite_and_boolean_numeric_inputs(self):
        for bad in ([float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                oracle.rms_norm(bad, [1])
        with self.assertRaises(ValueError):
            oracle.swiglu([1, 2], [1])
        with self.assertRaises(ValueError):
            oracle.causal_attention([[1]], [[1], [2]], [[1], [2]])
        with self.assertRaises(ValueError):
            oracle.gated_delta_step(0, 1, 2, 2)
        for fn, args in ((oracle.rms_norm, ([1], [1], 0)),
                         (oracle.rms_norm, ([1], [1], float("inf"))),
                         (oracle.rope_pair, ([1, 0], 1, 0)),
                         (oracle.rope_pair, ([1, 0], 1, True)),
                         (oracle.gated_delta_step, (False, 1, 2, 0.5)),
                         (oracle.swiglu, ([1000], [1e308]))):
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises((TypeError, ValueError)):
                fn(*args)

    def test_empty_ragged_and_overflowing_inputs_fail_explicitly(self):
        huge = 10 ** 10000
        for fn, args in ((oracle.rms_norm, ([huge], [1])),
                         (oracle.rope_pair, ([1, 0], 1, huge)),
                         (oracle.rope_pair, ([1, 0], huge, 1)),
                         (oracle.causal_attention, ([], [], [])),
                         (oracle.causal_attention, ([[1], [1, 2]], [[1], [1]], [[1], [2]])),
                         (oracle.causal_depthwise_step, ([], [], [1])),
                         (oracle.causal_attention, ([[1e308]], [[1e308]], [[1]])),
                         (oracle.causal_depthwise_step, ([], [1e308], [1e308]))):
            with self.subTest(fn=fn.__name__, args=args), self.assertRaises((TypeError, ValueError)):
                fn(*args)


if __name__ == "__main__":
    unittest.main()
