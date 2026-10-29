"""CPU oracle for the locked recurrent GDN transition; no upstream code copied."""
import math


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real int or float")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} is outside finite binary64 range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(values, label, expected=None):
    try:
        items = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be a sequence") from exc
    if expected is not None and len(items) != expected:
        raise ValueError(f"{label} length must be {expected}")
    if not items:
        raise ValueError(f"{label} must not be empty")
    return tuple(_finite_number(item, label) for item in items)


def _matrix(values, label, rows=None, columns=None):
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be a matrix") from exc
    if not raw or (rows is not None and len(raw) != rows):
        raise ValueError(f"{label} must have the required nonempty row count")
    result = tuple(_vector(row, f"{label} row", columns) for row in raw)
    width = len(result[0])
    if any(len(row) != width for row in result):
        raise ValueError(f"{label} must be rectangular")
    return result


def _finite(value, label):
    if not math.isfinite(value):
        raise ValueError(f"{label} produced a nonfinite result")
    return value


def transition(state, key, value, decay, beta, query, output_scale=1.0):
    """Apply S'=decay*S + beta*(v-decay*(S@k)) outer k; emit (S'@q)*scale.

    `decay` is the already transformed exp(g) gate, and key/query vectors are
    already supplied in the reference operator's selected normalization mode.
    """
    matrix = _matrix(state, "state")
    dv, dk = len(matrix), len(matrix[0])
    if any(len(row) != dk for row in matrix):
        raise ValueError("state must be rectangular")
    k = _vector(key, "key", dk)
    v = _vector(value, "value", dv)
    q = _vector(query, "query", dk)
    a = _finite_number(decay, "decay")
    b = _finite_number(beta, "beta")
    scale = _finite_number(output_scale, "output_scale")
    if a < 0.0 or not 0.0 <= b <= 1.0:
        raise ValueError("decay must be nonnegative and beta must be in [0, 1]")

    updated = []
    deltas = []
    for r in range(dv):
        sk = sum(matrix[r][c] * k[c] for c in range(dk))
        delta = _finite(b * (v[r] - a * sk), "delta")
        deltas.append(delta)
        updated.append(tuple(_finite(a * matrix[r][c] + delta * k[c], "state") for c in range(dk)))
    updated = tuple(updated)
    output = tuple(_finite(sum(updated[r][c] * q[c] for c in range(dk)) * scale, "output") for r in range(dv))
    return updated, tuple(deltas), output


class GatedDeltaState:
    """Explicit mutable matrix-state lifecycle for supplied recurrent inputs."""

    def __init__(self, matrix):
        self.matrix = [list(row) for row in _matrix(matrix, "state")]

    def step(self, key, value, decay, beta, query, output_scale=1.0):
        updated, _deltas, output = transition(self.matrix, key, value, decay, beta, query, output_scale)
        self.matrix = [list(row) for row in updated]
        return output

    def checkpoint(self):
        return tuple(tuple(row) for row in self.matrix)

    def rollback(self, checkpoint):
        restored = _matrix(checkpoint, "checkpoint", len(self.matrix), len(self.matrix[0]))
        self.matrix = [list(row) for row in restored]

    def reset(self, matrix):
        restored = _matrix(matrix, "reset state", len(self.matrix), len(self.matrix[0]))
        self.matrix = [list(row) for row in restored]

    def clone(self):
        return GatedDeltaState(self.checkpoint())
