"""Small stdlib-only mathematical fixtures; conventions are not target qualification."""
import math


def _finite(value, name):
    if not math.isfinite(value):
        raise ValueError(f"{name} produced a nonfinite result")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} values must be real integers or floats, not bool")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the finite float range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vec(values, name):
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite numeric sequence") from exc
    if not raw:
        raise ValueError(f"{name} must be nonempty")
    return tuple(_number(x, name) for x in raw)


def rms_norm(x, weight, epsilon=1e-5):
    x, weight = _vec(x, "x"), _vec(weight, "weight")
    epsilon = _number(epsilon, "epsilon")
    if len(x) != len(weight) or epsilon <= 0:
        raise ValueError("RMSNorm dimensions must match and epsilon must be positive finite")
    peak = max(abs(v) for v in x)
    if peak <= math.sqrt(epsilon):
        denom = math.sqrt(sum(v * v for v in x) / len(x) + epsilon)
        result = tuple((v / denom) * w for v, w in zip(x, weight))
    else:
        scaled_mean = sum((v / peak) ** 2 for v in x) / len(x)
        eps_scaled = (math.sqrt(epsilon) / peak) ** 2
        result = tuple((v / peak) / math.sqrt(scaled_mean + eps_scaled) * w for v, w in zip(x, weight))
    return tuple(_finite(y, "RMSNorm") for y in result)


def rope_pair(x, position, theta=1.0):
    x = _vec(x, "x")
    if len(x) != 2 or isinstance(position, bool) or not isinstance(position, int):
        raise ValueError("RoPE fixture accepts one pair and an integer position")
    theta = _number(theta, "theta")
    if theta <= 0:
        raise ValueError("theta must be positive finite")
    try:
        angle = position / theta
    except OverflowError as exc:
        raise ValueError("RoPE angle is outside the finite float range") from exc
    if not math.isfinite(angle):
        raise ValueError("RoPE angle must be finite")
    c, s = math.cos(angle), math.sin(angle)
    return tuple(_finite(y, "RoPE") for y in (x[0] * c - x[1] * s, x[0] * s + x[1] * c))


def _silu_scalar(value):
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return value * exp_value / (1.0 + exp_value)


def silu(x):
    x = _vec(x, "x")
    return tuple(_finite(_silu_scalar(v), "SiLU") for v in x)


def swiglu(gate, up):
    gate, up = _vec(gate, "gate"), _vec(up, "up")
    if len(gate) != len(up):
        raise ValueError("SwiGLU dimensions must match")
    return tuple(_finite(_silu_scalar(g) * u, "SwiGLU") for g, u in zip(gate, up))


def causal_attention(query, keys, values):
    """Single-head causal attention; query row i sees keys/values 0..i."""
    q = tuple(_vec(row, "query row") for row in query)
    k = tuple(_vec(row, "key row") for row in keys)
    v = tuple(_vec(row, "value row") for row in values)
    if not q or len(q) != len(k) or len(k) != len(v):
        raise ValueError("attention requires equal nonempty sequence lengths")
    d = len(q[0])
    if any(len(row) != d for rows in (q, k) for row in rows) or any(len(row) != len(v[0]) for row in v):
        raise ValueError("attention dimensions must be rectangular and compatible")
    scale = math.sqrt(d)
    output = []
    for i, qi in enumerate(q):
        scores = [_finite(sum(a * b for a, b in zip(qi, kj)) / scale, "attention score") for kj in k[:i + 1]]
        peak = max(scores)
        probs = [math.exp(score - peak) for score in scores]
        denom = sum(probs)
        output.append(tuple(_finite(sum(p * vj[c] for p, vj in zip(probs, v[:i + 1])) / denom, "attention output") for c in range(len(v[0]))))
    return tuple(output)


def gated_delta_step(state, key, value, beta):
    """Toy scalar recurrence: s' = s + beta*(v-k*s); y=k*s'."""
    vals = tuple(_number(x, "state/key/value/beta") for x in (state, key, value, beta))
    s, k, v, b = vals
    if not 0.0 <= b <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    updated = _finite(s + b * (v - k * s), "gated-delta state")
    return _finite(k * updated, "gated-delta output"), updated


def causal_depthwise_step(history, sample, kernel):
    """Per-channel valid causal FIR; kernel[0] multiplies current sample."""
    h = tuple(_vec(row, "history row") for row in history)
    x, w = _vec(sample, "sample"), _vec(kernel, "kernel")
    if not w or any(len(row) != len(x) for row in h):
        raise ValueError("convolution channels/kernel must be nonempty and consistent")
    seq = (x,) + h
    y = tuple(_finite(sum(w[j] * seq[j][c] for j in range(min(len(w), len(seq)))), "depthwise convolution") for c in range(len(x)))
    new_history = (x,) + h
    return y, new_history[:len(w) - 1]


class ScalarRecurrenceState:
    """Mutable lifecycle wrapper for the explicitly toy scalar recurrence."""

    def __init__(self, value=0.0):
        self.value = _number(value, "state")

    def step(self, key, value, beta):
        output, self.value = gated_delta_step(self.value, key, value, beta)
        return output

    def checkpoint(self):
        return self.value

    def rollback(self, checkpoint):
        self.value = _number(checkpoint, "checkpoint")

    def reset(self, value=0.0):
        self.value = _number(value, "state")

    def clone(self):
        return ScalarRecurrenceState(self.value)
