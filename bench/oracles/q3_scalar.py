"""Independent scalar oracle for the locked 128 x signed-int3 storage contract."""

from fractions import Fraction

GROUP_SIZE = 128
PACKED_BYTES = 48


def _checked_codes(codes):
    try:
        values = tuple(codes)
    except TypeError as exc:
        raise TypeError("codes must be an iterable of integers") from exc
    for code in values:
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("codes must be integers")
        if code < -4 or code > 3:
            raise ValueError("signed three-bit code outside [-4, 3]")
    return values


def pack_signed3(codes):
    """Pack 128 codes into 48 bytes, code 0 in least-significant bits first.

    Each signed code is two's-complement in [-4, 3]. Bytes are emitted in
    increasing address order; the bit stream advances low bit to high bit.
    """
    values = _checked_codes(codes)
    if len(values) != GROUP_SIZE:
        raise ValueError("a group must contain exactly 128 codes")
    packed = bytearray(PACKED_BYTES)
    for index, code in enumerate(values):
        bits = code & 0b111
        bit_offset = index * 3
        byte_index, shift = divmod(bit_offset, 8)
        packed[byte_index] |= (bits << shift) & 0xFF
        if shift > 5:
            packed[byte_index + 1] |= bits >> (8 - shift)
    return bytes(packed)


def unpack_signed3(packed):
    """Decode one 48-byte group using low-bit-first, two's-complement rules."""
    if not isinstance(packed, (bytes, bytearray, memoryview)):
        raise TypeError("packed group must be bytes-like")
    try:
        view = memoryview(packed).cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError("packed group must be a contiguous byte buffer") from exc
    if len(view) != PACKED_BYTES:
        raise ValueError("packed group must contain exactly 48 bytes")
    decoded = []
    for index in range(GROUP_SIZE):
        bit_offset = index * 3
        byte_index, shift = divmod(bit_offset, 8)
        word = view[byte_index]
        if shift > 5:
            word |= view[byte_index + 1] << 8
        raw = (word >> shift) & 0b111
        decoded.append(raw - 8 if raw & 0b100 else raw)
    return tuple(decoded)


def fp16_fraction(bits):
    """Return the exact rational represented by finite IEEE-754 binary16 bits."""
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise TypeError("FP16 bits must be an integer")
    if not 0 <= bits <= 0xFFFF:
        raise ValueError("FP16 bit pattern must fit uint16")
    exponent = (bits >> 10) & 0x1F
    fraction = bits & 0x3FF
    sign = -1 if bits & 0x8000 else 1
    if exponent == 0x1F:
        raise ValueError("nonfinite FP16 has no exact finite Fraction")
    if exponent == 0:
        magnitude = Fraction(fraction, 1 << 24)
    else:
        magnitude = Fraction(1024 + fraction, 1024) * Fraction(2) ** (exponent - 15)
    return sign * magnitude


def represented_values(codes, scale_bits):
    """Return code * finite binary16 scale exactly; codes must be in [-4, 3]."""
    values = _checked_codes(codes)
    scale = fp16_fraction(scale_bits)
    return tuple(Fraction(code) * scale for code in values)


def dot_exact(left, right):
    """Exact dot product over integers and Fractions; bools and floats are rejected."""
    try:
        lhs, rhs = tuple(left), tuple(right)
    except TypeError as exc:
        raise TypeError("dot operands must be iterable") from exc
    if len(lhs) != len(rhs):
        raise ValueError("dot operands must have equal lengths")
    allowed = (int, Fraction)
    for value in lhs + rhs:
        if isinstance(value, bool) or not isinstance(value, allowed):
            raise TypeError("dot operands must contain only integers or Fractions")
    return sum((Fraction(a) * Fraction(b) for a, b in zip(lhs, rhs)), Fraction())
