"""Symmetric bit-level syntax coder.

A single syntax function (e.g. h264.sps(io, f)) describes the bitstream once; the same function parses when `io`
is in read mode and serialises when it is in write mode, so reading and writing can never drift apart.
Values live in a plain dict `f` (arrays as lists / nested dicts)."""
from __future__ import annotations

from typing import Optional


class BitError(ValueError):
    pass


class BitIO:
    def __init__(self, data: Optional[bytes] = None):
        self.reading = data is not None
        self.data = data or b""
        self.pos = 0                 # bit position (read)
        self._out: list[int] = []    # bits (write)

    # ---- raw -------------------------------------------------------------------------------------
    def read_bits(self, n: int) -> int:
        if n == 0:
            return 0
        if self.pos + n > len(self.data) * 8:
            raise BitError(f"read past end at bit {self.pos} (+{n}), length {len(self.data) * 8}")
        v = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3]
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def peek_bits(self, n: int) -> int:
        p = self.pos
        try:
            return self.read_bits(n)
        finally:
            self.pos = p

    def write_bits(self, v: int, n: int) -> None:
        if v < 0 or (n < 64 and v >= (1 << n)):
            raise BitError(f"value {v} does not fit in {n} bits")
        for i in range(n - 1, -1, -1):
            self._out.append((v >> i) & 1)

    @property
    def bitpos(self) -> int:
        return self.pos if self.reading else len(self._out)

    def byte_aligned(self) -> bool:
        return self.bitpos % 8 == 0

    def bits_left(self) -> int:
        return len(self.data) * 8 - self.pos

    def more_rbsp_data(self) -> bool:
        """True if there is more data before the rbsp_trailing_bits (last 1 bit in the buffer)."""
        if not self.reading:
            raise BitError("more_rbsp_data only valid when reading")
        if self.pos >= len(self.data) * 8:
            return False
        # find last set bit
        last = len(self.data) * 8 - 1
        while last >= 0:
            byte = self.data[last >> 3]
            if (byte >> (7 - (last & 7))) & 1:
                break
            last -= 1
        return self.pos < last

    def to_bytes(self) -> bytes:
        bits = self._out
        out = bytearray((len(bits) + 7) // 8)
        for i, b in enumerate(bits):
            if b:
                out[i >> 3] |= 0x80 >> (i & 7)
        return bytes(out)

    # ---- syntax elements (symmetric) -----------------------------------------------------------
    def u(self, n: int, f: dict, name: str) -> int:
        if self.reading:
            f[name] = self.read_bits(n)
        else:
            self.write_bits(int(f[name]), n)
        return f[name]

    def flag(self, f: dict, name: str) -> int:
        return self.u(1, f, name)

    def ue(self, f: dict, name: str) -> int:
        if self.reading:
            f[name] = self.read_ue()
        else:
            self.write_ue(int(f[name]))
        return f[name]

    def se(self, f: dict, name: str) -> int:
        if self.reading:
            f[name] = self.read_se()
        else:
            self.write_se(int(f[name]))
        return f[name]

    def u_list(self, n: int, f: dict, name: str, count: int) -> list[int]:
        if self.reading:
            f[name] = [self.read_bits(n) for _ in range(count)]
        else:
            for v in f[name][:count]:
                self.write_bits(int(v), n)
        return f[name]

    def read_ue(self) -> int:
        zeros = 0
        while self.read_bits(1) == 0:
            zeros += 1
            if zeros > 32:
                raise BitError("invalid exp-golomb code")
        return (1 << zeros) - 1 + (self.read_bits(zeros) if zeros else 0)

    def write_ue(self, v: int) -> None:
        if v < 0:
            raise BitError("ue(v) must be >= 0")
        x = v + 1
        n = x.bit_length()
        self.write_bits(0, n - 1)
        self.write_bits(x, n)

    def read_se(self) -> int:
        k = self.read_ue()
        return (k + 1) // 2 if k % 2 else -(k // 2)

    def write_se(self, v: int) -> None:
        self.write_ue(2 * v - 1 if v > 0 else -2 * v)

    # ---- trailing / alignment --------------------------------------------------------------------
    def rbsp_trailing_bits(self) -> None:
        if self.reading:
            if self.read_bits(1) != 1:
                raise BitError("rbsp_stop_one_bit missing")
            while not self.byte_aligned():
                if self.read_bits(1) != 0:
                    raise BitError("rbsp_alignment_zero_bit not zero")
        else:
            self.write_bits(1, 1)
            while not self.byte_aligned():
                self.write_bits(0, 1)

    def align_with(self, bit: int) -> None:
        """Consume/emit `bit`s until byte aligned (cabac_alignment_one_bit = 1, byte_alignment zeros = 0)."""
        if self.reading:
            while not self.byte_aligned():
                self.read_bits(1)
        else:
            while not self.byte_aligned():
                self.write_bits(bit, 1)

    def byte_alignment(self) -> None:
        """HEVC byte_alignment(): a 1 followed by zeros up to the byte boundary."""
        if self.reading:
            if self.read_bits(1) != 1:
                raise BitError("alignment_bit_equal_to_one missing")
            while not self.byte_aligned():
                self.read_bits(1)
        else:
            self.write_bits(1, 1)
            while not self.byte_aligned():
                self.write_bits(0, 1)


def ceil_log2(v: int) -> int:
    return (v - 1).bit_length() if v > 1 else 0
