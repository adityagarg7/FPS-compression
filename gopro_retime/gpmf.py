"""GPMF (GoPro Metadata Format) KLV parser / serializer. Lossless round trip is a hard requirement."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional, Union

# type char -> struct format / size
TYPE_SIZES = {
    b"b": 1, b"B": 1, b"c": 1, b"d": 8, b"f": 4, b"F": 4, b"G": 16, b"j": 8, b"J": 8, b"l": 4, b"L": 4,
    b"q": 4, b"Q": 8, b"s": 2, b"S": 2, b"U": 16, b"?": 1,
}
TYPE_FMT = {
    b"b": "b", b"B": "B", b"d": "d", b"f": "f", b"j": "q", b"J": "Q", b"l": "i", b"L": "I", b"q": "i", b"Q": "q",
    b"s": "h", b"S": "H",
}


@dataclass
class KLV:
    key: bytes                   # 4 chars
    type: int                    # type char code (0 = nested)
    size: int                    # struct size (bytes per sample)
    repeat: int                  # sample count
    data: bytes = b""            # raw payload (without padding) for leaves
    children: Optional[list["KLV"]] = None   # for nested (type 0)
    pad_bytes: bytes = b""       # the original padding bytes after data (GoPro leaves garbage here; kept for round trip)

    @property
    def is_nested(self) -> bool:
        return self.type == 0

    @property
    def type_char(self) -> bytes:
        return bytes([self.type]) if self.type else b"\x00"

    @property
    def payload_len(self) -> int:
        return self.size * self.repeat

    def child(self, key: Union[bytes, str]) -> Optional["KLV"]:
        if isinstance(key, str):
            key = key.encode("latin1")
        if self.children is None:
            return None
        for c in self.children:
            if c.key == key:
                return c
        return None

    def children_of(self, key: Union[bytes, str]) -> list["KLV"]:
        if isinstance(key, str):
            key = key.encode("latin1")
        return [c for c in (self.children or []) if c.key == key]

    def walk(self, depth: int = 0) -> Iterator[tuple[int, "KLV"]]:
        yield depth, self
        for c in self.children or []:
            yield from c.walk(depth + 1)

    # ---- value decoding (for the simple numeric/text types) --------------------------------------
    def values(self, type_override: Optional[bytes] = None) -> list:
        """Decode samples as a list (each sample = tuple of values, or scalar if one value per sample)."""
        t = type_override or self.type_char
        if t == b"c":
            return [self.data[i * self.size:(i + 1) * self.size] for i in range(self.repeat)]
        if t in TYPE_FMT:
            fmt = TYPE_FMT[t]
            n = self.size // struct.calcsize(fmt)
            out = []
            for i in range(self.repeat):
                vals = struct.unpack(">" + fmt * n, self.data[i * self.size:(i + 1) * self.size])
                out.append(vals[0] if n == 1 else vals)
            return out
        return [self.data[i * self.size:(i + 1) * self.size] for i in range(self.repeat)]

    def serialize(self) -> bytes:
        if self.is_nested:
            body = b"".join(c.serialize() for c in self.children or [])
            # nested: size=1, repeat=len(body) is the convention (struct size 1 byte, repeat = total bytes)
            size = 1
            if len(body) > 0xFFFF:
                size = 4
                while len(body) // size > 0xFFFF:
                    size *= 2
                if len(body) % size:
                    body += b"\x00" * (size - len(body) % size)
            repeat = len(body) // size
            hdr = self.key + b"\x00" + bytes([size]) + struct.pack(">H", repeat)
            return hdr + body
        hdr = self.key + bytes([self.type, self.size]) + struct.pack(">H", self.repeat)
        body = self.data
        padlen = (-len(body)) % 4
        pad = self.pad_bytes if len(self.pad_bytes) == padlen else b"\x00" * padlen
        return hdr + body + pad


class GPMFError(ValueError):
    pass


def parse_with_trailing(buf: bytes, start: int = 0, end: Optional[int] = None) -> tuple[list[KLV], bytes]:
    """Parse KLVs; stops at a zero key (end marker) and returns (klvs, trailing_bytes)."""
    if end is None:
        end = len(buf)
    out: list[KLV] = []
    pos = start
    while pos + 8 <= end:
        key = buf[pos:pos + 4]
        if key == b"\x00\x00\x00\x00":
            break
        typ = buf[pos + 4]
        size = buf[pos + 5]
        repeat = struct.unpack(">H", buf[pos + 6:pos + 8])[0]
        plen = size * repeat
        padded = (plen + 3) & ~3
        if pos + 8 + plen > end:
            raise GPMFError(f"KLV {key!r} at {pos} overruns payload ({plen} bytes, end {end})")
        if pos + 8 + padded > end:
            padded = end - pos - 8  # unpadded last element (seen in some udta GPMF blobs)
        if typ == 0:
            k = KLV(key, 0, size, repeat, b"", parse(buf, pos + 8, pos + 8 + plen))
        else:
            k = KLV(key, typ, size, repeat, bytes(buf[pos + 8:pos + 8 + plen]), None,
                    bytes(buf[pos + 8 + plen:pos + 8 + padded]))
        out.append(k)
        pos += 8 + padded
    return out, bytes(buf[pos:end])


def parse(buf: bytes, start: int = 0, end: Optional[int] = None) -> list[KLV]:
    klvs, _trailing = parse_with_trailing(buf, start, end)
    return klvs


def serialize(klvs: list[KLV]) -> bytes:
    return b"".join(k.serialize() for k in klvs)


def make(key: Union[bytes, str], type_char: Union[bytes, str], size: int, data: bytes) -> KLV:
    if isinstance(key, str):
        key = key.encode("latin1")
    if isinstance(type_char, str):
        type_char = type_char.encode("latin1")
    if size == 0:
        raise GPMFError("size must be > 0")
    if len(data) % size:
        raise GPMFError("data length not a multiple of struct size")
    return KLV(key, type_char[0], size, len(data) // size, data)


def make_nested(key: Union[bytes, str], children: list[KLV]) -> KLV:
    if isinstance(key, str):
        key = key.encode("latin1")
    return KLV(key, 0, 1, 0, b"", children)


def tree_string(klvs: list[KLV], max_vals: int = 6) -> str:
    lines = []
    for k in klvs:
        for d, n in k.walk():
            if n.is_nested:
                lines.append("  " * d + f"{n.key.decode('latin1')} (nested, {len(n.children or [])} children)")
            else:
                vals = n.values()
                shown = vals[:max_vals]
                if n.type_char == b"c":
                    shown = [v.rstrip(b"\x00").decode("latin1", "replace") for v in shown]
                lines.append("  " * d + f"{n.key.decode('latin1')} type={n.type_char.decode('latin1')} size={n.size} repeat={n.repeat} {shown}{'...' if len(vals) > max_vals else ''}")
    return "\n".join(lines)
