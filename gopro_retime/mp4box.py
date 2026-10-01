"""Minimal, lossless ISO BMFF (MP4/QuickTime) box tree parser and writer.

Design goals:
  * Byte-exact round trip: parse(serialize(tree)) == original bytes for every box we do not touch.
  * Containers are recursed into only where the GoPro firmware writes containers; everything else is kept as an
    opaque payload so unknown/private boxes survive untouched.
  * Small helpers for the handful of boxes whose content we must rewrite (sample tables, headers, durations).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterator, Optional

# Boxes that are pure containers (children start right after the 8/16 byte header).
_CONTAINERS = {
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"edts", b"dinf", b"tref", b"gmhd", b"mvex",
    b"moof", b"traf", b"mfra", b"skip", b"sinf", b"schi", b"wave",
}
# Containers whose children follow a FullBox header (4 bytes version/flags).
_FULL_CONTAINERS = {b"meta"}
# 'tmcd' is a container (holding 'tcmi') only when it is a child of 'gmhd'; in 'tref' it is a leaf.
_CONTEXT_CONTAINERS = {(b"gmhd", b"tmcd")}


def is_container(typ: bytes, parent: Optional[bytes]) -> bool:
    if (parent, typ) in _CONTEXT_CONTAINERS:
        return True
    if parent == b"tref":
        return False
    return typ in _CONTAINERS or typ in _FULL_CONTAINERS


@dataclass
class Box:
    type: bytes
    data: bytes = b""                       # leaf payload (bytes after the header); for FULL containers: the 4 header bytes
    children: Optional[list["Box"]] = None  # None => leaf
    largesize: bool = False                 # header uses 64-bit size (16 byte header)
    offset: int = -1                        # original absolute offset of the box header (parse only)
    size: int = -1                          # original total size (parse only)
    parent: Optional["Box"] = field(default=None, repr=False, compare=False)

    # ---- tree helpers -------------------------------------------------------------------------
    @property
    def is_container(self) -> bool:
        return self.children is not None

    @property
    def header_size(self) -> int:
        return 16 if self.largesize else 8

    @property
    def payload_offset(self) -> int:
        """Absolute offset of the payload in the ORIGINAL file (parse only)."""
        return self.offset + self.header_size + (len(self.data) if self.is_container else 0)

    def find(self, path: str) -> Optional["Box"]:
        """Find first box by '/'-separated path of 4cc, e.g. 'moov/trak/mdia/mdhd'. '*' matches any 4cc."""
        parts = path.split("/")
        node = self
        for p in parts:
            if node.children is None:
                return None
            nxt = None
            for c in node.children:
                if p == "*" or c.type == p.encode("latin1"):
                    nxt = c
                    break
            if nxt is None:
                return None
            node = nxt
        return node

    def find_all(self, path: str) -> list["Box"]:
        parts = path.split("/")
        nodes = [self]
        for p in parts:
            nxt: list[Box] = []
            for n in nodes:
                if n.children is None:
                    continue
                for c in n.children:
                    if p == "*" or c.type == p.encode("latin1"):
                        nxt.append(c)
            nodes = nxt
        return nodes

    def child(self, typ: str) -> Optional["Box"]:
        if self.children is None:
            return None
        for c in self.children:
            if c.type == typ.encode("latin1"):
                return c
        return None

    def children_of_type(self, typ: str) -> list["Box"]:
        if self.children is None:
            return []
        return [c for c in self.children if c.type == typ.encode("latin1")]

    def replace_child(self, typ: str, new: "Box") -> None:
        assert self.children is not None
        for i, c in enumerate(self.children):
            if c.type == typ.encode("latin1"):
                new.parent = self
                self.children[i] = new
                return
        raise KeyError(typ)

    def remove_child(self, box: "Box") -> None:
        assert self.children is not None
        self.children.remove(box)

    def walk(self, depth: int = 0) -> Iterator[tuple[int, "Box"]]:
        yield depth, self
        if self.children:
            for c in self.children:
                yield from c.walk(depth + 1)

    def tree_string(self, with_offsets: bool = True) -> str:
        lines = []
        for d, b in self.walk():
            extra = f" size={b.size} off={b.offset}" if with_offsets and b.size >= 0 else f" size={b.serialized_size()}"
            lines.append("  " * d + b.type.decode("latin1") + extra)
        return "\n".join(lines)

    # ---- serialization ------------------------------------------------------------------------
    def payload_size(self) -> int:
        if self.children is None:
            return len(self.data)
        return len(self.data) + sum(c.serialized_size() for c in self.children)

    def serialized_size(self) -> int:
        n = self.payload_size()
        hdr = 16 if (self.largesize or n + 8 > 0xFFFFFFFF) else 8
        return hdr + n

    def serialize(self) -> bytes:
        return b"".join(self.iter_serialize())

    def iter_serialize(self) -> Iterator[bytes]:
        n = self.payload_size()
        total = n + 8
        if self.largesize or total > 0xFFFFFFFF:
            yield struct.pack(">I4sQ", 1, self.type, n + 16)
        else:
            yield struct.pack(">I4s", total, self.type)
        if self.data:
            yield self.data
        if self.children:
            for c in self.children:
                yield from c.iter_serialize()


class ParseError(ValueError):
    pass


def parse_boxes(buf: bytes, start: int = 0, end: Optional[int] = None, parent: Optional[Box] = None,
                base_offset: int = 0) -> list[Box]:
    """Parse a sequence of boxes from buf[start:end]. Offsets recorded are base_offset + position."""
    if end is None:
        end = len(buf)
    out: list[Box] = []
    pos = start
    ptype = parent.type if parent is not None else None
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8])
        hsz = 8
        large = False
        if size == 1:
            if pos + 16 > end:
                raise ParseError(f"truncated largesize box at {pos}")
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]
            hsz = 16
            large = True
        elif size == 0:
            size = end - pos
        if size < hsz or pos + size > end:
            raise ParseError(f"bad box size {size} for {typ!r} at {base_offset + pos} (end={end})")
        box = Box(type=typ, largesize=large, offset=base_offset + pos, size=size, parent=parent)
        if is_container(typ, ptype):
            cstart = pos + hsz
            if typ in _FULL_CONTAINERS:
                box.data = bytes(buf[cstart:cstart + 4])
                cstart += 4
            box.children = parse_boxes(buf, cstart, pos + size, parent=box, base_offset=base_offset)
        else:
            box.data = bytes(buf[pos + hsz:pos + size])
        out.append(box)
        pos += size
    if pos != end:
        raise ParseError(f"trailing {end - pos} bytes after boxes at {base_offset + pos}")
    return out


@dataclass
class TopLevel:
    """Top-level layout of a file: list of boxes; 'mdat' payloads are NOT loaded (only offset/size)."""
    boxes: list[Box]
    path: str

    def find(self, typ: str) -> Optional[Box]:
        for b in self.boxes:
            if b.type == typ.encode("latin1"):
                return b
        return None

    def find_all(self, typ: str) -> list[Box]:
        return [b for b in self.boxes if b.type == typ.encode("latin1")]


def parse_file(path: str) -> TopLevel:
    """Parse top-level boxes of a file. moov/udta/etc are parsed fully; mdat is recorded as an empty leaf
    whose .offset/.size describe where the payload lives (payload_offset = offset + header_size)."""
    boxes: list[Box] = []
    with open(path, "rb") as f:
        f.seek(0, 2)
        fsize = f.tell()
        pos = 0
        while pos + 8 <= fsize:
            f.seek(pos)
            hdr = f.read(16)
            size, typ = struct.unpack(">I4s", hdr[:8])
            hsz = 8
            large = False
            if size == 1:
                size = struct.unpack(">Q", hdr[8:16])[0]
                hsz = 16
                large = True
            elif size == 0:
                size = fsize - pos
            if size < hsz or pos + size > fsize:
                raise ParseError(f"bad top-level box {typ!r} at {pos} size {size}")
            if typ == b"mdat":
                box = Box(type=typ, largesize=large, offset=pos, size=size)
            else:
                f.seek(pos)
                raw = f.read(size)
                box = parse_boxes(raw, 0, size, base_offset=pos)[0]
            boxes.append(box)
            pos += size
    return TopLevel(boxes=boxes, path=path)


# ---- FullBox helpers ----------------------------------------------------------------------------
def full_header(data: bytes) -> tuple[int, int]:
    return data[0], int.from_bytes(data[1:4], "big")


def u32(b: bytes, off: int) -> int:
    return struct.unpack(">I", b[off:off + 4])[0]


def u64(b: bytes, off: int) -> int:
    return struct.unpack(">Q", b[off:off + 8])[0]


def u16(b: bytes, off: int) -> int:
    return struct.unpack(">H", b[off:off + 2])[0]


def put_u32(b: bytearray, off: int, v: int) -> None:
    b[off:off + 4] = struct.pack(">I", v)


def put_u64(b: bytearray, off: int, v: int) -> None:
    b[off:off + 8] = struct.pack(">Q", v)


# ---- mvhd / tkhd / mdhd --------------------------------------------------------------------------
@dataclass
class MvhdInfo:
    version: int
    creation_time: int
    modification_time: int
    timescale: int
    duration: int
    next_track_id: int


def parse_mvhd(b: Box) -> MvhdInfo:
    d = b.data
    v = d[0]
    if v == 1:
        return MvhdInfo(1, u64(d, 4), u64(d, 12), u32(d, 20), u64(d, 24), u32(d, len(d) - 4))
    return MvhdInfo(0, u32(d, 4), u32(d, 8), u32(d, 12), u32(d, 16), u32(d, len(d) - 4))


def set_mvhd_duration(b: Box, duration: int) -> None:
    d = bytearray(b.data)
    if d[0] == 1:
        put_u64(d, 24, duration)
    else:
        put_u32(d, 16, duration)
    b.data = bytes(d)


def set_mvhd_times(b: Box, creation: int, modification: int) -> None:
    d = bytearray(b.data)
    if d[0] == 1:
        put_u64(d, 4, creation); put_u64(d, 12, modification)
    else:
        put_u32(d, 4, creation); put_u32(d, 8, modification)
    b.data = bytes(d)


@dataclass
class TkhdInfo:
    version: int
    flags: int
    creation_time: int
    modification_time: int
    track_id: int
    duration: int
    width: float
    height: float


def parse_tkhd(b: Box) -> TkhdInfo:
    d = b.data
    v, fl = full_header(d)
    if v == 1:
        ct, mt, tid, dur = u64(d, 4), u64(d, 12), u32(d, 20), u64(d, 28)
        rest = 36
    else:
        ct, mt, tid, dur = u32(d, 4), u32(d, 8), u32(d, 12), u32(d, 20)
        rest = 24
    w = u32(d, len(d) - 8) / 65536.0
    h = u32(d, len(d) - 4) / 65536.0
    return TkhdInfo(v, fl, ct, mt, tid, dur, w, h)


def set_tkhd_duration(b: Box, duration: int) -> None:
    d = bytearray(b.data)
    if d[0] == 1:
        put_u64(d, 28, duration)
    else:
        put_u32(d, 20, duration)
    b.data = bytes(d)


def set_tkhd_times(b: Box, creation: int, modification: int) -> None:
    d = bytearray(b.data)
    if d[0] == 1:
        put_u64(d, 4, creation); put_u64(d, 12, modification)
    else:
        put_u32(d, 4, creation); put_u32(d, 8, modification)
    b.data = bytes(d)


@dataclass
class MdhdInfo:
    version: int
    creation_time: int
    modification_time: int
    timescale: int
    duration: int
    language: int


def parse_mdhd(b: Box) -> MdhdInfo:
    d = b.data
    if d[0] == 1:
        return MdhdInfo(1, u64(d, 4), u64(d, 12), u32(d, 20), u64(d, 24), u16(d, 32))
    return MdhdInfo(0, u32(d, 4), u32(d, 8), u32(d, 12), u32(d, 16), u16(d, 20))


def set_mdhd(b: Box, timescale: Optional[int] = None, duration: Optional[int] = None) -> None:
    d = bytearray(b.data)
    if d[0] == 1:
        if timescale is not None: put_u32(d, 20, timescale)
        if duration is not None: put_u64(d, 24, duration)
    else:
        if timescale is not None: put_u32(d, 12, timescale)
        if duration is not None: put_u32(d, 16, duration)
    b.data = bytes(d)


def set_mdhd_times(b: Box, creation: int, modification: int) -> None:
    d = bytearray(b.data)
    if d[0] == 1:
        put_u64(d, 4, creation); put_u64(d, 12, modification)
    else:
        put_u32(d, 4, creation); put_u32(d, 8, modification)
    b.data = bytes(d)


def hdlr_type(b: Box) -> bytes:
    return b.data[8:12]


def hdlr_name(b: Box) -> str:
    """Handler name; supports QuickTime pascal strings and ISO null-terminated strings."""
    raw = b.data[24:]
    if raw and raw[0] == len(raw) - 1:
        return raw[1:].decode("latin1")
    return raw.rstrip(b"\x00").decode("latin1")


# ---- edit list --------------------------------------------------------------------------------
@dataclass
class ElstEntry:
    segment_duration: int
    media_time: int
    media_rate: int  # 16.16 fixed


def parse_elst(b: Box) -> list[ElstEntry]:
    d = b.data
    v = d[0]
    n = u32(d, 4)
    out = []
    off = 8
    for _ in range(n):
        if v == 1:
            sd, mt = u64(d, off), struct.unpack(">q", d[off + 8:off + 16])[0]
            rate = u32(d, off + 16); off += 20
        else:
            sd, mt = u32(d, off), struct.unpack(">i", d[off + 4:off + 8])[0]
            rate = u32(d, off + 8); off += 12
        out.append(ElstEntry(sd, mt, rate))
    return out


def build_elst(entries: list[ElstEntry], version: int = 0) -> Box:
    parts = [bytes([version, 0, 0, 0]), struct.pack(">I", len(entries))]
    for e in entries:
        if version == 1:
            parts.append(struct.pack(">QqI", e.segment_duration, e.media_time, e.media_rate))
        else:
            parts.append(struct.pack(">IiI", e.segment_duration, e.media_time, e.media_rate))
    return Box(type=b"elst", data=b"".join(parts))


# ---- sample tables -----------------------------------------------------------------------------
def parse_stts(b: Box) -> list[tuple[int, int]]:
    d = b.data
    n = u32(d, 4)
    return [struct.unpack(">II", d[8 + 8 * i:16 + 8 * i]) for i in range(n)]


def build_stts(entries: list[tuple[int, int]]) -> Box:
    # merge consecutive equal durations
    merged: list[list[int]] = []
    for cnt, dur in entries:
        if cnt == 0:
            continue
        if merged and merged[-1][1] == dur:
            merged[-1][0] += cnt
        else:
            merged.append([cnt, dur])
    data = b"\x00\x00\x00\x00" + struct.pack(">I", len(merged)) + b"".join(struct.pack(">II", c, d) for c, d in merged)
    return Box(type=b"stts", data=data)


def stts_from_durations(durs: list[int]) -> Box:
    return build_stts([(1, d) for d in durs])


def parse_ctts(b: Box) -> list[tuple[int, int]]:
    d = b.data
    v = d[0]
    n = u32(d, 4)
    fmt = ">Ii" if v == 1 else ">II"
    out = []
    for i in range(n):
        c, o = struct.unpack(fmt, d[8 + 8 * i:16 + 8 * i])
        if v == 0 and o >= 0x80000000:  # some writers store negative offsets in v0
            o -= 0x100000000
        out.append((c, o))
    return out


def build_ctts(offsets: list[int], version: int = 0) -> Box:
    merged: list[list[int]] = []
    for o in offsets:
        if merged and merged[-1][1] == o:
            merged[-1][0] += 1
        else:
            merged.append([1, o])
    fmt = ">Ii" if version == 1 else ">II"
    data = bytes([version, 0, 0, 0]) + struct.pack(">I", len(merged)) + b"".join(
        struct.pack(fmt, c, (o if version == 1 else o & 0xFFFFFFFF)) for c, o in merged)
    return Box(type=b"ctts", data=data)


def parse_stsc(b: Box) -> list[tuple[int, int, int]]:
    d = b.data
    n = u32(d, 4)
    return [struct.unpack(">III", d[8 + 12 * i:20 + 12 * i]) for i in range(n)]


def build_stsc(entries: list[tuple[int, int, int]]) -> Box:
    data = b"\x00\x00\x00\x00" + struct.pack(">I", len(entries)) + b"".join(struct.pack(">III", *e) for e in entries)
    return Box(type=b"stsc", data=data)


def parse_stsz(b: Box) -> list[int]:
    d = b.data
    ss = u32(d, 4)
    n = u32(d, 8)
    if ss != 0:
        return [ss] * n
    return list(struct.unpack(f">{n}I", d[12:12 + 4 * n]))


def build_stsz(sizes: list[int], force_table: bool = True) -> Box:
    if not force_table and sizes and all(s == sizes[0] for s in sizes):
        data = b"\x00\x00\x00\x00" + struct.pack(">II", sizes[0], len(sizes))
    else:
        data = b"\x00\x00\x00\x00" + struct.pack(">II", 0, len(sizes)) + struct.pack(f">{len(sizes)}I", *sizes)
    return Box(type=b"stsz", data=data)


def parse_stco(b: Box) -> list[int]:
    d = b.data
    n = u32(d, 4)
    if b.type == b"co64":
        return list(struct.unpack(f">{n}Q", d[8:8 + 8 * n]))
    return list(struct.unpack(f">{n}I", d[8:8 + 4 * n]))


def build_stco(offsets: list[int], force64: bool = False) -> Box:
    if force64 or (offsets and max(offsets) > 0xFFFFFFFF):
        data = b"\x00\x00\x00\x00" + struct.pack(">I", len(offsets)) + struct.pack(f">{len(offsets)}Q", *offsets)
        return Box(type=b"co64", data=data)
    data = b"\x00\x00\x00\x00" + struct.pack(">I", len(offsets)) + struct.pack(f">{len(offsets)}I", *offsets)
    return Box(type=b"stco", data=data)


def parse_stss(b: Box) -> list[int]:
    d = b.data
    n = u32(d, 4)
    return list(struct.unpack(f">{n}I", d[8:8 + 4 * n]))


def build_stss(sample_numbers_1based: list[int]) -> Box:
    data = b"\x00\x00\x00\x00" + struct.pack(">I", len(sample_numbers_1based)) + struct.pack(
        f">{len(sample_numbers_1based)}I", *sample_numbers_1based)
    return Box(type=b"stss", data=data)


# ---- stsd ---------------------------------------------------------------------------------------
@dataclass
class SampleEntry:
    format: bytes
    fixed: bytes            # bytes after the 8-byte (size+format) header up to the first child box
    children: list[Box]     # extension boxes (avcC, hvcC, colr, pasp, esds, ...)
    raw: bytes              # original full entry bytes

    def serialize(self) -> bytes:
        body = self.fixed + b"".join(c.serialize() for c in self.children)
        return struct.pack(">I4s", 8 + len(body), self.format) + body


_VIDEO_FORMATS = {b"avc1", b"avc3", b"hvc1", b"hev1", b"mp4v", b"apcn", b"ap4h", b"av01", b"vp09"}
_AUDIO_FORMATS = {b"mp4a", b"ac-3", b"ec-3", b"twos", b"sowt", b"lpcm", b"Opus", b"fLaC", b"alac"}


def _fixed_len(fmt: bytes, entry: bytes) -> int:
    """Length of the fixed (non-box) part of a sample entry after the 8 byte header."""
    if fmt in _VIDEO_FORMATS:
        return 78
    if fmt in _AUDIO_FORMATS:
        version = u16(entry, 16)
        if version == 0:
            return 28
        if version == 1:
            return 28 + 16
        if version == 2:
            return 28 + 36 + 0  # 64-byte v2 header
        return 28
    if fmt == b"tmcd":
        return 26
    # unknown (gpmd, fdsc, ...): everything is fixed unless it parses as boxes cleanly
    return len(entry) - 8


def parse_stsd(b: Box) -> list[SampleEntry]:
    d = b.data
    n = u32(d, 4)
    off = 8
    out = []
    for _ in range(n):
        size = u32(d, off)
        fmt = d[off + 4:off + 8]
        entry = d[off:off + size]
        fl = _fixed_len(fmt, entry)
        fixed = entry[8:8 + fl]
        rest = entry[8 + fl:]
        children: list[Box] = []
        if rest:
            try:
                children = parse_boxes(rest, 0, len(rest))
            except ParseError:
                # not boxes: fold into fixed
                fixed = entry[8:]
                children = []
        out.append(SampleEntry(fmt, fixed, children, entry))
        off += size
    return out


def build_stsd(entries: list[SampleEntry], version_flags: bytes = b"\x00\x00\x00\x00") -> Box:
    data = version_flags + struct.pack(">I", len(entries)) + b"".join(e.serialize() for e in entries)
    return Box(type=b"stsd", data=data)


@dataclass
class VideoEntryInfo:
    width: int
    height: int
    compressor_name: str
    depth: int


def video_entry_info(e: SampleEntry) -> VideoEntryInfo:
    f = e.fixed
    w = u16(f, 24)
    h = u16(f, 26)
    name_raw = f[42:74]
    n = name_raw[0]
    name = name_raw[1:1 + n].decode("latin1") if n < 32 else name_raw.rstrip(b"\x00").decode("latin1")
    depth = u16(f, 74)
    return VideoEntryInfo(w, h, name, depth)


@dataclass
class TmcdEntryInfo:
    flags: int
    timescale: int
    frame_duration: int
    number_of_frames: int


def tmcd_entry_info(e: SampleEntry) -> TmcdEntryInfo:
    f = e.fixed
    return TmcdEntryInfo(u32(f, 12), u32(f, 16), u32(f, 20), f[24])


def set_tmcd_entry(e: SampleEntry, timescale: int, frame_duration: int, number_of_frames: int,
                   flags: Optional[int] = None) -> None:
    f = bytearray(e.fixed)
    if flags is not None:
        put_u32(f, 12, flags)
    put_u32(f, 16, timescale)
    put_u32(f, 20, frame_duration)
    f[24] = number_of_frames & 0xFF
    e.fixed = bytes(f)


def clone(box: Box) -> Box:
    """Deep copy of a box tree (parent links are rebuilt)."""
    nb = Box(type=box.type, data=box.data, largesize=box.largesize, offset=box.offset, size=box.size)
    if box.children is not None:
        nb.children = []
        for c in box.children:
            cc = clone(c)
            cc.parent = nb
            nb.children.append(cc)
    return nb
