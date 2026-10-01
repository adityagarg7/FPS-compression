"""NAL unit splitting/joining for Annex B and length-prefixed (MP4) H.264 / HEVC streams, SEI stripping,
and avcC / hvcC parsing & building."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator, Optional

# H.264 nal_unit_type values
H264_SLICE = 1
H264_IDR = 5
H264_SEI = 6
H264_SPS = 7
H264_PPS = 8
H264_AUD = 9
H264_END_SEQ = 10
H264_END_STREAM = 11
H264_FILLER = 12
H264_SPS_EXT = 13
H264_PREFIX = 14
H264_SUBSET_SPS = 15

# HEVC nal_unit_type values
HEVC_VPS = 32
HEVC_SPS = 33
HEVC_PPS = 34
HEVC_AUD = 35
HEVC_EOS = 36
HEVC_EOB = 37
HEVC_FD = 38
HEVC_PREFIX_SEI = 39
HEVC_SUFFIX_SEI = 40
HEVC_IDR_W_RADL = 19
HEVC_IDR_N_LP = 20
HEVC_CRA = 21
HEVC_BLA_W_LP = 16
HEVC_BLA_W_RADL = 17
HEVC_BLA_N_LP = 18


def h264_nal_type(nal: bytes) -> int:
    return nal[0] & 0x1F


def hevc_nal_type(nal: bytes) -> int:
    return (nal[0] >> 1) & 0x3F


def nal_type(nal: bytes, codec: str) -> int:
    return h264_nal_type(nal) if codec == "h264" else hevc_nal_type(nal)


def is_vcl(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return (1 <= t <= 5) if codec == "h264" else (t <= 31)


def is_irap(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return t == H264_IDR if codec == "h264" else (16 <= t <= 23)


def is_idr(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return t == H264_IDR if codec == "h264" else t in (HEVC_IDR_W_RADL, HEVC_IDR_N_LP)


def is_sei(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return t == H264_SEI if codec == "h264" else t in (HEVC_PREFIX_SEI, HEVC_SUFFIX_SEI)


def is_param_set(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return t in (H264_SPS, H264_PPS, H264_SPS_EXT, H264_SUBSET_SPS) if codec == "h264" else t in (HEVC_VPS, HEVC_SPS, HEVC_PPS)


def is_aud(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return t == H264_AUD if codec == "h264" else t == HEVC_AUD


def is_filler(nal: bytes, codec: str) -> bool:
    t = nal_type(nal, codec)
    return t == H264_FILLER if codec == "h264" else t == HEVC_FD


def is_first_slice_of_picture(nal: bytes, codec: str) -> bool:
    """Cheap check whether a VCL NAL starts a new picture (first_mb_in_slice == 0 / first_slice_segment_in_pic_flag)."""
    if not is_vcl(nal, codec):
        return False
    if codec == "h264":
        # first_mb_in_slice is ue(v): value 0 is encoded as a single '1' bit -> top bit of byte 1 set
        return len(nal) > 1 and (nal[1] & 0x80) != 0
    return len(nal) > 2 and (nal[2] & 0x80) != 0


# ---- Annex B ---------------------------------------------------------------------------------------
def split_annexb(buf: bytes) -> list[bytes]:
    """Split an Annex B byte stream into NAL units (without start codes)."""
    out: list[bytes] = []
    n = len(buf)
    i = buf.find(b"\x00\x00\x01")
    while i != -1 and i < n:
        start = i + 3
        j = buf.find(b"\x00\x00\x01", start)
        if j == -1:
            end = n
            nxt = -1
        else:
            end = j
            # trailing zero bytes before the next start code belong to the start code (00 00 00 01) / are trailing_zero
            while end > start and buf[end - 1] == 0:
                end -= 1
            nxt = j
        if end > start:
            out.append(bytes(buf[start:end]))
        i = nxt
    return out


def iter_annexb_file(path: str, chunk: int = 1 << 24) -> Iterator[bytes]:
    """Stream NAL units from a (possibly huge) Annex B file without loading it entirely."""
    with open(path, "rb") as f:
        buf = b""
        while True:
            data = f.read(chunk)
            if not data:
                break
            buf += data
            # find last start code; everything before it can be split safely
            last = buf.rfind(b"\x00\x00\x01")
            if last <= 0:
                continue
            head, buf = buf[:last], buf[last:]
            for nal in split_annexb(head + b"\x00\x00\x01"):
                yield nal
            # the artificial trailing start code yields nothing (empty)
        for nal in split_annexb(buf):
            yield nal


def to_length_prefixed(nals: list[bytes], length_size: int = 4) -> bytes:
    parts = []
    for n in nals:
        parts.append(len(n).to_bytes(length_size, "big"))
        parts.append(n)
    return b"".join(parts)


def split_length_prefixed(sample: bytes, length_size: int = 4) -> list[bytes]:
    out = []
    p = 0
    n = len(sample)
    while p + length_size <= n:
        ln = int.from_bytes(sample[p:p + length_size], "big")
        p += length_size
        out.append(bytes(sample[p:p + ln]))
        p += ln
    return out


def to_annexb(nals: list[bytes]) -> bytes:
    return b"".join(b"\x00\x00\x00\x01" + n for n in nals)


# ---- emulation prevention --------------------------------------------------------------------------
def remove_epb(nal: bytes) -> bytes:
    """Remove emulation prevention bytes (00 00 03 -> 00 00) returning RBSP including the NAL header."""
    out = bytearray()
    zeros = 0
    i = 0
    n = len(nal)
    while i < n:
        b = nal[i]
        if zeros >= 2 and b == 3:
            # skip the 03 if it is followed by 00..03 (or is the last byte)
            if i + 1 >= n or nal[i + 1] <= 3:
                zeros = 0
                i += 1
                continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
        i += 1
    return bytes(out)


def insert_epb(rbsp: bytes) -> bytes:
    out = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


# ---- access unit grouping --------------------------------------------------------------------------
@dataclass
class AccessUnit:
    nals: list[bytes]
    codec: str

    @property
    def vcl(self) -> list[bytes]:
        return [n for n in self.nals if is_vcl(n, self.codec)]

    @property
    def is_irap(self) -> bool:
        v = self.vcl
        return bool(v) and is_irap(v[0], self.codec)

    @property
    def is_idr(self) -> bool:
        v = self.vcl
        return bool(v) and is_idr(v[0], self.codec)


def group_access_units(nals: Iterator[bytes], codec: str) -> Iterator[AccessUnit]:
    """Group a NAL stream into access units. A new AU starts at an AUD, or at the first VCL NAL of a picture
    (preceded by any non-VCL NALs that follow the previous picture's last VCL NAL)."""
    cur: list[bytes] = []
    seen_vcl = False
    for nal in nals:
        t = nal_type(nal, codec)
        starts_new = False
        if is_aud(nal, codec):
            starts_new = True
        elif is_vcl(nal, codec):
            if seen_vcl and is_first_slice_of_picture(nal, codec):
                starts_new = True
        else:
            # non-VCL after VCL (SPS/PPS/prefix SEI/VPS) belongs to the NEXT AU, except suffix SEI / EOS / EOB / FD
            if seen_vcl:
                if codec == "hevc" and t in (HEVC_SUFFIX_SEI, HEVC_EOS, HEVC_EOB, HEVC_FD):
                    starts_new = False
                elif codec == "h264" and t in (H264_END_SEQ, H264_END_STREAM, H264_FILLER):
                    starts_new = False
                else:
                    starts_new = True
        if starts_new and cur:
            yield AccessUnit(cur, codec)
            cur = []
            seen_vcl = False
        cur.append(nal)
        if is_vcl(nal, codec):
            seen_vcl = True
    if cur:
        yield AccessUnit(cur, codec)


# ---- avcC / hvcC -----------------------------------------------------------------------------------
@dataclass
class AvcC:
    configuration_version: int
    profile_idc: int
    profile_compat: int
    level_idc: int
    length_size: int
    sps: list[bytes]
    pps: list[bytes]
    trailing: bytes = b""   # chroma_format/bit depth extension for high profiles (kept verbatim)


def parse_avcc(d: bytes) -> AvcC:
    ver, prof, compat, level = d[0], d[1], d[2], d[3]
    length_size = (d[4] & 3) + 1
    n_sps = d[5] & 0x1F
    off = 6
    sps = []
    for _ in range(n_sps):
        ln = struct.unpack(">H", d[off:off + 2])[0]
        sps.append(d[off + 2:off + 2 + ln]); off += 2 + ln
    n_pps = d[off]; off += 1
    pps = []
    for _ in range(n_pps):
        ln = struct.unpack(">H", d[off:off + 2])[0]
        pps.append(d[off + 2:off + 2 + ln]); off += 2 + ln
    return AvcC(ver, prof, compat, level, length_size, sps, pps, d[off:])


def build_avcc(c: AvcC) -> bytes:
    out = bytearray([c.configuration_version, c.profile_idc, c.profile_compat, c.level_idc,
                     0xFC | (c.length_size - 1), 0xE0 | len(c.sps)])
    for s in c.sps:
        out += struct.pack(">H", len(s)) + s
    out.append(len(c.pps))
    for p in c.pps:
        out += struct.pack(">H", len(p)) + p
    out += c.trailing
    return bytes(out)


@dataclass
class HvcC:
    head: bytes                      # the 22 bytes before numOfArrays (kept verbatim; contains profile/level/etc.)
    arrays: list[tuple[int, int, list[bytes]]]  # (array_completeness, nal_unit_type, nal units)

    @property
    def length_size(self) -> int:
        return (self.head[21] & 3) + 1

    def nals_of(self, t: int) -> list[bytes]:
        for _c, nt, nals in self.arrays:
            if nt == t:
                return nals
        return []


def parse_hvcc(d: bytes) -> HvcC:
    head = d[:22]
    n_arrays = d[22]
    off = 23
    arrays = []
    for _ in range(n_arrays):
        b = d[off]; off += 1
        completeness = b >> 7
        nt = b & 0x3F
        cnt = struct.unpack(">H", d[off:off + 2])[0]; off += 2
        nals = []
        for _ in range(cnt):
            ln = struct.unpack(">H", d[off:off + 2])[0]; off += 2
            nals.append(d[off:off + ln]); off += ln
        arrays.append((completeness, nt, nals))
    return HvcC(head, arrays)


def build_hvcc(c: HvcC) -> bytes:
    out = bytearray(c.head)
    out.append(len(c.arrays))
    for completeness, nt, nals in c.arrays:
        out.append((completeness << 7) | (nt & 0x3F))
        out += struct.pack(">H", len(nals))
        for n in nals:
            out += struct.pack(">H", len(n)) + n
    return bytes(out)


def parameter_sets_from_entry_children(children, codec: str) -> dict[str, list[bytes]]:
    """Extract {'vps','sps','pps'} lists from avcC / hvcC child boxes of a sample entry."""
    for c in children:
        if codec == "h264" and c.type == b"avcC":
            a = parse_avcc(c.data)
            return {"sps": a.sps, "pps": a.pps}
        if codec == "hevc" and c.type == b"hvcC":
            h = parse_hvcc(c.data)
            return {"vps": h.nals_of(HEVC_VPS), "sps": h.nals_of(HEVC_SPS), "pps": h.nals_of(HEVC_PPS)}
    return {}
