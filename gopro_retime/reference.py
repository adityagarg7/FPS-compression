"""Optional NATIVE reference recording (same camera, target frame rate): the source of every frame-rate-dependent
convention we cannot derive from the 50 fps file alone (SETT bits, mode-dependent Global Settings keys such as SROT,
the SOS header's mode fields, metadata payload period, MET write latency, tmcd numberOfFrames, video timescale).

Per-file identity stays with the SOURCE: creation times, MUID, GUMI, HiLight tags, GPS position, the SOS clock fields."""
from __future__ import annotations

import struct
from typing import Optional

from . import gpmf
from . import mp4box as mb
from .model import SourceFile

PER_FILE_UDTA = {b"\xa9xyz", b"free", b"MUID", b"HMMT", b"GUMI"}


def check_compatible(src: SourceFile, ref: SourceFile) -> list[str]:
    problems = []
    for key in (b"FIRM", b"LENS", b"CAME"):
        a, b = src.moov.find(f"udta/{key.decode('latin1')}"), ref.moov.find(f"udta/{key.decode('latin1')}")
        if a is not None and b is not None and a.data != b.data:
            problems.append(f"udta {key.decode()} differs between source and reference (different camera/firmware?)")
    sv, rv = src.video, ref.video
    if sv.format != rv.format:
        problems.append(f"video codec differs: {sv.format!r} vs {rv.format!r}")
    a, b = mb.video_entry_info(sv.stsd_entries[0]), mb.video_entry_info(rv.stsd_entries[0])
    if (a.width, a.height) != (b.width, b.height):
        problems.append(f"resolution differs: {a.width}x{a.height} vs {b.width}x{b.height}")
    if [t.kind for t in src.tracks] != [t.kind for t in ref.tracks]:
        problems.append("track layout differs")
    return problems


def merged_udta(src: SourceFile, ref: SourceFile, log=None) -> mb.Box:
    """Reference udta with the per-file atoms replaced by the source's."""
    su, ru = src.moov.child("udta"), ref.moov.child("udta")
    assert su is not None and ru is not None
    out = mb.clone(ru)
    src_by_type: dict[bytes, list[mb.Box]] = {}
    for c in su.children:
        src_by_type.setdefault(c.type, []).append(c)
    new_children: list[mb.Box] = []
    seen: dict[bytes, int] = {}
    for c in out.children:
        i = seen.get(c.type, 0)
        seen[c.type] = i + 1
        if c.type in PER_FILE_UDTA and c.type in src_by_type and i < len(src_by_type[c.type]):
            # the 30-byte first atom is '©xyz' or 'free' depending on the GPS fix of THIS recording
            if c.type in (b"\xa9xyz", b"free") and i == 0 and su.children and su.children[0].type in (b"\xa9xyz", b"free") and c is out.children[0]:
                new_children.append(mb.clone(su.children[0]))
            else:
                new_children.append(mb.clone(src_by_type[c.type][i]))
        elif c.type == b"GPMF":
            new_children.append(_merged_global_settings(src_by_type.get(b"GPMF", [None])[0], c, log))
        else:
            new_children.append(c)
    out.children = new_children
    for ch in out.children:
        ch.parent = out
    return out


def _merged_global_settings(src_box: Optional[mb.Box], ref_box: mb.Box, log=None) -> mb.Box:
    if src_box is None:
        return ref_box
    s_klv, _ = gpmf.parse_with_trailing(src_box.data)
    r_klv, r_trail = gpmf.parse_with_trailing(ref_box.data)
    s_devcs = {_dvnm(d): d for d in s_klv}
    out = []
    for d in r_klv:
        name = _dvnm(d)
        if name == b"Global Settings" and name in s_devcs:
            # mode/user settings from the reference; per-file identity (MUID) from the source
            s_muid = s_devcs[name].child("MUID")
            if s_muid is not None and d.child("MUID") is not None:
                d.child("MUID").data = s_muid.data
            out.append(d)
        elif name in s_devcs and name != b"Global Settings":
            out.append(s_devcs[name])  # Highlights etc.: this recording's own
        else:
            out.append(d)
    body = gpmf.serialize(out)
    data = body + r_trail if len(body) + len(r_trail) == len(ref_box.data) else body.ljust(len(ref_box.data), b"\x00")
    return mb.Box(type=b"GPMF", data=data[:len(ref_box.data)] if len(data) > len(ref_box.data) else data)


def _dvnm(d: gpmf.KLV) -> bytes:
    n = d.child("DVNM")
    return n.data.rstrip(b"\x00") if n is not None else b""


def merged_sos_header(src_header: bytes, ref_header: bytes, src_ts: int, ref_ts: int, log=None) -> bytes:
    """Reference SOS header (mode fields) with this recording's identity/time fields: MUID, GPS block, clock triple."""
    if len(src_header) != len(ref_header):
        if log:
            log(f"reference SOS header size {len(ref_header)} != source {len(src_header)}; keeping the source header")
        return src_header
    h = bytearray(ref_header)
    h[0x8C:0xAC] = src_header[0x8C:0xAC]          # MUID (per file)
    h[0xAD:0xC8] = src_header[0xAD:0xC8]          # GPS fix flag + ©xyz string
    si = src_header.find(struct.pack("<II", 1, src_ts))
    ri = ref_header.find(struct.pack("<II", 1, ref_ts))
    if si >= 0 and ri >= 0:
        h[ri + 52:ri + 64] = src_header[si + 52:si + 64]   # unix time, seconds since midnight, ms
    return bytes(h)


def apply(src: SourceFile, ref: SourceFile, log=None) -> None:
    """Mutate src.moov so the muxer's template carries the reference's mode-dependent udta."""
    probs = check_compatible(src, ref)
    for p in probs:
        if log:
            log(f"reference warning: {p}")
    su = src.moov.child("udta")
    if su is not None and ref.moov.child("udta") is not None:
        src.moov.replace_child("udta", merged_udta(src, ref, log))
        if log:
            log("udta: mode-dependent atoms (SETT, Global Settings, ...) taken from the reference; per-file atoms kept")
