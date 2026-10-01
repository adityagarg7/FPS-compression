"""GoPro 'SOS' recovery track (handler 'GoPro SOS', sample entry 'fdsc') regeneration.

Layout (verified on 10 firmware-written files):
  fdsc sample #0  : 'GPRO' header struct (firmware-specific, copied from the source; only the video timescale field and,
                    with a reference file, the mode-dependent settings are patched)
  fdsc sample #1  : descriptor of the FIRST video frame, type 3, extended with the parameter sets (u32 BE length + NAL in a
                    fixed-size zero-padded buffer, one block per parameter set)
  every other     : 16-byte descriptor "GP" u8 type u8 flags u32 size u32 duration u32 X written immediately BEFORE the
                    media sample it describes.  type: 0 video, 4 audio, 5 tmcd, 6 gpmd; flags: video 1 = sync, 3 = non-sync;
                    duration in the track's timescale (tmcd: the VIDEO frame duration; gpmd: the payload's real duration);
                    X = video frame duration for video on HD6+ firmware, 0 otherwise.
All conventions are LEARNED from the source file's own descriptors so firmware differences are followed automatically.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Callable, Optional

from .model import SourceFile
from .mux import OutTrack


@dataclass
class SosConventions:
    header: bytes                                  # sample #0 (verbatim)
    type_codes: dict[str, int]                     # kind -> descriptor type
    video_flags: tuple[int, int]                   # (sync, non-sync)
    other_flags: dict[str, int]
    x_rule: dict[str, str]                         # kind -> 'duration' | 'zero'
    tmcd_duration_rule: str                        # 'video_frame' | 'own'
    ps_blocks: list[tuple[int, int]]               # for sample #1: (length-field offset, buffer size) per parameter set
    ps_tail: bytes                                 # bytes the firmware leaves right after each parameter set (HD5: 00 00 00 01)
    sample1_prefix: bytes                          # the 16 descriptor bytes of sample #1 (type/flags learned)
    sample1_size: int
    descriptor_size: int = 16

    @property
    def tail_offset(self) -> int:
        return -1


def learn(src: SourceFile, codec: str, src_ps: dict[str, list[bytes]]) -> SosConventions:
    fd = src.track("fdsc")
    if fd is None:
        raise ValueError("source has no fdsc track")
    samples = src.read_samples(fd)
    header, s1 = samples[0], samples[1]
    media = [(t, s) for t, s in src.all_samples_in_file_order() if t.kind != "fdsc"]
    descs = samples[1:]
    if len(descs) != len(media):
        raise ValueError(f"fdsc descriptor count {len(descs)} != media sample count {len(media)}")
    type_codes: dict[str, int] = {}
    other_flags: dict[str, int] = {}
    x_rule: dict[str, str] = {}
    vflags = [None, None]
    tmcd_rule = "video_frame"
    vdur = src.video_frame_duration()
    for (t, s), d in zip(media, descs):
        typ, flags = d[2], d[3]
        size, dur, x = struct.unpack(">III", d[4:16])
        if size != s.size:
            raise ValueError("fdsc descriptor size does not match its sample; unknown layout")
        if t.kind == "video":
            type_codes.setdefault("video", typ if typ != 3 else 0)
            if s.is_sync:
                vflags[0] = flags if vflags[0] is None else vflags[0]
            else:
                vflags[1] = flags if vflags[1] is None else vflags[1]
            x_rule.setdefault("video", "duration" if x == dur and x != 0 else "zero")
        else:
            type_codes.setdefault(t.kind, typ)
            other_flags.setdefault(t.kind, flags)
            x_rule.setdefault(t.kind, "duration" if x == dur and x != 0 else "zero")
            if t.kind == "tmcd":
                tmcd_rule = "video_frame" if dur == vdur else "own"
    # sample #1 parameter-set blocks: locate each parameter set (in VPS, SPS, PPS order) inside sample #1
    order = ["vps", "sps", "pps"] if codec == "hevc" else ["sps", "pps"]
    nals = [n for k in order for n in src_ps.get(k, [])]
    blocks: list[tuple[int, int]] = []
    pos = 16
    starts: list[int] = []
    for n in nals:
        want = struct.pack(">I", len(n)) + n
        i = s1.find(want, pos)
        if i < 0:
            raise ValueError("parameter set not found in fdsc sample #1; unknown layout")
        starts.append(i)
        pos = i + 4 + len(n)
    for i, st in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(s1)
        blocks.append((st, end - st - 4))
    # bytes left after the NAL inside its buffer (HD5-era: Annex-B start code garbage)
    first_nal_end = starts[0] + 4 + len(nals[0])
    tail = s1[first_nal_end:first_nal_end + 4]
    ps_tail = b"\x00\x00\x00\x01" if tail == b"\x00\x00\x00\x01" else b""
    return SosConventions(header, type_codes, (vflags[0] if vflags[0] is not None else 1, vflags[1] if vflags[1] is not None else 3),
                          other_flags, x_rule, tmcd_rule, blocks, ps_tail, s1[:16], len(s1))


def patch_header_timescale(header: bytes, old_ts: int, new_ts: int) -> bytes:
    """The only frame-rate dependent field of sample #0 is the video mdhd timescale in the tail block (LE u32 right after a LE 1)."""
    if old_ts == new_ts:
        return header
    needle = struct.pack("<II", 1, old_ts)
    i = header.find(needle)
    if i < 0:
        return header
    return header[:i + 4] + struct.pack("<I", new_ts) + header[i + 8:]


def header_clock(header: bytes, video_ts: int) -> Optional[tuple[int, int, int]]:
    """(unix_time, seconds_since_midnight, milliseconds) from the tail block of sample #0, if found."""
    i = header.find(struct.pack("<II", 1, video_ts))
    if i < 0 or i + 64 > len(header):
        return None
    vals = struct.unpack("<16I", header[i:i + 64])
    return vals[13], vals[14], vals[15]


def build_sample1(conv: SosConventions, first_size: int, first_dur: int, codec: str, ps: dict[str, list[bytes]]) -> bytes:
    order = ["vps", "sps", "pps"] if codec == "hevc" else ["sps", "pps"]
    nals = [n for k in order for n in ps.get(k, [])]
    if len(nals) != len(conv.ps_blocks):
        raise ValueError(f"parameter set count {len(nals)} != source sample #1 blocks {len(conv.ps_blocks)}")
    x = first_dur if conv.x_rule.get("video") == "duration" else 0
    out = bytearray(b"GP" + bytes([3, conv.video_flags[0]]) + struct.pack(">III", first_size, first_dur, x))
    # keep any bytes between the descriptor and the first block (none in known layouts)
    out += conv.sample1_prefix[16:conv.ps_blocks[0][0]] if conv.ps_blocks[0][0] > 16 else b""
    for n, (_off, bufsize) in zip(nals, conv.ps_blocks):
        if len(n) > bufsize:
            raise ValueError(f"parameter set ({len(n)} bytes) exceeds the firmware buffer ({bufsize} bytes)")
        out += struct.pack(">I", len(n)) + (n + conv.ps_tail)[:bufsize].ljust(bufsize, b"\x00")
    return bytes(out)


def make_builder(src: SourceFile, codec: str, ps: dict[str, list[bytes]], out_video_ts: int, out_frame_dur: int,
                 conv: Optional[SosConventions] = None, header_override: Optional[bytes] = None) -> Callable:
    conv = conv or learn(src, codec, __import__("gopro_retime.h26x.nal", fromlist=["x"]).parameter_sets_from_entry_children(src.video.stsd_entries[0].children, codec))
    header = header_override or patch_header_timescale(conv.header, src.video.timescale, out_video_ts)

    def build(order: list[tuple[str, int]], tracks: dict[str, OutTrack]) -> list[bytes]:
        out = [header]
        for key, idx in order:
            t = tracks[key]
            kind = t.kind
            size = len(t.samples[idx])
            dur = t.durations[idx]
            if kind == "video":
                sync = t.sync[idx] if t.sync else True
                if idx == 0:
                    out.append(build_sample1(conv, size, dur, codec, ps))
                    continue
                flags = conv.video_flags[0] if sync else conv.video_flags[1]
                x = dur if conv.x_rule.get("video") == "duration" else 0
                out.append(b"GP" + bytes([conv.type_codes.get("video", 0), flags]) + struct.pack(">III", size, dur, x))
            else:
                if kind == "tmcd" and conv.tmcd_duration_rule == "video_frame":
                    dur = out_frame_dur
                x = dur if conv.x_rule.get(kind) == "duration" else 0
                out.append(b"GP" + bytes([conv.type_codes.get(kind, 6), conv.other_flags.get(kind, 0)]) + struct.pack(">III", size, dur, x))
        return out

    return build
