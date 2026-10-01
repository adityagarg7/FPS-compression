"""In-memory model of a parsed source MP4: tracks with fully expanded sample tables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from . import mp4box as mb


@dataclass
class Sample:
    index: int          # 0-based within track
    offset: int         # absolute file offset
    size: int
    dts: int            # decode time in track timescale
    duration: int
    cts_offset: int     # composition offset (0 when no ctts)
    is_sync: bool
    chunk: int          # 0-based chunk index


@dataclass
class Track:
    trak: mb.Box
    track_id: int
    kind: str                       # 'video' | 'audio' | 'tmcd' | 'gpmd' | 'fdsc' | 'other'
    handler: bytes                  # hdlr handler_type
    handler_name: str
    timescale: int
    media_duration: int
    tkhd_duration: int
    stsd_entries: list[mb.SampleEntry]
    samples: list[Sample]
    elst: Optional[list[mb.ElstEntry]]
    has_ctts: bool
    has_stss: bool
    chunk_offsets: list[int] = field(default_factory=list)
    stsc: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def format(self) -> bytes:
        return self.stsd_entries[0].format if self.stsd_entries else b""

    @property
    def total_bytes(self) -> int:
        return sum(s.size for s in self.samples)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def stbl(self) -> mb.Box:
        b = self.trak.find("mdia/minf/stbl")
        assert b is not None
        return b


def _classify(handler: bytes, fmt: bytes, name: str) -> str:
    if handler == b"vide":
        return "video"
    if handler == b"soun":
        return "audio"
    if fmt == b"tmcd" or handler == b"tmcd":
        return "tmcd"
    if fmt == b"gpmd":
        return "gpmd"
    if fmt == b"fdsc":
        return "fdsc"
    return "other"


def expand_track(trak: mb.Box) -> Track:
    tkhd = trak.child("tkhd"); assert tkhd is not None
    tk = mb.parse_tkhd(tkhd)
    mdia = trak.child("mdia"); assert mdia is not None
    mdhd = mdia.child("mdhd"); assert mdhd is not None
    md = mb.parse_mdhd(mdhd)
    hdlr = mdia.child("hdlr"); assert hdlr is not None
    handler = mb.hdlr_type(hdlr)
    hname = mb.hdlr_name(hdlr)
    stbl = trak.find("mdia/minf/stbl"); assert stbl is not None
    stsd = stbl.child("stsd"); assert stsd is not None
    entries = mb.parse_stsd(stsd)
    fmt = entries[0].format if entries else b""
    kind = _classify(handler, fmt, hname)

    stts = mb.parse_stts(stbl.child("stts"))
    stsc = mb.parse_stsc(stbl.child("stsc"))
    stsz = mb.parse_stsz(stbl.child("stsz"))
    stco_box = stbl.child("stco") or stbl.child("co64")
    stco = mb.parse_stco(stco_box) if stco_box is not None else []
    stss_box = stbl.child("stss")
    sync = set(mb.parse_stss(stss_box)) if stss_box is not None else None
    ctts_box = stbl.child("ctts")
    ctts = mb.parse_ctts(ctts_box) if ctts_box is not None else None

    # expand stsc
    nchunks = len(stco)
    spc: list[int] = []
    for i, (first, n, _sdi) in enumerate(stsc):
        nxt = stsc[i + 1][0] if i + 1 < len(stsc) else nchunks + 1
        spc.extend([n] * (nxt - first))
    durs: list[int] = []
    for cnt, d in stts:
        durs.extend([d] * cnt)
    offs: list[int] = []
    for i in range(len(durs)):
        pass
    cts: list[int] = []
    if ctts is not None:
        for cnt, o in ctts:
            cts.extend([o] * cnt)
    samples: list[Sample] = []
    si = 0
    dts = 0
    for ci, coff in enumerate(stco):
        o = coff
        for _ in range(spc[ci] if ci < len(spc) else 0):
            if si >= len(stsz):
                break
            d = durs[si] if si < len(durs) else 0
            samples.append(Sample(si, o, stsz[si], dts, d, cts[si] if si < len(cts) else 0,
                                  (sync is None) or ((si + 1) in sync), ci))
            o += stsz[si]
            dts += d
            si += 1
    if si != len(stsz):
        raise ValueError(f"track {tk.track_id}: chunk tables cover {si} samples but stsz has {len(stsz)}")
    edts = trak.child("edts")
    elst = None
    if edts is not None and edts.child("elst") is not None:
        elst = mb.parse_elst(edts.child("elst"))
    return Track(trak, tk.track_id, kind, handler, hname, md.timescale, md.duration, tk.duration, entries,
                 samples, elst, ctts is not None, stss_box is not None, stco, stsc)


@dataclass
class SourceFile:
    path: str
    top: mb.TopLevel
    ftyp: mb.Box
    moov: mb.Box
    mvhd: mb.MvhdInfo
    tracks: list[Track]
    mdat: mb.Box
    file_size: int

    @classmethod
    def open(cls, path: str) -> "SourceFile":
        top = mb.parse_file(path)
        ftyp = top.find("ftyp")
        moov = top.find("moov")
        mdats = top.find_all("mdat")
        if ftyp is None or moov is None or not mdats:
            raise ValueError(f"{path}: not a complete MP4 (need ftyp, mdat, moov)")
        if len(mdats) != 1:
            raise ValueError(f"{path}: expected exactly one mdat, found {len(mdats)}")
        mvhd = mb.parse_mvhd(moov.child("mvhd"))
        tracks = [expand_track(t) for t in moov.children_of_type("trak")]
        return cls(path, top, ftyp, moov, mvhd, tracks, mdats[0], os.path.getsize(path))

    def track(self, kind: str) -> Optional[Track]:
        for t in self.tracks:
            if t.kind == kind:
                return t
        return None

    def tracks_of(self, kind: str) -> list[Track]:
        return [t for t in self.tracks if t.kind == kind]

    def read(self, offset: int, size: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    def read_sample(self, s: Sample) -> bytes:
        return self.read(s.offset, s.size)

    def read_samples(self, track: Track) -> list[bytes]:
        out = []
        with open(self.path, "rb") as f:
            for s in track.samples:
                f.seek(s.offset)
                out.append(f.read(s.size))
        return out

    # ---- derived properties --------------------------------------------------------------------
    @property
    def video(self) -> Track:
        t = self.track("video")
        if t is None:
            raise ValueError("no video track")
        return t

    def video_frame_rate(self) -> Fraction:
        v = self.video
        durs = {}
        for s in v.samples:
            durs[s.duration] = durs.get(s.duration, 0) + 1
        d = max(durs, key=durs.get)
        return Fraction(v.timescale, d)

    def video_frame_duration(self) -> int:
        v = self.video
        durs = {}
        for s in v.samples:
            durs[s.duration] = durs.get(s.duration, 0) + 1
        return max(durs, key=durs.get)

    def all_samples_in_file_order(self) -> list[tuple[Track, Sample]]:
        allp = [(s.offset, t, s) for t in self.tracks for s in t.samples]
        allp.sort(key=lambda x: x[0])
        return [(t, s) for _, t, s in allp]

    def layout_string(self) -> str:
        """Interleave pattern string, one letter per sample in file order (V/A/T/M/S/O)."""
        letters = {"video": "V", "audio": "A", "tmcd": "T", "gpmd": "M", "fdsc": "S", "other": "O"}
        return "".join(letters[t.kind] for t, _ in self.all_samples_in_file_order())
