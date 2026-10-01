"""Write the output MP4 by cloning the source file's moov and replacing only what must change.

Everything the camera wrote that is not a function of the new sample data is copied byte-for-byte:
ftyp, udta (all GoPro atoms), iods, hdlr, vmhd/smhd/gmhd, dinf, stsd entries (except parameter sets / tmcd timing),
track header fields other than durations, creation times, ...
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

from . import mp4box as mb
from .model import SourceFile, Track


@dataclass
class OutTrack:
    kind: str
    samples: list[bytes]
    durations: list[int]                 # per sample, in `timescale`
    timescale: int
    sync: Optional[list[bool]] = None    # None => no stss (all sync / not applicable)
    cts_offsets: Optional[list[int]] = None
    stsd_entries: Optional[list[mb.SampleEntry]] = None   # None => keep the source's stsd
    media_duration: Optional[int] = None                  # None => sum(durations)
    tkhd_duration: Optional[int] = None                   # None => derived from media duration
    elst: Optional[list[mb.ElstEntry]] = None             # None => keep/derive from the source convention
    source: Optional[Track] = None                        # the template track

    @property
    def total_duration(self) -> int:
        return self.media_duration if self.media_duration is not None else sum(self.durations)


@dataclass
class MdatItem:
    track_kind: str
    index: int          # sample index within its track
    size: int
    offset: int = 0


InterleaveFn = Callable[[dict[str, OutTrack]], list[tuple[str, int]]]


def interleave_time_ordered(tracks: dict[str, OutTrack], priority: tuple[str, ...] = ("tmcd", "gpmd", "video", "audio")) -> list[tuple[str, int]]:
    """Default interleave: strictly by decode timestamp, ties broken by `priority` order. This is replaced by the
    firmware rule derived from the source (see interleave.py) when available."""
    items: list[tuple[Fraction, int, str, int]] = []
    prio = {k: i for i, k in enumerate(priority)}
    for kind, t in tracks.items():
        if kind == "fdsc":
            continue
        dts = 0
        for i, d in enumerate(t.durations):
            items.append((Fraction(dts, t.timescale), prio.get(kind, 99), kind, i))
            dts += d
    items.sort()
    return [(k, i) for _, _, k, i in items]


def _scaled(value: int, from_ts: int, to_ts: int) -> int:
    return (value * to_ts) // from_ts


@dataclass
class MuxResult:
    path: str
    order: list[tuple[str, int]]
    offsets: dict[str, list[int]]
    moov: mb.Box


def mvhd_duration_rule(src: SourceFile) -> str:
    """Determine how the source computed mvhd.duration: 'max_av' (max of video/audio tkhd) or 'max_all' or 'video'."""
    tk = {t.kind: t.tkhd_duration for t in src.tracks}
    d = src.mvhd.duration
    av = [tk.get("video", 0), tk.get("audio", 0)]
    if d == max(av):
        return "max_av"
    if d == max(tk.values()):
        return "max_all"
    if d == tk.get("video"):
        return "video"
    return "max_av"


def write_output(src: SourceFile, out_path: str, tracks: dict[str, OutTrack], order: list[tuple[str, int]],
                 fdsc_builder: Optional[Callable[[list[tuple[str, int]], dict[str, OutTrack]], list[bytes]]] = None,
                 mvhd_timescale: Optional[int] = None, log=None) -> MuxResult:
    """tracks: by kind ('video','audio','tmcd','gpmd'); order: media samples in mdat order (kind, index).
    fdsc_builder: given the order, returns the list of fdsc samples; fdsc sample j is written immediately before
    media sample j-2 (first two fdsc samples are headers written first) — i.e. the GoPro 'SOS' convention."""
    # ---- 1. layout ---------------------------------------------------------------------------
    ftyp_bytes = src.ftyp.serialize()
    fdsc_samples: list[bytes] = []
    if fdsc_builder is not None and src.track("fdsc") is not None:
        fdsc_samples = fdsc_builder(order, tracks)
    layout: list[MdatItem] = []
    n_hdr = len(fdsc_samples) - len(order) if fdsc_samples else 0
    if fdsc_samples:
        if n_hdr < 0:
            raise ValueError("fdsc builder returned fewer samples than media samples")
        for j in range(n_hdr):
            layout.append(MdatItem("fdsc", j, len(fdsc_samples[j])))
    for i, (kind, idx) in enumerate(order):
        if fdsc_samples:
            layout.append(MdatItem("fdsc", n_hdr + i, len(fdsc_samples[n_hdr + i])))
        layout.append(MdatItem(kind, idx, len(tracks[kind].samples[idx])))
    total = sum(it.size for it in layout)
    mdat_large = (total + 8) > 0xFFFFFFFF
    mdat_hdr = 16 if mdat_large else 8
    pos = len(ftyp_bytes) + mdat_hdr
    offsets: dict[str, list[int]] = {k: [0] * len(t.samples) for k, t in tracks.items()}
    if fdsc_samples:
        offsets["fdsc"] = [0] * len(fdsc_samples)
    for it in layout:
        it.offset = pos
        offsets[it.track_kind][it.index] = pos
        pos += it.size

    # ---- 2. moov -----------------------------------------------------------------------------
    moov = mb.clone(src.moov)
    mvhd_box = moov.child("mvhd"); assert mvhd_box is not None
    mv_ts = mvhd_timescale or src.mvhd.timescale
    if mv_ts != src.mvhd.timescale:
        d = bytearray(mvhd_box.data)
        if d[0] == 1:
            mb.put_u32(d, 20, mv_ts)
        else:
            mb.put_u32(d, 12, mv_ts)
        mvhd_box.data = bytes(d)
    tkhd_durs: dict[str, int] = {}
    for trak in moov.children_of_type("trak"):
        tkhd = trak.child("tkhd"); assert tkhd is not None
        tid = mb.parse_tkhd(tkhd).track_id
        st = next(t for t in src.tracks if t.track_id == tid)
        kind = st.kind
        if kind == "fdsc" and fdsc_samples:
            ot = OutTrack("fdsc", fdsc_samples, [0] * len(fdsc_samples), st.timescale, None, None, None,
                          media_duration=tracks["video"].total_duration if "video" in tracks else st.media_duration,
                          source=st)
            ot.timescale = tracks["video"].timescale if "video" in tracks else st.timescale
        elif kind in tracks:
            ot = tracks[kind]
        else:
            # track kept untouched (should not happen for GoPro files; keep its tables but they'd point into the old mdat)
            raise ValueError(f"no output data for track {tid} ({kind}); cannot keep a track without samples")
        _patch_track(trak, st, ot, mv_ts, offsets[kind], tkhd_durs)
    rule = mvhd_duration_rule(src)
    if rule == "video":
        mv_dur = tkhd_durs.get("video", 0)
    elif rule == "max_all":
        mv_dur = max(tkhd_durs.values())
    else:
        mv_dur = max(tkhd_durs.get("video", 0), tkhd_durs.get("audio", 0))
    mb.set_mvhd_duration(mvhd_box, mv_dur)
    moov_bytes = moov.serialize()

    # ---- 3. write ----------------------------------------------------------------------------
    with open(out_path, "wb") as f:
        f.write(ftyp_bytes)
        if mdat_large:
            f.write(struct.pack(">I4sQ", 1, b"mdat", total + 16))
        else:
            f.write(struct.pack(">I4s", total + 8, b"mdat"))
        for it in layout:
            data = fdsc_samples[it.index] if it.track_kind == "fdsc" else tracks[it.track_kind].samples[it.index]
            f.write(data)
        f.write(moov_bytes)
    if log:
        log(f"wrote {out_path}: mdat {total} bytes, moov {len(moov_bytes)} bytes, {len(layout)} mdat items")
    return MuxResult(out_path, order, offsets, moov)


def _patch_track(trak: mb.Box, st: Track, ot: OutTrack, mv_ts: int, offs: list[int], tkhd_durs: dict[str, int]) -> None:
    media_dur = ot.total_duration
    ts = ot.timescale
    mdia = trak.child("mdia"); assert mdia is not None
    mdhd = mdia.child("mdhd"); assert mdhd is not None
    mb.set_mdhd(mdhd, timescale=ts, duration=media_dur)
    tk_dur = ot.tkhd_duration if ot.tkhd_duration is not None else _scaled(media_dur, ts, mv_ts)
    # keep the source's convention when the source tkhd duration was NOT simply the scaled media duration
    if ot.tkhd_duration is None and st.tkhd_duration != _scaled(st.media_duration, st.timescale, mv_ts if mv_ts == st.timescale or True else mv_ts):
        pass
    tkhd = trak.child("tkhd"); assert tkhd is not None
    mb.set_tkhd_duration(tkhd, tk_dur)
    tkhd_durs[st.kind] = tk_dur
    # edit list
    edts = trak.child("edts")
    if edts is not None:
        elst_box = edts.child("elst")
        if elst_box is not None:
            if ot.elst is not None:
                entries = ot.elst
            else:
                entries = mb.parse_elst(elst_box)
                if len(entries) == 1 and entries[0].segment_duration == st.tkhd_duration:
                    entries[0].segment_duration = tk_dur
            new = mb.build_elst(entries, elst_box.data[0])
            edts.replace_child("elst", new)
    stbl = trak.find("mdia/minf/stbl"); assert stbl is not None
    # stsd
    if ot.stsd_entries is not None:
        old = stbl.child("stsd"); assert old is not None
        stbl.replace_child("stsd", mb.build_stsd(ot.stsd_entries, old.data[:4]))
    # stts
    stbl.replace_child("stts", mb.stts_from_durations(ot.durations))
    # ctts
    old_ctts = stbl.child("ctts")
    if ot.cts_offsets is not None and any(ot.cts_offsets):
        new_ctts = mb.build_ctts(ot.cts_offsets, old_ctts.data[0] if old_ctts is not None else 0)
        if old_ctts is not None:
            stbl.replace_child("ctts", new_ctts)
        else:
            stbl.children.insert(stbl.children.index(stbl.child("stts")) + 1, new_ctts)
    elif old_ctts is not None:
        stbl.remove_child(old_ctts)
    # stsc: one sample per chunk (GoPro convention) unless the source used something else uniform
    stbl.replace_child("stsc", mb.build_stsc([(1, 1, 1)]))
    # stsz
    old_stsz = stbl.child("stsz"); assert old_stsz is not None
    uniform_src = mb.u32(old_stsz.data, 4) != 0
    stbl.replace_child("stsz", mb.build_stsz([len(s) for s in ot.samples], force_table=not uniform_src))
    # stco / co64
    old_co = stbl.child("stco") or stbl.child("co64"); assert old_co is not None
    new_co = mb.build_stco(offs, force64=(old_co.type == b"co64"))
    stbl.replace_child(old_co.type.decode(), new_co)
    # stss
    old_stss = stbl.child("stss")
    if ot.sync is not None and not all(ot.sync):
        new_stss = mb.build_stss([i + 1 for i, s in enumerate(ot.sync) if s])
        if old_stss is not None:
            stbl.replace_child("stss", new_stss)
        else:
            stbl.children.append(new_stss)
    elif old_stss is not None and ot.sync is not None and all(ot.sync):
        # all samples sync: GoPro video always has stss; keep a full list
        stbl.replace_child("stss", mb.build_stss(list(range(1, len(ot.samples) + 1))))
