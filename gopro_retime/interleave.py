"""mdat sample ordering reproducing the camera's writer.

Verified rule (HD6..HD8 / MAX firmware): V0 first, the single tmcd sample second, then audio and video merged by decode
time with AUDIO FIRST on exact ties; every GoPro MET payload k is written a fixed latency after its window ends
(immediately before the first video frame whose dts >= window_end + latency); the final partial payload is written after
the last audio frame on HD8+ firmware.  The latency and the final-payload behaviour are measured on the source file.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from .model import SourceFile
from .mux import OutTrack


@dataclass
class InterleaveConventions:
    latency: Fraction                 # seconds between a MET window end and the video frame it is written before
    latency_lo: Fraction              # measured interval (lo, hi]; latency = midpoint
    latency_hi: Fraction
    final_payload_last: bool          # final MET payload after the last audio frame
    audio_first_on_tie: bool = True
    start_pattern: str = "VTA"        # first three media samples


def measure(src: SourceFile) -> InterleaveConventions:
    order = [(t, s) for t, s in src.all_samples_in_file_order() if t.kind != "fdsc"]
    v, m = src.track("video"), src.track("gpmd")
    los: list[Fraction] = []
    his: list[Fraction] = []
    final_last = False
    if m is not None and v is not None:
        positions = {(t.kind, s.index): i for i, (t, s) in enumerate(order)}
        vid_times = [Fraction(s.dts, v.timescale) for s in v.samples]
        for k, ms in enumerate(m.samples):
            end = Fraction(ms.dts + ms.duration, m.timescale)
            pos = positions[("gpmd", k)]
            if k == len(m.samples) - 1:
                after = order[pos + 1:]
                final_last = not any(t.kind == "audio" for t, _ in after)
                if not any(t.kind == "video" for t, _ in after):
                    continue
            # the next video frame written after this payload
            nxt = next((s for t, s in order[pos + 1:] if t.kind == "video"), None)
            prev = next((s for t, s in reversed(order[:pos]) if t.kind == "video"), None)
            if nxt is None or prev is None:
                continue
            his.append(vid_times[nxt.index] - end)
            los.append(vid_times[prev.index] - end)
    if his:
        lo, hi = max(los), min(his)
        if lo >= hi:  # inconsistent measurements (e.g. hero7-style anomaly): fall back to the most common hi
            hi = sorted(his)[len(his) // 2]
            lo = hi - Fraction(1, 1000)
        lat = (lo + hi) / 2
    else:
        lo, hi, lat = Fraction(0), Fraction(0), Fraction(0)
    start = "".join({"video": "V", "audio": "A", "tmcd": "T", "gpmd": "M"}.get(t.kind, "O") for t, _ in order[:3])
    return InterleaveConventions(lat, lo, hi, final_last, True, start)


class _MergedAudio:
    """All audio tracks merged by decode time (ties: track order), presented as one duration list."""

    def __init__(self, tracks: dict[str, OutTrack], keys: list[str]):
        items = []
        for ki, k in enumerate(keys):
            t = tracks[k]
            acc = 0
            for i, d in enumerate(t.durations):
                items.append((Fraction(acc, t.timescale), ki, k, i))
                acc += d
        items.sort(key=lambda x: (x[0], x[1], x[3]))
        self.times = [x[0] for x in items]
        self.items = [(x[2], x[3]) for x in items]
        self.timescale = 1
        self.durations = [0] * len(items)


def _merged_audio(tracks: dict[str, OutTrack], keys: list[str]) -> _MergedAudio:
    return _MergedAudio(tracks, keys)


def order_samples(src: SourceFile, tracks: dict[str, OutTrack], conv: Optional[InterleaveConventions] = None) -> list[tuple[str, int]]:
    conv = conv or measure(src)
    out: list[tuple[str, int]] = []
    v = tracks.get("video")
    audio_keys = [k for k, tr in tracks.items() if tr.kind == "audio"]
    a = _merged_audio(tracks, audio_keys) if audio_keys else None
    t = tracks.get("tmcd")
    m = tracks.get("gpmd")
    vt = [Fraction(sum(v.durations[:i]), v.timescale) for i in range(len(v.durations))] if v else []
    # prefix sums (faster)
    if v:
        acc = 0
        vt = []
        for d in v.durations:
            vt.append(Fraction(acc, v.timescale)); acc += d
    at: list[Fraction] = list(a.times) if a else []
    # MET insertion points: before video frame index j_k
    met_before: dict[int, list[int]] = {}
    met_at_end: list[int] = []
    if m:
        acc = 0
        for k, d in enumerate(m.durations):
            end = Fraction(acc + d, m.timescale)
            acc += d
            if k == len(m.durations) - 1 and conv.final_payload_last:
                met_at_end.append(k)
                continue
            target = end + conv.latency
            j = next((i for i, tt in enumerate(vt) if tt >= target), None)
            if j is None:
                met_at_end.append(k)
            else:
                met_before.setdefault(j, []).append(k)
    vi = ai = 0
    nv, na = len(vt), len(at)
    # V0, T0 first
    if nv:
        out.append(("video", 0)); vi = 1
        if t:
            out.append(("tmcd", 0))
    elif t:
        out.append(("tmcd", 0))
    while vi < nv or ai < na:
        take_video = False
        if vi < nv and ai < na:
            if vt[vi] < at[ai]:
                take_video = True
            elif vt[vi] > at[ai]:
                take_video = False
            else:
                take_video = not conv.audio_first_on_tie
        elif vi < nv:
            take_video = True
        if take_video:
            for k in met_before.get(vi, []):
                out.append(("gpmd", k))
            out.append(("video", vi)); vi += 1
        else:
            out.append(a.items[ai]); ai += 1
            if ai == na:
                # HD8+/MAX firmware flushes the final (partial) payload at the stop event, right after the last audio frame
                for k in met_at_end:
                    out.append(("gpmd", k))
                met_at_end = []
    for k in met_at_end:
        out.append(("gpmd", k))
    # any MET payload not yet placed (no video frame late enough): append in order
    placed = {i for kind, i in out if kind == "gpmd"}
    if m:
        for k in range(len(m.durations)):
            if k not in placed:
                out.append(("gpmd", k))
    return out
