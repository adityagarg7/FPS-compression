"""Timing plan: output frame rate, frame selection map, output durations."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

NTSC_30 = Fraction(30000, 1001)
EXACT_30 = Fraction(30, 1)


def parse_fps(s: str) -> Fraction:
    s = s.strip().lower()
    if s in ("29.97", "ntsc", "30000/1001", "29.970"):
        return NTSC_30
    if s in ("30", "30.0", "exact30", "30/1"):
        return EXACT_30
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    f = Fraction(s)
    # common decimals -> 1001-based
    for cand in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001)):
        if abs(float(cand) - float(f)) < 1e-3:
            return cand
    return f


@dataclass
class FramePlan:
    src_fps: Fraction
    out_fps: Fraction
    src_frames: int
    out_frames: int
    frame_map: list[int]           # out index -> source frame index (display order)
    keep_mask: list[bool]          # per source frame: kept?
    mode: str                      # 'realtime' | 'conform'

    @property
    def src_frame_time(self) -> Fraction:
        return 1 / self.src_fps

    @property
    def out_frame_time(self) -> Fraction:
        return 1 / self.out_fps

    def out_time(self, k: int) -> Fraction:
        return k * self.out_frame_time

    def src_time(self, j: int) -> Fraction:
        return j * self.src_frame_time


def nearest_frame_map(src_fps: Fraction, out_fps: Fraction, src_frames: int) -> list[int]:
    """For each output slot k at time k/out_fps (while < source duration), pick the nearest source frame."""
    ratio = src_fps / out_fps  # source frames per output frame, e.g. 50/(30000/1001) = 1001/600
    duration = Fraction(src_frames, 1) / src_fps
    out = []
    k = 0
    while k * (1 / out_fps) < duration:
        j = int((k * ratio) + Fraction(1, 2))  # floor(k*ratio + 0.5), nearest with ties up
        if j >= src_frames:
            break  # no source frame close enough: the output simply ends one slot earlier (never duplicate)
        out.append(j)
        k += 1
    return out


def make_plan(src_fps: Fraction, out_fps: Fraction, src_frames: int, mode: str = "realtime") -> FramePlan:
    if mode == "conform":
        fmap = list(range(src_frames))
    elif mode == "realtime":
        fmap = nearest_frame_map(src_fps, out_fps, src_frames)
    else:
        raise ValueError(mode)
    keep = [False] * src_frames
    for j in fmap:
        keep[j] = True
    return FramePlan(src_fps, out_fps, src_frames, len(fmap), fmap, keep, mode)


def select_expression(src_fps: Fraction, out_fps: Fraction) -> str:
    """ffmpeg 'select' filter expression that keeps exactly the frames in nearest_frame_map (by frame index n).
    Source frame n is kept iff n == floor(round(n*q/p)*p/q + 1/2) where p/q = src_fps/out_fps."""
    ratio = src_fps / out_fps
    p, q = ratio.numerator, ratio.denominator
    if p == q:
        return "1"
    # floor(k*p/q + 1/2) == floor((2*k*p + q) / (2*q))
    return f"eq(n\\,floor((2*round(n*{q}/{p})*{p}+{q})/(2*{q})))"


def frames_per_gpmf_payload(out_fps: Fraction) -> tuple[int, int]:
    """GoPro MET payload granularity: (frames per payload, payload duration in ms) — 1001 ms / 30 frames for 29.97,
    1000 ms / N frames for integer rates."""
    if out_fps.denominator == 1001:
        return out_fps.numerator // 1000, 1001
    return int(out_fps), 1000
