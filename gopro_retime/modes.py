"""GoPro HERO11/HERO12 nominal video bit rates per mode (Mbps), Standard / High bit-rate setting.
Source: GoPro support bit-rate chart for HERO11 (reported identical for HERO12). The camera writes the nominal
value into the SPS HRD (bit_rate_value) and its rate control lands on it, so a 29.97 output must carry the
29.97 variant's nominal value, not the 50 fps one."""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

# (max_width, aspect) -> {fps_family: (standard, high, ten_bit)}; fps families: 'hi' (>=100), 'mid' (60/50), 'lo' (<=30)
_TABLE = {
    (5312, "16:9"): {"mid": (60, 120, 120), "lo": (60, 100, 120)},
    (5312, "4:3"): {"lo": (60, 120, None)},
    (5312, "8:7"): {"lo": (60, 120, 120)},
    (3840, "16:9"): {"hi": (60, 120, None), "mid": (60, 100, 120), "lo": (45, 100, 120)},
    (3840, "4:3"): {"mid": (60, 120, None), "lo": (60, 100, None)},
    (3840, "8:7"): {"mid": (60, 120, 120), "lo": (60, 120, 120)},
    (2704, "16:9"): {"hi": (60, 100, None), "mid": (45, 100, None), "lo": (45, 60, None)},
    (2704, "4:3"): {"mid": (45, 100, None), "lo": (45, 60, None)},
    (1920, "16:9"): {"hi": (45, 60, None), "mid": (45, 60, None), "lo": (45, 60, None)},
}
_1080P240 = (60, 78)


def _aspect(w: int, h: int) -> str:
    r = w / h
    if abs(r - 16 / 9) < 0.05:
        return "16:9"
    if abs(r - 4 / 3) < 0.05:
        return "4:3"
    if abs(r - 8 / 7) < 0.05:
        return "8:7"
    if abs(r - 9 / 16) < 0.05:
        return "9:16"
    return "16:9"


def _family(fps: Fraction) -> str:
    f = float(fps)
    if f >= 99:
        return "hi"
    if f >= 49:
        return "mid"
    return "lo"


def nominal_bitrates(width: int, height: int, fps: Fraction, ten_bit: bool = False) -> Optional[tuple[int, int]]:
    """(standard, high) in bps for the mode, or None if unknown."""
    key = (max(width, height) if width < height else width, _aspect(width, height))
    if key[1] == "9:16":
        key = (max(width, height), "16:9")
    row = None
    for (w, a), fam in _TABLE.items():
        if a == key[1] and abs(w - key[0]) <= 64:
            row = fam
            break
    if row is None:
        return None
    if key[0] <= 1984 and float(fps) >= 199:
        std, high = _1080P240
    else:
        ent = row.get(_family(fps)) or row.get("lo")
        if ent is None:
            return None
        std, high = ent[0], ent[1]
        if ten_bit and ent[2]:
            std = high = ent[2]
    return std * 1_000_000, high * 1_000_000


def classify_setting(width: int, height: int, fps: Fraction, hrd_bitrate: int, ten_bit: bool = False) -> Optional[str]:
    """Which bit-rate setting ('standard' | 'high') the source HRD value corresponds to."""
    nb = nominal_bitrates(width, height, fps, ten_bit)
    if nb is None or not hrd_bitrate:
        return None
    std, high = nb
    return "standard" if abs(hrd_bitrate - std) <= abs(hrd_bitrate - high) else "high"


def target_bitrate(width: int, height: int, src_fps: Fraction, out_fps: Fraction, src_hrd: Optional[int], ten_bit: bool = False) -> Optional[int]:
    """Nominal bit rate of the output mode with the same bit-rate setting as the source, or None if unknown."""
    setting = classify_setting(width, height, src_fps, src_hrd or 0, ten_bit)
    nb = nominal_bitrates(width, height, out_fps, ten_bit)
    if setting is None or nb is None:
        return None
    return nb[0] if setting == "standard" else nb[1]
