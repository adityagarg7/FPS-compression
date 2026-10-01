"""GoPro 'SOS' (fdsc) recovery-track regeneration. Placeholder until the format spec lands: copies the two header
samples from the source and writes 16-byte descriptors with the observed layout (type, flags, size, duration, cts)."""
from __future__ import annotations

from fractions import Fraction
from typing import Callable

from .model import SourceFile
from .mux import OutTrack

TYPE_CODES = {"video": 0, "gpmd": 3, "audio": 4, "tmcd": 5}


def make_builder(src: SourceFile, codec: str, ps: dict[str, list[bytes]], out_fps: Fraction, out_ts: int, out_fdur: int) -> Callable:
    fd = src.track("fdsc")
    hdr = [src.read_sample(s) for s in fd.samples[:2]]

    def build(order: list[tuple[str, int]], tracks: dict[str, OutTrack]) -> list[bytes]:
        out = list(hdr)
        for kind, idx in order:
            t = tracks[kind]
            size = len(t.samples[idx])
            dur = t.durations[idx]
            if kind == "video":
                sync = t.sync[idx] if t.sync else True
                code, flags = (3, 1) if (sync and idx == 0) else (0, 3)
                out.append(b"GP" + bytes([code, flags]) + size.to_bytes(4, "big") + dur.to_bytes(4, "big") + dur.to_bytes(4, "big"))
            else:
                out.append(b"GP" + bytes([TYPE_CODES.get(kind, 6), 0]) + size.to_bytes(4, "big") + dur.to_bytes(4, "big") + b"\x00" * 4)
        return out

    return build
