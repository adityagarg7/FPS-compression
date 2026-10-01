"""mdat sample ordering reproducing the camera's writer. Placeholder rule = strict decode-time order with the
tie-break observed in GoPro files (video before audio at equal time; tmcd right after the first video frame;
MET payload before the video frame that starts its window). Replaced by the verified rule once derived."""
from __future__ import annotations

from fractions import Fraction

from .model import SourceFile
from .mux import OutTrack


def order_samples(src: SourceFile, tracks: dict[str, OutTrack]) -> list[tuple[str, int]]:
    items = []
    prio = {"tmcd": 0, "gpmd": 1, "video": 2, "audio": 3}
    for kind, t in tracks.items():
        dts = 0
        for i, d in enumerate(t.durations):
            tm = Fraction(dts, t.timescale)
            if kind == "tmcd":
                tm = Fraction(0)
                items.append((tm, 2.5, kind, i))  # after the first video frame
            else:
                items.append((tm, prio.get(kind, 9), kind, i))
            dts += d
    items.sort(key=lambda x: (x[0], x[1], x[3]))
    return [(k, i) for _, _, k, i in items]
