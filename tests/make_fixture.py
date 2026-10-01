"""Build a synthetic 50 fps 'PAL' GoPro-structured fixture from a real 29.97 fps sample by retiming its tables.

The frames are the same pictures; only the timing (stts, mvhd/tkhd/mdhd durations, tmcd, gpmd stts, elst) is rewritten
to what a 50 fps PAL recording uses (timescale 90000, frame duration 1800, 1000 ms metadata payloads).
This exercises every container code path for the real 50 -> 29.97 use case without owning a HERO12 file.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gopro_retime import mp4box as mb  # noqa: E402
from gopro_retime.model import SourceFile  # noqa: E402
from gopro_retime import mux  # noqa: E402


def make_50fps_fixture(src_path: str, out_path: str, fps: int = 50) -> str:
    src = SourceFile.open(src_path)
    v = src.video
    ts = 90000
    fdur = ts // fps
    nframes = v.sample_count
    tracks: dict[str, mux.OutTrack] = {}
    vid_entries = mb.parse_stsd(v.stbl.child("stsd"))
    tracks["video"] = mux.OutTrack("video", src.read_samples(v), [fdur] * nframes, ts,
                                   sync=[s.is_sync for s in v.samples], source=v)
    a = src.track("audio")
    # keep audio samples that fit the new (shorter) duration
    vid_secs = nframes / fps
    a_samples = src.read_samples(a)
    keep = int(vid_secs * a.timescale / 1024) + 1
    a_samples = a_samples[:keep]
    tracks["audio"] = mux.OutTrack("audio", a_samples, [1024] * len(a_samples), a.timescale, source=a)
    t = src.track("tmcd")
    t_entries = mb.parse_stsd(t.stbl.child("stsd"))
    mb.set_tmcd_entry(t_entries[0], ts, fdur, fps)
    old_tc = int.from_bytes(src.read_sample(t.samples[0]), "big")
    new_tc = int(old_tc / float(src.video_frame_rate()) * fps)
    tracks["tmcd"] = mux.OutTrack("tmcd", [new_tc.to_bytes(4, "big")], [nframes * fdur], ts, stsd_entries=t_entries, source=t)
    m = src.track("gpmd")
    m_samples = src.read_samples(m)
    n_pay = int(vid_secs) + (1 if vid_secs % 1 else 0)
    m_samples = m_samples[:n_pay]
    durs = [1000] * len(m_samples)
    rem = int(round((vid_secs - (len(m_samples) - 1)) * 1000))
    durs[-1] = rem
    tracks["gpmd"] = mux.OutTrack("gpmd", m_samples, durs, 1000, source=m)
    # HD8-style writer behaviour: MET payload k written ~117 ms after its window ends, final partial payload after the last audio frame
    from gopro_retime import interleave
    from fractions import Fraction
    hd8 = interleave.InterleaveConventions(Fraction(1168, 10000), Fraction(1001, 10000), Fraction(1335, 10000), True)
    order = interleave.order_samples(src, tracks, hd8)
    # regenerate the SOS track with the conventions learned from the real file (timescale patched to 50 fps)
    from gopro_retime import sos
    from gopro_retime.h26x import nal as N
    ps = N.parameter_sets_from_entry_children(v.stsd_entries[0].children, "h264")
    sos_conv = sos.learn(src, "h264", ps)
    fdsc_builder = sos.make_builder(src, "h264", ps, ts, fdur, sos_conv)

    mux.write_output(src, out_path, tracks, order, fdsc_builder=fdsc_builder, mvhd_timescale=ts)
    return out_path


if __name__ == "__main__":
    print(make_50fps_fixture(sys.argv[1], sys.argv[2]))
