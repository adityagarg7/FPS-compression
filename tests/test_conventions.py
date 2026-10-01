import os

import pytest

from gopro_retime import interleave, mux, sos
from gopro_retime.h26x import nal as N
from gopro_retime.model import SourceFile

# firmware anomalies the rule intentionally does not model
INTERLEAVE_EXCEPTIONS = {"hero7.mp4", "karma.mp4"}


def _tracks(src):
    tracks = {}
    for t in src.tracks:
        if t.kind == "fdsc":
            continue
        tracks[t.kind] = mux.OutTrack(t.kind, src.read_samples(t), [s.duration for s in t.samples], t.timescale,
                                      sync=[s.is_sync for s in t.samples] if t.has_stss else None, source=t)
    return tracks


def test_sos_regeneration_byte_exact(all_samples):
    for p in all_samples:
        src = SourceFile.open(p)
        ps = N.parameter_sets_from_entry_children(src.video.stsd_entries[0].children, "h264")
        conv = sos.learn(src, "h264", ps)
        order = [(t.kind, s.index) for t, s in src.all_samples_in_file_order() if t.kind != "fdsc"]
        regen = sos.make_builder(src, "h264", ps, src.video.timescale, src.video_frame_duration(), conv)(order, _tracks(src))
        assert regen == src.read_samples(src.track("fdsc")), p


def test_interleave_rule_reproduces_firmware_order(all_samples):
    for p in all_samples:
        if os.path.basename(p) in INTERLEAVE_EXCEPTIONS:
            continue
        src = SourceFile.open(p)
        true_order = [(t.kind, s.index) for t, s in src.all_samples_in_file_order() if t.kind != "fdsc"]
        assert interleave.order_samples(src, _tracks(src), interleave.measure(src)) == true_order, p


def test_header_timescale_patch(hero8):
    src = SourceFile.open(hero8)
    ps = N.parameter_sets_from_entry_children(src.video.stsd_entries[0].children, "h264")
    conv = sos.learn(src, "h264", ps)
    h2 = sos.patch_header_timescale(conv.header, 90000, 24000)
    assert h2 != conv.header and len(h2) == len(conv.header)
    assert sos.header_clock(h2, 24000) == sos.header_clock(conv.header, 90000)
    assert sos.patch_header_timescale(h2, 24000, 90000) == conv.header
