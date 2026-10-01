import os

import pytest

from gopro_retime import ffmpeg as ff, verify
from gopro_retime.h26x import nal as N
from gopro_retime.model import SourceFile
from gopro_retime.pipeline import Options, run


@pytest.mark.slow
def test_end_to_end_fake50_to_2997(fake50, tmp_path):
    out = tmp_path / "out.mp4"
    res = run(Options(src=fake50, out=str(out), fps="29.97", preset="veryfast", verify=True, external_tools=False), log=lambda s: None)
    assert res.transplant_applied, res.notes
    assert res.lossless_verified
    o = SourceFile.open(str(out))
    s = SourceFile.open(fake50)
    assert o.video.sample_count == res.plan.out_frames
    assert o.video_frame_rate() == res.plan.out_fps
    assert o.video.timescale == 90000 and o.video_frame_duration() == 3003
    assert [t.kind for t in o.tracks] == [t.kind for t in s.tracks]
    # parameter sets equal to the source's except timing
    rep = verify.Report()
    verify.compare_parameter_sets(fake50, str(out), rep)
    assert not [c for c in rep.checks if c.status == "FAIL"], rep.render()
    # decoded frames of the output are exactly the planned source frames? (lossy encode -> compare count & sync pattern)
    frames = ff.ffprobe_frames(str(out))
    assert len(frames) == res.plan.out_frames
    assert all(f["pict_type"] in ("I", "P") for f in frames)
    # every sample: AUD + 1 slice, no SEI
    for smp in o.read_samples(o.video)[:30]:
        types = [N.h264_nal_type(n) for n in N.split_length_prefixed(smp)]
        assert types in ([9, 5], [9, 1]), types


@pytest.mark.slow
def test_end_to_end_hevc_fake50_to_2997(fake50_hevc, tmp_path):
    out = tmp_path / "out.mp4"
    res = run(Options(src=fake50_hevc, out=str(out), fps="29.97", preset="veryfast", verify=True, external_tools=False), log=lambda s: None)
    assert res.transplant_applied, res.notes
    assert res.lossless_verified
    o = SourceFile.open(str(out))
    assert o.video.format == b"hvc1" and o.video.sample_count == res.plan.out_frames
    assert o.video.handler_name == "GoPro H.265"
    rep = verify.Report()
    verify.compare_parameter_sets(fake50_hevc, str(out), rep)
    assert not [c for c in rep.checks if c.status == "FAIL"], rep.render()
    for smp in o.read_samples(o.video)[:30]:
        types = [N.hevc_nal_type(n) for n in N.split_length_prefixed(smp)]
        assert types[0] == N.HEVC_AUD and all(t <= 31 for t in types[1:]), types
    # SOS sample #1 carries VPS/SPS/PPS in the source's 256-byte blocks
    s1 = o.read_sample(o.track("fdsc").samples[1])
    assert len(s1) == 16 + 3 * 260


@pytest.mark.slow
def test_reference_template_path(fake50, hero8, tmp_path):
    out = tmp_path / "out.mp4"
    res = run(Options(src=fake50, out=str(out), fps="29.97", preset="veryfast", verify=True, external_tools=False, reference=hero8), log=lambda s: None)
    assert res.transplant_applied and res.lossless_verified
    o = SourceFile.open(str(out))
    r = SourceFile.open(hero8)
    # frame-rate-dependent conventions come from the reference, identity atoms from the source
    assert o.video.timescale == r.video.timescale and o.video_frame_duration() == r.video_frame_duration()
    assert o.moov.find("udta/SETT").data == r.moov.find("udta/SETT").data
    s = SourceFile.open(fake50)
    assert o.moov.find("udta/MUID").data == s.moov.find("udta/MUID").data
    assert o.moov.find("udta/GUMI").data == s.moov.find("udta/GUMI").data


def test_inspect_runs(fake50, capsys):
    from gopro_retime.inspect_cmd import inspect
    inspect(fake50)
    text = capsys.readouterr().out
    assert "SOS:" in text and "gpmd:" in text and "SPS:" in text
