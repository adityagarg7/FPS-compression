import os

from gopro_retime import gpmf, gpmf_rebuild, plan as P
from gopro_retime.model import SourceFile


def _norm(b: bytes) -> bytes:
    """Mask STMP (sub-ms jitter) and pad bytes (old firmware leaves garbage; we zero-fill like HD8+)."""
    k, tr = gpmf.parse_with_trailing(b)
    for d in k:
        for _, it in d.walk():
            if it.key == b"STMP":
                it.data = b"\x00" * 8
            it.pad_bytes = b""
    return gpmf.serialize(k) + tr


# hero7: firmware's own FACE TSMP counter is inconsistent in its last payload; karma: metadata longer than the video
EXPECT_EXACT = {"hero8.mp4", "max-heromode.mp4", "max-360mode.mp4", "hero6.mp4", "hero6a.mp4", "hero6+ble.mp4", "hero5.mp4", "Fusion.mp4"}


def test_identity_rebuild_reproduces_firmware_payloads(all_samples):
    for p in all_samples:
        name = os.path.basename(p)
        s = SourceFile.open(p)
        fps = s.video_frame_rate()
        pl = P.make_plan(fps, fps, s.video.sample_count, "conform")
        out, durs = gpmf_rebuild.rebuild(s, pl, fps, False, False, log=lambda m: None)
        orig = s.read_samples(s.track("gpmd"))
        gp = s.track("gpmd")
        if name in EXPECT_EXACT:
            assert durs == [x.duration for x in gp.samples], name
            assert len(out) == len(orig), name
            bad = [i for i, (a, b) in enumerate(zip(out, orig)) if _norm(a) != _norm(b)]
            assert not bad, (name, bad)
        # STMP within a few microseconds everywhere
        for a, b in zip(out, orig):
            for (_, ia), (_, ib) in zip(list(gpmf.parse(a)[0].walk()), list(gpmf.parse(b)[0].walk())):
                if ia.key == b"STMP" and ib.key == b"STMP":
                    assert abs(int.from_bytes(ia.data, "big") - int.from_bytes(ib.data, "big")) <= 5, name


def test_decimation_50_to_2997_counts(fake50):
    s = SourceFile.open(fake50)
    pl = P.make_plan(s.video_frame_rate(), P.NTSC_30, s.video.sample_count)
    out, durs = gpmf_rebuild.rebuild(s, pl, P.NTSC_30, drop_imu=True, drop_gps=False, log=lambda m: None)
    assert durs[:-1] == [1001] * (len(durs) - 1)
    for b in out:
        devc = gpmf.parse(b)[0]
        keys = [st.children[-1].key for st in devc.children_of("STRM")]
        assert b"ACCL" not in keys and b"GYRO" not in keys and b"CORI" not in keys
        assert b"SHUT" in keys
    # SHUT total == output frames
    total = 0
    for b in out:
        devc = gpmf.parse(b)[0]
        for st in devc.children_of("STRM"):
            if st.children[-1].key == b"SHUT":
                total += st.children[-1].repeat
    assert total == pl.out_frames


def test_vfps_patch_keeps_length():
    devc = gpmf.make_nested(b"DEVC", [gpmf.make(b"DVID", b"L", 4, (1).to_bytes(4, "big")),
                                       gpmf.make(b"VFPS", b"L", 4, (50).to_bytes(4, "big") + (1).to_bytes(4, "big")),
                                       gpmf.make(b"ORDP", b"c", 1, b"Y")])
    blob = gpmf.serialize([devc]).ljust(256, b"\x00")
    out = gpmf_rebuild.patch_global_settings_fps(blob, P.NTSC_30, True, log=lambda m: None)
    assert len(out) == len(blob)
    d = gpmf.parse(out)[0]
    assert d.child("VFPS").values() == [30000, 1001]
    assert d.child("ORDP").data == b"N"


def test_audio_clock_streams_stay_10hz_at_50fps(fake50):
    """At exactly 50 fps the 10 Hz audio-clock streams coincide with every 5th frame; they must still be re-binned by time."""
    s = SourceFile.open(fake50)
    pl = P.make_plan(s.video_frame_rate(), P.NTSC_30, s.video.sample_count)
    out, durs = gpmf_rebuild.rebuild(s, pl, P.NTSC_30, drop_imu=True, drop_gps=False, log=lambda m: None)
    for b in out[:-1]:
        devc = gpmf.parse(b)[0]
        for st in devc.children_of("STRM"):
            key = st.children[-1].key
            if key in (b"WNDM", b"MWET", b"AALP"):
                assert st.children[-1].repeat in (10, 11), (key, st.children[-1].repeat)


def test_hero11_pal_track_classification():
    """Real HERO11 25 fps track (1040 ms / 26-frame payloads, off-grid final payload): per-frame streams stay per-frame."""
    import os
    raw_path = "/tmp/claude-0/-home-user-FPS-compression/323d0a30-0e99-596b-9ec6-bf370f087cf7/scratchpad/samples/hero11_gpmd_track.raw"
    if not os.path.exists(raw_path):
        import pytest
        pytest.skip("HERO11 raw track not available")
    from fractions import Fraction
    devcs = gpmf.parse(open(raw_path, "rb").read())
    payloads = [[d] for d in devcs]
    # SHUT total across the track tells how many frames the metadata covers
    shut_total = 0
    for d in devcs:
        for st in d.children_of("STRM"):
            if st.children[-1].key == b"SHUT":
                shut_total += st.children[-1].repeat
    streams = gpmf_rebuild.analyze(payloads, shut_total, 26, 1040000, Fraction(25), 0, covered_frames=shut_total)
    cls = {st.key: st.cls for st in streams}
    for k in (b"SHUT", b"ISOE", b"CORI", b"IORI", b"GRAV", b"MSKP"):
        assert cls[k] == "per_frame", (k, cls[k])
    for k in (b"WNDM", b"MWET", b"AALP", b"ACCL", b"GYRO", b"GPS9"):
        assert cls[k] == "timed", (k, cls[k])
