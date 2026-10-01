from fractions import Fraction

from gopro_retime import gpmf, mp4box as mb, mux, plan as P
from gopro_retime.model import SourceFile


def test_box_roundtrip_all_samples(all_samples):
    for p in all_samples:
        tl = mb.parse_file(p)
        buf = open(p, "rb").read()
        for b in tl.boxes:
            if b.type == b"mdat":
                continue
            assert b.serialize() == buf[b.offset:b.offset + b.size], (p, b.type)


def test_stsd_roundtrip_all_samples(all_samples):
    for p in all_samples:
        for stsd in mb.parse_file(p).find("moov").find_all("trak/mdia/minf/stbl/stsd"):
            assert mb.build_stsd(mb.parse_stsd(stsd), stsd.data[:4]).data == stsd.data


def test_sample_tables_contiguous(all_samples):
    for p in all_samples:
        s = SourceFile.open(p)
        items = sorted((smp.offset, smp.size) for t in s.tracks for smp in t.samples)
        pos = s.mdat.offset + s.mdat.header_size
        for o, sz in items:
            assert o == pos, p
            pos += sz
        assert pos == s.mdat.offset + s.mdat.size


def test_identity_remux_is_byte_identical(hero8, tmp_path):
    src = SourceFile.open(hero8)
    tracks = {}
    for t in src.tracks:
        if t.kind == "fdsc":
            continue
        tracks[t.kind] = mux.OutTrack(t.kind, src.read_samples(t), [s.duration for s in t.samples], t.timescale,
                                      sync=[s.is_sync for s in t.samples] if t.has_stss else None,
                                      cts_offsets=[s.cts_offset for s in t.samples] if t.has_ctts else None,
                                      source=t, media_duration=t.media_duration, tkhd_duration=t.tkhd_duration)
    fd = src.read_samples(src.track("fdsc"))
    order = [(t.kind, s.index) for t, s in src.all_samples_in_file_order() if t.kind != "fdsc"]
    out = tmp_path / "remux.mp4"
    mux.write_output(src, str(out), tracks, order, fdsc_builder=lambda o, t: fd, mvhd_timescale=src.mvhd.timescale)
    assert out.read_bytes() == open(hero8, "rb").read()


def test_gpmf_roundtrip_all_samples(all_samples):
    for p in all_samples:
        s = SourceFile.open(p)
        for smp in s.read_samples(s.track("gpmd")):
            k, tr = gpmf.parse_with_trailing(smp)
            assert gpmf.serialize(k) + tr == smp
        u = s.moov.find("udta/GPMF")
        if u is not None:
            k, tr = gpmf.parse_with_trailing(u.data)
            assert gpmf.serialize(k) + tr == u.data


def test_plan_nearest_no_duplicates():
    import math
    for sf, of, n in [(Fraction(50), P.NTSC_30, 379), (Fraction(30000, 1001), Fraction(24000, 1001), 379),
                      (Fraction(50), Fraction(30), 1000), (Fraction(60000, 1001), P.NTSC_30, 777), (Fraction(50), Fraction(25), 101)]:
        fm = P.nearest_frame_map(sf, of, n)
        assert len(set(fm)) == len(fm)
        assert fm == sorted(fm)
        r = sf / of
        p, q = r.numerator, r.denominator
        kept = [j for j in range(n) if j == math.floor((2 * math.floor(j * q / p + 0.5) * p + q) / (2 * q))]
        assert kept == fm
        # every output slot's chosen frame is the nearest in time
        for k, j in enumerate(fm):
            t_out = k / of
            assert abs(j / sf - t_out) <= (1 / sf) / 2 + Fraction(1, 10**9)


def test_parse_fps():
    assert P.parse_fps("29.97") == Fraction(30000, 1001)
    assert P.parse_fps("30") == Fraction(30)
    assert P.parse_fps("24000/1001") == Fraction(24000, 1001)
    assert P.parse_fps("23.976") == Fraction(24000, 1001)
