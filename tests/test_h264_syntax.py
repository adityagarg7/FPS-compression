import subprocess

from gopro_retime.h26x import h264, nal as N
from gopro_retime.model import SourceFile


def _roundtrip_stream(nals):
    sp = next(n for n in nals if N.h264_nal_type(n) == N.H264_SPS)
    pp = next(n for n in nals if N.h264_nal_type(n) == N.H264_PPS)
    spsf = h264.parse_sps_nal(sp)
    assert h264.write_sps_nal(spsf) == sp
    ppsf = h264.parse_pps_nal(pp, spsf)
    assert h264.write_pps_nal(ppsf, spsf) == pp
    cnt = 0
    for n in nals:
        if N.is_vcl(n, "h264"):
            f, data, hb = h264.parse_slice_nal(n, spsf, ppsf)
            assert h264.write_slice_nal(f, data, hb, spsf, ppsf) == n
            cnt += 1
    return cnt


def test_gopro_h264_roundtrip(all_samples):
    for p in all_samples:
        s = SourceFile.open(p)
        v = s.video
        ps = N.parameter_sets_from_entry_children(v.stsd_entries[0].children, "h264")
        nals = ps["sps"] + ps["pps"]
        for smp in s.read_samples(v):
            nals += N.split_length_prefixed(smp)
        assert _roundtrip_stream(nals) == v.sample_count


def _x264(tmp_path, params, extra=()):
    out = tmp_path / "t.h264"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-t", "1.2",
                    "-c:v", "libx264", "-x264-params", params, *extra, "-f", "h264", str(out)], check=True)
    return list(N.iter_annexb_file(str(out)))


def test_x264_variants_roundtrip(tmp_path):
    for params, extra in [
        ("aud=1:nal-hrd=vbr:vbv-maxrate=2000:vbv-bufsize=1000:bframes=0:ref=2", ["-profile:v", "main"]),
        ("cabac=0:8x8dct=1:weightp=2:bframes=2:b-pyramid=0:aq-mode=1:deblock=1,1:slices=2", ["-profile:v", "high"]),
        ("bframes=3:b-pyramid=2:weightb=1:ref=4:open-gop=1:keyint=12", ["-profile:v", "high"]),
        ("interlaced=1:bframes=1", ["-profile:v", "high"]),
        ("cqm=jvt:8x8dct=1", ["-profile:v", "high"]),
        ("bframes=0:ref=1:chroma-qp-offset=3:constrained-intra=1", ["-profile:v", "main"]),
    ]:
        assert _roundtrip_stream(_x264(tmp_path, params, extra)) > 0


def test_epb_roundtrip():
    for b in (b"\x00\x00\x00\x01", b"\x00\x00\x02\x00\x00\x03\x00\x00\x00", b"\x00\x00", bytes(range(256)) * 3):
        assert N.remove_epb(N.insert_epb(b)) == b
