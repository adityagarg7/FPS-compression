"""Build a GoPro-structured HEVC fixture: the frames of a real GoPro sample re-encoded with x265 using camera-like
settings that DIFFER from what the tool derives (so the transplant has real work to do), muxed as 'hvc1' with the
GoPro HEVC handler/compressor names, a 50 fps PAL timeline and a regenerated SOS track (3 parameter-set blocks)."""
from __future__ import annotations

import os
import struct
import subprocess
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gopro_retime import gpmf_rebuild, interleave, mp4box as mb, mux, plan as P, sos  # noqa: E402
from gopro_retime.h26x import nal as N  # noqa: E402
from gopro_retime.model import SourceFile  # noqa: E402

X265_CAMERA_LIKE = ("bframes=0:ref=2:keyint=25:min-keyint=25:no-open-gop=1:aud=1:repeat-headers=1:info=0:log-level=error:hrd=1:"
                    "vbv-maxrate=2500:vbv-bufsize=1000:bitrate=2200:wpp=0:sao=1:amp=0:rect=0:tskip=0:signhide=1:aq-mode=0:"
                    "cutree=0:ctu=32:min-cu-size=8:max-tu-size=32:tu-intra-depth=2:tu-inter-depth=2:log2-max-poc-lsb=8:"
                    "temporal-mvp=1:strong-intra-smoothing=1:deblock=0:0:range=full:videoformat=5:colorprim=1:transfer=1:colormatrix=1")


def make(src_path: str, out_path: str, fps: int = 50, workdir: str | None = None) -> str:
    workdir = workdir or os.path.dirname(os.path.abspath(out_path))
    src = SourceFile.open(src_path)
    v = src.video
    es = os.path.join(workdir, "fixture.h265")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src_path, "-map", "0:v:0", "-an", "-vf", f"setpts=N/{fps}/TB", "-r", str(fps),
                    "-pix_fmt", "yuvj420p", "-c:v", "libx265", "-preset", "fast", "-x265-params", X265_CAMERA_LIKE, "-f", "hevc", es], check=True)
    nals = list(N.iter_annexb_file(es))
    aus = list(N.group_access_units(iter(nals), "hevc"))
    ps = {"vps": [], "sps": [], "pps": []}
    for n in nals:
        t = N.hevc_nal_type(n)
        key = {N.HEVC_VPS: "vps", N.HEVC_SPS: "sps", N.HEVC_PPS: "pps"}.get(t)
        if key and n not in ps[key]:
            ps[key].append(n)
    samples, sync = [], []
    for au in aus:
        keep = [n for n in au.nals if N.is_aud(n, "hevc") or N.is_vcl(n, "hevc")]
        samples.append(N.to_length_prefixed(keep))
        sync.append(au.is_irap)
    nframes = len(samples)
    ts, fdur = 90000, 90000 // fps
    # hvc1 sample entry: fixed part cloned from avc1 (same width/height) with GoPro HEVC names
    ent = mb.parse_stsd(v.stbl.child("stsd"))[0]
    fixed = bytearray(ent.fixed)
    name = b"GoPro H.265 encoder"
    fixed[42:74] = bytes([len(name)]) + name.ljust(31, b"\x00")
    hvcc = _make_hvcc(ps)
    children = [c for c in ent.children if c.type == b"colr"] + [mb.Box(type=b"hvcC", data=hvcc)]
    vid_entry = mb.SampleEntry(b"hvc1", bytes(fixed), children, b"")
    # handler name
    hdlr = v.trak.find("mdia/hdlr")
    hdlr.data = hdlr.data[:24] + bytes([len(b"GoPro H.265")]) + b"GoPro H.265"
    tracks: dict[str, mux.OutTrack] = {}
    tracks["video"] = mux.OutTrack("video", samples, [fdur] * nframes, ts, sync=sync, stsd_entries=[vid_entry], source=v)
    a = src.track("audio")
    a_samples = src.read_samples(a)[: int(nframes / fps * a.timescale / 1024) + 1]
    tracks["audio"] = mux.OutTrack("audio", a_samples, [1024] * len(a_samples), a.timescale, source=a)
    t = src.track("tmcd")
    t_entries = mb.parse_stsd(t.stbl.child("stsd"))
    mb.set_tmcd_entry(t_entries[0], ts, fdur, fps)
    tracks["tmcd"] = mux.OutTrack("tmcd", [int(int.from_bytes(src.read_sample(t.samples[0]), "big") / float(src.video_frame_rate()) * fps).to_bytes(4, "big")],
                                  [nframes * fdur], ts, stsd_entries=t_entries, source=t)
    pl = P.make_plan(src.video_frame_rate(), Fraction(fps), nframes, "realtime")
    pl.frame_map = [min(v.sample_count - 1, round(i * float(src.video_frame_rate()) / fps)) for i in range(nframes)]
    pl.out_frames = nframes
    pl.src_frames = v.sample_count
    m_samples, durs = gpmf_rebuild.rebuild(src, pl, Fraction(fps), False, False, log=lambda s_: None)
    tracks["gpmd"] = mux.OutTrack("gpmd", m_samples, durs, 1000, source=src.track("gpmd"))
    hd8 = interleave.InterleaveConventions(Fraction(1168, 10000), Fraction(1001, 10000), Fraction(1335, 10000), True)
    order = interleave.order_samples(src, tracks, hd8)
    # SOS: header from the real file (timescale patched), sample #1 with VPS/SPS/PPS blocks of 256 bytes
    src_ps = N.parameter_sets_from_entry_children(v.stsd_entries[0].children, "h264")
    conv = sos.learn(src, "h264", src_ps)
    header = sos.patch_header_timescale(conv.header, src.video.timescale, ts)

    def fdsc_builder(order_, tracks_):
        out = [header]
        for kind, idx in order_:
            tr = tracks_[kind]
            size, dur = len(tr.samples[idx]), tr.durations[idx]
            if kind == "video":
                if idx == 0:
                    s1 = b"GP" + bytes([3, 1]) + struct.pack(">III", size, dur, dur)
                    for n in ps["vps"] + ps["sps"] + ps["pps"]:
                        s1 += struct.pack(">I", len(n)) + n.ljust(256, b"\x00")
                    out.append(s1)
                else:
                    out.append(b"GP" + bytes([0, 1 if tr.sync[idx] else 3]) + struct.pack(">III", size, dur, dur))
            else:
                d = fdur if kind == "tmcd" else dur
                out.append(b"GP" + bytes([{"audio": 4, "tmcd": 5, "gpmd": 6}[kind], 0]) + struct.pack(">III", size, d, 0))
        return out

    mux.write_output(src, out_path, tracks, order, fdsc_builder=fdsc_builder, mvhd_timescale=ts)
    return out_path


def _make_hvcc(ps: dict[str, list[bytes]]) -> bytes:
    from gopro_retime.h26x import hevc
    sps_f = hevc.parse_sps_nal(ps["sps"][0])
    ptl = sps_f
    head = bytearray(22)
    head[0] = 1
    head[1] = (ptl.get("general_profile_space", 0) << 6) | (ptl.get("general_tier_flag", 0) << 5) | ptl.get("general_profile_idc", 1)
    flags = ptl.get("general_profile_compatibility_flag", [0] * 32)
    val = 0
    for i, b in enumerate(flags[:32]):
        val |= (b & 1) << (31 - i)
    head[2:6] = val.to_bytes(4, "big")
    head[6:12] = b"\x90\x00\x00\x00\x00\x00"
    head[12] = ptl.get("general_level_idc", 93)
    head[13:15] = (0xF000).to_bytes(2, "big")
    head[15] = 0xFC; head[16] = 0xFC | sps_f["chroma_format_idc"]
    head[17] = 0xF8 | sps_f["bit_depth_luma_minus8"]; head[18] = 0xF8 | sps_f["bit_depth_chroma_minus8"]
    head[19:21] = (0).to_bytes(2, "big")
    head[21] = 0x0F  # constantFrameRate 0, numTemporalLayers 1, temporalIdNested 1, lengthSizeMinusOne 3
    arrays = [(1, N.HEVC_VPS, ps["vps"]), (1, N.HEVC_SPS, ps["sps"]), (1, N.HEVC_PPS, ps["pps"])]
    return N.build_hvcc(N.HvcC(bytes(head), arrays))


if __name__ == "__main__":
    print(make(sys.argv[1], sys.argv[2]))
