"""`gopro-retime-inspect FILE`: print everything the converter would learn from a GoPro recording."""
from __future__ import annotations

import sys
from fractions import Fraction

from . import gpmf, gpmf_rebuild, interleave, mp4box as mb, sos
from .h26x import nal as N
from .h26x import params as P
from .model import SourceFile
from .pipeline import codec_of
from .samples import detect_conventions


def inspect(path: str, out=sys.stdout) -> None:
    src = SourceFile.open(path)
    w = lambda s="": print(s, file=out)  # noqa: E731
    v = src.video
    fps = src.video_frame_rate()
    codec = codec_of(v)
    w(f"file: {path}")
    w(f"ftyp: {src.ftyp.data!r}   moov children: {[c.type.decode() for c in src.moov.children]}")
    w(f"mvhd: timescale {src.mvhd.timescale} duration {src.mvhd.duration} next_track_id {src.mvhd.next_track_id}")
    for t in src.tracks:
        info = f"  track {t.track_id} {t.key:6} {t.handler_name!r:16} fmt {t.format.decode()} ts {t.timescale} dur {t.media_duration} samples {t.sample_count} stsc {t.stsc} ctts={t.has_ctts} stss={t.has_stss} elst={t.elst}"
        w(info)
    vi = mb.video_entry_info(v.stsd_entries[0])
    w(f"video: {codec} {vi.width}x{vi.height} {float(fps):.3f} fps ({fps}) frame duration {src.video_frame_duration()} compressor {vi.compressor_name!r} children {[c.type.decode() for c in v.stsd_entries[0].children]}")
    ps = N.parameter_sets_from_entry_children(v.stsd_entries[0].children, codec)
    first = [src.read_sample(s) for s in v.samples[:40]]
    conv = detect_conventions(codec, first)
    w(f"video samples: AUD={conv.keep_aud} in-band parameter sets={conv.inband_param_sets} SEI types={conv.inband_sei_types} nal_ref_idc slice/idr/ps={conv.nal_ref_idc_slice}/{conv.nal_ref_idc_idr}/{conv.nal_ref_idc_ps}")
    sync = [i for i, s in enumerate(v.samples) if s.is_sync]
    gaps = sorted({b - a for a, b in zip(sync, sync[1:])})
    w(f"GOP: sync samples {len(sync)}, intervals {gaps[:5]}")
    for k, lst in ps.items():
        for n in lst:
            w(f"  {k.upper()} {len(n)} bytes: {n.hex()}")
    try:
        sps_f = P.parse_sps(ps["sps"][0], codec)
        pps_f = P.parse_pps(ps["pps"][0], codec, sps_f)
        keys = [k for k in sps_f if not k.startswith("_") and not isinstance(sps_f[k], (dict, list))]
        w("  SPS: " + ", ".join(f"{k}={sps_f[k]}" for k in keys))
        vui = sps_f.get("vui") or {}
        w("  VUI: " + ", ".join(f"{k}={vui[k]}" for k in vui if not isinstance(vui[k], (dict, list))))
        w(f"  HRD: bit rate {P.hrd_values(sps_f)}")
        w("  PPS: " + ", ".join(f"{k}={pps_f[k]}" for k in pps_f if not k.startswith("_") and not isinstance(pps_f[k], (dict, list))))
    except Exception as e:  # noqa: BLE001
        w(f"  (parameter set parse failed: {e})")
    tm = src.track("tmcd")
    if tm is not None:
        ti = mb.tmcd_entry_info(tm.stsd_entries[0])
        w(f"tmcd: flags {ti.flags:#x} timescale {ti.timescale} frame duration {ti.frame_duration} numberOfFrames {ti.number_of_frames} value {int.from_bytes(src.read_sample(tm.samples[0]), 'big')}")
    fd = src.track("fdsc")
    if fd is not None:
        try:
            sc = sos.learn(src, codec, ps)
            clk = sos.header_clock(sc.header, v.timescale)
            w(f"SOS: header {len(sc.header)} bytes, sample #1 {sc.sample1_size} bytes blocks {sc.ps_blocks}, types {sc.type_codes}, video flags {sc.video_flags}, X rule {sc.x_rule}, clock {clk}")
        except Exception as e:  # noqa: BLE001
            w(f"SOS: could not learn conventions: {e}")
    ic = interleave.measure(src)
    w(f"interleave: start {ic.start_pattern}, MET latency ({float(ic.latency_lo) * 1000:.1f}, {float(ic.latency_hi) * 1000:.1f}] ms, final payload after last audio={ic.final_payload_last}")
    w(f"layout: {src.layout_string()[:80]}...")
    gp = src.track("gpmd")
    if gp is not None:
        durs = [s.duration for s in gp.samples]
        period = max(set(durs), key=durs.count)
        payloads = [gpmf.parse(b) for b in src.read_samples(gp)]
        f_s = round(Fraction(period, 1000) * fps)
        w(f"gpmd: {len(durs)} payloads, period {period} ms ({f_s} frames), last {durs[-1]} ms, sizes {min(s.size for s in gp.samples)}-{max(s.size for s in gp.samples)} bytes")
        for d in range(max(len(p) for p in payloads)):
            streams = gpmf_rebuild.analyze(payloads, v.sample_count, f_s, period * 1000, fps, d)
            dv = payloads[0][d] if d < len(payloads[0]) else None
            name = dv.child("DVNM").data.rstrip(b"\x00").decode("latin1", "replace") if dv is not None and dv.child("DVNM") else "?"
            w(f"  device {d} {name!r}:")
            for st in streams:
                cnts = st.counts[:4]
                w(f"    {(st.key or b'----').decode('latin1'):4} {st.cls:9} stride {st.stride} lag {st.lag_src_frames} tail {st.extra_tail:+d} grouped={st.grouped} stmp={st.has_stmp} counts {cnts}... total {sum(st.counts)}  {st.name.rstrip(b'\x00')[:40]!r}")
    u = src.moov.find("udta")
    if u is not None:
        w(f"udta: {[c.type.decode('latin1') + '(' + str(c.serialized_size()) + ')' for c in u.children]}")
        g = u.child("GPMF")
        if g is not None:
            klvs, _tr = gpmf.parse_with_trailing(g.data)
            for d in klvs:
                nm = d.child("DVNM")
                keys = [c.key.decode("latin1") for c in d.children or []]
                w(f"  GPMF DEVC {nm.data.rstrip(b'\x00') if nm else b''!r}: {keys}")


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: gopro-retime-inspect FILE.MP4", file=sys.stderr)
        return 2
    inspect(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
