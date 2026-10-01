"""End-to-end conversion pipeline."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Optional

from . import ffmpeg as ff
from . import mp4box as mb
from . import mux
from .encode import EncoderSettings, encode
from .h26x import nal as N
from .h26x import derive, params, rewrite
from .model import SourceFile, Track
from .plan import FramePlan, make_plan, parse_fps
from .samples import VideoConventions, build_samples, detect_conventions

Log = Callable[[str], None]


@dataclass
class Options:
    src: str
    out: str
    fps: str = "29.97"
    mode: str = "realtime"            # realtime | conform
    imu: str = "drop"                 # drop | keep
    gps: str = "keep"                 # keep | drop
    gpmf: str = "rebuild"             # rebuild | drop
    reference: Optional[str] = None
    bitrate: Optional[int] = None
    maxrate: Optional[int] = None
    bufsize: Optional[int] = None
    gop: Optional[int] = None
    preset: str = "slow"
    two_pass: bool = False
    verify: bool = True
    external_tools: bool = True
    keep_temp: bool = False
    workdir: Optional[str] = None
    no_transplant: bool = False
    threads: int = 0
    encoder_params: list[str] = field(default_factory=list)


@dataclass
class Result:
    out: str
    plan: FramePlan
    settings: EncoderSettings
    transplant_applied: bool
    lossless_verified: Optional[bool]
    report_text: str = ""
    notes: list[str] = field(default_factory=list)


def codec_of(track: Track) -> str:
    fmt = track.format
    if fmt in (b"avc1", b"avc3"):
        return "h264"
    if fmt in (b"hvc1", b"hev1"):
        return "hevc"
    raise ValueError(f"unsupported video sample entry {fmt!r}")


def _color_from_probe(stream: dict) -> dict:
    rng = stream.get("color_range", "tv")
    return {
        "range": "pc" if rng in ("pc", "jpeg", "full") else "tv",
        "primaries": stream.get("color_primaries", "bt709"),
        "trc": stream.get("color_transfer", "bt709"),
        "space": stream.get("color_space", "bt709"),
        "chroma_location": stream.get("chroma_location"),
    }


def _src_gop(video: Track) -> int:
    sync = [i for i, s in enumerate(video.samples) if s.is_sync]
    if len(sync) < 2:
        return max(1, video.sample_count)
    gaps = [b - a for a, b in zip(sync, sync[1:])]
    return max(set(gaps), key=gaps.count)


def _hrd_from_sps(sps_nal: bytes, codec: str) -> tuple[Optional[int], Optional[int]]:
    try:
        f = params.parse_sps(sps_nal, codec)
    except params.NotImplementedYet:
        return None, None
    return f.get("hrd_bit_rate"), f.get("hrd_cpb_size")


def run(opts: Options, log: Log = print) -> Result:
    t0 = time.time()
    ff.check_tools()
    src = SourceFile.open(opts.src)
    video = src.video
    codec = codec_of(video)
    out_fps = parse_fps(opts.fps)
    src_fps = src.video_frame_rate()
    plan = make_plan(src_fps, out_fps, video.sample_count, opts.mode)
    log(f"source: {codec} {video.sample_count} frames @ {src_fps} ({float(src_fps):.3f} fps) -> "
        f"{plan.out_frames} frames @ {out_fps} ({float(out_fps):.3f} fps), mode={plan.mode}")
    ref = SourceFile.open(opts.reference) if opts.reference else None

    workdir = opts.workdir or tempfile.mkdtemp(prefix="gopro-retime-")
    os.makedirs(workdir, exist_ok=True)
    notes: list[str] = []
    try:
        # ---- 1. source facts --------------------------------------------------------------------
        probe = ff.ffprobe_json(opts.src)
        vstream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
        color = _color_from_probe(vstream)
        pix_fmt = vstream.get("pix_fmt", "yuv420p")
        ps_src = N.parameter_sets_from_entry_children(video.stsd_entries[0].children, codec)
        first_samples = [src.read_sample(s) for s in video.samples[:40]]
        conv = detect_conventions(codec, first_samples)
        log(f"source conventions: AUD={conv.keep_aud} in-band parameter sets={conv.inband_param_sets} SEI types={conv.inband_sei_types}")
        src_gop = _src_gop(video)
        src_bitrate = int(video.total_bytes * 8 * video.timescale / max(1, video.media_duration))
        hrd_br, hrd_cpb = _hrd_from_sps(ps_src["sps"][0], codec) if ps_src.get("sps") else (None, None)

        # ---- 2. encoder settings ----------------------------------------------------------------
        if opts.bitrate:
            bitrate = opts.bitrate
        else:
            # same bits per frame as the source scaled by frame-rate ratio ^ 0.8 (fewer frames need slightly more bits each)
            bitrate = int(src_bitrate * float(out_fps / src_fps) ** 0.8) if plan.mode == "realtime" else src_bitrate
            if hrd_br:
                bitrate = min(bitrate, hrd_br)
        gop_frames = opts.gop or (ref and _src_gop(ref.video)) or src_gop
        settings = derive.derive_settings(codec, ps_src, int(vstream["width"]), int(vstream["height"]), pix_fmt,
                                          src_fps, out_fps, src_gop, bitrate, opts.maxrate or hrd_br, opts.bufsize or hrd_cpb,
                                          gop_frames, opts.preset, color, samples=first_samples)
        settings.threads = opts.threads
        for kv in opts.encoder_params:
            k, _, v = kv.partition("=")
            (settings.x264_params if codec == "h264" else settings.x265_params)[k] = v
        notes.extend(settings.notes)
        if not opts.no_transplant:
            from .h26x import calibrate
            cal = calibrate.calibrate(settings, ps_src, log=log)
            notes.extend(f"calibration: {a}" for a in cal.adjusted)
            if not cal.ok:
                notes.extend(f"decode-affecting mismatch (cannot be fixed by rewriting): {r}" for r in cal.residual)
        log(f"encoder: {settings.codec} bitrate={settings.bitrate} maxrate={settings.maxrate} bufsize={settings.bufsize} gop={settings.gop} refs={settings.refs} bframes={settings.bframes}")

        # ---- 3. encode --------------------------------------------------------------------------
        es_path = os.path.join(workdir, "video" + settings.es_suffix)
        log("encoding ...")
        encode(opts.src, plan, settings, es_path, log=log, two_pass=opts.two_pass, workdir=workdir)
        aus = list(N.group_access_units(N.iter_annexb_file(es_path), codec))
        if len(aus) != plan.out_frames:
            raise RuntimeError(f"encoder produced {len(aus)} access units, plan expects {plan.out_frames}")
        enc_ps = _collect_param_sets(aus, codec)
        log(f"encoded {len(aus)} access units in {time.time() - t0:.1f}s")

        # ---- 4. parameter-set transplant + slice header rewrite ---------------------------------
        target_ps = _patched_target_ps(ps_src, codec, out_fps)
        applied = False
        lossless: Optional[bool] = None
        if not opts.no_transplant:
            try:
                conv_slices = _measure_slice_conventions(codec, first_samples, ps_src)
                tr = rewrite.transplant(aus, codec, enc_ps, target_ps, conv_slices, log=log)
            except rewrite.RewriteUnsafe as e:
                tr = rewrite.TransplantResult(aus, enc_ps, [], [f"unsafe: {e}"], False)
            notes.extend(tr.residual_differences)
            if tr.applied:
                # mandatory losslessness proof: decode both and compare
                rew_path = os.path.join(workdir, "video.rewritten" + settings.es_suffix)
                with open(rew_path, "wb") as f:
                    for au in tr.aus:
                        f.write(N.to_annexb(_with_param_sets(au, tr.param_sets, codec)))
                m1 = ff.decode_md5(es_path)
                m2 = ff.decode_md5(rew_path)
                lossless = (m1 == m2)
                if lossless:
                    aus, final_ps, applied = tr.aus, tr.param_sets, True
                    log(f"transplant applied ({len(tr.rewritten_fields)} field groups rewritten); decode verified identical")
                else:
                    notes.append("transplant produced a different decode; discarded (encoder parameter sets kept)")
                    log("WARNING: transplanted stream decodes differently; keeping encoder parameter sets")
                    final_ps = enc_ps
            else:
                final_ps = enc_ps
        else:
            final_ps = enc_ps

        # ---- 5. build video samples -------------------------------------------------------------
        from . import interleave, reference, sos
        sos_conv = sos.learn(src, codec, ps_src) if src.track("fdsc") is not None else None
        il_conv = interleave.measure(src)
        sos_header_override = None
        if ref is not None:
            ref_ps = N.parameter_sets_from_entry_children(ref.video.stsd_entries[0].children, codec)
            if ref.track("fdsc") is not None and sos_conv is not None:
                ref_sos = sos.learn(ref, codec, ref_ps)
                sos_header_override = reference.merged_sos_header(sos_conv.header, ref_sos.header, video.timescale, ref.video.timescale, log)
            il_conv = interleave.measure(ref)
            reference.apply(src, ref, log)
        log(f"source writer conventions: MET latency {float(il_conv.latency) * 1000:.1f} ms, final payload after last audio={il_conv.final_payload_last}, "
            f"SOS types={sos_conv.type_codes if sos_conv else None}")
        built = build_samples(aus, conv, final_ps)
        out_ts, out_fdur = _video_timescale(out_fps, ref)
        vid_entries = mb.parse_stsd(video.stbl.child("stsd"))
        _replace_codec_config(vid_entries[0], codec, final_ps)
        tracks: dict[str, mux.OutTrack] = {}
        tracks["video"] = mux.OutTrack("video", built.samples, [out_fdur] * len(built.samples), out_ts,
                                       sync=built.sync, cts_offsets=None, stsd_entries=vid_entries, source=video)

        # ---- 6. audio ---------------------------------------------------------------------------
        audio = src.track("audio")
        if audio is not None:
            if plan.mode == "realtime":
                tracks["audio"] = mux.OutTrack("audio", src.read_samples(audio), [s.duration for s in audio.samples],
                                               audio.timescale, source=audio, media_duration=audio.media_duration)
            else:
                raise NotImplementedError("conform mode audio stretching not implemented yet")

        # ---- 7. timecode ------------------------------------------------------------------------
        tmcd = src.track("tmcd")
        if tmcd is not None:
            t_entries = mb.parse_stsd(tmcd.stbl.child("stsd"))
            info = mb.tmcd_entry_info(t_entries[0])
            n_frames_field = _tmcd_number_of_frames(out_fps, ref)
            mb.set_tmcd_entry(t_entries[0], out_ts, out_fdur, n_frames_field)
            old = int.from_bytes(src.read_sample(tmcd.samples[0]), "big")
            new_val = int(Fraction(old) / src_fps * out_fps) if plan.mode == "realtime" else old
            clk = sos_conv and sos.header_clock(sos_conv.header, video.timescale)
            if clk and plan.mode == "realtime":
                # the firmware derives the start timecode from its RTC (seconds since midnight + ms) at the recording rate
                new_val = int((clk[1] + Fraction(clk[2], 1000)) * out_fps)
            tracks["tmcd"] = mux.OutTrack("tmcd", [new_val.to_bytes(4, "big")], [len(built.samples) * out_fdur], out_ts,
                                          stsd_entries=t_entries, source=tmcd)

        # ---- 8. GPMF metadata -------------------------------------------------------------------
        gp = src.track("gpmd")
        if gp is not None and opts.gpmf != "drop":
            from . import gpmf_rebuild
            payloads, durations = gpmf_rebuild.rebuild(src, plan, out_fps, drop_imu=(opts.imu == "drop"),
                                                       drop_gps=(opts.gps == "drop"), reference=ref, log=log)
            tracks["gpmd"] = mux.OutTrack("gpmd", payloads, durations, gp.timescale, source=gp)
        udta_gpmf = src.moov.find("udta/GPMF")
        if udta_gpmf is not None and plan.mode == "realtime":
            from . import gpmf_rebuild
            udta_gpmf.data = gpmf_rebuild.patch_global_settings_fps(udta_gpmf.data, out_fps, opts.imu == "drop", log=log)
        elif gp is not None:
            notes.append("gpmd track dropped by request (a native file always has one)")

        # ---- 9. interleave + SOS + write -------------------------------------------------------
        order = interleave.order_samples(src, tracks, il_conv)
        fdsc_builder = sos.make_builder(src, codec, final_ps, out_ts, out_fdur, sos_conv, header_override=sos_header_override) if sos_conv else None
        if gp is not None and opts.gpmf == "drop":
            _drop_track(src, "gpmd")
        res = mux.write_output(src, opts.out, tracks, order, fdsc_builder=fdsc_builder, mvhd_timescale=out_ts, log=log)

        # ---- 10. verification -------------------------------------------------------------------
        report_text = ""
        if opts.verify:
            from . import verify
            m_es = ff.decode_md5(es_path) if lossless is None else None
            m_out = ff.decode_md5(opts.out)
            m_ref = ff.decode_md5(es_path if not applied else os.path.join(workdir, "video.rewritten" + settings.es_suffix))
            ok = (m_out == m_ref)
            rep = verify.full_report(opts.src, opts.out, codec, opts.reference, external_tools=opts.external_tools,
                                     imu_dropped=(opts.imu == "drop" and opts.gpmf != "drop"), gps_dropped=(opts.gps == "drop"))
            rep.add("output video decodes identically to the encoder's elementary stream", "PASS" if ok else "FAIL", f"{m_ref} vs {m_out}")
            report_text = rep.render()
            log(report_text)
        log(f"done in {time.time() - t0:.1f}s -> {opts.out}")
        return Result(opts.out, plan, settings, applied, lossless, report_text, notes)
    finally:
        if not opts.keep_temp and not opts.workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _measure_slice_conventions(codec: str, samples: list[bytes], ps: dict[str, list[bytes]]):
    if codec == "h264":
        from .h26x import h264
        sps_f = h264.parse_sps_nal(ps["sps"][0])
        pps_f = h264.parse_pps_nal(ps["pps"][0], sps_f)
        return rewrite.measure_h264_conventions(samples, sps_f, pps_f)
    return None


def _collect_param_sets(aus: list[N.AccessUnit], codec: str) -> dict[str, list[bytes]]:
    out: dict[str, list[bytes]] = {"vps": [], "sps": [], "pps": []}
    for au in aus[:2]:
        for n in au.nals:
            t = N.nal_type(n, codec)
            if codec == "h264":
                if t == N.H264_SPS and n not in out["sps"]:
                    out["sps"].append(n)
                elif t == N.H264_PPS and n not in out["pps"]:
                    out["pps"].append(n)
            else:
                if t == N.HEVC_VPS and n not in out["vps"]:
                    out["vps"].append(n)
                elif t == N.HEVC_SPS and n not in out["sps"]:
                    out["sps"].append(n)
                elif t == N.HEVC_PPS and n not in out["pps"]:
                    out["pps"].append(n)
    if codec == "h264":
        out.pop("vps")
    return out


def _with_param_sets(au: N.AccessUnit, ps: dict[str, list[bytes]], codec: str) -> list[bytes]:
    """NALs of an AU for an Annex B dump: parameter sets prepended on IRAP, SEI/filler dropped."""
    nals = [n for n in au.nals if not N.is_sei(n, codec) and not N.is_filler(n, codec) and not N.is_param_set(n, codec)]
    if au.is_irap:
        order = ["vps", "sps", "pps"] if codec == "hevc" else ["sps", "pps"]
        pre = [n for k in order for n in ps.get(k, [])]
        aud = [n for n in nals if N.is_aud(n, codec)]
        rest = [n for n in nals if not N.is_aud(n, codec)]
        return aud + pre + rest
    return nals


def _patched_target_ps(ps_src: dict[str, list[bytes]], codec: str, out_fps: Fraction) -> dict[str, list[bytes]]:
    """Source parameter sets with VUI timing rewritten for the output frame rate (H.264 counts fields: 2x)."""
    out = {k: list(v) for k, v in ps_src.items()}
    try:
        if codec == "h264":
            num, ts = out_fps.denominator, out_fps.numerator * 2
        else:
            num, ts = out_fps.denominator, out_fps.numerator
        out["sps"] = [params.patch_vui_timing(s, codec, num, ts) for s in out.get("sps", [])]
        if codec == "hevc" and out.get("vps"):
            out["vps"] = [params.patch_vui_timing(v, "hevc-vps", num, ts) for v in out["vps"]]
    except params.NotImplementedYet:
        pass
    return out


def _video_timescale(out_fps: Fraction, ref: Optional[SourceFile]) -> tuple[int, int]:
    """GoPro convention: 29.97 -> 90000/3003; 23.976 -> 24000/1001; integer rates -> 90000/(90000/fps)."""
    if ref is not None:
        return ref.video.timescale, ref.video_frame_duration()
    if out_fps == Fraction(30000, 1001):
        return 90000, 3003
    if out_fps == Fraction(24000, 1001):
        return 24000, 1001
    if out_fps == Fraction(60000, 1001):
        return 90000, 1501  # not exact (1501.5); GoPro uses 60000/1001 for 59.94 — use that instead
    if out_fps.denominator == 1001:
        return out_fps.numerator, 1001
    if 90000 % out_fps.numerator == 0 and out_fps.denominator == 1:
        return 90000, 90000 // out_fps.numerator
    return out_fps.numerator * 1000, out_fps.denominator * 1000


def _tmcd_number_of_frames(out_fps: Fraction, ref: Optional[SourceFile]) -> int:
    if ref is not None and ref.track("tmcd") is not None:
        return mb.tmcd_entry_info(ref.track("tmcd").stsd_entries[0]).number_of_frames
    return int(out_fps)  # floor: GoPro writes 29 for 29.97 and 23 for 23.976


def _replace_codec_config(entry: mb.SampleEntry, codec: str, ps: dict[str, list[bytes]]) -> None:
    for c in entry.children:
        if codec == "h264" and c.type == b"avcC":
            a = N.parse_avcc(c.data)
            a.sps, a.pps = ps["sps"], ps["pps"]
            # profile/level bytes mirror the SPS
            sps_rbsp = N.remove_epb(ps["sps"][0])
            a.profile_idc, a.profile_compat, a.level_idc = sps_rbsp[1], sps_rbsp[2], sps_rbsp[3]
            c.data = N.build_avcc(a)
        elif codec == "hevc" and c.type == b"hvcC":
            h = N.parse_hvcc(c.data)
            new_arrays = []
            for completeness, nt, nals in h.arrays:
                key = {N.HEVC_VPS: "vps", N.HEVC_SPS: "sps", N.HEVC_PPS: "pps"}.get(nt)
                new_arrays.append((completeness, nt, ps.get(key, nals) if key else nals))
            h.arrays = new_arrays
            c.data = N.build_hvcc(h)


def _drop_track(src: SourceFile, kind: str) -> None:
    for t in list(src.tracks):
        if t.kind == kind:
            src.moov.remove_child(t.trak)
            src.tracks.remove(t)
