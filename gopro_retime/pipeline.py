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
    failed_checks: int = 0


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
    return params.hrd_values(params.parse_sps(sps_nal, codec))


def run(opts: Options, log: Log = print) -> Result:
    t0 = time.time()
    ff.check_tools()
    if not os.path.exists(opts.src):
        raise FileNotFoundError(f"source not found: {opts.src}")
    if opts.reference and not os.path.exists(opts.reference):
        raise FileNotFoundError(f"reference not found: {opts.reference}")
    src = SourceFile.open(opts.src)
    _require_gopro(src)
    video = src.video
    codec = codec_of(video)
    out_fps = parse_fps(opts.fps)
    if out_fps <= 0:
        raise ValueError(f"invalid output frame rate {opts.fps}")
    src_fps = src.video_frame_rate()
    if out_fps > src_fps:
        raise ValueError(f"output rate {out_fps} is higher than the source's {src_fps}: frames would have to be duplicated or "
                         f"interpolated, which a camera never does; only down-conversion is supported")
    plan = make_plan(src_fps, out_fps, video.sample_count, opts.mode)
    log(f"source: {codec} {video.sample_count} frames @ {src_fps} ({float(src_fps):.3f} fps) -> "
        f"{plan.out_frames} frames @ {out_fps} ({float(out_fps):.3f} fps), mode={plan.mode}")
    ref = SourceFile.open(opts.reference) if opts.reference else None
    if ref is not None:
        from . import reference as _reference
        _reference.check_compatible(src, ref, out_fps)   # raises IncompatibleReference with a clear message

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
        from . import modes
        ten_bit = "10" in pix_fmt
        width, height = int(vstream["width"]), int(vstream["height"])
        ref_hrd = None
        if ref is not None:
            ref_ps = N.parameter_sets_from_entry_children(ref.video.stsd_entries[0].children, codec)
            ref_hrd = _hrd_from_sps(ref_ps["sps"][0], codec) if ref_ps.get("sps") else (None, None)
        if opts.bitrate:
            bitrate = opts.bitrate
            notes.append(f"bitrate {bitrate} bps set on the command line")
        elif ref_hrd and ref_hrd[0]:
            bitrate = ref_hrd[0]
            notes.append(f"bitrate {bitrate} bps taken from the reference recording's HRD")
        else:
            nominal = modes.target_bitrate(width, height, src_fps, out_fps, hrd_br, ten_bit)
            if nominal:
                bitrate = nominal
                notes.append(f"bitrate {bitrate} bps = nominal GoPro {modes.classify_setting(width, height, src_fps, hrd_br or 0, ten_bit)} bit-rate setting for {width}x{height} @ {float(out_fps):.3g}")
            else:
                bitrate = hrd_br or src_bitrate
                notes.append(f"bitrate {bitrate} bps kept from the source (mode not in the nominal bit-rate table)")
        # HRD: the camera writes the nominal rate into the SPS; CPB keeps the source's CPB/bit-rate ratio
        hrd_target_br = (opts.maxrate or (ref_hrd[0] if ref_hrd and ref_hrd[0] else None) or (bitrate if hrd_br else None))
        hrd_target_cpb = (opts.bufsize or (ref_hrd[1] if ref_hrd and ref_hrd[1] else None)
                          or (int(hrd_target_br * hrd_cpb / hrd_br) if (hrd_br and hrd_cpb and hrd_target_br) else None))
        # GOP: the reference's (same camera, native rate) when given; otherwise the source's length IN FRAMES.
        # No cross-rate rule is verified on native files: Ambarella-era cameras keep 8 frames at 29.97 and 23.976,
        # GP1 uses 10 @ 29.97 and 12 @ 23.976 -- "constant in seconds" is contradicted by both generations, while the
        # source's own length is at least a value this camera writes. --reference or --gop is the only certain choice.
        if opts.gop:
            gop_frames = opts.gop
        elif ref is not None:
            gop_frames = _src_gop(ref.video)
        else:
            gop_frames = src_gop
            notes.append(f"keyframe interval {gop_frames} frames copied from the source: the camera's GOP at {float(out_fps):.4g} fps is "
                         "not known without a native recording; pass --reference (or --gop) to be certain")
        settings = derive.derive_settings(codec, ps_src, width, height, pix_fmt,
                                          src_fps, out_fps, src_gop, bitrate, hrd_target_br or opts.maxrate or hrd_br,
                                          hrd_target_cpb or opts.bufsize or hrd_cpb, gop_frames, opts.preset, color, samples=first_samples)
        settings.threads = opts.threads
        for kv in opts.encoder_params:
            k, _, v = kv.partition("=")
            (settings.x264_params if codec == "h264" else settings.x265_params)[k] = v
        notes.extend(settings.notes)
        if settings.bitrate != bitrate:
            notes.append(f"encode target {settings.bitrate} bps (requested {bitrate} bps capped at 97% of the declared HRD max rate {settings.maxrate} bps so the measured average stays under it)")
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
        target_ps = _patched_target_ps(ps_src, codec, out_fps, hrd_target_br if hrd_br else None, hrd_target_cpb if hrd_cpb else None)
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
            reference.apply(src, ref, log, out_fps)
        log(f"source writer conventions: MET latency {float(il_conv.latency) * 1000:.1f} ms, final payload after last audio={il_conv.final_payload_last}, "
            f"SOS types={sos_conv.type_codes if sos_conv else None}")
        built = build_samples(aus, conv, final_ps, has_b_frames=bool(settings.bframes))
        out_ts, out_fdur = _video_timescale(out_fps, ref)
        vid_entries = mb.parse_stsd(video.stbl.child("stsd"))
        _replace_codec_config(vid_entries[0], codec, final_ps)
        tracks: dict[str, mux.OutTrack] = {}
        cts = _composition_offsets(aus, codec, final_ps, out_fdur) if settings.bframes else None
        if cts and not video.has_ctts:
            notes.append("output carries B-frames (ctts) although the source has none: structure differs from the camera's")
        tracks["video"] = mux.OutTrack("video", built.samples, [out_fdur] * len(built.samples), out_ts,
                                       sync=built.sync, cts_offsets=cts, stsd_entries=vid_entries, source=video)

        # ---- 6. audio ---------------------------------------------------------------------------
        out_video_dur = Fraction(len(built.samples) * out_fdur, out_ts)
        for audio in src.tracks_of("audio"):
            # copied verbatim (same AAC frames, same esds), only re-chunked by the interleaver; when the camera ends the
            # audio within one AAC frame after the video (HD8+ rule), keep that relation for the new video length
            a_samples = src.read_samples(audio)
            a_durs = [s.duration for s in audio.samples]
            frame = a_durs[0] if a_durs else 1024
            src_video_dur = Fraction(video.media_duration, video.timescale)
            src_rule = len(a_durs) == -(-int(src_video_dur * audio.timescale) // frame)
            if src_rule and plan.mode == "realtime":
                want = -(-int(out_video_dur * audio.timescale) // frame)
                if want < len(a_durs):
                    a_samples, a_durs = a_samples[:want], a_durs[:want]
                    notes.append(f"audio trimmed to {want} AAC frames so it ends within one frame after the video (camera rule)")
            tracks[audio.key] = mux.OutTrack("audio", a_samples, a_durs, audio.timescale, source=audio,
                                             media_duration=sum(a_durs))

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
            out_audio_ms = None
            a0 = next((t for t in tracks.values() if t.kind == "audio"), None)
            if a0 is not None:
                out_audio_ms = sum(a0.durations) * 1000 // a0.timescale
            payloads, durations = gpmf_rebuild.rebuild(src, plan, out_fps, drop_imu=(opts.imu == "drop"),
                                                       drop_gps=(opts.gps == "drop"), reference=ref, log=log,
                                                       out_audio_ms=out_audio_ms)
            tracks["gpmd"] = mux.OutTrack("gpmd", payloads, durations, gp.timescale, source=gp)
        elif gp is not None:
            notes.append("gpmd track dropped by request (a native file always has one)")
        udta_gpmf = src.moov.find("udta/GPMF")
        if udta_gpmf is not None and plan.mode == "realtime":
            from . import gpmf_rebuild
            udta_gpmf.data = gpmf_rebuild.patch_global_settings_fps(udta_gpmf.data, out_fps, opts.imu == "drop", log=log)

        # ---- 9. interleave + SOS + write -------------------------------------------------------
        order = interleave.order_samples(src, tracks, il_conv)
        fdsc_builder = sos.make_builder(src, codec, final_ps, out_ts, out_fdur, sos_conv, header_override=sos_header_override) if sos_conv else None
        if gp is not None and opts.gpmf == "drop":
            _drop_track(src, "gpmd")
        res = mux.write_output(src, opts.out, tracks, order, fdsc_builder=fdsc_builder, mvhd_timescale=out_ts, log=log)
        _touch_like_camera(opts.out, src, tracks["video"].total_duration, out_ts)

        # ---- 10. verification -------------------------------------------------------------------
        report_text = ""
        if opts.verify:
            from . import verify
            m_out = ff.decode_md5(opts.out)
            m_ref = ff.decode_md5(es_path if not applied else os.path.join(workdir, "video.rewritten" + settings.es_suffix))
            ok = (m_out == m_ref)
            rep = verify.full_report(opts.src, opts.out, codec, opts.reference, external_tools=opts.external_tools,
                                     imu_dropped=(opts.imu == "drop" and opts.gpmf != "drop"), gps_dropped=(opts.gps == "drop"),
                                     interleave_conv=il_conv, sos_conv=sos_conv, sos_header=sos_header_override)
            rep.add("output video decodes identically to the encoder's elementary stream", "PASS" if ok else "FAIL", f"{m_ref} vs {m_out}")
            report_text = rep.render()
            log(report_text)
            failed = rep.failed
        else:
            failed = 0
        log(f"done in {time.time() - t0:.1f}s -> {opts.out}")
        return Result(opts.out, plan, settings, applied, lossless, report_text, notes, failed)
    finally:
        if not opts.keep_temp and not opts.workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _measure_slice_conventions(codec: str, samples: list[bytes], ps: dict[str, list[bytes]]):
    if codec == "h264":
        from .h26x import h264
        sps_f = h264.parse_sps_nal(ps["sps"][0])
        pps_f = h264.parse_pps_nal(ps["pps"][0], sps_f)
        return rewrite.measure_h264_conventions(samples, sps_f, pps_f, ps)
    from .h26x import hevc, hevc_rewrite
    sps_f = hevc.parse_sps_nal(ps["sps"][0])
    pps_f = hevc.parse_pps_nal(ps["pps"][0], sps_f)
    return hevc_rewrite.measure_hevc_conventions(samples, sps_f, pps_f)


def _require_gopro(src: SourceFile) -> None:
    names = " ".join(t.handler_name for t in src.tracks)
    brand = src.ftyp.data[:4]
    if "GoPro" not in names or src.track("tmcd") is None or brand != b"mp41":
        raise ValueError(f"input does not look like a GoPro camera original (ftyp brand {brand!r}, handlers {names.strip()!r}); "
                         "re-muxed or edited files cannot be converted faithfully")


def _composition_offsets(aus: list[N.AccessUnit], codec: str, ps: dict[str, list[bytes]], fdur: int) -> Optional[list[int]]:
    """ctts offsets from the picture order counts (display order) of the final stream; None when nothing is reordered."""
    if codec == "h264":
        from .h26x import h264
        sps_f = h264.parse_sps_nal(ps["sps"][0]); pps_f = h264.parse_pps_nal(ps["pps"][0], sps_f)
        pics = rewrite._h264_poc_sequence(aus, sps_f, pps_f)
    else:
        from .h26x import hevc, hevc_rewrite
        sps_f = hevc.parse_sps_nal(ps["sps"][0]); pps_f = hevc.parse_pps_nal(ps["pps"][0], sps_f)
        pics = hevc_rewrite._poc_sequence(aus, sps_f, pps_f)
    period = -1
    keys = []
    for i, (info, _s) in enumerate(pics):
        if info["idr"] or info.get("irap"):
            period += 1
        keys.append((period, info["poc"], i))
    display = {i: pos for pos, (_p, _poc, i) in enumerate(sorted(keys))}
    delay = max(i - display[i] for i in range(len(pics)))
    if delay <= 0:
        return None
    return [(display[i] - i + delay) * fdur for i in range(len(pics))]


def _touch_like_camera(path: str, src: SourceFile, video_duration: int, video_ts: int) -> None:
    """The camera closes the file at the end of the recording: mtime = creation time (RTC, stored as if UTC) + duration."""
    try:
        unix = src.mvhd.creation_time - 2082844800
        if unix > 0:
            end = unix + video_duration / video_ts
            os.utime(path, (end, end))
    except OSError:
        pass


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


def _patched_target_ps(ps_src: dict[str, list[bytes]], codec: str, out_fps: Fraction, hrd_bitrate: Optional[int] = None,
                       hrd_cpb: Optional[int] = None) -> dict[str, list[bytes]]:
    """Source parameter sets with VUI timing (and HRD bit rate / CPB when given) rewritten for the output mode.
    H.264 timing counts fields (time_scale = 2 x fps numerator)."""
    out = {k: list(v) for k, v in ps_src.items()}
    if codec == "h264":
        num, ts = out_fps.denominator, out_fps.numerator * 2
    else:
        num, ts = out_fps.denominator, out_fps.numerator
    out["sps"] = [params.patch_vui_timing(s, codec, num, ts) for s in out.get("sps", [])]
    if hrd_bitrate and hrd_cpb:
        out["sps"] = [params.patch_hrd(s, codec, hrd_bitrate, hrd_cpb) for s in out["sps"]]
    if codec == "hevc" and out.get("vps"):
        out["vps"] = [params.patch_vui_timing(v, "hevc-vps", num, ts) for v in out["vps"]]
    return out


def _video_timescale(out_fps: Fraction, ref: Optional[SourceFile]) -> tuple[int, int]:
    """GoPro convention: 29.97 -> 90000/3003; 23.976 -> 24000/1001; integer rates -> 90000/(90000/fps)."""
    if ref is not None:
        return ref.video.timescale, ref.video_frame_duration()
    # GoPro rule (verified 29.97/25/50 -> 90000; 23.976 -> 24000; 59.94 -> 60000): the 90 kHz clock whenever the
    # frame duration is an integer number of 90 kHz ticks, else the nominal numerator with 1001-tick frames.
    ticks = Fraction(90000) / out_fps
    if ticks.denominator == 1:
        return 90000, int(ticks)
    return out_fps.numerator, out_fps.denominator


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
