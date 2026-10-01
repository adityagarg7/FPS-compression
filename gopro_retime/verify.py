"""Forensic self-check: compare the output file against the source (and optionally a native reference file) on every
dimension a tool or examiner could use to tell a camera original from a re-encode."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from . import ffmpeg as ff
from . import mp4box as mb
from .h26x import nal as N
from .model import SourceFile, Track

# boxes whose payload is EXPECTED to change between source and output
_VARIABLE_BOXES = {b"stts", b"stsz", b"stco", b"co64", b"stss", b"ctts", b"stsc", b"mvhd", b"tkhd", b"mdhd", b"elst", b"stsd"}


@dataclass
class Check:
    name: str
    status: str          # 'PASS' | 'WARN' | 'FAIL' | 'INFO'
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(Check(name, status, detail))

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def warned(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    def render(self, verbose: bool = False) -> str:
        lines = []
        for c in self.checks:
            if c.status == "INFO" and not verbose:
                continue
            d = c.detail if (verbose or c.status != "PASS") else ""
            lines.append(f"[{c.status:4}] {c.name}" + (f": {d}" if d else ""))
        lines.append(f"-- {self.failed} FAIL, {self.warned} WARN, {len(self.checks)} checks")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps([c.__dict__ for c in self.checks], indent=1)


def _box_signature(box: mb.Box) -> list[str]:
    return [("  " * d) + b.type.decode("latin1") for d, b in box.walk()]


def _leaf_map(box: mb.Box, path: str = "") -> dict[str, bytes]:
    """Map of path -> payload for all leaf boxes (path includes sibling index for repeated types)."""
    out: dict[str, bytes] = {}
    counts: Counter = Counter()

    def rec(b: mb.Box, p: str) -> None:
        if b.children is None:
            out[p] = b.data
            return
        if b.data:
            out[p + "#hdr"] = b.data
        seen: Counter = Counter()
        for c in b.children:
            t = c.type.decode("latin1")
            k = f"{p}/{t}" + (f"[{seen[t]}]" if seen[t] else "")
            seen[t] += 1
            rec(c, k)

    rec(box, path or box.type.decode("latin1"))
    return out


def compare_container(src: SourceFile, out: SourceFile, rep: Report, strict_udta: bool = True) -> None:
    # ftyp
    rep.add("ftyp bytes identical", "PASS" if src.ftyp.serialize() == out.ftyp.serialize() else "FAIL",
            f"{src.ftyp.data!r} vs {out.ftyp.data!r}")
    # top-level layout
    st = [b.type.decode() for b in src.top.boxes]
    ot = [b.type.decode() for b in out.top.boxes]
    rep.add("top-level box order", "PASS" if st == ot else "FAIL", f"{st} vs {ot}")
    # moov tree
    ss, os_ = _box_signature(src.moov), _box_signature(out.moov)
    if ss == os_:
        rep.add("moov box tree identical (order and nesting)", "PASS", f"{len(ss)} boxes")
    else:
        import difflib
        diff = "\n".join(difflib.unified_diff(ss, os_, lineterm="", n=1))
        rep.add("moov box tree identical (order and nesting)", "FAIL", diff[:2000])
    # leaf payloads
    sm, om = _leaf_map(src.moov), _leaf_map(out.moov)
    changed = []
    for k, v in sm.items():
        if k not in om:
            continue
        leaf = k.split("/")[-1].split("[")[0].encode()
        if om[k] != v and leaf not in _VARIABLE_BOXES:
            changed.append(k)
    rep.add("non-table moov boxes byte-identical (udta, hdlr, dinf, vmhd/smhd/gmhd, iods, ...)",
            "PASS" if not changed else "FAIL", ", ".join(changed) if changed else f"{len(sm)} leaves compared")
    # stsd per track (fixed part + non-codec-config children)
    for t_s, t_o in zip(src.tracks, out.tracks):
        es, eo = t_s.stsd_entries, t_o.stsd_entries
        if len(es) != len(eo) or es[0].format != eo[0].format:
            rep.add(f"stsd[{t_s.kind}] entry format", "FAIL", f"{[e.format for e in es]} vs {[e.format for e in eo]}")
            continue
        if t_s.kind == "tmcd":
            continue  # timing-dependent, checked separately
        same_fixed = es[0].fixed == eo[0].fixed
        rep.add(f"stsd[{t_s.kind}] fixed fields identical", "PASS" if same_fixed else "FAIL",
                "" if same_fixed else f"{es[0].fixed.hex()} vs {eo[0].fixed.hex()}")
        sc = [(c.type, c.data) for c in es[0].children if c.type not in (b"avcC", b"hvcC")]
        oc = [(c.type, c.data) for c in eo[0].children if c.type not in (b"avcC", b"hvcC")]
        rep.add(f"stsd[{t_s.kind}] extension boxes identical (colr/pasp/esds/...)", "PASS" if sc == oc else "FAIL",
                f"{[c[0] for c in sc]} vs {[c[0] for c in oc]}")
        for cs in es[0].children:
            if cs.type in (b"avcC", b"hvcC"):
                co = next((c for c in eo[0].children if c.type == cs.type), None)
                if co is None:
                    rep.add(f"{cs.type.decode()} present", "FAIL")
                elif cs.data == co.data:
                    rep.add(f"{cs.type.decode()} byte-identical to source", "PASS")
                else:
                    rep.add(f"{cs.type.decode()} differs from source (expected: VUI timing changes only)", "INFO",
                            f"{len(cs.data)} vs {len(co.data)} bytes")
    # track-level conventions
    for t_s, t_o in zip(src.tracks, out.tracks):
        k = t_s.kind
        rep.add(f"{k}: handler name", "PASS" if t_s.handler_name == t_o.handler_name else "FAIL",
                f"{t_s.handler_name!r} vs {t_o.handler_name!r}")
        rep.add(f"{k}: timescale", "PASS" if t_s.timescale == t_o.timescale else ("INFO" if k in ("video", "tmcd", "fdsc") else "WARN"),
                f"{t_s.timescale} vs {t_o.timescale}")
        rep.add(f"{k}: stsc pattern", "PASS" if t_s.stsc == t_o.stsc else "FAIL", f"{t_s.stsc} vs {t_o.stsc}")
        rep.add(f"{k}: has ctts", "PASS" if t_s.has_ctts == t_o.has_ctts else "FAIL", f"{t_s.has_ctts} vs {t_o.has_ctts}")
        rep.add(f"{k}: has stss", "PASS" if t_s.has_stss == t_o.has_stss else "FAIL")
        rep.add(f"{k}: has elst", "PASS" if (t_s.elst is None) == (t_o.elst is None) else "FAIL")
        if t_s.elst and t_o.elst:
            ok = len(t_s.elst) == len(t_o.elst) and all(a.media_time == b.media_time and a.media_rate == b.media_rate for a, b in zip(t_s.elst, t_o.elst))
            ok = ok and t_o.elst[0].segment_duration == t_o.tkhd_duration
            rep.add(f"{k}: elst convention (segment = track duration, media_time, rate)", "PASS" if ok else "FAIL",
                    f"{t_o.elst} tkhd={t_o.tkhd_duration}")
        # duration consistency
        mv_ts_o = out.mvhd.timescale
        exp_tk = (t_o.media_duration * mv_ts_o) // t_o.timescale
        rep.add(f"{k}: tkhd duration == scaled mdhd duration", "PASS" if exp_tk == t_o.tkhd_duration else "WARN",
                f"{t_o.tkhd_duration} vs {exp_tk}")
    # mvhd
    rep.add("mvhd timescale == video timescale", "PASS" if out.mvhd.timescale == out.video.timescale else "WARN",
            f"{out.mvhd.timescale} vs {out.video.timescale}")
    rep.add("mvhd next_track_id", "PASS" if out.mvhd.next_track_id == src.mvhd.next_track_id else "FAIL")
    rep.add("creation/modification times preserved", "PASS" if (out.mvhd.creation_time, out.mvhd.modification_time) == (src.mvhd.creation_time, src.mvhd.modification_time) else "WARN")
    # interleave
    ls, lo = src.layout_string(), out.layout_string()
    rep.add("interleave: fdsc descriptor precedes every media sample", "PASS" if _fdsc_precedes(lo) == _fdsc_precedes(ls) else "FAIL",
            f"{lo[:40]}...")
    rep.add("interleave: starts V0, tmcd, audio like the firmware", "PASS" if lo[:8] == ls[:8] else "FAIL", f"{ls[:24]} vs {lo[:24]}")
    # the output must obey the writer rule measured on the source (latency, final payload placement, tie-break)
    try:
        from . import interleave as _il
        from .mux import OutTrack as _OT
        conv = _il.measure(src)
        otracks = {t.kind: _OT(t.kind, [b""] * t.sample_count, [s.duration for s in t.samples], t.timescale) for t in out.tracks if t.kind != "fdsc"}
        expected = _il.order_samples(out, otracks, conv)
        actual = [(t.kind, s.index) for t, s in out.all_samples_in_file_order() if t.kind != "fdsc"]
        first = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        rep.add("interleave: output follows the writer rule measured on the source", "PASS" if expected == actual else "FAIL",
                f"latency {float(conv.latency) * 1000:.1f} ms, final-after-audio={conv.final_payload_last}" + (f"; first deviation at item {first}" if first is not None else ""))
    except Exception as e:  # noqa: BLE001
        rep.add("interleave: output follows the writer rule measured on the source", "WARN", f"could not evaluate: {e}")
    # contiguity of mdat
    rep.add("mdat fully covered by samples (no gaps/garbage)", "PASS" if _contiguous(out) else "FAIL")
    # gpmd stts pattern
    g_s, g_o = src.track("gpmd"), out.track("gpmd")
    if g_s and g_o:
        ds = Counter(s.duration for s in g_s.samples[:-1]); do = Counter(s.duration for s in g_o.samples[:-1])
        rep.add("gpmd payload durations", "INFO", f"src {dict(ds)} last {g_s.samples[-1].duration} | out {dict(do)} last {g_o.samples[-1].duration}")


def _fdsc_precedes(layout: str) -> bool:
    if "S" not in layout:
        return True
    body = layout[2:]
    return all(body[i] == "S" for i in range(0, len(body) - 1, 2)) and all(body[i] != "S" for i in range(1, len(body), 2))


def _contiguous(f: SourceFile) -> bool:
    items = sorted((s.offset, s.size) for t in f.tracks for s in t.samples)
    pos = f.mdat.offset + f.mdat.header_size
    for o, sz in items:
        if o != pos:
            return False
        pos = o + sz
    return pos == f.mdat.offset + f.mdat.size


def video_stream_facts(f: SourceFile, codec: str, n: int = 40) -> dict:
    v = f.video
    smp = f.read_samples(v)
    facts: dict = {}
    nal_layouts = Counter()
    sei = 0
    inband_ps = 0
    slices_per_pic = Counter()
    for i, s in enumerate(smp[:n]):
        nals = N.split_length_prefixed(s)
        types = tuple(N.nal_type(x, codec) for x in nals)
        nal_layouts[types] += 1
        sei += sum(1 for x in nals if N.is_sei(x, codec))
        inband_ps += sum(1 for x in nals if N.is_param_set(x, codec))
        slices_per_pic[sum(1 for x in nals if N.is_vcl(x, codec))] += 1
    facts["nal_layouts"] = {str(k): c for k, c in nal_layouts.most_common(4)}
    facts["sei_nals_in_first_%d" % n] = sei
    facts["inband_param_sets_in_first_%d" % n] = inband_ps
    facts["slices_per_picture"] = dict(slices_per_pic)
    sync = [i for i, s in enumerate(v.samples) if s.is_sync]
    gaps = Counter(b - a for a, b in zip(sync, sync[1:]))
    facts["gop_lengths"] = dict(gaps.most_common(3))
    facts["frame_duration"] = f.video_frame_duration()
    facts["frames"] = v.sample_count
    sizes = [s.size for s in v.samples]
    facts["mean_frame_bytes"] = sum(sizes) // max(1, len(sizes))
    facts["bitrate_bps"] = int(sum(sizes) * 8 / (v.media_duration / v.timescale)) if v.media_duration else 0
    return facts


def compare_video_streams(src: SourceFile, out: SourceFile, codec: str, rep: Report) -> None:
    fs, fo = video_stream_facts(src, codec), video_stream_facts(out, codec)
    rep.add("video: NAL layout per sample (AUD/VCL structure)", "PASS" if set(fs["nal_layouts"]) == set(fo["nal_layouts"]) else "FAIL",
            f"src {fs['nal_layouts']} | out {fo['nal_layouts']}")
    rep.add("video: no SEI NALs in samples", "PASS" if fo["sei_nals_in_first_40"] == fs["sei_nals_in_first_40"] else "FAIL",
            f"src {fs['sei_nals_in_first_40']} out {fo['sei_nals_in_first_40']}")
    rep.add("video: in-band parameter sets convention", "PASS" if (fs["inband_param_sets_in_first_40"] > 0) == (fo["inband_param_sets_in_first_40"] > 0) else "FAIL",
            f"src {fs['inband_param_sets_in_first_40']} out {fo['inband_param_sets_in_first_40']}")
    rep.add("video: slices per picture", "PASS" if fs["slices_per_picture"] == fo["slices_per_picture"] else "WARN",
            f"src {fs['slices_per_picture']} out {fo['slices_per_picture']}")
    rep.add("video: GOP length (frames between sync samples)", "INFO" if fs["gop_lengths"] != fo["gop_lengths"] else "PASS",
            f"src {fs['gop_lengths']} out {fo['gop_lengths']}")
    rep.add("video: bitrate", "INFO", f"src {fs['bitrate_bps']/1e6:.2f} Mbps out {fo['bitrate_bps']/1e6:.2f} Mbps")


# ---- parameter set field diff via ffmpeg trace_headers ------------------------------------------
_TRACE_RE = re.compile(r"\]\s+(\d+)\s+(\S+)\s+(?:[01]+|[0-9a-fA-F]+)\s*=\s*(-?\d+)")
_SECTION_RE = re.compile(r"\]\s+(Sequence Parameter Set|Picture Parameter Set|Video Parameter Set|Supplemental Enhancement Information|Slice Header|Slice Segment Header|Access Unit Delimiter|Packet:)")


def trace_headers(path: str, max_packets: int = 3) -> dict[str, list[tuple[str, int]]]:
    """Return {section: [(field, value), ...]} for the parameter sets (extradata) and first slice headers."""
    cmd = [ff.FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
           "-bsf:v", "trace_headers", "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    text = p.stderr.decode("utf-8", "replace")
    out: dict[str, list[tuple[str, int]]] = {}
    section = None
    packets = 0
    for line in text.splitlines():
        m = _SECTION_RE.search(line)
        if m:
            name = m.group(1)
            if name == "Packet:":
                packets += 1
                if packets > max_packets:
                    break
                section = None
                continue
            key = name
            idx = 0
            while f"{key}#{idx}" in out:
                idx += 1
            section = f"{key}#{idx}"
            out[section] = []
            continue
        if section is None:
            continue
        m2 = _TRACE_RE.search(line)
        if m2:
            out[section].append((m2.group(2), int(m2.group(3))))
    return out


def compare_parameter_sets(src_path: str, out_path: str, rep: Report, allowed: tuple[str, ...] = ("num_units_in_tick", "time_scale", "vps_num_units_in_tick", "vps_time_scale", "vui_num_units_in_tick", "vui_time_scale")) -> None:
    ts, to = trace_headers(src_path), trace_headers(out_path)
    for sec in sorted(set(ts) | set(to)):
        if sec.startswith("Slice") or sec.startswith("Access Unit"):
            continue
        a, b = ts.get(sec), to.get(sec)
        if a is None or b is None:
            rep.add(f"paramset {sec} present in both", "FAIL", f"src={a is not None} out={b is not None}")
            continue
        da, db = dict(a), dict(b)
        diffs = [(k, da.get(k), db.get(k)) for k in dict.fromkeys(list(da) + list(db)) if da.get(k) != db.get(k)]
        bad = [d for d in diffs if d[0] not in allowed]
        detail = "; ".join(f"{k}: {x}->{y}" for k, x, y in diffs)
        rep.add(f"paramset {sec}: fields identical (except frame-rate timing)", "PASS" if not bad else "FAIL", detail)
    # slice header conventions of the first pictures
    for sec in sorted(set(ts) & set(to)):
        if not sec.startswith("Slice"):
            continue
        da, db = dict(ts[sec]), dict(to[sec])
        keys = ("nal_ref_idc", "nal_unit_type", "slice_type", "num_ref_idx_active_override_flag", "cabac_init_idc",
                "disable_deblocking_filter_idc", "pic_order_cnt_lsb", "slice_sao_luma_flag", "num_entry_point_offsets",
                "short_term_ref_pic_set_sps_flag", "slice_temporal_mvp_enabled_flag", "cabac_init_flag",
                "slice_loop_filter_across_slices_enabled_flag", "num_ref_idx_l0_active_minus1")
        diffs = [(k, da.get(k), db.get(k)) for k in keys if k in da and da.get(k) != db.get(k) and k != "pic_order_cnt_lsb"]
        rep.add(f"{sec}: convention fields", "PASS" if not diffs else "WARN", "; ".join(f"{k}: {x}->{y}" for k, x, y in diffs))


# ---- external tools ----------------------------------------------------------------------------
def _run_text(cmd: list[str]) -> str:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        return p.stdout.decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return f"<error {e}>"


_FP_IGNORE = {"duration", "bit_rate", "nb_frames", "size", "filename", "start_time", "DURATION", "nb_read_frames",
              "r_frame_rate", "avg_frame_rate", "time_base", "max_bit_rate", "duration_ts", "start_pts"}


def compare_ffprobe(src_path: str, out_path: str, rep: Report) -> None:
    a, b = ff.ffprobe_json(src_path), ff.ffprobe_json(out_path)
    fa, fb = {k: v for k, v in a["format"].items() if k not in _FP_IGNORE}, {k: v for k, v in b["format"].items() if k not in _FP_IGNORE}
    diffs = {k: (fa.get(k), fb.get(k)) for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
    rep.add("ffprobe format (brands, tags) identical", "PASS" if not diffs else "FAIL", str(diffs))
    for sa, sb in zip(a["streams"], b["streams"]):
        da = {k: v for k, v in sa.items() if k not in _FP_IGNORE and k != "tags"}
        db = {k: v for k, v in sb.items() if k not in _FP_IGNORE and k != "tags"}
        ta = {k: v for k, v in sa.get("tags", {}).items() if k != "timecode"}
        tb = {k: v for k, v in sb.get("tags", {}).items() if k != "timecode"}
        diffs = {k: (da.get(k), db.get(k)) for k in set(da) | set(db) if da.get(k) != db.get(k)}
        tdiffs = {k: (ta.get(k), tb.get(k)) for k in set(ta) | set(tb) if ta.get(k) != tb.get(k)}
        kind = sa.get("codec_type", "?") + ":" + sa.get("codec_tag_string", "")
        rep.add(f"ffprobe stream {kind}: fields identical", "PASS" if not diffs else "WARN", str(diffs))
        rep.add(f"ffprobe stream {kind}: tags identical (handler_name, encoder, vendor_id, ...)", "PASS" if not tdiffs else "FAIL", str(tdiffs))


_MI_IGNORE = re.compile(r"^(mdhd_Duration|Bits-\(Pixel\*Frame\)|CompleteName|FileName|FileNameExtension|FileExtension|File_Modified_Date|File_Modified_Date_Local|FolderName|Complete name|File name|File size|Duration|Overall bit rate|Frame rate|Frame count|Stream size|Bit rate|Bits/\(Pixel\*Frame\)|FrameRate|Delay|File last modification|Proportion of this stream|DataSize|FooterSize|HeaderSize|Count|Samples count|Source duration|Source stream size|Source_StreamSize|Duration_|StreamSize|OverallBitRate|TimeCode|Time code|Format settings, GOP|Minimum frame rate|Maximum frame rate|SamplesPerFrame|Encoded date|Tagged date|Delay_|FrameCount|BitRate|FileSize|Buffer size|BufferSize|Maximum bit rate|Nominal bit rate|Original frame rate|Frame rate mode)")


def compare_mediainfo(src_path: str, out_path: str, rep: Report) -> None:
    if shutil.which("mediainfo") is None:
        rep.add("mediainfo available", "INFO", "mediainfo not installed; skipped")
        return
    def parse(path: str) -> dict[str, str]:
        d: dict[str, str] = {}
        section = ""
        for line in _run_text(["mediainfo", "--Full", "--Language=raw", path]).splitlines():
            if not line.strip():
                continue
            if ":" not in line:
                section = line.strip(); continue
            k, v = line.split(":", 1)
            k = k.strip(); v = v.strip()
            if _MI_IGNORE.match(k):
                continue
            d[f"{section}/{k}"] = v
        return d
    a, b = parse(src_path), parse(out_path)
    diffs = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    rep.add("mediainfo --Full identical (ignoring size/duration/bitrate/frame-rate fields)", "PASS" if not diffs else "FAIL",
            "; ".join(f"{k}: {v[0]!r}->{v[1]!r}" for k, v in sorted(diffs.items()))[:3000])
    for key in ("Encoded_Library", "Encoded_Library_Settings", "Encoded_Application", "Writing library", "Encoding settings"):
        hits = [k for k in b if k.endswith("/" + key)]
        rep.add(f"mediainfo: no '{key}' fingerprint", "PASS" if not hits else "FAIL", "; ".join(f"{h}={b[h]}" for h in hits))


_ET_IGNORE = re.compile(r"^(File|System|ExifTool):|Duration|FileSize|FileName|Directory|FileModifyDate|FileAccessDate|FileInodeChangeDate|MediaDataSize|FrameRate|FrameCount|VideoFrameRate|AvgBitrate|MaxBitrate|TimeCode|StartTimecode|TrackDuration|MediaDuration|PlaybackFrameRate|SampleDuration|SampleTime|TimeStamp|GPS|Accelerometer|Gyroscope|Exposure|ISOSpeeds|ColorTemperatures|WhiteBalanceRGB|LumaAverage|InputUniformity|SceneClassification|PrediminantHue|CameraTemperature|ImageOrientation|CameraOrientation|GravityVector|MicrophoneWet|WindProcessing|AudioLevel|ShutterSpeeds|AverageBitrate")


def compare_exiftool(src_path: str, out_path: str, rep: Report) -> None:
    if shutil.which("exiftool") is None:
        rep.add("exiftool available", "INFO", "exiftool not installed; skipped")
        return
    def parse(path: str) -> dict[str, str]:
        d: dict[str, str] = {}
        for line in _run_text(["exiftool", "-a", "-G1", "-s", "-ee3", path]).splitlines():
            m = re.match(r"^\[(\S+)\]\s+(\S+)\s*:\s*(.*)$", line)
            if not m:
                continue
            key = f"{m.group(1)}:{m.group(2)}"
            if _ET_IGNORE.search(key):
                continue
            if key in d:
                i = 2
                while f"{key}#{i}" in d:
                    i += 1
                key = f"{key}#{i}"
            d[key] = m.group(3)
        return d
    a, b = parse(src_path), parse(out_path)
    diffs = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    rep.add("exiftool tags identical (ignoring durations/sizes/telemetry values)", "PASS" if not diffs else "FAIL",
            "; ".join(f"{k}: {v[0]!r}->{v[1]!r}" for k, v in sorted(diffs.items()))[:3000])


def full_report(src_path: str, out_path: str, codec: str, reference_path: Optional[str] = None,
                external_tools: bool = True) -> Report:
    rep = Report()
    src, out = SourceFile.open(src_path), SourceFile.open(out_path)
    compare_container(src, out, rep)
    compare_video_streams(src, out, codec, rep)
    compare_parameter_sets(src_path, out_path, rep)
    if external_tools:
        compare_ffprobe(src_path, out_path, rep)
        compare_mediainfo(src_path, out_path, rep)
        compare_exiftool(src_path, out_path, rep)
    if reference_path:
        ref = SourceFile.open(reference_path)
        rep.add("reference: video timescale/frame duration", "PASS" if (ref.video.timescale, ref.video_frame_duration()) == (out.video.timescale, out.video_frame_duration()) else "FAIL",
                f"ref {ref.video.timescale}/{ref.video_frame_duration()} out {out.video.timescale}/{out.video_frame_duration()}")
        rep.add("reference: mvhd timescale", "PASS" if ref.mvhd.timescale == out.mvhd.timescale else "FAIL")
        tr, to = ref.track("tmcd"), out.track("tmcd")
        if tr and to:
            a, b = mb.tmcd_entry_info(tr.stsd_entries[0]), mb.tmcd_entry_info(to.stsd_entries[0])
            rep.add("reference: tmcd stsd (flags, timescale, frame duration, frames)", "PASS" if a == b else "FAIL", f"{a} vs {b}")
        gr, go = ref.track("gpmd"), out.track("gpmd")
        if gr and go:
            rep.add("reference: gpmd payload duration", "PASS" if gr.samples[0].duration == go.samples[0].duration else "FAIL",
                    f"{gr.samples[0].duration} vs {go.samples[0].duration}")
    return rep
