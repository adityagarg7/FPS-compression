"""Video re-encode step: decode the source, select frames per the plan, encode to a raw elementary stream."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from . import ffmpeg as ff
from .plan import FramePlan, select_expression


@dataclass
class EncoderSettings:
    codec: str                      # 'h264' | 'hevc'
    pix_fmt: str                    # e.g. 'yuvj420p', 'yuv420p', 'yuv420p10le'
    width: int
    height: int
    bitrate: int                    # target average bitrate (bps)
    maxrate: int                    # HRD max bitrate (bps)
    bufsize: int                    # HRD cpb size (bits)
    gop: int                        # keyframe interval in frames
    refs: int
    bframes: int
    color_range: str = "pc"         # 'pc' (full) | 'tv' (limited)
    color_primaries: str = "bt709"
    color_trc: str = "bt709"
    colorspace: str = "bt709"
    chroma_location: Optional[str] = None
    profile: Optional[str] = None   # x264/x265 profile name
    level: Optional[str] = None     # e.g. '4.2' / '5.1'
    tier: Optional[str] = None      # hevc: 'main' | 'high'
    x264_params: dict[str, str] = field(default_factory=dict)   # extra key=value pairs for -x264-params
    x265_params: dict[str, str] = field(default_factory=dict)   # extra key=value pairs for -x265-params
    preset: str = "slow"
    threads: int = 0
    extra_args: list[str] = field(default_factory=list)       # extra ffmpeg output args
    notes: list[str] = field(default_factory=list)            # human-readable derivation notes

    @property
    def es_suffix(self) -> str:
        return ".h264" if self.codec == "h264" else ".h265"

    @property
    def es_format(self) -> str:
        return "h264" if self.codec == "h264" else "hevc"


def _params_str(d: dict[str, str]) -> str:
    return ":".join(f"{k}={v}" for k, v in d.items())


def build_ffmpeg_command(src_path: str, plan: FramePlan, st: EncoderSettings, out_es: str,
                         pass_num: int = 0, passlog: Optional[str] = None, decode_threads: int = 0) -> list[str]:
    out_fps = plan.out_fps
    fps_str = f"{out_fps.numerator}/{out_fps.denominator}"
    vf = []
    if plan.mode == "realtime":
        expr = select_expression(plan.src_fps, plan.out_fps)
        if expr != "1":
            vf.append(f"select='{expr}'")
    vf.append(f"setpts=N/({fps_str})/TB")
    cmd = [ff.FFMPEG, "-y", "-hide_banner", "-nostdin", "-loglevel", "warning", "-nostats"]
    if decode_threads:
        cmd += ["-threads", str(decode_threads)]
    cmd += ["-i", src_path, "-map", "0:v:0", "-an", "-sn", "-dn",
            "-vf", ",".join(vf), "-r", fps_str, "-fps_mode", "cfr", "-pix_fmt", st.pix_fmt,
            "-color_range", st.color_range, "-color_primaries", st.color_primaries, "-color_trc", st.color_trc,
            "-colorspace", st.colorspace]
    if st.chroma_location:
        cmd += ["-chroma_sample_location", st.chroma_location]
    cmd += ["-g", str(st.gop), "-keyint_min", str(st.gop), "-sc_threshold", "0", "-bf", str(st.bframes),
            "-refs", str(st.refs), "-b:v", str(st.bitrate)]
    if st.maxrate and st.bufsize:
        cmd += ["-maxrate", str(st.maxrate), "-bufsize", str(st.bufsize)]
    cmd += ["-preset", st.preset]
    if st.threads:
        cmd += ["-threads", str(st.threads)]
    if st.codec == "h264":
        cmd += ["-c:v", "libx264"]
        if st.profile:
            cmd += ["-profile:v", st.profile]
        if st.level:
            cmd += ["-level:v", st.level]
        params = dict(st.x264_params)
        if pass_num:
            params["pass"] = str(pass_num)
            params["stats"] = passlog or "x264pass.log"
        if params:
            cmd += ["-x264-params", _params_str(params)]
    else:
        cmd += ["-c:v", "libx265"]
        if st.profile:
            cmd += ["-profile:v", st.profile]
        params = dict(st.x265_params)
        if st.level:
            params.setdefault("level-idc", st.level)
        if st.tier:
            params.setdefault("high-tier", "1" if st.tier == "high" else "0")
        if pass_num:
            params["pass"] = str(pass_num)
            params["stats"] = passlog or "x265pass.log"
        if params:
            cmd += ["-x265-params", _params_str(params)]
    cmd += st.extra_args
    cmd += ["-f", st.es_format, out_es]
    return cmd


def encode(src_path: str, plan: FramePlan, st: EncoderSettings, out_es: str, log=None, two_pass: bool = False,
           workdir: Optional[str] = None) -> str:
    """Run the encode; returns the path to the raw elementary stream."""
    workdir = workdir or os.path.dirname(os.path.abspath(out_es))
    if two_pass:
        passlog = os.path.join(workdir, "ratecontrol.log")
        cmd1 = build_ffmpeg_command(src_path, plan, st, os.devnull, pass_num=1, passlog=passlog)
        _report_encoder_warnings(ff.run(cmd1, log=log, capture=True), log)
        cmd2 = build_ffmpeg_command(src_path, plan, st, out_es, pass_num=2, passlog=passlog)
        _report_encoder_warnings(ff.run(cmd2, log=log, capture=True), log)
    else:
        cmd = build_ffmpeg_command(src_path, plan, st, out_es)
        p = ff.run(cmd, log=log, capture=True)
        _report_encoder_warnings(p, log)
    return out_es


def _report_encoder_warnings(p, log) -> None:
    """Encoder option problems are only warnings for ffmpeg; they must never pass silently here."""
    err = (p.stderr or b"").decode("utf-8", "replace")
    bad = [ln for ln in err.splitlines() if "Error parsing option" in ln or "invalid" in ln.lower() or "unknown option" in ln.lower()]
    if bad:
        raise ff.ToolError("encoder rejected options:\n" + "\n".join(bad[:10]))
    if log:
        for ln in err.splitlines():
            if ln.strip() and "deprecated" not in ln:
                log("ffmpeg: " + ln.strip())
