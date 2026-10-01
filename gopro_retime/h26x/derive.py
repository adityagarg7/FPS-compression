"""Derive encoder settings (x264 / x265 via ffmpeg) from the SOURCE parameter sets so that every DECODE-AFFECTING
coding tool matches the camera's stream, and the syntax-only remainder can be fixed by `rewrite`.

Interface contract:

    derive_settings(codec, param_sets: dict[str, list[bytes]], width, height, pix_fmt, src_fps, out_fps,
                    src_gop_frames, bitrate_hint, options) -> EncoderSettings
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..encode import EncoderSettings


def derive_settings(codec: str, param_sets: dict[str, list[bytes]], width: int, height: int, pix_fmt: str,
                    src_fps: Fraction, out_fps: Fraction, src_gop_frames: int, bitrate: Optional[int],
                    maxrate: Optional[int], bufsize: Optional[int], gop: Optional[int], preset: str,
                    color: dict) -> EncoderSettings:
    """Baseline implementation (used until the full derivation lands): generic settings + conservative tool choices."""
    is_h264 = codec == "h264"
    br = bitrate or 60_000_000
    st = EncoderSettings(
        codec=codec, pix_fmt=pix_fmt, width=width, height=height, bitrate=br, maxrate=maxrate or br,
        bufsize=bufsize or br, gop=gop or src_gop_frames, refs=1, bframes=0, preset=preset,
        color_range=color.get("range", "pc"), color_primaries=color.get("primaries", "bt709"),
        color_trc=color.get("trc", "bt709"), colorspace=color.get("space", "bt709"),
        chroma_location=color.get("chroma_location"),
    )
    if is_h264:
        st.x264_params = {"aud": "1", "nal-hrd": "vbr", "weightp": "0", "open-gop": "0", "scenecut": "0",
                          "psy": "0", "chroma-qp-offset": "0", "8x8dct": "0", "cabac": "1", "bframes": "0"}
    else:
        st.x265_params = {"aud": "1", "info": "0", "open-gop": "0", "scenecut": "0", "bframes": "0",
                          "repeat-headers": "1", "hrd": "1", "b-pyramid": "0", "no-psy-rd": "1"}
    st.notes.append("baseline derivation (full source-matched derivation not available)")
    return st
