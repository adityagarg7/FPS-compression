"""Derive encoder settings (x264 / x265 via ffmpeg) from the SOURCE parameter sets so that every DECODE-AFFECTING
coding tool matches the camera's stream; the syntax-only remainder is fixed by `rewrite`."""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from ..encode import EncoderSettings
from . import h264
from . import nal as N

H264_PROFILES = {66: "baseline", 77: "main", 88: "extended", 100: "high", 110: "high10", 122: "high422", 244: "high444"}


def _level_str(level_idc: int, constraint_set3: int = 0) -> str:
    if level_idc == 9 or (level_idc == 11 and constraint_set3):
        return "1b"
    return f"{level_idc // 10}.{level_idc % 10}" if level_idc % 10 else str(level_idc // 10)


def h264_stream_facts(samples: list[bytes], sps_f: dict, pps_f: dict, max_samples: int = 400) -> dict:
    """Slice-level habits of the source: slice types, B-run length, deblocking params, slices per picture, QP."""
    types: list[int] = []
    deblock = None
    slices_per_pic: list[int] = []
    qps: list[int] = []
    max_refs = 0
    for smp in samples[:max_samples]:
        cnt = 0
        for n in N.split_length_prefixed(smp):
            if not N.is_vcl(n, "h264"):
                continue
            f, _d, _b = h264.parse_slice_nal(n, sps_f, pps_f)
            cnt += 1
            max_refs = max(max_refs, f.get("_num_ref_idx_l0_active", 0), f.get("_num_ref_idx_l1_active", 0))
            if f["first_mb_in_slice"] == 0:
                types.append(h264.slice_type_base(f["slice_type"]))
                qps.append(26 + pps_f["pic_init_qp_minus26"] + f["slice_qp_delta"])
                if deblock is None and pps_f["deblocking_filter_control_present_flag"]:
                    deblock = (f.get("disable_deblocking_filter_idc", 0), f.get("slice_alpha_c0_offset_div2", 0), f.get("slice_beta_offset_div2", 0))
        slices_per_pic.append(cnt)
    # max run of B pictures in decode order
    run = best = 0
    for t in types:
        if t == h264.SLICE_B:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return {"slice_types": types, "max_b_run": best, "deblock": deblock or (0, 0, 0), "max_refs": max_refs,
            "slices_per_pic": max(set(slices_per_pic), key=slices_per_pic.count) if slices_per_pic else 1,
            "mean_qp": sum(qps) / len(qps) if qps else None}


def derive_h264(ps: dict[str, list[bytes]], samples: list[bytes], width: int, height: int, pix_fmt: str,
                bitrate: int, maxrate: Optional[int], bufsize: Optional[int], gop: int, preset: str, color: dict) -> EncoderSettings:
    sps_f = h264.parse_sps_nal(ps["sps"][0])
    pps_f = h264.parse_pps_nal(ps["pps"][0], sps_f)
    facts = h264_stream_facts(samples, sps_f, pps_f)
    vui = sps_f.get("vui", {})
    notes: list[str] = []
    profile = H264_PROFILES.get(sps_f["profile_idc"], "high")
    level = _level_str(sps_f["level_idc"], sps_f.get("constraint_set3_flag", 0))
    # active references actually used by the camera's slices (PPS default unless slices override)
    refs = facts["max_refs"] or (pps_f["num_ref_idx_l0_default_active_minus1"] + 1)
    refs = max(1, min(refs, sps_f["max_num_ref_frames"] or 1))
    bframes = facts["max_b_run"]
    hrd_br, hrd_cpb = sps_f.get("_nal_hrd_bit_rate"), sps_f.get("_nal_hrd_cpb_size")
    nal_hrd = vui.get("nal_hrd_parameters_present_flag") or vui.get("vcl_hrd_parameters_present_flag")
    cbr = bool(vui.get("nal_hrd", {}).get("cbr_flag", [0])[0]) if nal_hrd else False
    maxrate = maxrate or hrd_br or bitrate
    bufsize = bufsize or hrd_cpb or maxrate
    if bitrate > maxrate:
        bitrate = maxrate
    full_range = bool(vui.get("video_full_range_flag", 0)) if vui.get("video_signal_type_present_flag") else color.get("range") == "pc"
    st = EncoderSettings(
        codec="h264", pix_fmt=pix_fmt, width=width, height=height, bitrate=bitrate, maxrate=maxrate, bufsize=bufsize,
        gop=gop, refs=refs, bframes=bframes, preset=preset,
        color_range="pc" if full_range else "tv",
        color_primaries=color.get("primaries", "bt709"), color_trc=color.get("trc", "bt709"), colorspace=color.get("space", "bt709"),
        chroma_location=color.get("chroma_location"), profile=profile, level=level,
    )
    p: dict[str, str] = {}
    p["aud"] = "1"
    p["cabac"] = str(pps_f["entropy_coding_mode_flag"])
    p["8x8dct"] = str(pps_f.get("transform_8x8_mode_flag") or 0)
    p["weightp"] = "2" if pps_f["weighted_pred_flag"] else "0"
    p["weightb"] = "1" if pps_f["weighted_bipred_idc"] == 1 else "0"
    p["constrained-intra"] = str(pps_f["constrained_intra_pred_flag"])
    p["chroma-qp-offset"] = str(pps_f["chroma_qp_index_offset"])  # x264 shifts this with psy-rd: fixed by calibrate()
    p["open-gop"] = "0"
    p["scenecut"] = "0"
    p["keyint"] = str(gop)
    p["min-keyint"] = str(gop)
    p["bframes"] = str(bframes)
    if bframes:
        p["b-pyramid"] = "none"
        p["b-adapt"] = "0"
    p["ref"] = str(refs)
    p["slices"] = str(facts["slices_per_pic"])
    idc, alpha, beta = facts["deblock"]
    if idc == 1:
        p["deblock"] = "0:0"
        p["no-deblock"] = "1"
    else:
        p["deblock"] = f"{alpha}:{beta}"
    if nal_hrd:
        p["nal-hrd"] = "cbr" if cbr else "vbr"
        p["vbv-maxrate"] = str(max(1, maxrate // 1000))
        p["vbv-bufsize"] = str(max(1, bufsize // 1000))
    p["fullrange"] = "on" if full_range else "off"
    if vui.get("video_signal_type_present_flag"):
        p["videoformat"] = ["component", "pal", "ntsc", "secam", "mac", "undef"][vui.get("video_format", 5)]
    if sps_f.get("pic_scaling_matrix_present_flag") or sps_f.get("seq_scaling_matrix_present_flag"):
        notes.append("source uses custom scaling matrices; x264 cannot reproduce them exactly (cqm not derived)")
    if pps_f.get("transform_8x8_mode_flag") and profile == "main":
        profile = "high"
    st.x264_params = p
    st.notes = notes + [f"x264 derived from source SPS/PPS: profile={profile} level={level} refs={refs} bframes={bframes} "
                        f"cabac={p['cabac']} 8x8dct={p['8x8dct']} weightp={p['weightp']} deblock={p['deblock']} slices={p['slices']} "
                        f"hrd={'cbr' if cbr else 'vbr' if nal_hrd else 'none'} mean source QP={facts['mean_qp']}"]
    return st


def derive_settings(codec: str, param_sets: dict[str, list[bytes]], width: int, height: int, pix_fmt: str,
                    src_fps: Fraction, out_fps: Fraction, src_gop_frames: int, bitrate: Optional[int],
                    maxrate: Optional[int], bufsize: Optional[int], gop: Optional[int], preset: str,
                    color: dict, samples: Optional[list[bytes]] = None) -> EncoderSettings:
    gop = gop or src_gop_frames
    br = bitrate or 60_000_000
    if codec == "h264":
        return derive_h264(param_sets, samples or [], width, height, pix_fmt, br, maxrate, bufsize, gop, preset, color)
    from . import hevc_derive  # provided separately
    return hevc_derive.derive_hevc(param_sets, samples or [], width, height, pix_fmt, br, maxrate, bufsize, gop, preset, color)
