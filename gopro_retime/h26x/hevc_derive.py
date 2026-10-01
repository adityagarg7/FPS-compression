"""x265 settings derived from the camera's HEVC VPS/SPS/PPS (decode-affecting tools) — verified by calibrate()."""
from __future__ import annotations

from typing import Optional

from ..encode import EncoderSettings
from . import nal as N
from . import params as P

HEVC_PROFILES = {1: "main", 2: "main10", 3: "mainstillpicture", 4: "main422-10"}


def hevc_stream_facts(samples: list[bytes], sps_f: dict, pps_f: dict, max_samples: int = 300) -> dict:
    from . import hevc
    types: list[int] = []
    slices_per_pic: list[int] = []
    qps: list[int] = []
    deblock = None
    sao_used = False
    tmvp = None
    cabac_init = False
    max_refs = 0
    for smp in samples[:max_samples]:
        cnt = 0
        for n in N.split_length_prefixed(smp):
            if not N.is_vcl(n, "hevc"):
                continue
            f, _d, _b = hevc.parse_slice_nal(n, sps_f, pps_f)
            if f.get("dependent_slice_segment_flag"):
                continue
            cnt += 1
            max_refs = max(max_refs, f.get("_num_ref_idx_l0_active", 0), f.get("_num_ref_idx_l1_active", 0))
            if f.get("first_slice_segment_in_pic_flag"):
                st = f.get("slice_type", 2)
                types.append(st)
                qps.append(26 + pps_f["init_qp_minus26"] + f.get("slice_qp_delta", 0))
                sao_used |= bool(f.get("slice_sao_luma_flag") or f.get("slice_sao_chroma_flag"))
                if tmvp is None:
                    tmvp = f.get("slice_temporal_mvp_enabled_flag")
                cabac_init |= bool(f.get("cabac_init_flag"))
                if deblock is None:
                    deblock = (f.get("slice_deblocking_filter_disabled_flag", pps_f.get("pps_deblocking_filter_disabled_flag", 0)),
                               f.get("slice_beta_offset_div2", pps_f.get("pps_beta_offset_div2", 0)),
                               f.get("slice_tc_offset_div2", pps_f.get("pps_tc_offset_div2", 0)))
        slices_per_pic.append(cnt)
    run = best = 0
    for t in types:  # HEVC slice_type: 0 = B, 1 = P, 2 = I
        if t == 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return {"slice_types": types, "max_b_run": best, "deblock": deblock or (0, 0, 0), "sao_used": sao_used,
            "slices_per_pic": max(set(slices_per_pic), key=slices_per_pic.count) if slices_per_pic else 1,
            "mean_qp": sum(qps) / len(qps) if qps else None, "cabac_init_used": cabac_init, "max_refs": max_refs}


def derive_hevc(ps: dict[str, list[bytes]], samples: list[bytes], width: int, height: int, pix_fmt: str,
                bitrate: int, maxrate: Optional[int], bufsize: Optional[int], gop: int, preset: str, color: dict) -> EncoderSettings:
    sps_f = P.parse_sps(ps["sps"][0], "hevc")
    pps_f = P.parse_pps(ps["pps"][0], "hevc", sps_f)
    facts = hevc_stream_facts(samples, sps_f, pps_f) if samples else {"max_b_run": 0, "deblock": (0, 0, 0), "slices_per_pic": 1, "mean_qp": None, "sao_used": False, "cabac_init_used": False, "max_refs": 0}
    vui = sps_f.get("vui", {})
    notes: list[str] = []
    ptl = sps_f.get("ptl", {})
    profile_idc = ptl.get("general_profile_idc", 1)
    tier = "high" if ptl.get("general_tier_flag") else "main"
    level_idc = ptl.get("general_level_idc", 0)
    hrd_br, hrd_cpb = P.hrd_values(sps_f)
    maxrate = maxrate or hrd_br or bitrate
    bufsize = bufsize or hrd_cpb or maxrate
    if bitrate > maxrate:
        bitrate = maxrate
    full_range = bool(vui.get("video_full_range_flag", 0)) if vui.get("video_signal_type_present_flag") else color.get("range") == "pc"
    min_cb = 1 << (sps_f["log2_min_luma_coding_block_size_minus3"] + 3)
    ctu = min_cb << sps_f["log2_diff_max_min_luma_coding_block_size"]
    min_tb = 1 << (sps_f["log2_min_luma_transform_block_size_minus2"] + 2)
    max_tb = min_tb << sps_f["log2_diff_max_min_luma_transform_block_size"]
    st = EncoderSettings(
        codec="hevc", pix_fmt=pix_fmt, width=width, height=height, bitrate=bitrate, maxrate=maxrate, bufsize=bufsize,
        gop=gop, refs=max(1, facts["max_refs"] or _refs_from_rps(sps_f, pps_f)), bframes=facts["max_b_run"], preset=preset,
        color_range="pc" if full_range else "tv",
        color_primaries=color.get("primaries", "bt709"), color_trc=color.get("trc", "bt709"), colorspace=color.get("space", "bt709"),
        chroma_location=color.get("chroma_location"), profile=HEVC_PROFILES.get(profile_idc), level=str(level_idc), tier=tier,
    )
    p: dict[str, str] = {}
    p["aud"] = "1"
    p["info"] = "0"                      # no x265 version SEI
    p["repeat-headers"] = "1"            # VPS/SPS/PPS on every IDR (dropped/placed later per source convention)
    p["open-gop"] = "0"
    p["scenecut"] = "0"
    p["keyint"] = str(gop)
    p["min-keyint"] = str(gop)
    p["bframes"] = str(facts["max_b_run"])
    if facts["max_b_run"]:
        p["b-pyramid"] = "0"
        p["b-adapt"] = "0"
    p["ref"] = str(st.refs)
    p["ctu"] = str(ctu)
    p["min-cu-size"] = str(min_cb)
    p["max-tu-size"] = str(max_tb)
    if min_tb != 4:
        notes.append(f"source min transform block size {min_tb} cannot be reproduced by x265 (always 4)")
    p["tu-intra-depth"] = str(max(1, min(4, sps_f["max_transform_hierarchy_depth_intra"] + 1)))
    p["tu-inter-depth"] = str(max(1, min(4, sps_f["max_transform_hierarchy_depth_inter"] + 1)))
    p["amp"] = str(sps_f["amp_enabled_flag"])
    p["sao"] = str(sps_f["sample_adaptive_offset_enabled_flag"])
    p["strong-intra-smoothing"] = str(sps_f["strong_intra_smoothing_enabled_flag"])
    p["temporal-mvp"] = str(sps_f["sps_temporal_mvp_enabled_flag"])
    if sps_f.get("scaling_list_enabled_flag"):
        p["scaling-list"] = "default"
        notes.append("source enables scaling lists; x265 'default' lists used (custom lists cannot be reproduced)")
    if sps_f.get("pcm_enabled_flag"):
        notes.append("source enables PCM coding; x265 cannot reproduce pcm_enabled_flag (decode-affecting)")
    p["log2-max-poc-lsb"] = str(sps_f["log2_max_pic_order_cnt_lsb_minus4"] + 4)
    p["signhide"] = str(pps_f["sign_data_hiding_enabled_flag"])
    p["tskip"] = str(pps_f["transform_skip_enabled_flag"])
    p["constrained-intra"] = str(pps_f["constrained_intra_pred_flag"])
    p["weightp"] = str(pps_f["weighted_pred_flag"])
    p["weightb"] = str(pps_f["weighted_bipred_flag"])
    p["cbqpoffs"] = str(pps_f["pps_cb_qp_offset"])
    p["crqpoffs"] = str(pps_f["pps_cr_qp_offset"])
    if pps_f["cu_qp_delta_enabled_flag"]:
        p["aq-mode"] = "1"
        p["qg-size"] = str(max(8, ctu >> pps_f["diff_cu_qp_delta_depth"]))
    else:
        p["aq-mode"] = "0"
        p["cutree"] = "0"
    p["wpp"] = str(pps_f["entropy_coding_sync_enabled_flag"])
    if pps_f.get("tiles_enabled_flag"):
        notes.append("source uses tiles; x265 cannot encode tiles (decode-affecting)")
    if pps_f.get("transquant_bypass_enabled_flag"):
        p["cu-lossless"] = "1"
        notes.append("source enables transquant bypass; x265 cu-lossless used")
    dis, beta, tc = facts["deblock"]
    if dis:
        p["deblock"] = "0:0"; p["no-deblock"] = "1"
    else:
        p["deblock"] = f"{beta}:{tc}"
    p["slices"] = str(facts["slices_per_pic"])
    if vui.get("vui_hrd_parameters_present_flag") or hrd_br:
        p["hrd"] = "1"
        p["vbv-maxrate"] = str(max(1, maxrate // 1000))
        p["vbv-bufsize"] = str(max(1, bufsize // 1000))
    p["range"] = "full" if full_range else "limited"
    if vui.get("video_signal_type_present_flag"):
        p["videoformat"] = ["component", "pal", "ntsc", "secam", "mac", "undef"][vui.get("video_format", 5)]
    if sps_f.get("long_term_ref_pics_present_flag"):
        notes.append("source SPS signals long-term reference pictures (syntax only; rewritten)")
    st.x265_params = p
    st.notes = notes + [f"x265 derived from source VPS/SPS/PPS: profile={st.profile} tier={tier} level={level_idc} ctu={ctu} min-cu={min_cb} "
                        f"max-tu={max_tb} tu-depth intra/inter={p['tu-intra-depth']}/{p['tu-inter-depth']} amp={p['amp']} sao={p['sao']} "
                        f"wpp={p['wpp']} signhide={p['signhide']} tskip={p['tskip']} cuqp={pps_f['cu_qp_delta_enabled_flag']} "
                        f"refs={st.refs} bframes={st.bframes} deblock={p['deblock']} slices={p['slices']} mean source QP={facts['mean_qp']}"]
    return st


def _refs_from_rps(sps_f: dict, pps_f: dict) -> int:
    """Number of active references: PPS default (what slices use unless overridden)."""
    return pps_f.get("num_ref_idx_l0_default_active_minus1", 0) + 1
