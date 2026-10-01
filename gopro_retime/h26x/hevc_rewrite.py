"""HEVC parameter-set transplant + slice-segment-header rewrite (see rewrite.py for the H.264 counterpart)."""
from __future__ import annotations

from typing import Optional

from . import hevc
from . import nal as N
from .nal import AccessUnit
from .rewrite import RewriteUnsafe, SliceConventions, TransplantResult

SPS_MUST_MATCH = [
    "chroma_format_idc", "separate_colour_plane_flag", "pic_width_in_luma_samples", "pic_height_in_luma_samples",
    "conformance_window_flag", "conf_win_left_offset", "conf_win_right_offset", "conf_win_top_offset", "conf_win_bottom_offset",
    "bit_depth_luma_minus8", "bit_depth_chroma_minus8", "log2_min_luma_coding_block_size_minus3",
    "log2_diff_max_min_luma_coding_block_size", "log2_min_luma_transform_block_size_minus2",
    "log2_diff_max_min_luma_transform_block_size", "max_transform_hierarchy_depth_inter", "max_transform_hierarchy_depth_intra",
    "scaling_list_enabled_flag", "sps_scaling_list_data_present_flag", "amp_enabled_flag", "sample_adaptive_offset_enabled_flag",
    "pcm_enabled_flag", "strong_intra_smoothing_enabled_flag", "sps_range_extension_flag", "sps_scc_extension_flag",
]
PPS_MUST_MATCH = [
    "sign_data_hiding_enabled_flag", "constrained_intra_pred_flag", "transform_skip_enabled_flag", "cu_qp_delta_enabled_flag",
    "diff_cu_qp_delta_depth", "weighted_pred_flag_", "transquant_bypass_enabled_flag", "tiles_enabled_flag",
    "entropy_coding_sync_enabled_flag", "pps_scaling_list_data_present_flag", "log2_parallel_merge_level_minus2",
    "pps_range_extension_flag", "pps_scc_extension_flag", "num_tile_columns_minus1", "num_tile_rows_minus1", "uniform_spacing_flag",
]
SUBLAYER_NONREF = {0, 2, 4, 6, 8, 10, 12, 14}
RASL_RADL = {6, 7, 8, 9}


def _check(enc: dict, tgt: dict, keys: list[str], what: str) -> None:
    for k in keys:
        if k.endswith("_"):
            continue
        a, b = enc.get(k, 0) or 0, tgt.get(k, 0) or 0
        if a != b:
            raise RewriteUnsafe(f"{what}.{k}: encoder {a} != target {b} (decode-affecting)")
    for sub in ("range_ext", "scc_ext", "scaling_list"):
        if (enc.get(sub) or tgt.get(sub)) and enc.get(sub) != tgt.get(sub):
            raise RewriteUnsafe(f"{what}.{sub} differs (decode-affecting)")


def measure_hevc_conventions(samples: list[bytes], sps_f: dict, pps_f: dict) -> dict:
    """NAL-type and slice-header habits of the camera stream."""
    conv = {"idr_type": None, "trail_ref_type": 1, "uses_sps_rps": None, "temporal_id_plus1": 1}
    for smp in samples:
        for n in N.split_length_prefixed(smp):
            if not N.is_vcl(n, "hevc"):
                continue
            t = N.hevc_nal_type(n)
            if t in (N.HEVC_IDR_W_RADL, N.HEVC_IDR_N_LP) and conv["idr_type"] is None:
                conv["idr_type"] = t
            if t == 1:
                conv["trail_ref_type"] = 1
            f, _d, _b = hevc.parse_slice_nal(n, sps_f, pps_f)
            conv["temporal_id_plus1"] = f.get("nuh_temporal_id_plus1", 1)
            if t not in (N.HEVC_IDR_W_RADL, N.HEVC_IDR_N_LP) and conv["uses_sps_rps"] is None and "short_term_ref_pic_set_sps_flag" in f:
                conv["uses_sps_rps"] = bool(f["short_term_ref_pic_set_sps_flag"])
    return conv


def _poc_sequence(aus: list[AccessUnit], sps_f: dict, pps_f: dict) -> list[tuple[dict, list]]:
    """Full PicOrderCnt per picture under the encoder SPS (8.3.1)."""
    max_lsb = 1 << (sps_f["log2_max_pic_order_cnt_lsb_minus4"] + 4)
    out = []
    prev_tid0_poc = 0
    first = True
    for au in aus:
        slices = []
        for n in au.nals:
            if N.is_vcl(n, "hevc"):
                f, data, hb = hevc.parse_slice_nal(n, sps_f, pps_f)
                slices.append((n, f, data, hb))
        if not slices:
            raise RewriteUnsafe("access unit without VCL NAL")
        f0 = slices[0][1]
        t = f0["nal_unit_type"]
        irap = 16 <= t <= 23
        idr = t in (19, 20)
        if idr:
            poc = 0
        else:
            lsb = f0["slice_pic_order_cnt_lsb"]
            if irap and (first or 16 <= t <= 18):   # CRA first in stream / BLA: NoRaslOutputFlag -> msb 0
                msb = 0
            else:
                prev_lsb = prev_tid0_poc % max_lsb
                prev_msb = prev_tid0_poc - prev_lsb
                if lsb < prev_lsb and (prev_lsb - lsb) >= max_lsb // 2:
                    msb = prev_msb + max_lsb
                elif lsb > prev_lsb and (lsb - prev_lsb) > max_lsb // 2:
                    msb = prev_msb - max_lsb
                else:
                    msb = prev_msb
            poc = msb + lsb
        tid = f0.get("nuh_temporal_id_plus1", 1) - 1
        if tid == 0 and t not in RASL_RADL and t not in SUBLAYER_NONREF:
            prev_tid0_poc = poc
        first = False
        out.append(({"poc": poc, "idr": idr, "irap": irap, "type": t}, slices))
    return out


def _rps_equal(a: dict, b: dict) -> bool:
    return (a["delta_poc_s0"] == b["delta_poc_s0"] and a["delta_poc_s1"] == b["delta_poc_s1"]
            and a["used_by_curr_pic_s0"] == b["used_by_curr_pic_s0"] and a["used_by_curr_pic_s1"] == b["used_by_curr_pic_s1"])


def transplant_hevc(aus: list[AccessUnit], enc_ps: dict[str, list[bytes]], tgt_ps: dict[str, list[bytes]],
                    conv: Optional[dict] = None, log=None) -> TransplantResult:
    conv = conv or {}
    for k in ("vps", "sps", "pps"):
        if len(enc_ps.get(k, [])) != 1 or len(tgt_ps.get(k, [])) != 1:
            raise RewriteUnsafe(f"exactly one {k.upper()} required on both sides")
    es, ts = hevc.parse_sps_nal(enc_ps["sps"][0]), hevc.parse_sps_nal(tgt_ps["sps"][0])
    ep, tp = hevc.parse_pps_nal(enc_ps["pps"][0], es), hevc.parse_pps_nal(tgt_ps["pps"][0], ts)
    _check(es, ts, SPS_MUST_MATCH, "SPS")
    _check(ep, tp, PPS_MUST_MATCH, "PPS")
    if ep.get("tiles_enabled_flag") and (ep.get("column_width_minus1") != tp.get("column_width_minus1") or ep.get("row_height_minus1") != tp.get("row_height_minus1") or ep.get("loop_filter_across_tiles_enabled_flag") != tp.get("loop_filter_across_tiles_enabled_flag")):
        raise RewriteUnsafe("tile layout differs")
    # DPB capacity of the target must cover the encoder's needs
    e_dpb = (es.get("sps_max_dec_pic_buffering_minus1") or [0])[-1]
    t_dpb = (ts.get("sps_max_dec_pic_buffering_minus1") or [0])[-1]
    if e_dpb > t_dpb:
        raise RewriteUnsafe(f"encoder needs DPB {e_dpb + 1} > target {t_dpb + 1}")
    e_reo = (es.get("sps_max_num_reorder_pics") or [0])[-1]
    t_reo = (ts.get("sps_max_num_reorder_pics") or [0])[-1]
    if e_reo > t_reo:
        raise RewriteUnsafe(f"encoder reorder depth {e_reo} > target {t_reo}")
    pics = _poc_sequence(aus, es, ep)
    t_max_lsb = 1 << (ts["log2_max_pic_order_cnt_lsb_minus4"] + 4)
    qp_shift = ep["init_qp_minus26"] - tp["init_qp_minus26"]
    changed: set[str] = set()
    idr_type = conv.get("idr_type")
    new_aus: list[AccessUnit] = []
    # verify POC lsb width under the target with the final nal types: simulate the decoder
    prev_tid0 = 0
    first = True
    for info, slices in pics:
        f0 = slices[0][1]
        t = f0["nal_unit_type"]
        if idr_type and t in (19, 20):
            t = idr_type
        if not info["idr"] and not (info["irap"] and (first or 16 <= t <= 18)):
            diff = info["poc"] - prev_tid0
            if abs(diff) >= t_max_lsb // 2:
                raise RewriteUnsafe(f"POC jump {diff} too large for target log2_max_pic_order_cnt_lsb {ts['log2_max_pic_order_cnt_lsb_minus4'] + 4}")
        tid = f0.get("nuh_temporal_id_plus1", 1) - 1
        if tid == 0 and t not in RASL_RADL and t not in SUBLAYER_NONREF:
            prev_tid0 = info["poc"]
        first = False
    for ai, (info, slices) in enumerate(pics):
        au_src = aus[ai]
        new_nals = [n for n in au_src.nals if not N.is_vcl(n, "hevc") and not N.is_sei(n, "hevc")
                    and not N.is_filler(n, "hevc") and not N.is_param_set(n, "hevc")]
        for n, f, data, hb in slices:
            g = dict(f)
            if idr_type and g["nal_unit_type"] in (19, 20) and g["nal_unit_type"] != idr_type:
                g["nal_unit_type"] = idr_type
                changed.add("idr nal type")
            if not info["idr"]:
                new_lsb = info["poc"] % t_max_lsb
                if new_lsb != f.get("slice_pic_order_cnt_lsb"):
                    changed.add("slice_pic_order_cnt_lsb")
                g["slice_pic_order_cnt_lsb"] = new_lsb
                # reference picture set
                if f.get("short_term_ref_pic_set_sps_flag"):
                    rps = es["st_ref_pic_sets"][f["short_term_ref_pic_set_idx"]]
                else:
                    rps = f["st_ref_pic_set"]
                match = next((i for i, r in enumerate(ts.get("st_ref_pic_sets", [])) if _rps_equal(r, rps)), None)
                if match is not None:
                    g["short_term_ref_pic_set_sps_flag"] = 1
                    g["short_term_ref_pic_set_idx"] = match
                    g.pop("st_ref_pic_set", None)
                else:
                    g["short_term_ref_pic_set_sps_flag"] = 0
                    g["st_ref_pic_set"] = {"inter_ref_pic_set_prediction_flag": 0, "delta_poc_s0": list(rps["delta_poc_s0"]),
                                           "delta_poc_s1": list(rps["delta_poc_s1"]), "used_by_curr_pic_s0": list(rps["used_by_curr_pic_s0"]),
                                           "used_by_curr_pic_s1": list(rps["used_by_curr_pic_s1"])}
                if bool(g["short_term_ref_pic_set_sps_flag"]) != bool(f.get("short_term_ref_pic_set_sps_flag")):
                    changed.add("rps signalling")
                # long-term pictures
                if ts["long_term_ref_pics_present_flag"]:
                    if not es["long_term_ref_pics_present_flag"]:
                        g["num_long_term_sps"] = 0
                        g["num_long_term_pics"] = 0
                        g["long_term_pics"] = []
                        changed.add("long-term fields added")
                elif es["long_term_ref_pics_present_flag"] and (f.get("num_long_term_sps", 0) or f.get("num_long_term_pics", 0)):
                    raise RewriteUnsafe("encoder uses long-term references but the target SPS has none")
                # temporal MVP
                if ts["sps_temporal_mvp_enabled_flag"] and not es["sps_temporal_mvp_enabled_flag"]:
                    g["slice_temporal_mvp_enabled_flag"] = 0
                    changed.add("slice_temporal_mvp_enabled_flag added")
                elif es["sps_temporal_mvp_enabled_flag"] and not ts["sps_temporal_mvp_enabled_flag"] and f.get("slice_temporal_mvp_enabled_flag"):
                    raise RewriteUnsafe("encoder uses temporal MVP but the target SPS disables it")
            # extra header bits / output flag / dependent slices
            g["slice_reserved_flag"] = [0] * tp["num_extra_slice_header_bits"]
            if tp["output_flag_present_flag"]:
                g.setdefault("pic_output_flag", 1)
            if f.get("dependent_slice_segment_flag") and not tp["dependent_slice_segments_enabled_flag"]:
                raise RewriteUnsafe("dependent slice segments cannot be expressed under the target PPS")
            # SAO
            if ts["sample_adaptive_offset_enabled_flag"] and not es["sample_adaptive_offset_enabled_flag"]:
                g["slice_sao_luma_flag"] = g["slice_sao_chroma_flag"] = 0
                changed.add("sao flags added")
            elif es["sample_adaptive_offset_enabled_flag"] and not ts["sample_adaptive_offset_enabled_flag"] and (f.get("slice_sao_luma_flag") or f.get("slice_sao_chroma_flag")):
                raise RewriteUnsafe("encoder slices use SAO but the target SPS disables it")
            st = f["slice_type"]
            if st in (hevc.SLICE_P, hevc.SLICE_B):
                n_l0, n_l1 = f["_num_ref_idx_l0_active"], f["_num_ref_idx_l1_active"]
                need = n_l0 != tp["num_ref_idx_l0_default_active_minus1"] + 1 or (st == hevc.SLICE_B and n_l1 != tp["num_ref_idx_l1_default_active_minus1"] + 1)
                g["num_ref_idx_active_override_flag"] = 1 if need else 0
                if need:
                    g["num_ref_idx_l0_active_minus1"] = n_l0 - 1
                    g["num_ref_idx_l1_active_minus1"] = n_l1 - 1
                if bool(need) != bool(f.get("num_ref_idx_active_override_flag")):
                    changed.add("num_ref_idx override")
                # list modification
                if tp["lists_modification_present_flag"] and not ep["lists_modification_present_flag"]:
                    g["ref_pic_list_modification_flag_l0"] = 0
                    g["ref_pic_list_modification_flag_l1"] = 0
                    changed.add("list modification flags added")
                elif ep["lists_modification_present_flag"] and not tp["lists_modification_present_flag"] and (f.get("ref_pic_list_modification_flag_l0") or f.get("ref_pic_list_modification_flag_l1")):
                    raise RewriteUnsafe("encoder modifies reference lists but the target PPS has no list modification")
                # cabac init
                if not tp["cabac_init_present_flag"] and f.get("cabac_init_flag"):
                    raise RewriteUnsafe("encoder slice uses cabac_init_flag=1 but the target PPS has no cabac_init_present_flag")
                if tp["cabac_init_present_flag"]:
                    g.setdefault("cabac_init_flag", 0)
                # weighted prediction
                tw = tp["weighted_pred_flag"] if st == hevc.SLICE_P else tp["weighted_bipred_flag"]
                ew = ep["weighted_pred_flag"] if st == hevc.SLICE_P else ep["weighted_bipred_flag"]
                if tw and not ew:
                    g["luma_log2_weight_denom"] = 0
                    g["delta_chroma_log2_weight_denom"] = 0
                    g["pred_weight_l0"] = [{"luma_weight_flag": 0, "chroma_weight_flag": 0} for _ in range(n_l0)]
                    g["pred_weight_l1"] = [{"luma_weight_flag": 0, "chroma_weight_flag": 0} for _ in range(n_l1)]
                    changed.add("default weight table added")
                elif ew and not tw:
                    if any(e.get("luma_weight_flag") or e.get("chroma_weight_flag") for e in f.get("pred_weight_l0", []) + f.get("pred_weight_l1", [])):
                        raise RewriteUnsafe("encoder uses explicit weights but the target PPS disables weighted prediction")
                    changed.add("weight table dropped")
            # QP / chroma offsets
            if qp_shift:
                g["slice_qp_delta"] = f["slice_qp_delta"] + qp_shift
                changed.add("slice_qp_delta (init_qp)")
            e_cb = ep["pps_cb_qp_offset"] + f.get("slice_cb_qp_offset", 0)
            e_cr = ep["pps_cr_qp_offset"] + f.get("slice_cr_qp_offset", 0)
            if tp["pps_slice_chroma_qp_offsets_present_flag"]:
                g["slice_cb_qp_offset"] = e_cb - tp["pps_cb_qp_offset"]
                g["slice_cr_qp_offset"] = e_cr - tp["pps_cr_qp_offset"]
                if (g["slice_cb_qp_offset"], g["slice_cr_qp_offset"]) != (f.get("slice_cb_qp_offset", 0), f.get("slice_cr_qp_offset", 0)):
                    changed.add("slice chroma qp offsets")
            elif (e_cb, e_cr) != (tp["pps_cb_qp_offset"], tp["pps_cr_qp_offset"]):
                raise RewriteUnsafe(f"chroma QP offsets {e_cb}/{e_cr} cannot be expressed under the target PPS ({tp['pps_cb_qp_offset']}/{tp['pps_cr_qp_offset']})")
            # deblocking
            e_dis = f.get("slice_deblocking_filter_disabled_flag", ep.get("pps_deblocking_filter_disabled_flag", 0))
            e_beta = f.get("slice_beta_offset_div2", ep.get("pps_beta_offset_div2", 0)) if not e_dis else 0
            e_tc = f.get("slice_tc_offset_div2", ep.get("pps_tc_offset_div2", 0)) if not e_dis else 0
            t_dis, t_beta, t_tc = tp.get("pps_deblocking_filter_disabled_flag", 0), tp.get("pps_beta_offset_div2", 0), tp.get("pps_tc_offset_div2", 0)
            same = (e_dis, e_beta, e_tc) == (t_dis, t_beta, t_tc)
            if tp["deblocking_filter_override_enabled_flag"]:
                g["deblocking_filter_override_flag"] = 0 if same else 1
                if not same:
                    g["slice_deblocking_filter_disabled_flag"] = e_dis
                    g["slice_beta_offset_div2"] = e_beta
                    g["slice_tc_offset_div2"] = e_tc
                if bool(g["deblocking_filter_override_flag"]) != bool(f.get("deblocking_filter_override_flag")):
                    changed.add("deblocking override")
            elif not same:
                raise RewriteUnsafe("deblocking parameters differ and the target PPS allows no slice override")
            else:
                g["deblocking_filter_override_flag"] = 0
            g["slice_deblocking_filter_disabled_flag"] = e_dis if g.get("deblocking_filter_override_flag") else t_dis
            # loop filter across slices
            e_lf = f.get("slice_loop_filter_across_slices_enabled_flag", ep["pps_loop_filter_across_slices_enabled_flag"]) if ep["pps_loop_filter_across_slices_enabled_flag"] else 0
            if tp["pps_loop_filter_across_slices_enabled_flag"]:
                g["slice_loop_filter_across_slices_enabled_flag"] = e_lf
            elif e_lf and len(slices) > 1:
                raise RewriteUnsafe("loop filter across slices used but the target PPS disables it")
            # header extension
            if tp["slice_segment_header_extension_present_flag"]:
                g.setdefault("slice_segment_header_extension_length", 0)
                g.setdefault("slice_segment_header_extension_data_byte", [])
            # merge candidates / mvd / collocated: copied from the encoder slice
            new_nals.append(hevc.write_slice_nal(g, data, hb, ts, tp))
        new_aus.append(AccessUnit(new_nals, "hevc"))
    if log:
        log(f"hevc transplant: rewrote {sorted(changed)}")
    return TransplantResult(new_aus, {"vps": [tgt_ps["vps"][0]], "sps": [tgt_ps["sps"][0]], "pps": [tgt_ps["pps"][0]]}, sorted(changed), [], True)
