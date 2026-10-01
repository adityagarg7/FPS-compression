"""HEVC (ITU-T H.265 / ISO/IEC 23008-2) VPS / SPS / PPS / slice segment header syntax, symmetric parse & write.

Same conventions as h264.py: one function per syntax structure serves both directions (see bits.py), values live in
plain dicts, derived convenience values use underscore keys. Syntax element names follow the specification."""
from __future__ import annotations

from typing import Optional

from .bits import BitIO, BitError, ceil_log2
from . import nal as N

SLICE_B, SLICE_P, SLICE_I = 0, 1, 2
EXTENDED_SAR = 255


# ---- small symmetric helpers -----------------------------------------------------------------------
def _ue_list(io: BitIO, f: dict, name: str, count: int) -> list[int]:
    if io.reading:
        f[name] = [io.read_ue() for _ in range(count)]
    else:
        for v in f[name][:count]:
            io.write_ue(int(v))
    return f[name]


def _se_list(io: BitIO, f: dict, name: str, count: int) -> list[int]:
    if io.reading:
        f[name] = [io.read_se() for _ in range(count)]
    else:
        for v in f[name][:count]:
            io.write_se(int(v))
    return f[name]


def _dict_list(io: BitIO, f: dict, name: str, count: int) -> list[dict]:
    """A list of `count` sub-dicts (created when reading, taken from `f` when writing)."""
    if io.reading:
        f[name] = [{} for _ in range(count)]
    return f[name][:count]


def _extension_data(io: BitIO, f: dict, name: str) -> None:
    """Opaque *_extension_data_flag bits up to the rbsp trailing bits (kept verbatim as a list of bits)."""
    if io.reading:
        f[name] = []
        while io.more_rbsp_data():
            f[name].append(io.read_bits(1))
    else:
        for b in f[name]:
            io.write_bits(int(b), 1)


def _extension_flags(io: BitIO, f: dict, p: str) -> None:
    names = (p + "range_extension_flag", p + "multilayer_extension_flag", p + "3d_extension_flag", p + "scc_extension_flag")
    if io.flag(f, p + "extension_present_flag"):
        for k in names:
            io.flag(f, k)
        io.u(4, f, p + "extension_4bits")
    else:
        for k in names + (p + "extension_4bits",):
            f[k] = 0


def _nal_header(io: BitIO, f: dict) -> None:
    f.setdefault("forbidden_zero_bit", 0)
    f.setdefault("nuh_layer_id", 0)
    f.setdefault("nuh_temporal_id_plus1", 1)
    io.u(1, f, "forbidden_zero_bit")
    io.u(6, f, "nal_unit_type")
    io.u(6, f, "nuh_layer_id")
    io.u(3, f, "nuh_temporal_id_plus1")


def _chroma_array_type(sps_f: dict) -> int:
    return 0 if sps_f.get("separate_colour_plane_flag") else sps_f["chroma_format_idc"]


def _pic_size_in_ctbs(sps_f: dict) -> int:
    ctb = 1 << (sps_f["log2_min_luma_coding_block_size_minus3"] + 3 + sps_f["log2_diff_max_min_luma_coding_block_size"])
    return -(-sps_f["pic_width_in_luma_samples"] // ctb) * -(-sps_f["pic_height_in_luma_samples"] // ctb)


# ---- profile_tier_level (7.3.3) --------------------------------------------------------------------
def _profile_body(io: BitIO, f: dict, p: str) -> None:
    io.u(2, f, p + "profile_space")
    io.flag(f, p + "tier_flag")
    io.u(5, f, p + "profile_idc")
    io.u(32, f, p + "profile_compatibility_flags")
    io.flag(f, p + "progressive_source_flag")
    io.flag(f, p + "interlaced_source_flag")
    io.flag(f, p + "non_packed_constraint_flag")
    io.flag(f, p + "frame_only_constraint_flag")
    io.u(43, f, p + "constraint_flags")      # profile-specific constraint flags / reserved_zero_43bits
    io.flag(f, p + "inbld_flag")             # or reserved_zero_bit


def profile_tier_level(io: BitIO, f: dict, profile_present: bool, max_sub_layers_minus1: int) -> None:
    if profile_present:
        _profile_body(io, f, "general_")
    io.u(8, f, "general_level_idc")
    subs = _dict_list(io, f, "sub_layers", max_sub_layers_minus1)
    for s in subs:
        io.flag(s, "sub_layer_profile_present_flag")
        io.flag(s, "sub_layer_level_present_flag")
    if max_sub_layers_minus1 > 0:
        io.u_list(2, f, "reserved_zero_2bits", 8 - max_sub_layers_minus1)
    for s in subs:
        if s["sub_layer_profile_present_flag"]:
            _profile_body(io, s, "sub_layer_")
        if s["sub_layer_level_present_flag"]:
            io.u(8, s, "sub_layer_level_idc")


# ---- HRD (E.2.2) -----------------------------------------------------------------------------------
def sub_layer_hrd_parameters(io: BitIO, f: dict, cpb_cnt: int, sub_pic: int) -> None:
    cpbs = _dict_list(io, f, "cpb", cpb_cnt)
    for c in cpbs:
        io.ue(c, "bit_rate_value_minus1")
        io.ue(c, "cpb_size_value_minus1")
        if sub_pic:
            io.ue(c, "cpb_size_du_value_minus1")
            io.ue(c, "bit_rate_du_value_minus1")
        io.flag(c, "cbr_flag")


def hrd_parameters(io: BitIO, f: dict, common_inf_present: bool, max_sub_layers_minus1: int) -> None:
    if common_inf_present:
        io.flag(f, "nal_hrd_parameters_present_flag")
        io.flag(f, "vcl_hrd_parameters_present_flag")
        f["sub_pic_hrd_params_present_flag"] = 0
        if f["nal_hrd_parameters_present_flag"] or f["vcl_hrd_parameters_present_flag"]:
            if io.flag(f, "sub_pic_hrd_params_present_flag"):
                io.u(8, f, "tick_divisor_minus2")
                io.u(5, f, "du_cpb_removal_delay_increment_length_minus1")
                io.flag(f, "sub_pic_cpb_params_in_pic_timing_sei_flag")
                io.u(5, f, "dpb_output_delay_du_length_minus1")
            io.u(4, f, "bit_rate_scale")
            io.u(4, f, "cpb_size_scale")
            if f["sub_pic_hrd_params_present_flag"]:
                io.u(4, f, "cpb_size_du_scale")
            io.u(5, f, "initial_cpb_removal_delay_length_minus1")
            io.u(5, f, "au_cpb_removal_delay_length_minus1")
            io.u(5, f, "dpb_output_delay_length_minus1")
    for s in _dict_list(io, f, "sub_layers", max_sub_layers_minus1 + 1):
        if io.flag(s, "fixed_pic_rate_general_flag"):
            s["fixed_pic_rate_within_cvs_flag"] = 1
        else:
            io.flag(s, "fixed_pic_rate_within_cvs_flag")
        if s["fixed_pic_rate_within_cvs_flag"]:
            io.ue(s, "elemental_duration_in_tc_minus1")
            s["low_delay_hrd_flag"] = 0
        else:
            io.flag(s, "low_delay_hrd_flag")
        if s["low_delay_hrd_flag"]:
            s["cpb_cnt_minus1"] = 0
        else:
            io.ue(s, "cpb_cnt_minus1")
        for key, present in (("nal", f["nal_hrd_parameters_present_flag"]), ("vcl", f["vcl_hrd_parameters_present_flag"])):
            if present:
                sub_layer_hrd_parameters(io, s.setdefault(key, {}), s["cpb_cnt_minus1"] + 1, f["sub_pic_hrd_params_present_flag"])


# ---- scaling lists (7.3.4) -------------------------------------------------------------------------
def scaling_list_data(io: BitIO, f: dict) -> None:
    """Stored as f['scaling_lists']: one dict per coded (size_id, matrix_id) in bitstream order."""
    if io.reading:
        f["scaling_lists"] = []
    k = 0
    for size_id in range(4):
        for matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            if io.reading:
                f["scaling_lists"].append({"size_id": size_id, "matrix_id": matrix_id})
            e = f["scaling_lists"][k]
            k += 1
            if not io.flag(e, "scaling_list_pred_mode_flag"):
                io.ue(e, "scaling_list_pred_matrix_id_delta")
            else:
                if size_id > 1:
                    io.se(e, "scaling_list_dc_coef_minus8")
                _se_list(io, e, "scaling_list_delta_coef", min(64, 1 << (4 + (size_id << 1))))


# ---- short-term reference picture sets (7.3.7 / 7.4.8) ---------------------------------------------
def _expand_inter_rps(r: dict, ref: dict) -> None:
    """Derive DeltaPocS0/S1 + UsedByCurrPicS0/S1 of an inter-predicted RPS from its reference RPS (7-61, 7-62)."""
    delta_rps = (1 - 2 * r["delta_rps_sign"]) * (r["abs_delta_rps_minus1"] + 1)
    used, use = r["used_by_curr_pic_flag"], r["use_delta_flag"]
    ref_s0, ref_s1 = ref["delta_poc_s0"], ref["delta_poc_s1"]
    n_neg, n_all = len(ref_s0), len(ref_s0) + len(ref_s1)
    s0: list[tuple[int, int]] = []
    s1: list[tuple[int, int]] = []
    for j in range(len(ref_s1) - 1, -1, -1):
        d = ref_s1[j] + delta_rps
        if d < 0 and use[n_neg + j]:
            s0.append((d, used[n_neg + j]))
    if delta_rps < 0 and use[n_all]:
        s0.append((delta_rps, used[n_all]))
    for j in range(n_neg):
        d = ref_s0[j] + delta_rps
        if d < 0 and use[j]:
            s0.append((d, used[j]))
    for j in range(n_neg - 1, -1, -1):
        d = ref_s0[j] + delta_rps
        if d > 0 and use[j]:
            s1.append((d, used[j]))
    if delta_rps > 0 and use[n_all]:
        s1.append((delta_rps, used[n_all]))
    for j in range(len(ref_s1)):
        d = ref_s1[j] + delta_rps
        if d > 0 and use[n_neg + j]:
            s1.append((d, used[n_neg + j]))
    r["delta_poc_s0"], r["used_by_curr_pic_s0"] = [d for d, _ in s0], [u for _, u in s0]
    r["delta_poc_s1"], r["used_by_curr_pic_s1"] = [d for d, _ in s1], [u for _, u in s1]


def _explicit_rps(io: BitIO, r: dict) -> None:
    if io.reading:
        n_neg, n_pos = io.read_ue(), io.read_ue()
        r["delta_poc_s0"], r["used_by_curr_pic_s0"], r["delta_poc_s1"], r["used_by_curr_pic_s1"] = [], [], [], []
        poc = 0
        for _ in range(n_neg):
            poc -= io.read_ue() + 1
            r["delta_poc_s0"].append(poc)
            r["used_by_curr_pic_s0"].append(io.read_bits(1))
        poc = 0
        for _ in range(n_pos):
            poc += io.read_ue() + 1
            r["delta_poc_s1"].append(poc)
            r["used_by_curr_pic_s1"].append(io.read_bits(1))
    else:
        io.write_ue(len(r["delta_poc_s0"]))
        io.write_ue(len(r["delta_poc_s1"]))
        prev = 0
        for d, u in zip(r["delta_poc_s0"], r["used_by_curr_pic_s0"]):
            io.write_ue(prev - d - 1)
            io.write_bits(u, 1)
            prev = d
        prev = 0
        for d, u in zip(r["delta_poc_s1"], r["used_by_curr_pic_s1"]):
            io.write_ue(d - prev - 1)
            io.write_bits(u, 1)
            prev = d


def st_ref_pic_set(io: BitIO, r: dict, idx: int, num_sets: int, sps_rps_list: list[dict]) -> None:
    """st_ref_pic_set(idx). The RPS dict always carries the expanded `delta_poc_s0/s1` (signed POC deltas, closest
    first) and `used_by_curr_pic_s0/s1` lists; with inter_ref_pic_set_prediction_flag = 0 those lists ARE the coded
    content, otherwise the coded prediction elements are stored too and the lists are derived from the reference RPS."""
    if idx != 0:
        io.flag(r, "inter_ref_pic_set_prediction_flag")
    else:
        r["inter_ref_pic_set_prediction_flag"] = 0
    if not r["inter_ref_pic_set_prediction_flag"]:
        _explicit_rps(io, r)
        return
    if idx == num_sets:
        io.ue(r, "delta_idx_minus1")
    else:
        r["delta_idx_minus1"] = 0
    io.flag(r, "delta_rps_sign")
    io.ue(r, "abs_delta_rps_minus1")
    ref_idx = idx - (r["delta_idx_minus1"] + 1)
    if not 0 <= ref_idx < len(sps_rps_list):
        raise BitError(f"st_ref_pic_set {idx}: reference RPS index {ref_idx} out of range")
    ref = sps_rps_list[ref_idx]
    n_ref = len(ref["delta_poc_s0"]) + len(ref["delta_poc_s1"])
    if io.reading:
        r["used_by_curr_pic_flag"], r["use_delta_flag"] = [], []
        for _ in range(n_ref + 1):
            used = io.read_bits(1)
            r["used_by_curr_pic_flag"].append(used)
            r["use_delta_flag"].append(1 if used else io.read_bits(1))
    else:
        for j in range(n_ref + 1):
            io.write_bits(r["used_by_curr_pic_flag"][j], 1)
            if not r["used_by_curr_pic_flag"][j]:
                io.write_bits(r["use_delta_flag"][j], 1)
    _expand_inter_rps(r, ref)


# ---- VPS (7.3.2.1) ---------------------------------------------------------------------------------
def vps(io: BitIO, f: dict) -> None:
    io.u(4, f, "vps_video_parameter_set_id")
    io.flag(f, "vps_base_layer_internal_flag")
    io.flag(f, "vps_base_layer_available_flag")
    io.u(6, f, "vps_max_layers_minus1")
    io.u(3, f, "vps_max_sub_layers_minus1")
    io.flag(f, "vps_temporal_id_nesting_flag")
    io.u(16, f, "vps_reserved_0xffff_16bits")
    profile_tier_level(io, f.setdefault("ptl", {}), True, f["vps_max_sub_layers_minus1"])
    io.flag(f, "vps_sub_layer_ordering_info_present_flag")
    _sub_layer_ordering_info(io, f, "vps_", f["vps_max_sub_layers_minus1"], f["vps_sub_layer_ordering_info_present_flag"])
    io.u(6, f, "vps_max_layer_id")
    io.ue(f, "vps_num_layer_sets_minus1")
    sets = _dict_list(io, f, "layer_sets", f["vps_num_layer_sets_minus1"])
    for s in sets:
        io.u_list(1, s, "layer_id_included_flag", f["vps_max_layer_id"] + 1)
    if io.flag(f, "vps_timing_info_present_flag"):
        io.u(32, f, "vps_num_units_in_tick")
        io.u(32, f, "vps_time_scale")
        if io.flag(f, "vps_poc_proportional_to_timing_flag"):
            io.ue(f, "vps_num_ticks_poc_diff_one_minus1")
        io.ue(f, "vps_num_hrd_parameters")
        hrds = _dict_list(io, f, "hrd", f["vps_num_hrd_parameters"])
        for i, h in enumerate(hrds):
            io.ue(h, "hrd_layer_set_idx")
            if i > 0:
                io.flag(h, "cprms_present_flag")
            else:
                h["cprms_present_flag"] = 1
            if not h["cprms_present_flag"]:
                for k in ("nal_hrd_parameters_present_flag", "vcl_hrd_parameters_present_flag", "sub_pic_hrd_params_present_flag"):
                    h[k] = hrds[i - 1][k]
            hrd_parameters(io, h, bool(h["cprms_present_flag"]), f["vps_max_sub_layers_minus1"])
    if io.flag(f, "vps_extension_flag"):
        _extension_data(io, f, "vps_extension_data_flag")
    io.rbsp_trailing_bits()


def _sub_layer_ordering_info(io: BitIO, f: dict, p: str, max_sub_layers_minus1: int, present: int) -> None:
    """Lists indexed by sub-layer; entries below the first coded one carry the inferred (highest sub-layer) value."""
    names = (p + "max_dec_pic_buffering_minus1", p + "max_num_reorder_pics", p + "max_latency_increase_plus1")
    start = 0 if present else max_sub_layers_minus1
    if io.reading:
        for k in names:
            f[k] = [0] * (max_sub_layers_minus1 + 1)
    for i in range(start, max_sub_layers_minus1 + 1):
        for k in names:
            if io.reading:
                f[k][i] = io.read_ue()
            else:
                io.write_ue(f[k][i])
    if io.reading and not present:
        for k in names:
            f[k] = [f[k][max_sub_layers_minus1]] * (max_sub_layers_minus1 + 1)


# ---- SPS (7.3.2.2) ---------------------------------------------------------------------------------
def vui_parameters(io: BitIO, f: dict, max_sub_layers_minus1: int) -> None:
    if io.flag(f, "aspect_ratio_info_present_flag"):
        io.u(8, f, "aspect_ratio_idc")
        if f["aspect_ratio_idc"] == EXTENDED_SAR:
            io.u(16, f, "sar_width")
            io.u(16, f, "sar_height")
    if io.flag(f, "overscan_info_present_flag"):
        io.flag(f, "overscan_appropriate_flag")
    if io.flag(f, "video_signal_type_present_flag"):
        io.u(3, f, "video_format")
        io.flag(f, "video_full_range_flag")
        if io.flag(f, "colour_description_present_flag"):
            io.u(8, f, "colour_primaries")
            io.u(8, f, "transfer_characteristics")
            io.u(8, f, "matrix_coeffs")
    if io.flag(f, "chroma_loc_info_present_flag"):
        io.ue(f, "chroma_sample_loc_type_top_field")
        io.ue(f, "chroma_sample_loc_type_bottom_field")
    io.flag(f, "neutral_chroma_indication_flag")
    io.flag(f, "field_seq_flag")
    io.flag(f, "frame_field_info_present_flag")
    if io.flag(f, "default_display_window_flag"):
        io.ue(f, "def_disp_win_left_offset")
        io.ue(f, "def_disp_win_right_offset")
        io.ue(f, "def_disp_win_top_offset")
        io.ue(f, "def_disp_win_bottom_offset")
    if io.flag(f, "vui_timing_info_present_flag"):
        io.u(32, f, "vui_num_units_in_tick")
        io.u(32, f, "vui_time_scale")
        if io.flag(f, "vui_poc_proportional_to_timing_flag"):
            io.ue(f, "vui_num_ticks_poc_diff_one_minus1")
        if io.flag(f, "vui_hrd_parameters_present_flag"):
            hrd_parameters(io, f.setdefault("hrd", {}), True, max_sub_layers_minus1)
    if io.flag(f, "bitstream_restriction_flag"):
        io.flag(f, "tiles_fixed_structure_flag")
        io.flag(f, "motion_vectors_over_pic_boundaries_flag")
        io.flag(f, "restricted_ref_pic_lists_flag")
        io.ue(f, "min_spatial_segmentation_idc")
        io.ue(f, "max_bytes_per_pic_denom")
        io.ue(f, "max_bits_per_min_cu_denom")
        io.ue(f, "log2_max_mv_length_horizontal")
        io.ue(f, "log2_max_mv_length_vertical")


def sps_range_extension(io: BitIO, f: dict) -> None:
    for k in ("transform_skip_rotation_enabled_flag", "transform_skip_context_enabled_flag", "implicit_rdpcm_enabled_flag",
              "explicit_rdpcm_enabled_flag", "extended_precision_processing_flag", "intra_smoothing_disabled_flag",
              "high_precision_offsets_enabled_flag", "persistent_rice_adaptation_enabled_flag",
              "cabac_bypass_alignment_enabled_flag"):
        io.flag(f, k)


def sps_3d_extension(io: BitIO, f: dict) -> None:
    for d in _dict_list(io, f, "sps_3d", 2):
        io.flag(d, "iv_di_mc_enabled_flag")
        io.flag(d, "iv_mv_scal_enabled_flag")
        if d is f["sps_3d"][0]:
            io.ue(d, "log2_ivmc_sub_pb_size_minus3")
            for k in ("iv_res_pred_enabled_flag", "depth_ref_enabled_flag", "vsp_mc_enabled_flag", "dbbp_enabled_flag"):
                io.flag(d, k)
        else:
            io.flag(d, "tex_mc_enabled_flag")
            io.ue(d, "log2_texmc_sub_pb_size_minus3")
            for k in ("intra_contour_enabled_flag", "intra_dc_only_wedge_enabled_flag", "cqt_cu_part_pred_enabled_flag",
                      "inter_dc_only_enabled_flag", "skip_intra_enabled_flag"):
                io.flag(d, k)


def sps_scc_extension(io: BitIO, f: dict, chroma_format_idc: int, bit_depth_luma: int, bit_depth_chroma: int) -> None:
    io.flag(f, "sps_curr_pic_ref_enabled_flag")
    if io.flag(f, "palette_mode_enabled_flag"):
        io.ue(f, "palette_max_size")
        io.ue(f, "delta_palette_max_predictor_size")
        if io.flag(f, "sps_palette_predictor_initializers_present_flag"):
            io.ue(f, "sps_num_palette_predictor_initializers_minus1")
            n = f["sps_num_palette_predictor_initializers_minus1"] + 1
            comps = _dict_list(io, f, "sps_palette_predictor_initializer", 1 if chroma_format_idc == 0 else 3)
            for comp, c in enumerate(comps):
                io.u_list(bit_depth_luma if comp == 0 else bit_depth_chroma, c, "value", n)
    io.u(2, f, "motion_vector_resolution_control_idc")
    io.flag(f, "intra_boundary_filtering_disabled_flag")


def sps(io: BitIO, f: dict) -> None:
    io.u(4, f, "sps_video_parameter_set_id")
    io.u(3, f, "sps_max_sub_layers_minus1")
    if f.get("nuh_layer_id", 0) != 0 and f["sps_max_sub_layers_minus1"] == 7:
        raise BitError("multi-layer extension SPS (sps_ext_or_max_sub_layers_minus1 == 7) not supported")
    msl = f["sps_max_sub_layers_minus1"]
    io.flag(f, "sps_temporal_id_nesting_flag")
    profile_tier_level(io, f.setdefault("ptl", {}), True, msl)
    io.ue(f, "sps_seq_parameter_set_id")
    io.ue(f, "chroma_format_idc")
    if f["chroma_format_idc"] == 3:
        io.flag(f, "separate_colour_plane_flag")
    else:
        f["separate_colour_plane_flag"] = 0
    io.ue(f, "pic_width_in_luma_samples")
    io.ue(f, "pic_height_in_luma_samples")
    if io.flag(f, "conformance_window_flag"):
        io.ue(f, "conf_win_left_offset")
        io.ue(f, "conf_win_right_offset")
        io.ue(f, "conf_win_top_offset")
        io.ue(f, "conf_win_bottom_offset")
    io.ue(f, "bit_depth_luma_minus8")
    io.ue(f, "bit_depth_chroma_minus8")
    io.ue(f, "log2_max_pic_order_cnt_lsb_minus4")
    io.flag(f, "sps_sub_layer_ordering_info_present_flag")
    _sub_layer_ordering_info(io, f, "sps_", msl, f["sps_sub_layer_ordering_info_present_flag"])
    io.ue(f, "log2_min_luma_coding_block_size_minus3")
    io.ue(f, "log2_diff_max_min_luma_coding_block_size")
    io.ue(f, "log2_min_luma_transform_block_size_minus2")
    io.ue(f, "log2_diff_max_min_luma_transform_block_size")
    io.ue(f, "max_transform_hierarchy_depth_inter")
    io.ue(f, "max_transform_hierarchy_depth_intra")
    if io.flag(f, "scaling_list_enabled_flag"):
        if io.flag(f, "sps_scaling_list_data_present_flag"):
            scaling_list_data(io, f)
    io.flag(f, "amp_enabled_flag")
    io.flag(f, "sample_adaptive_offset_enabled_flag")
    if io.flag(f, "pcm_enabled_flag"):
        io.u(4, f, "pcm_sample_bit_depth_luma_minus1")
        io.u(4, f, "pcm_sample_bit_depth_chroma_minus1")
        io.ue(f, "log2_min_pcm_luma_coding_block_size_minus3")
        io.ue(f, "log2_diff_max_min_pcm_luma_coding_block_size")
        io.flag(f, "pcm_loop_filter_disabled_flag")
    io.ue(f, "num_short_term_ref_pic_sets")
    rps_list = _dict_list(io, f, "st_ref_pic_sets", f["num_short_term_ref_pic_sets"])
    for i, r in enumerate(rps_list):
        st_ref_pic_set(io, r, i, f["num_short_term_ref_pic_sets"], rps_list)
    if io.flag(f, "long_term_ref_pics_present_flag"):
        io.ue(f, "num_long_term_ref_pics_sps")
        for lt in _dict_list(io, f, "long_term_ref_pics_sps", f["num_long_term_ref_pics_sps"]):
            io.u(f["log2_max_pic_order_cnt_lsb_minus4"] + 4, lt, "lt_ref_pic_poc_lsb_sps")
            io.flag(lt, "used_by_curr_pic_lt_sps_flag")
    else:
        f["num_long_term_ref_pics_sps"] = 0
    io.flag(f, "sps_temporal_mvp_enabled_flag")
    io.flag(f, "strong_intra_smoothing_enabled_flag")
    if io.flag(f, "vui_parameters_present_flag"):
        vui_parameters(io, f.setdefault("vui", {}), msl)
    _extension_flags(io, f, "sps_")
    if f["sps_range_extension_flag"]:
        sps_range_extension(io, f.setdefault("range_ext", {}))
    if f["sps_multilayer_extension_flag"]:
        io.flag(f, "inter_view_mv_vert_constraint_flag")
    if f["sps_3d_extension_flag"]:
        sps_3d_extension(io, f)
    if f["sps_scc_extension_flag"]:
        sps_scc_extension(io, f.setdefault("scc_ext", {}), f["chroma_format_idc"], f["bit_depth_luma_minus8"] + 8, f["bit_depth_chroma_minus8"] + 8)
    if f["sps_extension_4bits"]:
        _extension_data(io, f, "sps_extension_data_flag")
    io.rbsp_trailing_bits()


# ---- PPS (7.3.2.3) ---------------------------------------------------------------------------------
def pps_range_extension(io: BitIO, f: dict, transform_skip_enabled: int) -> None:
    if transform_skip_enabled:
        io.ue(f, "log2_max_transform_skip_block_size_minus2")
    io.flag(f, "cross_component_prediction_enabled_flag")
    if io.flag(f, "chroma_qp_offset_list_enabled_flag"):
        io.ue(f, "diff_cu_chroma_qp_offset_depth")
        io.ue(f, "chroma_qp_offset_list_len_minus1")
        for e in _dict_list(io, f, "chroma_qp_offset_list", f["chroma_qp_offset_list_len_minus1"] + 1):
            io.se(e, "cb_qp_offset_list")
            io.se(e, "cr_qp_offset_list")
    io.ue(f, "log2_sao_offset_scale_luma")
    io.ue(f, "log2_sao_offset_scale_chroma")


def pps_scc_extension(io: BitIO, f: dict) -> None:
    io.flag(f, "pps_curr_pic_ref_enabled_flag")
    if io.flag(f, "residual_adaptive_colour_transform_enabled_flag"):
        io.flag(f, "pps_slice_act_qp_offsets_present_flag")
        io.se(f, "pps_act_y_qp_offset_plus5")
        io.se(f, "pps_act_cb_qp_offset_plus5")
        io.se(f, "pps_act_cr_qp_offset_plus3")
    else:
        f["pps_slice_act_qp_offsets_present_flag"] = 0
    if io.flag(f, "pps_palette_predictor_initializers_present_flag"):
        io.ue(f, "pps_num_palette_predictor_initializers")
        n = f["pps_num_palette_predictor_initializers"]
        if n > 0:
            io.flag(f, "monochrome_palette_flag")
            io.ue(f, "luma_bit_depth_entry_minus8")
            if not f["monochrome_palette_flag"]:
                io.ue(f, "chroma_bit_depth_entry_minus8")
            comps = _dict_list(io, f, "pps_palette_predictor_initializer", 1 if f["monochrome_palette_flag"] else 3)
            for comp, c in enumerate(comps):
                io.u_list((f["luma_bit_depth_entry_minus8"] if comp == 0 else f["chroma_bit_depth_entry_minus8"]) + 8, c, "value", n)


def pps(io: BitIO, f: dict, sps_f: Optional[dict] = None) -> None:
    """pic_parameter_set_rbsp(). `sps_f` is accepted for API parity with h264.pps; the HEVC PPS syntax does not
    depend on the SPS."""
    io.ue(f, "pps_pic_parameter_set_id")
    io.ue(f, "pps_seq_parameter_set_id")
    io.flag(f, "dependent_slice_segments_enabled_flag")
    io.flag(f, "output_flag_present_flag")
    io.u(3, f, "num_extra_slice_header_bits")
    io.flag(f, "sign_data_hiding_enabled_flag")
    io.flag(f, "cabac_init_present_flag")
    io.ue(f, "num_ref_idx_l0_default_active_minus1")
    io.ue(f, "num_ref_idx_l1_default_active_minus1")
    io.se(f, "init_qp_minus26")
    io.flag(f, "constrained_intra_pred_flag")
    io.flag(f, "transform_skip_enabled_flag")
    if io.flag(f, "cu_qp_delta_enabled_flag"):
        io.ue(f, "diff_cu_qp_delta_depth")
    io.se(f, "pps_cb_qp_offset")
    io.se(f, "pps_cr_qp_offset")
    io.flag(f, "pps_slice_chroma_qp_offsets_present_flag")
    io.flag(f, "weighted_pred_flag")
    io.flag(f, "weighted_bipred_flag")
    io.flag(f, "transquant_bypass_enabled_flag")
    io.flag(f, "tiles_enabled_flag")
    io.flag(f, "entropy_coding_sync_enabled_flag")
    if f["tiles_enabled_flag"]:
        io.ue(f, "num_tile_columns_minus1")
        io.ue(f, "num_tile_rows_minus1")
        if not io.flag(f, "uniform_spacing_flag"):
            _ue_list(io, f, "column_width_minus1", f["num_tile_columns_minus1"])
            _ue_list(io, f, "row_height_minus1", f["num_tile_rows_minus1"])
        io.flag(f, "loop_filter_across_tiles_enabled_flag")
    io.flag(f, "pps_loop_filter_across_slices_enabled_flag")
    if io.flag(f, "deblocking_filter_control_present_flag"):
        io.flag(f, "deblocking_filter_override_enabled_flag")
        if not io.flag(f, "pps_deblocking_filter_disabled_flag"):
            io.se(f, "pps_beta_offset_div2")
            io.se(f, "pps_tc_offset_div2")
    else:
        f["deblocking_filter_override_enabled_flag"] = f["pps_deblocking_filter_disabled_flag"] = 0
    if io.flag(f, "pps_scaling_list_data_present_flag"):
        scaling_list_data(io, f)
    io.flag(f, "lists_modification_present_flag")
    io.ue(f, "log2_parallel_merge_level_minus2")
    io.flag(f, "slice_segment_header_extension_present_flag")
    _extension_flags(io, f, "pps_")
    if f["pps_range_extension_flag"]:
        pps_range_extension(io, f.setdefault("range_ext", {}), f["transform_skip_enabled_flag"])
    if f["pps_multilayer_extension_flag"] or f["pps_3d_extension_flag"]:
        raise BitError("pps_multilayer_extension / pps_3d_extension not supported")
    if f["pps_scc_extension_flag"]:
        pps_scc_extension(io, f.setdefault("scc_ext", {}))
    if f["pps_extension_4bits"]:
        _extension_data(io, f, "pps_extension_data_flag")
    io.rbsp_trailing_bits()


# ---- slice segment header (7.3.6) ------------------------------------------------------------------
def ref_pic_lists_modification(io: BitIO, f: dict, is_b: bool, n_l0: int, n_l1: int, num_pic_total_curr: int) -> None:
    bits = ceil_log2(num_pic_total_curr)
    if io.flag(f, "ref_pic_list_modification_flag_l0"):
        io.u_list(bits, f, "list_entry_l0", n_l0)
    if is_b and io.flag(f, "ref_pic_list_modification_flag_l1"):
        io.u_list(bits, f, "list_entry_l1", n_l1)


def pred_weight_table(io: BitIO, f: dict, is_b: bool, chroma_array_type: int, n_l0: int, n_l1: int) -> None:
    """Offsets are se(v) regardless of high_precision_offsets_enabled_flag (it only widens their value range)."""
    io.ue(f, "luma_log2_weight_denom")
    if chroma_array_type != 0:
        io.se(f, "delta_chroma_log2_weight_denom")
    for lst, n in (("l0", n_l0), ("l1", n_l1)) if is_b else (("l0", n_l0),):
        entries = _dict_list(io, f, f"pred_weight_{lst}", n)
        for e in entries:
            io.flag(e, "luma_weight_flag")
        for e in entries:
            if chroma_array_type != 0:
                io.flag(e, "chroma_weight_flag")
            else:
                e["chroma_weight_flag"] = 0
        for e in entries:
            if e["luma_weight_flag"]:
                io.se(e, "delta_luma_weight")
                io.se(e, "luma_offset")
            if e["chroma_weight_flag"]:
                if io.reading:
                    e["delta_chroma_weight"], e["delta_chroma_offset"] = [], []
                for j in range(2):
                    if io.reading:
                        e["delta_chroma_weight"].append(io.read_se())
                        e["delta_chroma_offset"].append(io.read_se())
                    else:
                        io.write_se(e["delta_chroma_weight"][j])
                        io.write_se(e["delta_chroma_offset"][j])


def _long_term_pics(io: BitIO, f: dict, sps_f: dict) -> int:
    """Long-term reference picture part of the slice header; returns the number of entries used by the current picture."""
    n_sps = sps_f["num_long_term_ref_pics_sps"]
    if n_sps > 0:
        io.ue(f, "num_long_term_sps")
    else:
        f["num_long_term_sps"] = 0
    io.ue(f, "num_long_term_pics")
    used = 0
    for i, lt in enumerate(_dict_list(io, f, "long_term_pics", f["num_long_term_sps"] + f["num_long_term_pics"])):
        if i < f["num_long_term_sps"]:
            if n_sps > 1:
                io.u(ceil_log2(n_sps), lt, "lt_idx_sps")
            else:
                lt["lt_idx_sps"] = 0
            used += sps_f["long_term_ref_pics_sps"][lt["lt_idx_sps"]]["used_by_curr_pic_lt_sps_flag"]
        else:
            io.u(sps_f["log2_max_pic_order_cnt_lsb_minus4"] + 4, lt, "poc_lsb_lt")
            used += io.flag(lt, "used_by_curr_pic_lt_flag")
        if io.flag(lt, "delta_poc_msb_present_flag"):
            io.ue(lt, "delta_poc_msb_cycle_lt")
    return used


def slice_segment_header(io: BitIO, f: dict, nal_unit_type: int, sps_f: dict, pps_f: dict) -> None:
    io.flag(f, "first_slice_segment_in_pic_flag")
    if N.HEVC_BLA_W_LP <= nal_unit_type <= 23:
        io.flag(f, "no_output_of_prior_pics_flag")
    io.ue(f, "slice_pic_parameter_set_id")
    if f["first_slice_segment_in_pic_flag"] or not pps_f["dependent_slice_segments_enabled_flag"]:
        f["dependent_slice_segment_flag"] = 0
    if not f["first_slice_segment_in_pic_flag"]:
        if pps_f["dependent_slice_segments_enabled_flag"]:
            io.flag(f, "dependent_slice_segment_flag")
        io.u(ceil_log2(_pic_size_in_ctbs(sps_f)), f, "slice_segment_address")
    if not f["dependent_slice_segment_flag"]:
        _independent_slice_fields(io, f, nal_unit_type, sps_f, pps_f)
    if pps_f["tiles_enabled_flag"] or pps_f["entropy_coding_sync_enabled_flag"]:
        io.ue(f, "num_entry_point_offsets")
        if f["num_entry_point_offsets"] > 0:
            io.ue(f, "offset_len_minus1")
            io.u_list(f["offset_len_minus1"] + 1, f, "entry_point_offset_minus1", f["num_entry_point_offsets"])
    if pps_f["slice_segment_header_extension_present_flag"]:
        io.ue(f, "slice_segment_header_extension_length")
        io.u_list(8, f, "slice_segment_header_extension_data_byte", f["slice_segment_header_extension_length"])
    io.byte_alignment()


def _independent_slice_fields(io: BitIO, f: dict, nal_unit_type: int, sps_f: dict, pps_f: dict) -> None:
    scc_sps, scc_pps, range_pps = sps_f.get("scc_ext", {}), pps_f.get("scc_ext", {}), pps_f.get("range_ext", {})
    chroma_array_type = _chroma_array_type(sps_f)
    io.u_list(1, f, "slice_reserved_flag", pps_f["num_extra_slice_header_bits"])
    io.ue(f, "slice_type")
    if pps_f["output_flag_present_flag"]:
        io.flag(f, "pic_output_flag")
    if sps_f["separate_colour_plane_flag"]:
        io.u(2, f, "colour_plane_id")
    num_pic_total_curr = 1 if scc_pps.get("pps_curr_pic_ref_enabled_flag") else 0
    idr = nal_unit_type in (N.HEVC_IDR_W_RADL, N.HEVC_IDR_N_LP)
    if idr or not sps_f["sps_temporal_mvp_enabled_flag"]:
        f["slice_temporal_mvp_enabled_flag"] = 0
    if not idr:
        io.u(sps_f["log2_max_pic_order_cnt_lsb_minus4"] + 4, f, "slice_pic_order_cnt_lsb")
        num_sets = sps_f["num_short_term_ref_pic_sets"]
        if not io.flag(f, "short_term_ref_pic_set_sps_flag"):
            st_ref_pic_set(io, f.setdefault("st_ref_pic_set", {}), num_sets, num_sets, sps_f["st_ref_pic_sets"])
            rps = f["st_ref_pic_set"]
        else:
            if num_sets > 1:
                io.u(ceil_log2(num_sets), f, "short_term_ref_pic_set_idx")
            else:
                f["short_term_ref_pic_set_idx"] = 0
            rps = sps_f["st_ref_pic_sets"][f["short_term_ref_pic_set_idx"]]
        num_pic_total_curr += sum(rps["used_by_curr_pic_s0"]) + sum(rps["used_by_curr_pic_s1"])
        if sps_f["long_term_ref_pics_present_flag"]:
            num_pic_total_curr += _long_term_pics(io, f, sps_f)
        if sps_f["sps_temporal_mvp_enabled_flag"]:
            io.flag(f, "slice_temporal_mvp_enabled_flag")
    f["_num_pic_total_curr"] = num_pic_total_curr
    if sps_f["sample_adaptive_offset_enabled_flag"]:
        io.flag(f, "slice_sao_luma_flag")
        if chroma_array_type != 0:
            io.flag(f, "slice_sao_chroma_flag")
        else:
            f["slice_sao_chroma_flag"] = 0
    else:
        f["slice_sao_luma_flag"] = f["slice_sao_chroma_flag"] = 0
    st = f["slice_type"]
    is_b = st == SLICE_B
    n_l0 = pps_f["num_ref_idx_l0_default_active_minus1"] + 1
    n_l1 = pps_f["num_ref_idx_l1_default_active_minus1"] + 1
    if st in (SLICE_P, SLICE_B):
        if io.flag(f, "num_ref_idx_active_override_flag"):
            io.ue(f, "num_ref_idx_l0_active_minus1")
            n_l0 = f["num_ref_idx_l0_active_minus1"] + 1
            if is_b:
                io.ue(f, "num_ref_idx_l1_active_minus1")
                n_l1 = f["num_ref_idx_l1_active_minus1"] + 1
        if pps_f["lists_modification_present_flag"] and num_pic_total_curr > 1:
            ref_pic_lists_modification(io, f, is_b, n_l0, n_l1, num_pic_total_curr)
        if is_b:
            io.flag(f, "mvd_l1_zero_flag")
        if pps_f["cabac_init_present_flag"]:
            io.flag(f, "cabac_init_flag")
        if f["slice_temporal_mvp_enabled_flag"]:
            if is_b:
                io.flag(f, "collocated_from_l0_flag")
            else:
                f["collocated_from_l0_flag"] = 1
            if (n_l0 if f["collocated_from_l0_flag"] else n_l1) > 1:
                io.ue(f, "collocated_ref_idx")
        if (pps_f["weighted_pred_flag"] and st == SLICE_P) or (pps_f["weighted_bipred_flag"] and is_b):
            if scc_pps.get("pps_curr_pic_ref_enabled_flag"):
                raise BitError("pred_weight_table with current-picture referencing not supported")
            pred_weight_table(io, f, is_b, chroma_array_type, n_l0, n_l1)
        io.ue(f, "five_minus_max_num_merge_cand")
        if scc_sps.get("motion_vector_resolution_control_idc") == 2:
            io.flag(f, "use_integer_mv_flag")
    f["_num_ref_idx_l0_active"] = n_l0 if st in (SLICE_P, SLICE_B) else 0
    f["_num_ref_idx_l1_active"] = n_l1 if is_b else 0
    io.se(f, "slice_qp_delta")
    if pps_f["pps_slice_chroma_qp_offsets_present_flag"]:
        io.se(f, "slice_cb_qp_offset")
        io.se(f, "slice_cr_qp_offset")
    if scc_pps.get("pps_slice_act_qp_offsets_present_flag"):
        io.se(f, "slice_act_y_qp_offset")
        io.se(f, "slice_act_cb_qp_offset")
        io.se(f, "slice_act_cr_qp_offset")
    if range_pps.get("chroma_qp_offset_list_enabled_flag"):
        io.flag(f, "cu_chroma_qp_offset_enabled_flag")
    if pps_f["deblocking_filter_override_enabled_flag"]:
        io.flag(f, "deblocking_filter_override_flag")
    else:
        f["deblocking_filter_override_flag"] = 0
    if f["deblocking_filter_override_flag"]:
        if not io.flag(f, "slice_deblocking_filter_disabled_flag"):
            io.se(f, "slice_beta_offset_div2")
            io.se(f, "slice_tc_offset_div2")
    else:
        f["slice_deblocking_filter_disabled_flag"] = pps_f["pps_deblocking_filter_disabled_flag"]
    if pps_f["pps_loop_filter_across_slices_enabled_flag"] and (
            f["slice_sao_luma_flag"] or f["slice_sao_chroma_flag"] or not f["slice_deblocking_filter_disabled_flag"]):
        io.flag(f, "slice_loop_filter_across_slices_enabled_flag")


# ---- NAL-level API ---------------------------------------------------------------------------------
def _parse_param_set(nal_bytes: bytes, expected_type: int) -> tuple[BitIO, dict]:
    io = BitIO(N.remove_epb(nal_bytes))
    f: dict = {}
    _nal_header(io, f)
    if f["nal_unit_type"] != expected_type:
        raise BitError(f"expected nal_unit_type {expected_type}, got {f['nal_unit_type']}")
    return io, f


def _write_nal(f: dict, default_type: int, body, tail: bytes = b"") -> bytes:
    f.setdefault("nal_unit_type", default_type)
    io = BitIO()
    _nal_header(io, f)
    body(io)
    return N.insert_epb(io.to_bytes() + tail)


def parse_vps_nal(nal_bytes: bytes) -> dict:
    io, f = _parse_param_set(nal_bytes, N.HEVC_VPS)
    vps(io, f)
    return f


def write_vps_nal(f: dict) -> bytes:
    return _write_nal(f, N.HEVC_VPS, lambda io: vps(io, f))


def parse_sps_nal(nal_bytes: bytes) -> dict:
    io, f = _parse_param_set(nal_bytes, N.HEVC_SPS)
    sps(io, f)
    _derive_sps(f)
    return f


def write_sps_nal(f: dict) -> bytes:
    return _write_nal(f, N.HEVC_SPS, lambda io: sps(io, f))


def parse_pps_nal(nal_bytes: bytes, sps_f: Optional[dict] = None) -> dict:
    io, f = _parse_param_set(nal_bytes, N.HEVC_PPS)
    pps(io, f, sps_f)
    return f


def write_pps_nal(f: dict, sps_f: Optional[dict] = None) -> bytes:
    return _write_nal(f, N.HEVC_PPS, lambda io: pps(io, f, sps_f))


def _derive_sps(f: dict) -> None:
    sub_w, sub_h = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}[f["chroma_format_idc"]]
    if f["separate_colour_plane_flag"]:
        sub_w = sub_h = 1
    f["_width"] = f["pic_width_in_luma_samples"] - sub_w * (f.get("conf_win_left_offset", 0) + f.get("conf_win_right_offset", 0))
    f["_height"] = f["pic_height_in_luma_samples"] - sub_h * (f.get("conf_win_top_offset", 0) + f.get("conf_win_bottom_offset", 0))
    f["_min_cb_log2"] = f["log2_min_luma_coding_block_size_minus3"] + 3
    f["_ctb_log2"] = f["_min_cb_log2"] + f["log2_diff_max_min_luma_coding_block_size"]
    f["_chroma_array_type"] = _chroma_array_type(f)
    vui = f.get("vui", {})
    if vui.get("vui_timing_info_present_flag"):
        f["_fps"] = (vui["vui_time_scale"], vui["vui_num_units_in_tick"])
    hrd = vui.get("hrd", {})
    for key in ("nal", "vcl"):
        cpb = hrd.get("sub_layers", [{}])[0].get(key, {}).get("cpb")
        if cpb:
            f[f"_{key}_hrd_bit_rate"] = (cpb[0]["bit_rate_value_minus1"] + 1) << (6 + hrd["bit_rate_scale"])
            f[f"_{key}_hrd_cpb_size"] = (cpb[0]["cpb_size_value_minus1"] + 1) << (4 + hrd["cpb_size_scale"])


def slice_pps_id(nal_bytes: bytes) -> int:
    """slice_pic_parameter_set_id of a VCL NAL (so the right PPS/SPS can be picked before parsing the header)."""
    io = BitIO(N.remove_epb(nal_bytes[:16]))
    f: dict = {}
    _nal_header(io, f)
    io.read_bits(1 + (1 if N.HEVC_BLA_W_LP <= f["nal_unit_type"] <= 23 else 0))
    return io.read_ue()


def parse_slice_nal(nal_bytes: bytes, sps_f: dict, pps_f: dict) -> tuple[dict, bytes, int]:
    """Returns (header_fields, slice_segment_data_bytes, header_bit_length). HEVC slice data always starts at the byte
    boundary after byte_alignment(); header_bit_length counts the bits after the 2-byte NAL header."""
    io = BitIO(N.remove_epb(nal_bytes))
    f: dict = {}
    _nal_header(io, f)
    if f["nal_unit_type"] > 31:
        raise BitError(f"nal_unit_type {f['nal_unit_type']} is not a VCL NAL")
    slice_segment_header(io, f, f["nal_unit_type"], sps_f, pps_f)
    return f, io.data[io.pos // 8:], io.pos - 16


def write_slice_nal(f: dict, data: bytes, header_bits: int, sps_f: dict, pps_f: dict) -> bytes:
    """New NAL header + slice_segment_header() (incl. byte_alignment) followed by `data`, EPB re-inserted.
    `header_bits` exists for signature parity with h264.write_slice_nal (where CAVLC data needs bit-shifting); HEVC
    slice data is always byte aligned so its value is irrelevant here."""
    body = lambda io: slice_segment_header(io, f, f["nal_unit_type"], sps_f, pps_f)  # noqa: E731
    return _write_nal(f, f["nal_unit_type"], body, data)
