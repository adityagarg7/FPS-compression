"""H.264 (ISO/IEC 14496-10) SPS / PPS / slice header syntax, symmetric parse & write (see bits.py)."""
from __future__ import annotations

from typing import Optional

from .bits import BitIO, BitError
from . import nal as N

HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}


# ---- helpers -------------------------------------------------------------------------------------
def _scaling_list(io: BitIO, f: dict, name: str, size: int) -> None:
    """scaling_list(): stored as the list of delta_scale values actually coded."""
    if io.reading:
        deltas = []
        last = 8
        nxt = 8
        for j in range(size):
            if nxt != 0:
                d = io.read_se()
                deltas.append(d)
                nxt = (last + d + 256) % 256
                if nxt == 0:
                    break
            last = nxt if nxt != 0 else last
        f[name] = deltas
    else:
        for d in f[name]:
            io.write_se(d)


def hrd_parameters(io: BitIO, f: dict) -> None:
    io.ue(f, "cpb_cnt_minus1")
    io.u(4, f, "bit_rate_scale")
    io.u(4, f, "cpb_size_scale")
    n = f["cpb_cnt_minus1"] + 1
    if io.reading:
        f["bit_rate_value_minus1"], f["cpb_size_value_minus1"], f["cbr_flag"] = [], [], []
        for _ in range(n):
            f["bit_rate_value_minus1"].append(io.read_ue())
            f["cpb_size_value_minus1"].append(io.read_ue())
            f["cbr_flag"].append(io.read_bits(1))
    else:
        for i in range(n):
            io.write_ue(f["bit_rate_value_minus1"][i])
            io.write_ue(f["cpb_size_value_minus1"][i])
            io.write_bits(f["cbr_flag"][i], 1)
    io.u(5, f, "initial_cpb_removal_delay_length_minus1")
    io.u(5, f, "cpb_removal_delay_length_minus1")
    io.u(5, f, "dpb_output_delay_length_minus1")
    io.u(5, f, "time_offset_length")


def vui_parameters(io: BitIO, f: dict) -> None:
    if io.flag(f, "aspect_ratio_info_present_flag"):
        io.u(8, f, "aspect_ratio_idc")
        if f["aspect_ratio_idc"] == 255:
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
            io.u(8, f, "matrix_coefficients")
    if io.flag(f, "chroma_loc_info_present_flag"):
        io.ue(f, "chroma_sample_loc_type_top_field")
        io.ue(f, "chroma_sample_loc_type_bottom_field")
    if io.flag(f, "timing_info_present_flag"):
        io.u(32, f, "num_units_in_tick")
        io.u(32, f, "time_scale")
        io.flag(f, "fixed_frame_rate_flag")
    if io.flag(f, "nal_hrd_parameters_present_flag"):
        f.setdefault("nal_hrd", {})
        hrd_parameters(io, f["nal_hrd"])
    if io.flag(f, "vcl_hrd_parameters_present_flag"):
        f.setdefault("vcl_hrd", {})
        hrd_parameters(io, f["vcl_hrd"])
    if f["nal_hrd_parameters_present_flag"] or f["vcl_hrd_parameters_present_flag"]:
        io.flag(f, "low_delay_hrd_flag")
    io.flag(f, "pic_struct_present_flag")
    if io.flag(f, "bitstream_restriction_flag"):
        io.flag(f, "motion_vectors_over_pic_boundaries_flag")
        io.ue(f, "max_bytes_per_pic_denom")
        io.ue(f, "max_bits_per_mb_denom")
        io.ue(f, "log2_max_mv_length_horizontal")
        io.ue(f, "log2_max_mv_length_vertical")
        io.ue(f, "max_num_reorder_frames")
        io.ue(f, "max_dec_frame_buffering")


def sps(io: BitIO, f: dict) -> None:
    """seq_parameter_set_data() (after the NAL header byte)."""
    io.u(8, f, "profile_idc")
    for i in range(6):
        io.flag(f, f"constraint_set{i}_flag")
    io.u(2, f, "reserved_zero_2bits")
    io.u(8, f, "level_idc")
    io.ue(f, "seq_parameter_set_id")
    f.setdefault("chroma_format_idc", 1)
    f.setdefault("separate_colour_plane_flag", 0)
    f.setdefault("bit_depth_luma_minus8", 0)
    f.setdefault("bit_depth_chroma_minus8", 0)
    if f["profile_idc"] in HIGH_PROFILES:
        io.ue(f, "chroma_format_idc")
        if f["chroma_format_idc"] == 3:
            io.flag(f, "separate_colour_plane_flag")
        io.ue(f, "bit_depth_luma_minus8")
        io.ue(f, "bit_depth_chroma_minus8")
        io.flag(f, "qpprime_y_zero_transform_bypass_flag")
        if io.flag(f, "seq_scaling_matrix_present_flag"):
            n = 8 if f["chroma_format_idc"] != 3 else 12
            io.u_list(1, f, "seq_scaling_list_present_flag", n) if io.reading else None
            if not io.reading:
                for i in range(n):
                    io.write_bits(f["seq_scaling_list_present_flag"][i], 1)
                    if f["seq_scaling_list_present_flag"][i]:
                        _scaling_list(io, f, f"seq_scaling_list_{i}", 16 if i < 6 else 64)
            else:
                # re-read properly interleaved (flags and lists interleave in the bitstream)
                io.pos -= n
                f["seq_scaling_list_present_flag"] = []
                for i in range(n):
                    pf = io.read_bits(1)
                    f["seq_scaling_list_present_flag"].append(pf)
                    if pf:
                        _scaling_list(io, f, f"seq_scaling_list_{i}", 16 if i < 6 else 64)
    io.ue(f, "log2_max_frame_num_minus4")
    io.ue(f, "pic_order_cnt_type")
    if f["pic_order_cnt_type"] == 0:
        io.ue(f, "log2_max_pic_order_cnt_lsb_minus4")
    elif f["pic_order_cnt_type"] == 1:
        io.flag(f, "delta_pic_order_always_zero_flag")
        io.se(f, "offset_for_non_ref_pic")
        io.se(f, "offset_for_top_to_bottom_field")
        io.ue(f, "num_ref_frames_in_pic_order_cnt_cycle")
        if io.reading:
            f["offset_for_ref_frame"] = [io.read_se() for _ in range(f["num_ref_frames_in_pic_order_cnt_cycle"])]
        else:
            for v in f["offset_for_ref_frame"]:
                io.write_se(v)
    io.ue(f, "max_num_ref_frames")
    io.flag(f, "gaps_in_frame_num_value_allowed_flag")
    io.ue(f, "pic_width_in_mbs_minus1")
    io.ue(f, "pic_height_in_map_units_minus1")
    io.flag(f, "frame_mbs_only_flag")
    if not f["frame_mbs_only_flag"]:
        io.flag(f, "mb_adaptive_frame_field_flag")
    io.flag(f, "direct_8x8_inference_flag")
    if io.flag(f, "frame_cropping_flag"):
        io.ue(f, "frame_crop_left_offset")
        io.ue(f, "frame_crop_right_offset")
        io.ue(f, "frame_crop_top_offset")
        io.ue(f, "frame_crop_bottom_offset")
    if io.flag(f, "vui_parameters_present_flag"):
        f.setdefault("vui", {})
        vui_parameters(io, f["vui"])
    io.rbsp_trailing_bits()


def pps(io: BitIO, f: dict, sps_f: Optional[dict] = None) -> None:
    io.ue(f, "pic_parameter_set_id")
    io.ue(f, "seq_parameter_set_id")
    io.flag(f, "entropy_coding_mode_flag")
    io.flag(f, "bottom_field_pic_order_in_frame_present_flag")
    io.ue(f, "num_slice_groups_minus1")
    if f["num_slice_groups_minus1"] > 0:
        io.ue(f, "slice_group_map_type")
        t = f["slice_group_map_type"]
        n = f["num_slice_groups_minus1"]
        if t == 0:
            if io.reading:
                f["run_length_minus1"] = [io.read_ue() for _ in range(n + 1)]
            else:
                for v in f["run_length_minus1"]:
                    io.write_ue(v)
        elif t == 2:
            if io.reading:
                f["top_left"], f["bottom_right"] = [], []
                for _ in range(n):
                    f["top_left"].append(io.read_ue()); f["bottom_right"].append(io.read_ue())
            else:
                for a, b in zip(f["top_left"], f["bottom_right"]):
                    io.write_ue(a); io.write_ue(b)
        elif t in (3, 4, 5):
            io.flag(f, "slice_group_change_direction_flag")
            io.ue(f, "slice_group_change_rate_minus1")
        elif t == 6:
            io.ue(f, "pic_size_in_map_units_minus1")
            bits = (n).bit_length()
            io.u_list(bits, f, "slice_group_id", f["pic_size_in_map_units_minus1"] + 1)
    io.ue(f, "num_ref_idx_l0_default_active_minus1")
    io.ue(f, "num_ref_idx_l1_default_active_minus1")
    io.flag(f, "weighted_pred_flag")
    io.u(2, f, "weighted_bipred_idc")
    io.se(f, "pic_init_qp_minus26")
    io.se(f, "pic_init_qs_minus26")
    io.se(f, "chroma_qp_index_offset")
    io.flag(f, "deblocking_filter_control_present_flag")
    io.flag(f, "constrained_intra_pred_flag")
    io.flag(f, "redundant_pic_cnt_present_flag")
    has_ext = io.more_rbsp_data() if io.reading else f.get("transform_8x8_mode_flag") is not None
    f["_has_extension"] = bool(has_ext)
    if has_ext:
        io.flag(f, "transform_8x8_mode_flag")
        if io.flag(f, "pic_scaling_matrix_present_flag"):
            cfi = (sps_f or {}).get("chroma_format_idc", 1)
            n = 6 + (2 if cfi != 3 else 6) * f["transform_8x8_mode_flag"]
            if io.reading:
                f["pic_scaling_list_present_flag"] = []
                for i in range(n):
                    pf = io.read_bits(1)
                    f["pic_scaling_list_present_flag"].append(pf)
                    if pf:
                        _scaling_list(io, f, f"pic_scaling_list_{i}", 16 if i < 6 else 64)
            else:
                for i in range(n):
                    io.write_bits(f["pic_scaling_list_present_flag"][i], 1)
                    if f["pic_scaling_list_present_flag"][i]:
                        _scaling_list(io, f, f"pic_scaling_list_{i}", 16 if i < 6 else 64)
        io.se(f, "second_chroma_qp_index_offset")
    io.rbsp_trailing_bits()


# ---- slice header ------------------------------------------------------------------------------
SLICE_P, SLICE_B, SLICE_I, SLICE_SP, SLICE_SI = 0, 1, 2, 3, 4


def slice_type_base(t: int) -> int:
    return t % 5


def ref_pic_list_modification(io: BitIO, f: dict, is_b: bool) -> None:
    for lst in (["l0", "l1"] if is_b else ["l0"]):
        key = f"ref_pic_list_modification_flag_{lst}"
        if io.flag(f, key):
            ops_key = f"ref_pic_list_modification_{lst}"
            if io.reading:
                ops = []
                while True:
                    idc = io.read_ue()
                    if idc == 3:
                        break
                    if idc in (0, 1):
                        ops.append((idc, io.read_ue()))
                    elif idc == 2:
                        ops.append((idc, io.read_ue()))
                    else:
                        raise BitError(f"bad modification_of_pic_nums_idc {idc}")
                f[ops_key] = ops
            else:
                for idc, v in f[ops_key]:
                    io.write_ue(idc); io.write_ue(v)
                io.write_ue(3)


def pred_weight_table(io: BitIO, f: dict, is_b: bool, chroma_array_type: int, n_l0: int, n_l1: int) -> None:
    io.ue(f, "luma_log2_weight_denom")
    if chroma_array_type != 0:
        io.ue(f, "chroma_log2_weight_denom")
    for lst, n in (("l0", n_l0), ("l1", n_l1)) if is_b else (("l0", n_l0),):
        key = f"pred_weight_{lst}"
        if io.reading:
            entries = []
            for _ in range(n):
                e: dict = {}
                e["luma_weight_flag"] = io.read_bits(1)
                if e["luma_weight_flag"]:
                    e["luma_weight"] = io.read_se(); e["luma_offset"] = io.read_se()
                if chroma_array_type != 0:
                    e["chroma_weight_flag"] = io.read_bits(1)
                    if e["chroma_weight_flag"]:
                        e["chroma"] = [(io.read_se(), io.read_se()) for _ in range(2)]
                entries.append(e)
            f[key] = entries
        else:
            for e in f[key][:n]:
                io.write_bits(e["luma_weight_flag"], 1)
                if e["luma_weight_flag"]:
                    io.write_se(e["luma_weight"]); io.write_se(e["luma_offset"])
                if chroma_array_type != 0:
                    io.write_bits(e["chroma_weight_flag"], 1)
                    if e["chroma_weight_flag"]:
                        for w, o in e["chroma"]:
                            io.write_se(w); io.write_se(o)


def dec_ref_pic_marking(io: BitIO, f: dict, idr: bool) -> None:
    if idr:
        io.flag(f, "no_output_of_prior_pics_flag")
        io.flag(f, "long_term_reference_flag")
    else:
        if io.flag(f, "adaptive_ref_pic_marking_mode_flag"):
            if io.reading:
                ops = []
                while True:
                    op = io.read_ue()
                    if op == 0:
                        break
                    args = []
                    if op in (1, 3):
                        args.append(io.read_ue())
                    if op == 2:
                        args.append(io.read_ue())
                    if op in (3, 6):
                        args.append(io.read_ue())
                    if op == 4:
                        args.append(io.read_ue())
                    ops.append((op, args))
                f["mmco"] = ops
            else:
                for op, args in f["mmco"]:
                    io.write_ue(op)
                    for a in args:
                        io.write_ue(a)
                io.write_ue(0)


def slice_header(io: BitIO, f: dict, nal_unit_type: int, nal_ref_idc: int, sps_f: dict, pps_f: dict) -> None:
    idr = nal_unit_type == 5
    io.ue(f, "first_mb_in_slice")
    io.ue(f, "slice_type")
    io.ue(f, "pic_parameter_set_id")
    st = slice_type_base(f["slice_type"])
    is_b, is_p = st == SLICE_B, st in (SLICE_P, SLICE_SP)
    if sps_f.get("separate_colour_plane_flag"):
        io.u(2, f, "colour_plane_id")
    io.u(sps_f["log2_max_frame_num_minus4"] + 4, f, "frame_num")
    f.setdefault("field_pic_flag", 0)
    if not sps_f["frame_mbs_only_flag"]:
        if io.flag(f, "field_pic_flag"):
            io.flag(f, "bottom_field_flag")
    if idr:
        io.ue(f, "idr_pic_id")
    if sps_f["pic_order_cnt_type"] == 0:
        io.u(sps_f["log2_max_pic_order_cnt_lsb_minus4"] + 4, f, "pic_order_cnt_lsb")
        if pps_f["bottom_field_pic_order_in_frame_present_flag"] and not f["field_pic_flag"]:
            io.se(f, "delta_pic_order_cnt_bottom")
    if sps_f["pic_order_cnt_type"] == 1 and not sps_f.get("delta_pic_order_always_zero_flag"):
        io.se(f, "delta_pic_order_cnt_0")
        if pps_f["bottom_field_pic_order_in_frame_present_flag"] and not f["field_pic_flag"]:
            io.se(f, "delta_pic_order_cnt_1")
    if pps_f["redundant_pic_cnt_present_flag"]:
        io.ue(f, "redundant_pic_cnt")
    if is_b:
        io.flag(f, "direct_spatial_mv_pred_flag")
    n_l0 = pps_f["num_ref_idx_l0_default_active_minus1"] + 1
    n_l1 = pps_f["num_ref_idx_l1_default_active_minus1"] + 1
    if is_p or is_b:
        if io.flag(f, "num_ref_idx_active_override_flag"):
            io.ue(f, "num_ref_idx_l0_active_minus1")
            n_l0 = f["num_ref_idx_l0_active_minus1"] + 1
            if is_b:
                io.ue(f, "num_ref_idx_l1_active_minus1")
                n_l1 = f["num_ref_idx_l1_active_minus1"] + 1
    f["_num_ref_idx_l0_active"] = n_l0 if (is_p or is_b) else 0
    f["_num_ref_idx_l1_active"] = n_l1 if is_b else 0
    if nal_unit_type in (20, 21):
        raise BitError("MVC slices not supported")
    if is_p or is_b:
        ref_pic_list_modification(io, f, is_b)
    chroma_array_type = 0 if sps_f.get("separate_colour_plane_flag") else sps_f.get("chroma_format_idc", 1)
    if (pps_f["weighted_pred_flag"] and is_p) or (pps_f["weighted_bipred_idc"] == 1 and is_b):
        pred_weight_table(io, f, is_b, chroma_array_type, n_l0, n_l1)
    if nal_ref_idc != 0:
        dec_ref_pic_marking(io, f, idr)
    if pps_f["entropy_coding_mode_flag"] and st not in (SLICE_I, SLICE_SI):
        io.ue(f, "cabac_init_idc")
    io.se(f, "slice_qp_delta")
    if st in (SLICE_SP, SLICE_SI):
        if st == SLICE_SP:
            io.flag(f, "sp_for_switch_flag")
        io.se(f, "slice_qs_delta")
    if pps_f["deblocking_filter_control_present_flag"]:
        io.ue(f, "disable_deblocking_filter_idc")
        if f["disable_deblocking_filter_idc"] != 1:
            io.se(f, "slice_alpha_c0_offset_div2")
            io.se(f, "slice_beta_offset_div2")
    if pps_f["num_slice_groups_minus1"] > 0 and pps_f.get("slice_group_map_type", 0) in (3, 4, 5):
        pic_size = (sps_f["pic_width_in_mbs_minus1"] + 1) * (sps_f["pic_height_in_map_units_minus1"] + 1)
        rate = pps_f["slice_group_change_rate_minus1"] + 1
        from .bits import ceil_log2
        bits = ceil_log2(-(-pic_size // rate) + 1)
        io.u(bits, f, "slice_group_change_cycle")


# ---- NAL-level API -------------------------------------------------------------------------------
def parse_sps_nal(nal_bytes: bytes) -> dict:
    rbsp = N.remove_epb(nal_bytes)
    io = BitIO(rbsp[1:])
    f: dict = {"nal_ref_idc": (nal_bytes[0] >> 5) & 3}
    sps(io, f)
    _derive_sps(f)
    return f


def write_sps_nal(f: dict) -> bytes:
    io = BitIO()
    sps(io, f)
    hdr = bytes([(f.get("nal_ref_idc", 3) << 5) | N.H264_SPS])
    return N.insert_epb(hdr + io.to_bytes())


def parse_pps_nal(nal_bytes: bytes, sps_f: Optional[dict] = None) -> dict:
    rbsp = N.remove_epb(nal_bytes)
    io = BitIO(rbsp[1:])
    f: dict = {"nal_ref_idc": (nal_bytes[0] >> 5) & 3}
    pps(io, f, sps_f)
    return f


def write_pps_nal(f: dict, sps_f: Optional[dict] = None) -> bytes:
    io = BitIO()
    pps(io, f, sps_f)
    hdr = bytes([(f.get("nal_ref_idc", 3) << 5) | N.H264_PPS])
    return N.insert_epb(hdr + io.to_bytes())


def _derive_sps(f: dict) -> None:
    f["_width"] = (f["pic_width_in_mbs_minus1"] + 1) * 16
    f["_height"] = (2 - f["frame_mbs_only_flag"]) * (f["pic_height_in_map_units_minus1"] + 1) * 16
    vui = f.get("vui", {})
    for key in ("nal_hrd", "vcl_hrd"):
        h = vui.get(key)
        if h:
            f[f"_{key}_bit_rate"] = (h["bit_rate_value_minus1"][0] + 1) << (6 + h["bit_rate_scale"])
            f[f"_{key}_cpb_size"] = (h["cpb_size_value_minus1"][0] + 1) << (4 + h["cpb_size_scale"])
    if vui.get("timing_info_present_flag"):
        f["_fps"] = (vui["time_scale"], vui["num_units_in_tick"] * 2)


def parse_slice_nal(nal_bytes: bytes, sps_f: dict, pps_f: dict) -> tuple[dict, bytes, int]:
    """Returns (header_fields, slice_data_bytes, header_bit_length). For CABAC streams slice data starts at the next
    byte boundary after the header (cabac_alignment_one_bits consumed); for CAVLC the returned data starts at the
    byte containing the first slice_data bit and header_bit_length tells the bit offset within it."""
    rbsp = N.remove_epb(nal_bytes)
    nut = nal_bytes[0] & 0x1F
    nri = (nal_bytes[0] >> 5) & 3
    io = BitIO(rbsp[1:])
    f: dict = {"nal_unit_type": nut, "nal_ref_idc": nri}
    slice_header(io, f, nut, nri, sps_f, pps_f)
    hdr_bits = io.pos
    if pps_f["entropy_coding_mode_flag"]:
        io.align_with(1)
        data = rbsp[1 + io.pos // 8:]
    else:
        data = rbsp[1 + hdr_bits // 8:]
    return f, data, hdr_bits


def write_slice_nal(f: dict, data: bytes, hdr_bits_in_data: int, sps_f: dict, pps_f: dict) -> bytes:
    nut = f["nal_unit_type"]
    nri = f["nal_ref_idc"]
    io = BitIO()
    slice_header(io, f, nut, nri, sps_f, pps_f)
    hdr = bytes([(nri << 5) | nut])
    if pps_f["entropy_coding_mode_flag"]:
        io.align_with(1)
        body = io.to_bytes() + data
    else:
        # CAVLC: slice data is not byte aligned; bit-shift it after the new header
        nbits_hdr = io.bitpos
        skip = hdr_bits_in_data % 8
        total_bits = len(data) * 8 - skip
        val = int.from_bytes(data, "big")
        val &= (1 << total_bits) - 1  # drop the old header bits in the first byte
        hb = int.from_bytes(io.to_bytes(), "big") >> ((8 - nbits_hdr % 8) % 8)
        combined = (hb << total_bits) | val
        nb = nbits_hdr + total_bits
        combined <<= (8 - nb % 8) % 8
        body = combined.to_bytes((nb + 7) // 8, "big")
        # the rbsp_stop_one_bit is the last 1 bit: whole zero bytes created by the shift must go (7.4.1: last byte != 0)
        body = body.rstrip(b"\x00") or body
    return N.insert_epb(hdr + body)
