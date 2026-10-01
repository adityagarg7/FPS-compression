"""Parameter-set transplant + slice-header rewrite.

After encoding with x264/x265 (configured so every DECODE-AFFECTING tool matches the camera stream), the encoder's
VPS/SPS/PPS are replaced by the camera's own (VUI timing patched) and every slice header is re-serialised so that it
is decodable under the camera's parameter sets. Only syntax-level fields change; the entropy-coded slice data is
copied verbatim. The caller verifies losslessness by decoding both streams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import h264
from . import nal as N
from .nal import AccessUnit


class RewriteUnsafe(RuntimeError):
    pass


@dataclass
class TransplantResult:
    aus: list[AccessUnit]
    param_sets: dict[str, list[bytes]]
    rewritten_fields: list[str] = field(default_factory=list)
    residual_differences: list[str] = field(default_factory=list)
    applied: bool = False


@dataclass
class SliceConventions:
    """Slice-header habits of the camera, measured on the source stream, reproduced in the output."""
    idr_pic_id_pattern: str = "increment"      # 'increment' | 'alternate' | 'zero'
    nal_ref_idc_idr: int = 1
    nal_ref_idc_ref: int = 1
    nal_ref_idc_ps: int = 1
    poc_lsb_at_idr: int = 0


def measure_h264_conventions(samples: list[bytes], sps_f: dict, pps_f: dict, ps: Optional[dict[str, list[bytes]]] = None) -> SliceConventions:
    conv = SliceConventions()
    if ps and ps.get("sps"):
        conv.nal_ref_idc_ps = (ps["sps"][0][0] >> 5) & 3
    idr_ids: list[int] = []
    for smp in samples:
        for n in N.split_length_prefixed(smp):
            t = N.h264_nal_type(n)
            if t in (N.H264_SPS, N.H264_PPS):
                conv.nal_ref_idc_ps = (n[0] >> 5) & 3
            if not N.is_vcl(n, "h264"):
                continue
            f, _d, _b = h264.parse_slice_nal(n, sps_f, pps_f)
            if f["first_mb_in_slice"] != 0:
                continue
            if t == N.H264_IDR:
                conv.nal_ref_idc_idr = f["nal_ref_idc"]
                idr_ids.append(f["idr_pic_id"])
                conv.poc_lsb_at_idr = f.get("pic_order_cnt_lsb", 0)
            elif f["nal_ref_idc"]:
                conv.nal_ref_idc_ref = f["nal_ref_idc"]
    if len(idr_ids) >= 3:
        if all(b == (a + 1) % 65536 for a, b in zip(idr_ids, idr_ids[1:])):
            conv.idr_pic_id_pattern = "increment"
        elif all(a != b for a, b in zip(idr_ids, idr_ids[1:])) and len(set(idr_ids)) == 2:
            conv.idr_pic_id_pattern = "alternate"
        elif all(x == 0 for x in idr_ids):
            conv.idr_pic_id_pattern = "zero"
    return conv


# ---- H.264 ---------------------------------------------------------------------------------------
_H264_SPS_MUST_MATCH = [
    "profile_idc", "chroma_format_idc", "separate_colour_plane_flag", "bit_depth_luma_minus8", "bit_depth_chroma_minus8",
    "qpprime_y_zero_transform_bypass_flag", "seq_scaling_matrix_present_flag", "pic_width_in_mbs_minus1", "direct_8x8_inference_flag",
    "pic_height_in_map_units_minus1", "frame_mbs_only_flag", "mb_adaptive_frame_field_flag", "frame_cropping_flag",
    "frame_crop_left_offset", "frame_crop_right_offset", "frame_crop_top_offset", "frame_crop_bottom_offset",
]
_H264_PPS_MUST_MATCH = [
    "entropy_coding_mode_flag", "num_slice_groups_minus1", "weighted_pred_flag", "weighted_bipred_idc",
    "chroma_qp_index_offset", "constrained_intra_pred_flag", "transform_8x8_mode_flag", "pic_scaling_matrix_present_flag",
    "second_chroma_qp_index_offset",
]


def _check_match(enc: dict, tgt: dict, keys: list[str], what: str, skip_if_equal_default: Optional[dict] = None) -> None:
    for k in keys:
        a, b = enc.get(k), tgt.get(k)
        if a is None and b is None:
            continue
        if k == "second_chroma_qp_index_offset":
            # inferred equal to chroma_qp_index_offset when the PPS extension is absent (7.4.2.2)
            a = a if a is not None else enc.get("chroma_qp_index_offset", 0)
            b = b if b is not None else tgt.get("chroma_qp_index_offset", 0)
        # a missing PPS extension means transform_8x8_mode_flag == 0 etc.
        a = a if a is not None else 0
        b = b if b is not None else 0
        if a != b:
            raise RewriteUnsafe(f"{what}.{k}: encoder {a} != target {b} (decode-affecting)")


def _h264_poc_sequence(aus: list[AccessUnit], sps_f: dict, pps_f: dict) -> list[tuple[dict, list[tuple[bytes, dict, bytes, int]]]]:
    """Parse all slice headers (per AU) and derive the full PicOrderCnt of every picture under the encoder SPS."""
    out = []
    poc_type = sps_f["pic_order_cnt_type"]
    max_frame_num = 1 << (sps_f["log2_max_frame_num_minus4"] + 4)
    max_poc_lsb = 1 << (sps_f.get("log2_max_pic_order_cnt_lsb_minus4", 0) + 4)
    prev_msb = prev_lsb = 0
    prev_frame_num = 0
    prev_frame_num_offset = 0
    prev_ref_was_mmco5 = False
    for au in aus:
        slices = []
        for n in au.nals:
            if N.is_vcl(n, "h264"):
                f, data, hb = h264.parse_slice_nal(n, sps_f, pps_f)
                slices.append((n, f, data, hb))
        if not slices:
            raise RewriteUnsafe("access unit without VCL NAL")
        f0 = slices[0][1]
        idr = f0["nal_unit_type"] == N.H264_IDR
        is_ref = f0["nal_ref_idc"] != 0
        if poc_type == 0:
            lsb = f0["pic_order_cnt_lsb"]
            if idr:
                prev_msb = prev_lsb = 0
            if lsb < prev_lsb and (prev_lsb - lsb) >= max_poc_lsb // 2:
                msb = prev_msb + max_poc_lsb
            elif lsb > prev_lsb and (lsb - prev_lsb) > max_poc_lsb // 2:
                msb = prev_msb - max_poc_lsb
            else:
                msb = prev_msb
            poc = msb + lsb
            if is_ref:
                prev_msb, prev_lsb = msb, lsb
        elif poc_type == 2:
            fn = f0["frame_num"]
            if idr:
                frame_num_offset = 0
            elif prev_frame_num > fn:
                frame_num_offset = prev_frame_num_offset + max_frame_num
            else:
                frame_num_offset = prev_frame_num_offset
            if idr:
                poc = 0
            elif is_ref:
                poc = 2 * (frame_num_offset + fn)
            else:
                poc = 2 * (frame_num_offset + fn) - 1
            prev_frame_num, prev_frame_num_offset = fn, frame_num_offset
        else:
            raise RewriteUnsafe("encoder uses pic_order_cnt_type 1; not supported")
        for n, f, d, hb in slices:
            if f.get("mmco") and any(op == 5 for op, _ in f["mmco"]):
                raise RewriteUnsafe("memory_management_control_operation 5 not supported")
        info = {"poc": poc, "idr": idr, "is_ref": is_ref}
        out.append((info, slices))
    return out


def transplant_h264(aus: list[AccessUnit], enc_ps: dict[str, list[bytes]], tgt_ps: dict[str, list[bytes]],
                    conv: Optional[SliceConventions] = None, log=None) -> TransplantResult:
    conv = conv or SliceConventions()
    if len(enc_ps["sps"]) != 1 or len(enc_ps["pps"]) != 1 or len(tgt_ps["sps"]) != 1 or len(tgt_ps["pps"]) != 1:
        raise RewriteUnsafe("exactly one SPS and one PPS required on both sides")
    es, ts = h264.parse_sps_nal(enc_ps["sps"][0]), h264.parse_sps_nal(tgt_ps["sps"][0])
    ep, tp = h264.parse_pps_nal(enc_ps["pps"][0], es), h264.parse_pps_nal(tgt_ps["pps"][0], ts)
    _check_match(es, ts, _H264_SPS_MUST_MATCH, "SPS")
    _check_match(ep, tp, _H264_PPS_MUST_MATCH, "PPS")
    if es["max_num_ref_frames"] > ts["max_num_ref_frames"]:
        raise RewriteUnsafe(f"encoder max_num_ref_frames {es['max_num_ref_frames']} > target {ts['max_num_ref_frames']}")
    if ts["pic_order_cnt_type"] == 1:
        raise RewriteUnsafe("target pic_order_cnt_type 1 not supported")
    if tp["num_slice_groups_minus1"] or ep["num_slice_groups_minus1"]:
        raise RewriteUnsafe("slice groups not supported")
    pics = _h264_poc_sequence(aus, es, ep)
    # target type 2 requires output order == decode order
    if ts["pic_order_cnt_type"] == 2:
        pocs = [p[0]["poc"] for p in pics]
        if any(b < a for a, b in zip(pocs, pocs[1:]) if True) and any(not p[0]["idr"] and p[0]["poc"] < q[0]["poc"] for p, q in zip(pics[1:], pics)):
            raise RewriteUnsafe("target pic_order_cnt_type 2 but encoder stream reorders pictures")
    t_max_frame_num = 1 << (ts["log2_max_frame_num_minus4"] + 4)
    t_max_poc_lsb = 1 << (ts.get("log2_max_pic_order_cnt_lsb_minus4", 0) + 4)
    if ts["pic_order_cnt_type"] == 0:
        # the decoder reconstructs POC from prevPicOrderCnt of the previous REFERENCE picture: every step must stay
        # strictly inside half the lsb range (8.2.1.1)
        prev_ref_poc = 0
        base = 0
        for info, _sl in pics:
            if info["idr"]:
                base = info["poc"]; prev_ref_poc = 0
                continue
            rel = info["poc"] - base
            if abs(rel - prev_ref_poc) >= t_max_poc_lsb // 2:
                raise RewriteUnsafe(f"POC step {rel - prev_ref_poc} too large for the target log2_max_pic_order_cnt_lsb {ts.get('log2_max_pic_order_cnt_lsb_minus4', 0) + 4}")
            if info["is_ref"]:
                prev_ref_poc = rel
    qp_shift = ep["pic_init_qp_minus26"] - tp["pic_init_qp_minus26"]
    changed: set[str] = set()
    new_aus: list[AccessUnit] = []
    idr_count = 0
    last_idr_poc_base = 0
    prev_ref_frame_num = 0
    for info, slices in pics:
        # frame_num renumbered under the target width: 0 at IDR, PrevRefFrameNum + 1 otherwise (7.4.3)
        fn_new = 0 if info["idr"] else (prev_ref_frame_num + 1) % t_max_frame_num
        if info["is_ref"]:
            prev_ref_frame_num = fn_new
        if info["idr"]:
            last_idr_poc_base = info["poc"]
            idr_pic_id = {"increment": idr_count % 65536, "alternate": idr_count % 2, "zero": 0}[conv.idr_pic_id_pattern]
            idr_count += 1
        rel_poc = info["poc"] - last_idr_poc_base
        new_nals: list[bytes] = []
        au_src = aus[len(new_aus)]
        for n in au_src.nals:
            if N.is_vcl(n, "h264"):
                continue
            if N.is_sei(n, "h264") or N.is_filler(n, "h264") or N.is_param_set(n, "h264"):
                continue
            new_nals.append(n)  # AUD, end-of-seq etc.
        for n, f, data, hb in slices:
            g = dict(f)
            if fn_new != f["frame_num"]:
                changed.add("frame_num")
            g["frame_num"] = fn_new
            # POC
            if ts["pic_order_cnt_type"] == 0:
                g["pic_order_cnt_lsb"] = (conv.poc_lsb_at_idr if info["idr"] else rel_poc) % t_max_poc_lsb
                if es["pic_order_cnt_type"] != 0 or ts.get("log2_max_pic_order_cnt_lsb_minus4") != es.get("log2_max_pic_order_cnt_lsb_minus4"):
                    changed.add("pic_order_cnt_lsb")
                if tp["bottom_field_pic_order_in_frame_present_flag"] and not g.get("field_pic_flag"):
                    g.setdefault("delta_pic_order_cnt_bottom", 0)
            else:
                g.pop("pic_order_cnt_lsb", None)
                if es["pic_order_cnt_type"] != 2:
                    changed.add("poc type -> 2")
            # idr_pic_id convention
            if info["idr"]:
                if g.get("idr_pic_id") != idr_pic_id:
                    changed.add("idr_pic_id")
                g["idr_pic_id"] = idr_pic_id
            # num_ref_idx override vs target PPS defaults
            st = h264.slice_type_base(f["slice_type"])
            if st in (h264.SLICE_P, h264.SLICE_SP, h264.SLICE_B):
                n_l0 = f["_num_ref_idx_l0_active"]
                n_l1 = f["_num_ref_idx_l1_active"]
                default_l0 = tp["num_ref_idx_l0_default_active_minus1"] + 1
                default_l1 = tp["num_ref_idx_l1_default_active_minus1"] + 1
                need = n_l0 != default_l0 or (st == h264.SLICE_B and n_l1 != default_l1)
                if need:
                    g["num_ref_idx_active_override_flag"] = 1
                    g["num_ref_idx_l0_active_minus1"] = n_l0 - 1
                    if st == h264.SLICE_B:
                        g["num_ref_idx_l1_active_minus1"] = n_l1 - 1
                else:
                    g["num_ref_idx_active_override_flag"] = 0
                if bool(need) != bool(f.get("num_ref_idx_active_override_flag")):
                    changed.add("num_ref_idx_active_override")
            # pic_init_qp compensation
            if qp_shift:
                g["slice_qp_delta"] = f["slice_qp_delta"] + qp_shift
                changed.add("slice_qp_delta (pic_init_qp)")
            # deblocking control
            if tp["deblocking_filter_control_present_flag"] and not ep["deblocking_filter_control_present_flag"]:
                g["disable_deblocking_filter_idc"] = 0
                g["slice_alpha_c0_offset_div2"] = 0
                g["slice_beta_offset_div2"] = 0
                changed.add("deblocking fields added")
            elif ep["deblocking_filter_control_present_flag"] and not tp["deblocking_filter_control_present_flag"]:
                if f.get("disable_deblocking_filter_idc", 0) != 0 or f.get("slice_alpha_c0_offset_div2", 0) or f.get("slice_beta_offset_div2", 0):
                    raise RewriteUnsafe("target PPS has no deblocking control but slices use non-default deblocking")
                changed.add("deblocking fields dropped")
            # redundant_pic_cnt
            if tp["redundant_pic_cnt_present_flag"]:
                g.setdefault("redundant_pic_cnt", 0)
            # nal_ref_idc convention
            if g["nal_ref_idc"]:
                want = conv.nal_ref_idc_idr if info["idr"] else conv.nal_ref_idc_ref
                if want and want != g["nal_ref_idc"]:
                    g["nal_ref_idc"] = want
                    changed.add("nal_ref_idc")
            if ep["entropy_coding_mode_flag"]:
                stripped = _strip_cabac_zero_words(data)
                if len(stripped) != len(data):
                    changed.add("cabac_zero_words stripped")
                data = stripped
            new_nals.append(h264.write_slice_nal(g, data, hb, ts, tp))
        new_aus.append(AccessUnit(new_nals, "h264"))
    # target parameter sets with the camera's nal_ref_idc
    tsps = _set_ref_idc(tgt_ps["sps"][0], conv.nal_ref_idc_ps)
    tpps = _set_ref_idc(tgt_ps["pps"][0], conv.nal_ref_idc_ps)
    if log:
        log(f"h264 transplant: rewrote {sorted(changed)}")
    return TransplantResult(new_aus, {"sps": [tsps], "pps": [tpps]}, sorted(changed), [], True)


def _strip_cabac_zero_words(data: bytes) -> bytes:
    """Remove trailing cabac_zero_words (0x0000 pairs after rbsp_slice_trailing_bits); cameras never write them."""
    end = len(data)
    while end >= 3 and data[end - 2:end] == b"\x00\x00":
        end -= 2
    return data[:end]


def _set_ref_idc(n: bytes, idc: int) -> bytes:
    return bytes([(n[0] & 0x9F) | ((idc & 3) << 5)]) + n[1:]


# ---- dispatch ------------------------------------------------------------------------------------
def transplant(aus: list[AccessUnit], codec: str, encoder_ps: dict[str, list[bytes]], target_ps: dict[str, list[bytes]],
               conv: Optional[SliceConventions] = None, log=None) -> TransplantResult:
    if codec == "h264":
        return transplant_h264(aus, encoder_ps, target_ps, conv, log)
    from . import hevc_rewrite  # provided separately
    return hevc_rewrite.transplant_hevc(aus, encoder_ps, target_ps, conv, log)
