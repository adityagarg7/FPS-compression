"""Calibration: test-encode a tiny clip with the derived settings and compare the DECODE-AFFECTING parameter-set fields
against the camera's. Encoders apply hidden adjustments (x264 shifts chroma_qp_index_offset by -2 with psy-rd, x265
derives cu_qp_delta / TU depths from several options), so we measure instead of trusting the mapping, fix what we can
by adjusting options, and report whatever remains so the transplant can refuse unsafe streams early."""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from .. import ffmpeg as ff
from ..encode import EncoderSettings, build_ffmpeg_command
from ..plan import FramePlan, make_plan
from . import h264
from . import nal as N
from . import params as P

H264_KEYS_SPS = ["profile_idc", "chroma_format_idc", "bit_depth_luma_minus8", "bit_depth_chroma_minus8",
                 "qpprime_y_zero_transform_bypass_flag", "seq_scaling_matrix_present_flag", "frame_mbs_only_flag",
                 "direct_8x8_inference_flag"]
H264_KEYS_PPS = ["entropy_coding_mode_flag", "weighted_pred_flag", "weighted_bipred_idc", "chroma_qp_index_offset",
                 "constrained_intra_pred_flag", "transform_8x8_mode_flag", "pic_scaling_matrix_present_flag",
                 "second_chroma_qp_index_offset", "num_slice_groups_minus1"]
HEVC_KEYS_SPS = ["chroma_format_idc", "bit_depth_luma_minus8", "bit_depth_chroma_minus8",
                 "log2_min_luma_coding_block_size_minus3", "log2_diff_max_min_luma_coding_block_size",
                 "log2_min_luma_transform_block_size_minus2", "log2_diff_max_min_luma_transform_block_size",
                 "max_transform_hierarchy_depth_inter", "max_transform_hierarchy_depth_intra", "scaling_list_enabled_flag",
                 "amp_enabled_flag", "sample_adaptive_offset_enabled_flag", "pcm_enabled_flag",
                 "sps_temporal_mvp_enabled_flag", "strong_intra_smoothing_enabled_flag"]
HEVC_KEYS_PPS = ["sign_data_hiding_enabled_flag", "init_qp_minus26", "constrained_intra_pred_flag", "transform_skip_enabled_flag",
                 "cu_qp_delta_enabled_flag", "diff_cu_qp_delta_depth", "pps_cb_qp_offset", "pps_cr_qp_offset",
                 "weighted_pred_flag", "weighted_bipred_flag", "transquant_bypass_enabled_flag", "tiles_enabled_flag",
                 "entropy_coding_sync_enabled_flag", "pps_loop_filter_across_slices_enabled_flag",
                 "pps_deblocking_filter_disabled_flag", "pps_beta_offset_div2", "pps_tc_offset_div2",
                 "pps_scaling_list_data_present_flag", "log2_parallel_merge_level_minus2"]
# init_qp_minus26 is syntax-only (slice_qp_delta compensates) but listed so we can see it; excluded from 'residual'
_SYNTAX_ONLY = {"init_qp_minus26", "direct_8x8_inference_flag"}


@dataclass
class Calibration:
    ok: bool
    residual: list[str] = field(default_factory=list)
    adjusted: list[str] = field(default_factory=list)
    encoder_sps: Optional[dict] = None
    encoder_pps: Optional[dict] = None


def _probe_encode(st: EncoderSettings, workdir: str) -> tuple[dict, dict]:
    """Encode 3 frames of a synthetic source with the exact encoder options; return parsed (sps, pps)."""
    size = f"{max(128, (st.width // 8) & ~15)}x{max(128, (st.height // 8) & ~15)}"
    fps = "25"
    out = os.path.join(workdir, "calib" + st.es_suffix)
    plan = make_plan(__import__("fractions").Fraction(25), __import__("fractions").Fraction(25), 3, "conform")
    cmd = build_ffmpeg_command("lavfi-src", plan, st, out)
    # swap the input for a lavfi generator with the right pixel format
    i = cmd.index("-i")
    cmd[i:i + 2] = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}", "-t", "0.12"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nals = list(N.iter_annexb_file(out))
    codec = st.codec
    sps_n = next(n for n in nals if N.nal_type(n, codec) == (N.H264_SPS if codec == "h264" else N.HEVC_SPS))
    pps_n = next(n for n in nals if N.nal_type(n, codec) == (N.H264_PPS if codec == "h264" else N.HEVC_PPS))
    sps_f = P.parse_sps(sps_n, codec)
    pps_f = P.parse_pps(pps_n, codec, sps_f)
    return sps_f, pps_f


def _diff(enc: dict, tgt: dict, keys: list[str]) -> list[tuple[str, object, object]]:
    out = []
    for k in keys:
        a, b = enc.get(k, 0) or 0, tgt.get(k, 0) or 0
        if a != b:
            out.append((k, a, b))
    return out


def calibrate(st: EncoderSettings, target_ps: dict[str, list[bytes]], log=None, max_iter: int = 3) -> Calibration:
    codec = st.codec
    tsps = P.parse_sps(target_ps["sps"][0], codec)
    tpps = P.parse_pps(target_ps["pps"][0], codec, tsps)
    keys_sps, keys_pps = (H264_KEYS_SPS, H264_KEYS_PPS) if codec == "h264" else (HEVC_KEYS_SPS, HEVC_KEYS_PPS)
    adjusted: list[str] = []
    cal = Calibration(False)
    with tempfile.TemporaryDirectory(prefix="gopro-retime-calib-") as wd:
        for _ in range(max_iter):
            try:
                esps, epps = _probe_encode(st, wd)
            except subprocess.CalledProcessError as e:
                cal.residual = [f"calibration encode failed: {e.stderr.decode('utf-8', 'replace')[-300:].strip()}"]
                break
            cal.encoder_sps, cal.encoder_pps = esps, epps
            diffs = _diff(esps, tsps, keys_sps) + _diff(epps, tpps, keys_pps)
            diffs = [d for d in diffs if d[0] not in _SYNTAX_ONLY]
            if not diffs:
                cal.ok = True
                break
            fixed = False
            for k, a, b in diffs:
                if codec == "h264" and k == "chroma_qp_index_offset":
                    cur = int(st.x264_params.get("chroma-qp-offset", "0"))
                    st.x264_params["chroma-qp-offset"] = str(cur + (b - a))
                    adjusted.append(f"chroma-qp-offset -> {st.x264_params['chroma-qp-offset']} (encoder wrote {a}, target {b})")
                    fixed = True
                elif codec == "hevc":
                    fixed |= _fix_hevc(st, k, a, b, adjusted)
            if not fixed:
                cal.residual = [f"{k}: encoder {a} != target {b}" for k, a, b in diffs]
                break
        else:
            cal.residual = [f"{k}: encoder {a} != target {b}" for k, a, b in diffs]
    cal.adjusted = adjusted
    if log:
        if cal.ok:
            log("calibration: all decode-affecting parameter-set fields match the camera's" + (f" after adjusting {adjusted}" if adjusted else ""))
        else:
            log("calibration: residual decode-affecting differences (transplant will be refused): " + "; ".join(cal.residual))
    return cal


def _fix_hevc(st: EncoderSettings, k: str, a, b, adjusted: list[str]) -> bool:
    p = st.x265_params
    if k == "max_transform_hierarchy_depth_intra":
        cur = int(p.get("tu-intra-depth", "1"))
        p["tu-intra-depth"] = str(max(1, min(4, cur + (b - a))))
    elif k == "max_transform_hierarchy_depth_inter":
        cur = int(p.get("tu-inter-depth", "1"))
        p["tu-inter-depth"] = str(max(1, min(4, cur + (b - a))))
    elif k == "cu_qp_delta_enabled_flag":
        if b == 0:
            p["aq-mode"] = "0"; p["cutree"] = "0"; p["rc-grain"] = "0"
            for key in ("vbv-maxrate", "vbv-bufsize", "hrd"):
                p.pop(key, None)
            st.maxrate = 0; st.bufsize = 0
        else:
            p.setdefault("aq-mode", "1")
    elif k == "diff_cu_qp_delta_depth":
        ctu = int(p.get("ctu", "64"))
        p["qg-size"] = str(max(8, ctu >> int(b)))
    elif k == "pps_cb_qp_offset":
        p["cbqpoffs"] = str(b)
    elif k == "pps_cr_qp_offset":
        p["crqpoffs"] = str(b)
    elif k == "sign_data_hiding_enabled_flag":
        p["signhide"] = str(b)
    elif k == "transform_skip_enabled_flag":
        p["tskip"] = str(b)
    elif k == "amp_enabled_flag":
        p["amp"] = str(b)
    elif k == "sample_adaptive_offset_enabled_flag":
        p["sao"] = str(b)
    elif k == "strong_intra_smoothing_enabled_flag":
        p["strong-intra-smoothing"] = str(b)
    elif k == "sps_temporal_mvp_enabled_flag":
        p["temporal-mvp"] = str(b)
    elif k == "entropy_coding_sync_enabled_flag":
        p["wpp"] = str(b)
    elif k == "constrained_intra_pred_flag":
        p["constrained-intra"] = str(b)
    elif k == "weighted_pred_flag":
        p["weightp"] = str(b)
    elif k == "weighted_bipred_flag":
        p["weightb"] = str(b)
    elif k == "pps_loop_filter_across_slices_enabled_flag":
        return False
    elif k == "pps_deblocking_filter_disabled_flag":
        if b:
            p["deblock"] = "0:0"; p["no-deblock"] = "1"
        else:
            p.pop("no-deblock", None)
    elif k in ("pps_beta_offset_div2", "pps_tc_offset_div2"):
        beta = p.get("deblock", "0:0").split(":")[0]
        tc = p.get("deblock", "0:0").split(":")[-1]
        if k == "pps_beta_offset_div2":
            beta = str(b)
        else:
            tc = str(b)
        p["deblock"] = f"{beta}:{tc}"
    else:
        return False
    adjusted.append(f"{k}: {a} -> {b}")
    return True
