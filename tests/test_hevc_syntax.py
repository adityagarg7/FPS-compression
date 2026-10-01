"""Byte-exact parse/write round trips of gopro_retime.h26x.hevc against libx265 streams, plus synthetic coverage of
the syntax x265 never emits (SPS RPS lists, inter-RPS prediction, long-term pictures, list modification, extensions)."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Callable

import pytest

from gopro_retime.h26x import hevc
from gopro_retime.h26x import nal as N
from gopro_retime.h26x.bits import BitIO, BitError

FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
pytestmark = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed")

SOURCE = ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-t", "2"]
FRAMES = 50

# name -> (extra ffmpeg args, x265 params). This libx265 build refuses multiple slices without WPP, hence slices=3:wpp=1.
# ':' inside a value (sar) must be escaped for ffmpeg's x265-params splitter.
VARIANTS: dict[str, tuple[list[str], str]] = {
    "default": ([], ""),
    "hrd_aud_repeat": ([], "bframes=0:ref=1:keyint=10:min-keyint=10:no-open-gop=1:aud=1:repeat-headers=1:hrd=1"
                           ":vbv-maxrate=2000:vbv-bufsize=1000:info=0"),
    "bpyramid_tools": ([], "bframes=4:b-pyramid=1:ref=4:wpp=1:no-sao=1:amp=0:rect=0:tskip=1:signhide=0:aq-mode=0:cutree=0"
                           ":ctu=32:min-cu-size=16:max-tu-size=16:tu-intra-depth=2:tu-inter-depth=2"),
    "slices3": ([], "slices=3:wpp=1"),
    "slices2_wpp": ([], "slices=2:wpp=1"),
    "weighted": (["-vf", "fade=t=in:st=0:d=1,fade=t=out:st=1:d=1"], "weightp=1:weightb=1:bframes=2"),  # fades -> real weights
    "main10": (["-pix_fmt", "yuv420p10le", "-profile:v", "main10"], ""),
    "main422_10": (["-pix_fmt", "yuv422p10le", "-profile:v", "main422-10"], ""),
    "scaling_default": ([], "scaling-list=default"),
    "scaling_custom": ([], "scaling-list=@SCALING_FILE@"),
    "intra_tools": ([], "strong-intra-smoothing=0:temporal-mvp=0:constrained-intra=1:deblock=2,2:aq-mode=2:qg-size=16"),
    "lossless": ([], "limit-refs=0:lossless=1"),
    "vui": ([], "rc-lookahead=5:no-info=1:sar=4\\:3:videoformat=5:range=full:colorprim=bt709:transfer=bt709"
                ":colormatrix=bt709:chromaloc=1"),
    "intra_refresh": ([], "intra-refresh=1"),
    "open_gop": ([], "open-gop=1:keyint=10:min-keyint=10:bframes=3"),
    "temporal_layers_hrd": ([], "temporal-layers=1:bframes=3:hrd=1:vbv-maxrate=2000:vbv-bufsize=1000"),
}


def _scaling_list_file() -> str:
    """HM/x265 scaling-list text with non-default values in every matrix so that all lists get coded."""
    lines: list[str] = []
    kinds = ["INTRA", "INTER"]
    comps = ["LUMA", "CHROMAU", "CHROMAV"]
    for size, n, per_row in (("4X4", 16, 4), ("8X8", 64, 8), ("16X16", 64, 8), ("32X32", 64, 8)):
        for k in kinds:
            for c in (comps if size != "32X32" else comps[:1]):
                name = f"{k}{size}_{c}"
                vals = [str(10 + (i * 7 + len(name)) % 23) for i in range(n)]
                lines.append(f"{name} =")
                lines += [",".join(vals[r:r + per_row]) + "," for r in range(0, n, per_row)]
                if size in ("16X16", "32X32"):
                    lines += [f"{name}_DC =", f"{20 + len(name) % 5},"]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def streams(tmp_path_factory) -> Callable[[str], Path]:
    """Lazily encode each variant once; skips the test when this ffmpeg/libx265 build cannot produce it."""
    d = tmp_path_factory.mktemp("hevc")
    sl = d / "scaling.txt"
    sl.write_text(_scaling_list_file())
    cache: dict[str, Path | str] = {}

    def get(name: str) -> Path:
        if name not in cache:
            extra, params = VARIANTS[name]
            params = ":".join(p for p in (params.replace("@SCALING_FILE@", str(sl)), "log-level=error") if p)
            out = d / f"{name}.h265"
            r = subprocess.run([FFMPEG, "-y", "-v", "error", *SOURCE, *extra, "-c:v", "libx265", "-x265-params", params,
                                "-f", "hevc", str(out)], capture_output=True, text=True)
            cache[name] = out if r.returncode == 0 and out.exists() and out.stat().st_size else r.stderr.strip()
        v = cache[name]
        if isinstance(v, str):
            pytest.skip(f"libx265 cannot encode variant {name!r}: {v}")
        return v
    return get


def _roundtrip(path: Path) -> dict:
    """Parse + rewrite every VPS/SPS/PPS/VCL NAL, assert byte equality, and collect what was seen."""
    vps_by, sps_by, pps_by = {}, {}, {}
    types: Counter = Counter()
    slices: list[dict] = []
    for i, nal in enumerate(N.iter_annexb_file(str(path))):
        t = N.hevc_nal_type(nal)
        types[t] += 1
        if t == N.HEVC_VPS:
            f = hevc.parse_vps_nal(nal)
            out = hevc.write_vps_nal(f)
            vps_by[f["vps_video_parameter_set_id"]] = f
        elif t == N.HEVC_SPS:
            f = hevc.parse_sps_nal(nal)
            out = hevc.write_sps_nal(f)
            sps_by[f["sps_seq_parameter_set_id"]] = f
        elif t == N.HEVC_PPS:
            f = hevc.parse_pps_nal(nal)
            out = hevc.write_pps_nal(f)
            pps_by[f["pps_pic_parameter_set_id"]] = f
        elif N.is_vcl(nal, "hevc"):
            pps_f = pps_by[hevc.slice_pps_id(nal)]
            sps_f = sps_by[pps_f["pps_seq_parameter_set_id"]]
            f, data, hb = hevc.parse_slice_nal(nal, sps_f, pps_f)
            assert hb % 8 == 0 and f["nal_unit_type"] == t
            out = hevc.write_slice_nal(f, data, hb, sps_f, pps_f)
            slices.append(f)
        else:
            continue
        assert out == nal, f"NAL #{i} (type {t}) does not round-trip"
    aus = list(N.group_access_units(N.iter_annexb_file(str(path)), "hevc"))
    return {"types": types, "vps": vps_by, "sps": sps_by, "pps": pps_by, "slices": slices, "aus": aus}


def _assert_subdict(expected, actual, path: str = "") -> None:
    """Every (non-derived) value in `expected` must appear identically in `actual`."""
    if isinstance(expected, dict):
        for k, v in expected.items():
            if not str(k).startswith("_"):
                assert k in actual, f"{path}.{k} missing"
                _assert_subdict(v, actual[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), f"{path} length {len(actual)} != {len(expected)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_subdict(e, a, f"{path}[{i}]")
    else:
        assert expected == actual, f"{path}: {actual!r} != {expected!r}"


# ---- encoder streams ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(VARIANTS))
def test_roundtrip_every_nal(streams, name):
    st = _roundtrip(streams(name))
    assert st["vps"] and st["sps"] and st["pps"]
    assert len(st["aus"]) == FRAMES
    assert all(au.vcl for au in st["aus"])
    assert sum(1 for s in st["slices"] if s["first_slice_segment_in_pic_flag"]) == FRAMES
    assert any(s.get("st_ref_pic_set") for s in st["slices"])          # inline explicit RPS
    assert any(s.get("num_entry_point_offsets") for s in st["slices"])  # WPP entry points


def _first(d: dict) -> dict:
    return next(iter(d.values()))


def test_hrd_aud_repeat_headers(streams):
    st = _roundtrip(streams("hrd_aud_repeat"))
    assert st["types"][N.HEVC_AUD] == FRAMES
    assert all(N.is_aud(au.nals[0], "hevc") for au in st["aus"])
    idr_aus = [au for au in st["aus"] if au.is_idr]
    assert len(idr_aus) == 5 and st["types"][N.HEVC_SPS] == 5
    assert all(any(N.hevc_nal_type(n) == N.HEVC_VPS for n in au.nals) for au in idr_aus)
    sps_f = _first(st["sps"])
    hrd = sps_f["vui"]["hrd"]
    assert hrd["nal_hrd_parameters_present_flag"] == 1 and hrd["sub_layers"][0]["fixed_pic_rate_general_flag"] == 1
    assert all(s["slice_type"] in (hevc.SLICE_P, hevc.SLICE_I) for s in st["slices"])
    assert any(s.get("pred_weight_l0") for s in st["slices"])  # x265 weightp default


def test_derive_sps_matches_ffprobe(streams):
    path = streams("hrd_aud_repeat")
    probe = json.loads(subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    sps_f = _first(_roundtrip(path)["sps"])
    assert (sps_f["_width"], sps_f["_height"]) == (probe["width"], probe["height"]) == (320, 240)
    assert Fraction(*sps_f["_fps"]) == Fraction(probe["r_frame_rate"])
    assert sps_f["_nal_hrd_bit_rate"] == 2000 * 1000
    assert sps_f["_nal_hrd_cpb_size"] == 1000 * 1000
    assert (sps_f["_ctb_log2"], sps_f["_min_cb_log2"], sps_f["_chroma_array_type"]) == (6, 3, 1)


def test_bpyramid_tools(streams):
    st = _roundtrip(streams("bpyramid_tools"))
    sps_f, pps_f = _first(st["sps"]), _first(st["pps"])
    assert sps_f["amp_enabled_flag"] == 0 and sps_f["sample_adaptive_offset_enabled_flag"] == 0
    assert (sps_f["_ctb_log2"], sps_f["_min_cb_log2"]) == (5, 4)
    assert pps_f["transform_skip_enabled_flag"] == 1 and pps_f["sign_data_hiding_enabled_flag"] == 0
    b = [s for s in st["slices"] if s["slice_type"] == hevc.SLICE_B]
    assert b and any(s["num_ref_idx_active_override_flag"] for s in b)
    assert any(s.get("collocated_ref_idx") is not None for s in b)
    assert all("slice_sao_luma_flag" in s and s["slice_sao_luma_flag"] == 0 for s in st["slices"])


@pytest.mark.parametrize("name,per_pic", [("slices3", 3), ("slices2_wpp", 2)])
def test_multiple_slices(streams, name, per_pic):
    st = _roundtrip(streams(name))
    assert all(len(au.vcl) == per_pic for au in st["aus"])
    later = [s for s in st["slices"] if not s["first_slice_segment_in_pic_flag"]]
    assert len(later) == FRAMES * (per_pic - 1)
    assert all(s["slice_segment_address"] > 0 and s["dependent_slice_segment_flag"] == 0 for s in later)
    assert any(s["num_entry_point_offsets"] > 0 and len(s["entry_point_offset_minus1"]) == s["num_entry_point_offsets"]
               for s in st["slices"])


def test_weighted_prediction(streams):
    st = _roundtrip(streams("weighted"))
    pps_f = _first(st["pps"])
    assert pps_f["weighted_pred_flag"] == 1 and pps_f["weighted_bipred_flag"] == 1
    l1 = [s for s in st["slices"] if s.get("pred_weight_l1")]
    assert l1 and all(len(s["pred_weight_l1"]) == s["_num_ref_idx_l1_active"] for s in l1)
    entries = [e for s in l1 for e in s["pred_weight_l0"] + s["pred_weight_l1"]]
    assert any(e["luma_weight_flag"] for e in entries) and any(e["chroma_weight_flag"] for e in entries)


@pytest.mark.parametrize("name,chroma,depth,profile", [("main10", 1, 2, 2), ("main422_10", 2, 2, 4)])
def test_high_bit_depth(streams, name, chroma, depth, profile):
    sps_f = _first(_roundtrip(streams(name))["sps"])
    assert sps_f["chroma_format_idc"] == chroma and sps_f["bit_depth_luma_minus8"] == depth
    assert sps_f["ptl"]["general_profile_idc"] == profile


def test_scaling_lists(streams):
    sps_f = _first(_roundtrip(streams("scaling_default"))["sps"])
    assert sps_f["scaling_list_enabled_flag"] == 1 and sps_f["sps_scaling_list_data_present_flag"] == 0
    sps_f = _first(_roundtrip(streams("scaling_custom"))["sps"])
    assert sps_f["sps_scaling_list_data_present_flag"] == 1 and len(sps_f["scaling_lists"]) == 20
    coded = [e for e in sps_f["scaling_lists"] if e["scaling_list_pred_mode_flag"]]
    assert coded and all(len(e["scaling_list_delta_coef"]) == (16 if e["size_id"] == 0 else 64) for e in coded)
    assert all("scaling_list_dc_coef_minus8" in e for e in coded if e["size_id"] > 1)


def test_intra_tools(streams):
    st = _roundtrip(streams("intra_tools"))
    sps_f, pps_f = _first(st["sps"]), _first(st["pps"])
    assert sps_f["strong_intra_smoothing_enabled_flag"] == 0 and sps_f["sps_temporal_mvp_enabled_flag"] == 0
    assert pps_f["constrained_intra_pred_flag"] == 1 and pps_f["cu_qp_delta_enabled_flag"] == 1
    assert pps_f["deblocking_filter_control_present_flag"] == 1
    assert (pps_f["pps_beta_offset_div2"], pps_f["pps_tc_offset_div2"]) == (2, 2)
    assert all(s["slice_temporal_mvp_enabled_flag"] == 0 and "collocated_ref_idx" not in s for s in st["slices"])


def test_lossless(streams):
    st = _roundtrip(streams("lossless"))
    assert _first(st["pps"])["transquant_bypass_enabled_flag"] == 1


def test_vui(streams):
    vui = _first(_roundtrip(streams("vui"))["sps"])["vui"]
    assert vui["aspect_ratio_info_present_flag"] == 1 and vui["aspect_ratio_idc"] == 14  # 4:3
    assert vui["video_signal_type_present_flag"] == 1 and vui["video_format"] == 5 and vui["video_full_range_flag"] == 1
    assert (vui["colour_primaries"], vui["transfer_characteristics"], vui["matrix_coeffs"]) == (1, 1, 1)
    assert vui["chroma_loc_info_present_flag"] == 1 and vui["chroma_sample_loc_type_top_field"] == 1


def test_intra_refresh(streams):
    st = _roundtrip(streams("intra_refresh"))
    assert st["aus"][0].is_idr and sum(1 for au in st["aus"] if au.is_irap) == 1
    assert any(s["slice_type"] == hevc.SLICE_P for s in st["slices"])


def test_open_gop_leading_pictures(streams):
    st = _roundtrip(streams("open_gop"))
    assert st["types"][N.HEVC_CRA] > 0 and (st["types"][8] + st["types"][9]) > 0  # CRA + RASL_N/RASL_R
    cra = [s for s in st["slices"] if s["nal_unit_type"] == N.HEVC_CRA]
    assert all("no_output_of_prior_pics_flag" in s and "slice_pic_order_cnt_lsb" in s for s in cra)
    assert sum(1 for au in st["aus"] if au.is_irap) == len(cra) + 1


def test_temporal_sub_layers(streams):
    st = _roundtrip(streams("temporal_layers_hrd"))
    vps_f, sps_f = _first(st["vps"]), _first(st["sps"])
    assert vps_f["vps_max_sub_layers_minus1"] == 1 and sps_f["sps_max_sub_layers_minus1"] == 1
    assert len(sps_f["ptl"]["sub_layers"]) == 1 and len(sps_f["ptl"]["reserved_zero_2bits"]) == 7
    assert len(sps_f["vui"]["hrd"]["sub_layers"]) == 2 and len(sps_f["sps_max_dec_pic_buffering_minus1"]) == 2
    assert any(s["nuh_temporal_id_plus1"] == 2 for s in st["slices"])
    assert sps_f["_nal_hrd_bit_rate"] == 2_000_000


# ---- synthetic syntax --------------------------------------------------------------------------------
REF_RPS = {"inter_ref_pic_set_prediction_flag": 0, "delta_poc_s0": [-1, -3], "used_by_curr_pic_s0": [1, 0],
           "delta_poc_s1": [2], "used_by_curr_pic_s1": [1]}
INTER_RPS = {"inter_ref_pic_set_prediction_flag": 1, "delta_rps_sign": 1, "abs_delta_rps_minus1": 0,
             "used_by_curr_pic_flag": [1, 0, 1, 1], "use_delta_flag": [1, 0, 1, 1]}
# deltaRps = -1 applied to REF_RPS; entry j=1 (ref -3) is dropped by use_delta_flag = 0 (7-61 / 7-62)
INTER_EXPANDED = {"delta_poc_s0": [-1, -2], "used_by_curr_pic_s0": [1, 1], "delta_poc_s1": [1], "used_by_curr_pic_s1": [1]}


def test_inter_rps_prediction_expansion():
    w = BitIO()
    hevc.st_ref_pic_set(w, copy.deepcopy(REF_RPS), 0, 2, [])
    w.write_bits(1, 1)                 # inter_ref_pic_set_prediction_flag
    w.write_bits(1, 1)                 # delta_rps_sign
    w.write_ue(0)                      # abs_delta_rps_minus1
    for used, use in zip(INTER_RPS["used_by_curr_pic_flag"], INTER_RPS["use_delta_flag"]):
        w.write_bits(used, 1)
        if not used:
            w.write_bits(use, 1)
    bits = w.to_bytes()
    r = BitIO(bits)
    sets: list[dict] = [{}, {}]
    hevc.st_ref_pic_set(r, sets[0], 0, 2, sets)
    hevc.st_ref_pic_set(r, sets[1], 1, 2, sets)
    _assert_subdict(REF_RPS, sets[0])
    _assert_subdict(INTER_RPS, sets[1])
    _assert_subdict(INTER_EXPANDED, sets[1])
    w2 = BitIO()
    for i, s in enumerate(sets):
        hevc.st_ref_pic_set(w2, s, i, 2, sets)
    assert w2.to_bytes() == bits
    # the expanded lists written in explicit form parse back to the same lists
    w3 = BitIO()
    hevc.st_ref_pic_set(w3, dict(INTER_EXPANDED, inter_ref_pic_set_prediction_flag=0), 0, 1, [])
    explicit: dict = {}
    hevc.st_ref_pic_set(BitIO(w3.to_bytes()), explicit, 0, 1, [])
    _assert_subdict(INTER_EXPANDED, explicit)


def _synthetic_sps(base: dict) -> dict:
    f = copy.deepcopy(base)
    f["num_short_term_ref_pic_sets"] = 3
    f["st_ref_pic_sets"] = [copy.deepcopy(REF_RPS), copy.deepcopy(INTER_RPS),
                            {"inter_ref_pic_set_prediction_flag": 0, "delta_poc_s0": [-2], "used_by_curr_pic_s0": [1],
                             "delta_poc_s1": [], "used_by_curr_pic_s1": []}]
    f["long_term_ref_pics_present_flag"] = 1
    f["num_long_term_ref_pics_sps"] = 2
    f["long_term_ref_pics_sps"] = [{"lt_ref_pic_poc_lsb_sps": 5, "used_by_curr_pic_lt_sps_flag": 1},
                                   {"lt_ref_pic_poc_lsb_sps": 9, "used_by_curr_pic_lt_sps_flag": 0}]
    f["sps_extension_present_flag"] = 1
    f["sps_range_extension_flag"] = f["sps_scc_extension_flag"] = f["sps_multilayer_extension_flag"] = 1
    f["sps_3d_extension_flag"] = 0
    f["sps_extension_4bits"] = 1
    f["range_ext"] = {k: i % 2 for i, k in enumerate((
        "transform_skip_rotation_enabled_flag", "transform_skip_context_enabled_flag", "implicit_rdpcm_enabled_flag",
        "explicit_rdpcm_enabled_flag", "extended_precision_processing_flag", "intra_smoothing_disabled_flag",
        "high_precision_offsets_enabled_flag", "persistent_rice_adaptation_enabled_flag", "cabac_bypass_alignment_enabled_flag"))}
    f["inter_view_mv_vert_constraint_flag"] = 1
    f["scc_ext"] = {"sps_curr_pic_ref_enabled_flag": 0, "palette_mode_enabled_flag": 1, "palette_max_size": 64,
                    "delta_palette_max_predictor_size": 32, "sps_palette_predictor_initializers_present_flag": 1,
                    "sps_num_palette_predictor_initializers_minus1": 1,
                    "sps_palette_predictor_initializer": [{"value": [1, 255]}, {"value": [3, 4]}, {"value": [5, 6]}],
                    "motion_vector_resolution_control_idc": 2, "intra_boundary_filtering_disabled_flag": 1}
    f["sps_extension_data_flag"] = [1, 0, 1]
    return f


def _synthetic_pps(base: dict) -> dict:
    f = copy.deepcopy(base)
    f.update(dependent_slice_segments_enabled_flag=1, output_flag_present_flag=1, num_extra_slice_header_bits=2,
             cabac_init_present_flag=1, pps_slice_chroma_qp_offsets_present_flag=1, weighted_pred_flag=1,
             weighted_bipred_flag=1, transform_skip_enabled_flag=1, tiles_enabled_flag=1, num_tile_columns_minus1=1,
             num_tile_rows_minus1=0, uniform_spacing_flag=0, column_width_minus1=[2], row_height_minus1=[],
             loop_filter_across_tiles_enabled_flag=1, deblocking_filter_control_present_flag=1,
             deblocking_filter_override_enabled_flag=1, pps_deblocking_filter_disabled_flag=0, pps_beta_offset_div2=-1,
             pps_tc_offset_div2=2, pps_scaling_list_data_present_flag=1, lists_modification_present_flag=1,
             slice_segment_header_extension_present_flag=1, pps_extension_present_flag=1, pps_range_extension_flag=1,
             pps_scc_extension_flag=1, pps_extension_4bits=0)
    lists = []
    for size_id in range(4):
        for matrix_id in range(0, 6, 3 if size_id == 3 else 1):
            e = {"size_id": size_id, "matrix_id": matrix_id, "scaling_list_pred_mode_flag": (size_id + matrix_id) % 2}
            if e["scaling_list_pred_mode_flag"]:
                n = min(64, 1 << (4 + (size_id << 1)))
                e["scaling_list_delta_coef"] = [(i % 5) - 2 for i in range(n)]
                if size_id > 1:
                    e["scaling_list_dc_coef_minus8"] = 3
            else:
                e["scaling_list_pred_matrix_id_delta"] = matrix_id % 2
            lists.append(e)
    f["scaling_lists"] = lists
    f["range_ext"] = {"log2_max_transform_skip_block_size_minus2": 1, "cross_component_prediction_enabled_flag": 0,
                      "chroma_qp_offset_list_enabled_flag": 1, "diff_cu_chroma_qp_offset_depth": 1,
                      "chroma_qp_offset_list_len_minus1": 1,
                      "chroma_qp_offset_list": [{"cb_qp_offset_list": 1, "cr_qp_offset_list": -1},
                                                {"cb_qp_offset_list": 2, "cr_qp_offset_list": -2}],
                      "log2_sao_offset_scale_luma": 0, "log2_sao_offset_scale_chroma": 1}
    f["scc_ext"] = {"pps_curr_pic_ref_enabled_flag": 0, "residual_adaptive_colour_transform_enabled_flag": 1,
                    "pps_slice_act_qp_offsets_present_flag": 1, "pps_act_y_qp_offset_plus5": 0,
                    "pps_act_cb_qp_offset_plus5": 1, "pps_act_cr_qp_offset_plus3": -1,
                    "pps_palette_predictor_initializers_present_flag": 1, "pps_num_palette_predictor_initializers": 2,
                    "monochrome_palette_flag": 0, "luma_bit_depth_entry_minus8": 0, "chroma_bit_depth_entry_minus8": 2,
                    "pps_palette_predictor_initializer": [{"value": [7, 8]}, {"value": [1000, 0]}, {"value": [5, 6]}]}
    return f


def _weights(n: int) -> list[dict]:
    out = []
    for i in range(n):
        e = {"luma_weight_flag": i % 2, "chroma_weight_flag": (i + 1) % 2}
        if e["luma_weight_flag"]:
            e.update(delta_luma_weight=-3, luma_offset=4)
        if e["chroma_weight_flag"]:
            e.update(delta_chroma_weight=[1, -1], delta_chroma_offset=[-5, 6])
        out.append(e)
    return out


def _roundtrip_slice(sl: dict, sps_f: dict, pps_f: dict) -> dict:
    data = b"\x00\x00\x01\x00\x00\x03\x00\x00\x00\x02\xff" * 3   # exercises emulation prevention
    nal = hevc.write_slice_nal(sl, data, 0, sps_f, pps_f)
    f, out_data, hb = hevc.parse_slice_nal(nal, sps_f, pps_f)
    assert out_data == data and hb % 8 == 0
    assert hevc.write_slice_nal(f, out_data, hb, sps_f, pps_f) == nal
    assert hevc.slice_pps_id(nal) == sl["slice_pic_parameter_set_id"]
    _assert_subdict(sl, f)
    return f


def test_synthetic_parameter_sets_and_slices(streams):
    base = _roundtrip(streams("default"))
    sps_f = _synthetic_sps(_first(base["sps"]))
    nal = hevc.write_sps_nal(sps_f)
    sps2 = hevc.parse_sps_nal(nal)
    assert hevc.write_sps_nal(sps2) == nal
    _assert_subdict(sps_f, sps2)
    _assert_subdict(INTER_EXPANDED, sps2["st_ref_pic_sets"][1])
    pps_f = _synthetic_pps(_first(base["pps"]))
    pps_f["pps_seq_parameter_set_id"] = sps2["sps_seq_parameter_set_id"]
    nal = hevc.write_pps_nal(pps_f, sps2)
    pps2 = hevc.parse_pps_nal(nal, sps2)
    assert hevc.write_pps_nal(pps2, sps2) == nal
    _assert_subdict(pps_f, pps2)
    pid = pps2["pps_pic_parameter_set_id"]

    # B slice: SPS RPS by index (inter-predicted), long-term pics from SPS and explicit, list modification,
    # weighted bi-prediction, collocated from L1, SCC/range-extension slice elements, tiles entry points, extension bytes
    b_slice = {
        "nal_unit_type": 1, "first_slice_segment_in_pic_flag": 1, "slice_pic_parameter_set_id": pid,
        "slice_reserved_flag": [1, 0], "slice_type": hevc.SLICE_B, "pic_output_flag": 1, "slice_pic_order_cnt_lsb": 7,
        "short_term_ref_pic_set_sps_flag": 1, "short_term_ref_pic_set_idx": 1, "num_long_term_sps": 1, "num_long_term_pics": 1,
        "long_term_pics": [{"lt_idx_sps": 1, "delta_poc_msb_present_flag": 1, "delta_poc_msb_cycle_lt": 2},
                           {"poc_lsb_lt": 3, "used_by_curr_pic_lt_flag": 1, "delta_poc_msb_present_flag": 0}],
        "slice_temporal_mvp_enabled_flag": 1, "slice_sao_luma_flag": 1, "slice_sao_chroma_flag": 0,
        "num_ref_idx_active_override_flag": 1, "num_ref_idx_l0_active_minus1": 2, "num_ref_idx_l1_active_minus1": 1,
        "ref_pic_list_modification_flag_l0": 1, "list_entry_l0": [2, 0, 3], "ref_pic_list_modification_flag_l1": 1,
        "list_entry_l1": [1, 0], "mvd_l1_zero_flag": 1, "cabac_init_flag": 1, "collocated_from_l0_flag": 0,
        "collocated_ref_idx": 1, "luma_log2_weight_denom": 6, "delta_chroma_log2_weight_denom": -1,
        "pred_weight_l0": _weights(3), "pred_weight_l1": _weights(2), "five_minus_max_num_merge_cand": 1,
        "use_integer_mv_flag": 1, "slice_qp_delta": -3, "slice_cb_qp_offset": 1, "slice_cr_qp_offset": -1,
        "slice_act_y_qp_offset": 1, "slice_act_cb_qp_offset": 0, "slice_act_cr_qp_offset": -2,
        "cu_chroma_qp_offset_enabled_flag": 1, "deblocking_filter_override_flag": 1,
        "slice_deblocking_filter_disabled_flag": 0, "slice_beta_offset_div2": 1, "slice_tc_offset_div2": -1,
        "slice_loop_filter_across_slices_enabled_flag": 0, "num_entry_point_offsets": 2, "offset_len_minus1": 9,
        "entry_point_offset_minus1": [100, 1023], "slice_segment_header_extension_length": 2,
        "slice_segment_header_extension_data_byte": [0xAB, 0x00],
    }
    f = _roundtrip_slice(b_slice, sps2, pps2)
    assert f["_num_pic_total_curr"] == 4 and (f["_num_ref_idx_l0_active"], f["_num_ref_idx_l1_active"]) == (3, 2)

    # P slice with an inline inter-predicted RPS (delta_idx_minus1 coded), deblocking disabled by override
    p_slice = {
        "nal_unit_type": 0, "first_slice_segment_in_pic_flag": 0, "dependent_slice_segment_flag": 0,
        "slice_segment_address": 9, "slice_pic_parameter_set_id": pid, "slice_reserved_flag": [0, 1],
        "slice_type": hevc.SLICE_P, "pic_output_flag": 0, "slice_pic_order_cnt_lsb": 200,
        "short_term_ref_pic_set_sps_flag": 0,
        "st_ref_pic_set": {"inter_ref_pic_set_prediction_flag": 1, "delta_idx_minus1": 1, "delta_rps_sign": 0,
                           "abs_delta_rps_minus1": 1, "used_by_curr_pic_flag": [1, 1, 0, 1], "use_delta_flag": [1, 1, 1, 1]},
        "num_long_term_sps": 0, "num_long_term_pics": 0, "long_term_pics": [], "slice_temporal_mvp_enabled_flag": 1,
        "slice_sao_luma_flag": 0, "slice_sao_chroma_flag": 1, "num_ref_idx_active_override_flag": 1,
        "num_ref_idx_l0_active_minus1": 1, "ref_pic_list_modification_flag_l0": 1, "list_entry_l0": [1, 0],
        "cabac_init_flag": 0, "collocated_ref_idx": 1, "luma_log2_weight_denom": 0, "delta_chroma_log2_weight_denom": 0,
        "pred_weight_l0": _weights(2), "five_minus_max_num_merge_cand": 0, "use_integer_mv_flag": 0,
        "slice_qp_delta": 0, "slice_cb_qp_offset": 0, "slice_cr_qp_offset": 0, "slice_act_y_qp_offset": 0,
        "slice_act_cb_qp_offset": 0, "slice_act_cr_qp_offset": 0, "cu_chroma_qp_offset_enabled_flag": 0,
        "deblocking_filter_override_flag": 1, "slice_deblocking_filter_disabled_flag": 1,
        "slice_loop_filter_across_slices_enabled_flag": 1, "num_entry_point_offsets": 0,
        "slice_segment_header_extension_length": 0, "slice_segment_header_extension_data_byte": [],
    }
    f = _roundtrip_slice(p_slice, sps2, pps2)
    assert f["st_ref_pic_set"]["delta_poc_s0"] == [] and f["st_ref_pic_set"]["delta_poc_s1"] == [1, 2, 3]
    assert f["st_ref_pic_set"]["used_by_curr_pic_s1"] == [1, 1, 0] and f["_num_pic_total_curr"] == 2
    assert f["collocated_from_l0_flag"] == 1

    # dependent slice segment: only address + entry points + extension
    dep = {"nal_unit_type": 0, "first_slice_segment_in_pic_flag": 0, "dependent_slice_segment_flag": 1,
           "slice_segment_address": 5, "slice_pic_parameter_set_id": pid, "num_entry_point_offsets": 1,
           "offset_len_minus1": 31, "entry_point_offset_minus1": [0xFFFFFFFF], "slice_segment_header_extension_length": 0,
           "slice_segment_header_extension_data_byte": []}
    f = _roundtrip_slice(dep, sps2, pps2)
    assert "slice_type" not in f

    # IDR with no_output_of_prior_pics_flag, colour planes off, SAO + loop filter flags
    idr = {"nal_unit_type": N.HEVC_IDR_W_RADL, "first_slice_segment_in_pic_flag": 1, "no_output_of_prior_pics_flag": 1,
           "slice_pic_parameter_set_id": pid, "slice_reserved_flag": [0, 0], "slice_type": hevc.SLICE_I, "pic_output_flag": 1,
           "slice_sao_luma_flag": 1, "slice_sao_chroma_flag": 1, "slice_qp_delta": 2, "slice_cb_qp_offset": 0,
           "slice_cr_qp_offset": 0, "slice_act_y_qp_offset": 0, "slice_act_cb_qp_offset": 0, "slice_act_cr_qp_offset": 0,
           "cu_chroma_qp_offset_enabled_flag": 0, "deblocking_filter_override_flag": 0,
           "slice_loop_filter_across_slices_enabled_flag": 1, "num_entry_point_offsets": 0,
           "slice_segment_header_extension_length": 0, "slice_segment_header_extension_data_byte": []}
    f = _roundtrip_slice(idr, sps2, pps2)
    assert f["slice_temporal_mvp_enabled_flag"] == 0 and f["_num_pic_total_curr"] == 0
    assert f["slice_deblocking_filter_disabled_flag"] == pps2["pps_deblocking_filter_disabled_flag"]


def test_vps_timing_and_extension_roundtrip(streams):
    f = copy.deepcopy(_first(_roundtrip(streams("default"))["vps"]))
    f.update(vps_timing_info_present_flag=1, vps_num_units_in_tick=1001, vps_time_scale=60000,
             vps_poc_proportional_to_timing_flag=1, vps_num_ticks_poc_diff_one_minus1=0, vps_num_hrd_parameters=2,
             vps_extension_flag=1, vps_extension_data_flag=[1, 1, 0])
    sub = {"fixed_pic_rate_general_flag": 0, "fixed_pic_rate_within_cvs_flag": 0, "low_delay_hrd_flag": 0, "cpb_cnt_minus1": 1,
           "vcl": {"cpb": [{"bit_rate_value_minus1": 10, "cpb_size_value_minus1": 20, "cbr_flag": 1},
                           {"bit_rate_value_minus1": 11, "cpb_size_value_minus1": 21, "cbr_flag": 0}]}}
    f["hrd"] = [{"hrd_layer_set_idx": 0, "nal_hrd_parameters_present_flag": 0, "vcl_hrd_parameters_present_flag": 1,
                 "sub_pic_hrd_params_present_flag": 0, "bit_rate_scale": 2, "cpb_size_scale": 3,
                 "initial_cpb_removal_delay_length_minus1": 23, "au_cpb_removal_delay_length_minus1": 15,
                 "dpb_output_delay_length_minus1": 5, "sub_layers": [sub]},
                {"hrd_layer_set_idx": 0, "cprms_present_flag": 0, "sub_layers": [copy.deepcopy(sub)]}]
    nal = hevc.write_vps_nal(f)
    f2 = hevc.parse_vps_nal(nal)
    assert hevc.write_vps_nal(f2) == nal
    _assert_subdict(f, f2)
    assert f2["hrd"][1]["vcl_hrd_parameters_present_flag"] == 1   # inherited from hrd[0] when cprms_present_flag == 0


def test_unsupported_extensions_raise(streams):
    pps_f = copy.deepcopy(_first(_roundtrip(streams("default"))["pps"]))
    pps_f.update(pps_extension_present_flag=1, pps_multilayer_extension_flag=1)
    with pytest.raises(BitError, match="multilayer"):
        hevc.write_pps_nal(pps_f)
    sps_nal = N.insert_epb(bytes([0x42, 0x09]) + b"\x0f" * 8)   # nuh_layer_id = 1 with sps_ext_or_max_sub_layers_minus1 == 7
    with pytest.raises(BitError, match="multi-layer"):
        hevc.parse_sps_nal(sps_nal)
