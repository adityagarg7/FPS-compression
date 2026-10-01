"""Codec-agnostic parameter-set access: parse / write / patch VUI timing for H.264 and HEVC."""
from __future__ import annotations

from . import h264
from . import nal as N


class NotImplementedYet(NotImplementedError):
    pass


def _hevc():
    from . import hevc  # noqa: WPS433 (lazy: module provided separately)
    return hevc


def parse_sps(nal: bytes, codec: str) -> dict:
    if codec == "h264":
        return h264.parse_sps_nal(nal)
    return _hevc().parse_sps_nal(nal)


def write_sps(f: dict, codec: str) -> bytes:
    if codec == "h264":
        return h264.write_sps_nal(f)
    return _hevc().write_sps_nal(f)


def parse_pps(nal: bytes, codec: str, sps_f: dict) -> dict:
    if codec == "h264":
        return h264.parse_pps_nal(nal, sps_f)
    return _hevc().parse_pps_nal(nal, sps_f)


def write_pps(f: dict, codec: str, sps_f: dict) -> bytes:
    if codec == "h264":
        return h264.write_pps_nal(f, sps_f)
    return _hevc().write_pps_nal(f, sps_f)


def parse_vps(nal: bytes) -> dict:
    return _hevc().parse_vps_nal(nal)


def write_vps(f: dict) -> bytes:
    return _hevc().write_vps_nal(f)


def patch_vui_timing(sps_nal: bytes, codec: str, num_units_in_tick: int, time_scale: int) -> bytes:
    """Return the parameter set with its timing info rewritten. codec: 'h264' | 'hevc' | 'hevc-vps'.
    If the parameter set carries no timing info it is returned unchanged."""
    if codec == "h264":
        f = h264.parse_sps_nal(sps_nal)
        vui = f.get("vui")
        if not vui or not vui.get("timing_info_present_flag"):
            return sps_nal
        vui["num_units_in_tick"] = num_units_in_tick
        vui["time_scale"] = time_scale
        return h264.write_sps_nal(f)
    hevc = _hevc()
    if codec == "hevc-vps":
        f = hevc.parse_vps_nal(sps_nal)
        if not f.get("vps_timing_info_present_flag"):
            return sps_nal
        f["vps_num_units_in_tick"] = num_units_in_tick
        f["vps_time_scale"] = time_scale
        return hevc.write_vps_nal(f)
    f = hevc.parse_sps_nal(sps_nal)
    vui = f.get("vui")
    if not vui or not vui.get("vui_timing_info_present_flag"):
        return sps_nal
    vui["vui_num_units_in_tick"] = num_units_in_tick
    vui["vui_time_scale"] = time_scale
    return hevc.write_sps_nal(f)


def hrd_values(sps_f: dict) -> tuple[int | None, int | None]:
    """(bit_rate bps, cpb_size bits) from the NAL (or VCL) HRD of a parsed SPS, if present."""
    br = sps_f.get("_nal_hrd_bit_rate") or sps_f.get("_vcl_hrd_bit_rate")
    cpb = sps_f.get("_nal_hrd_cpb_size") or sps_f.get("_vcl_hrd_cpb_size")
    return br, cpb
