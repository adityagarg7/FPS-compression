"""Parameter-set field parsing for H.264 (SPS/PPS) and HEVC (VPS/SPS/PPS).

Interface contract (implemented in full by the bitstream module):

    parse_sps(rbsp_nal: bytes, codec) -> dict     # all syntax elements incl. VUI/HRD, plus derived values
    parse_pps(rbsp_nal: bytes, codec, sps) -> dict
    parse_vps(rbsp_nal: bytes) -> dict            # hevc only
    write_sps(fields: dict, codec) -> bytes       # NAL unit bytes (with header + emulation prevention)
    write_pps(fields: dict, codec, sps) -> bytes
    write_vps(fields: dict) -> bytes
    patch_vui_timing(sps_nal: bytes, codec, num_units_in_tick: int, time_scale: int) -> bytes

Round trip requirement: write_x(parse_x(nal)) == nal for every parameter set GoPro firmware writes and every one
x264/x265 write.
"""
from __future__ import annotations


class NotImplementedYet(NotImplementedError):
    pass


def parse_sps(nal: bytes, codec: str) -> dict:
    raise NotImplementedYet("h26x.params.parse_sps")


def parse_pps(nal: bytes, codec: str, sps: dict) -> dict:
    raise NotImplementedYet("h26x.params.parse_pps")


def parse_vps(nal: bytes) -> dict:
    raise NotImplementedYet("h26x.params.parse_vps")


def write_sps(fields: dict, codec: str) -> bytes:
    raise NotImplementedYet("h26x.params.write_sps")


def write_pps(fields: dict, codec: str, sps: dict) -> bytes:
    raise NotImplementedYet("h26x.params.write_pps")


def write_vps(fields: dict) -> bytes:
    raise NotImplementedYet("h26x.params.write_vps")


def patch_vui_timing(sps_nal: bytes, codec: str, num_units_in_tick: int, time_scale: int) -> bytes:
    raise NotImplementedYet("h26x.params.patch_vui_timing")
