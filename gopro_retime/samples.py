"""Turn an elementary stream (list of access units) into MP4 video samples following the source file's conventions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .h26x import nal as N


@dataclass
class VideoConventions:
    """What the camera's own samples look like (detected from the source file)."""
    codec: str
    keep_aud: bool                  # samples start with an AUD NAL
    inband_param_sets: bool         # IRAP samples carry VPS/SPS/PPS in-band
    inband_sei_types: list[int]     # SEI payload types present in source samples (informational)
    length_size: int = 4
    nal_ref_idc_slice: Optional[int] = None   # H.264: nal_ref_idc used by the camera for reference slices
    nal_ref_idc_ps: Optional[int] = None      # H.264: nal_ref_idc used for SPS/PPS
    nal_ref_idc_idr: Optional[int] = None


def detect_conventions(codec: str, first_samples: list[bytes], length_size: int = 4) -> VideoConventions:
    keep_aud = False
    inband = False
    sei_types: set[int] = set()
    ref_idc_slice = ref_idc_ps = ref_idc_idr = None
    for smp in first_samples:
        nals = N.split_length_prefixed(smp, length_size)
        if nals and N.is_aud(nals[0], codec):
            keep_aud = True
        for n in nals:
            if N.is_param_set(n, codec):
                inband = True
                if codec == "h264" and ref_idc_ps is None:
                    ref_idc_ps = (n[0] >> 5) & 3
            if N.is_sei(n, codec):
                sei_types.add(_first_sei_type(n, codec))
            if codec == "h264" and N.is_vcl(n, codec):
                r = (n[0] >> 5) & 3
                if N.is_idr(n, codec):
                    ref_idc_idr = r if ref_idc_idr is None else ref_idc_idr
                elif r and ref_idc_slice is None:
                    ref_idc_slice = r
    return VideoConventions(codec, keep_aud, inband, sorted(sei_types), length_size, ref_idc_slice, ref_idc_ps, ref_idc_idr)


def _first_sei_type(n: bytes, codec: str) -> int:
    off = 1 if codec == "h264" else 2
    t = 0
    while off < len(n) and n[off] == 0xFF:
        t += 255
        off += 1
    return t + (n[off] if off < len(n) else 0)


@dataclass
class BuiltVideo:
    samples: list[bytes]
    sync: list[bool]
    param_sets: dict[str, list[bytes]]
    poc: list[int] = field(default_factory=list)       # optional: picture order counts (for ctts)


def build_samples(aus: list[N.AccessUnit], conv: VideoConventions, param_sets: dict[str, list[bytes]],
                  irap_param_sets: Optional[list[bytes]] = None, has_b_frames: bool = False) -> BuiltVideo:
    """Assemble samples: [AUD] [VPS SPS PPS on IRAP if in-band] VCL NALs. SEI and filler are always dropped.
    param_sets: the FINAL parameter sets (after any rewrite) that go into avcC/hvcC.
    irap_param_sets: the NALs to insert in-band on IRAP samples when conv.inband_param_sets (defaults to param_sets)."""
    codec = conv.codec
    if irap_param_sets is None:
        order = ["vps", "sps", "pps"] if codec == "hevc" else ["sps", "pps"]
        irap_param_sets = [n for k in order for n in param_sets.get(k, [])]
    out: list[bytes] = []
    sync: list[bool] = []
    for au in aus:
        nals: list[bytes] = []
        aud = [n for n in au.nals if N.is_aud(n, codec)]
        if conv.keep_aud:
            if aud:
                nals.append(aud[0])
            else:
                nals.append(_make_aud(codec, au, has_b_frames))
        irap = au.is_irap
        if irap and conv.inband_param_sets:
            nals.extend(irap_param_sets)
        for n in au.nals:
            if N.is_vcl(n, codec):
                if codec == "h264" and conv.nal_ref_idc_slice is not None:
                    n = _patch_ref_idc(n, au, conv)
                nals.append(n)
        out.append(N.to_length_prefixed(nals, conv.length_size))
        sync.append(irap)
    return BuiltVideo(out, sync, param_sets)


def _patch_ref_idc(n: bytes, au: N.AccessUnit, conv: VideoConventions) -> bytes:
    cur = (n[0] >> 5) & 3
    if cur == 0:
        return n  # non-reference picture: must stay 0
    want = conv.nal_ref_idc_idr if (N.is_idr(n, "h264") and conv.nal_ref_idc_idr) else conv.nal_ref_idc_slice
    if not want or want == cur:
        return n
    return bytes([(n[0] & 0x9F) | (want << 5)]) + n[1:]


def _make_aud(codec: str, au: N.AccessUnit, has_b_frames: bool = False) -> bytes:
    # primary_pic_type / pic_type: 0 = I only, 1 = I/P, 2 = I/P/B
    t = 0 if au.is_irap else (2 if has_b_frames else 1)
    if codec == "h264":
        return bytes([0x09, (t << 5) | 0x10])
    return bytes([0x46, 0x01, (t << 5) | 0x10])
