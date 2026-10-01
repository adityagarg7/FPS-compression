"""Parameter-set transplant + slice-header rewrite.

Interface contract:

    transplant(aus: list[AccessUnit], codec: str, encoder_ps: dict[str, list[bytes]],
               target_ps: dict[str, list[bytes]], log=None) -> TransplantResult

    - encoder_ps: VPS/SPS/PPS NALs written by x264/x265 (as found in the ES).
    - target_ps : the camera's VPS/SPS/PPS NALs (already VUI-timing-patched for the new frame rate).
    - Returns new access units whose slice NALs are re-serialised to be decodable under target_ps, with SEI/filler
      removed and parameter-set NALs replaced by target_ps (in-band placement is decided later by samples.build_samples).
    - Raises RewriteUnsafe if any decode-affecting field differs between encoder_ps and target_ps (the caller then
      keeps the encoder's own parameter sets and reports the residual differences).
    - The caller ALWAYS verifies losslessness by decoding both streams (ffmpeg.decode_md5) before using the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


def transplant(aus: list[AccessUnit], codec: str, encoder_ps: dict[str, list[bytes]],
               target_ps: dict[str, list[bytes]], log=None) -> TransplantResult:
    """Baseline: no transplant available -> keep encoder parameter sets, strip nothing (samples.build_samples strips SEI)."""
    return TransplantResult(aus, encoder_ps, [], ["transplant not implemented: encoder parameter sets kept"], False)
