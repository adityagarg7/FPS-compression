"""GoPro MET payload rebuild for the new frame grid. Placeholder: copies the source payloads re-timed to the
output payload duration (per-frame streams are NOT yet resampled). Replaced by the full implementation."""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .model import SourceFile
from .plan import FramePlan, frames_per_gpmf_payload


def rebuild(src: SourceFile, plan: FramePlan, out_fps: Fraction, drop_imu: bool, drop_gps: bool,
            reference: Optional[SourceFile], log=print) -> tuple[list[bytes], list[int]]:
    gp = src.track("gpmd")
    payloads = src.read_samples(gp)
    fpp, ms = frames_per_gpmf_payload(out_fps)
    n_out = plan.out_frames
    n_pay = (n_out + fpp - 1) // fpp
    payloads = payloads[:n_pay] if len(payloads) >= n_pay else payloads + [payloads[-1]] * (n_pay - len(payloads))
    durs = [ms] * n_pay
    last_frames = n_out - fpp * (n_pay - 1)
    durs[-1] = int(round(last_frames * 1000 / float(out_fps)))
    log(f"gpmf: placeholder rebuild -> {n_pay} payloads x {ms} ms (last {durs[-1]} ms)")
    return payloads, durs
