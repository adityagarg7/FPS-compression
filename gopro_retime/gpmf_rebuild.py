"""Rebuild the GoPro MET (gpmd) payloads for a new frame grid.

Every stream is classified by MEASUREMENT on the source (nothing is hard-coded per camera):
  per_frame : one sample per video frame            -> the kept frames' samples
  stride k  : one sample every k frames             -> nearest source sample for each output frame multiple of k
  timed     : fixed-rate / aperiodic (IMU, audio, GPS, scene) -> re-binned by reconstructed timestamps into the new windows
  grouped   : several items with the data key per STRM (FACE/SCEN/HUES/DISP) -> each item is one sample of its class
Sticky items (STNM/SIUN/SCAL/TYPE/MTRX/.../TMPC/GPSU/VPTS...) are copied from the overlapping source payload; STMP/TSMP are
recomputed; delivery lag and stream phase are measured on the source and converted to the output frame rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from . import gpmf
from .model import SourceFile
from .plan import FramePlan, frames_per_gpmf_payload

IMU_KEYS = {b"ACCL", b"GYRO", b"MAGN", b"GRAV", b"CORI", b"IORI"}
GPS_KEYS = {b"GPS5", b"GPS9", b"GPSU", b"GPSF", b"GPSP", b"GPSA"}
META_KEYS = {b"STMP", b"TSMP", b"STNM", b"SIUN", b"UNIT", b"SCAL", b"TYPE", b"MTRX", b"ORIN", b"ORIO", b"TMPC", b"GPSF",
             b"GPSU", b"GPSP", b"GPSA", b"VPTS", b"TIMO", b"EMPT", b"RMRK", b"TICK", b"TOCK", b"DVID", b"DVNM", b"STPS"}


@dataclass
class Sample:
    t: Optional[Fraction]      # reconstructed time in microseconds (timed streams)
    ssize: int
    repeat: int
    data: bytes
    item: Optional[gpmf.KLV] = None   # original item for grouped/opaque samples


@dataclass
class Stream:
    index: int
    key: Optional[bytes]
    grouped: bool
    cls: str = "timed"
    stride: int = 1
    lag_src_frames: int = 0
    extra_tail: int = 0
    samples: list[Sample] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)       # per source payload (instances)
    stmps: list[Optional[int]] = field(default_factory=list)
    has_stmp: bool = False
    phase_us: Optional[Fraction] = None                   # B0 for timed streams
    t0: Optional[int] = None                              # first STMP (frame-locked streams)
    name: bytes = b""
    stmp_per_frame: Optional[Fraction] = None             # measured STMP units per video frame (frame-locked streams)


def _strm_data_key(strm: gpmf.KLV) -> tuple[Optional[bytes], bool]:
    """(data key, grouped). Data key = key of the last item unless it is metadata."""
    items = strm.children or []
    if not items:
        return None, False
    last = items[-1]
    if last.key in META_KEYS:
        return None, False
    same = [c for c in items if c.key == last.key]
    grouped = len(same) > 1 or last.type == ord("#") or (last.repeat == 0 and last.type == ord("?"))
    return last.key, grouped


def _stream_name(strm: gpmf.KLV) -> bytes:
    n = strm.child("STNM")
    return n.data if n is not None else b""


def analyze(payloads: list[list[gpmf.KLV]], src_frames: int, frames_per_payload: int, period_us: int,
            fps: Fraction, device: int = 0, covered_frames: Optional[int] = None) -> list[Stream]:
    """Flatten and classify every stream of the source (device = index of the DEVC within each payload).
    covered_frames: video frames spanned by the metadata track (older firmware stops the track before the video ends)."""
    devcs = [p[device] if device < len(p) else gpmf.make_nested(b"DEVC", []) for p in payloads]
    covered = covered_frames if covered_frames is not None else src_frames
    # streams identified by (data key, STNM) in order of first appearance
    streams: list[Stream] = []
    by_id: dict[tuple, Stream] = {}
    # pass 1: stream identities and whether the stream is grouped in ANY payload
    for devc in devcs:
        for strm in devc.children_of("STRM"):
            key, grouped = _strm_data_key(strm)
            ident = (key, _stream_name(strm))
            st = by_id.get(ident)
            if st is None:
                st = Stream(len(streams), key, grouped, name=_stream_name(strm))
                by_id[ident] = st
                streams.append(st)
            st.grouped = st.grouped or grouped
    # pass 2: samples per payload
    for pi, devc in enumerate(devcs):
        seen: set[int] = set()
        for strm in devc.children_of("STRM"):
            key, _g = _strm_data_key(strm)
            st = by_id[(key, _stream_name(strm))]
            if st.index in seen:
                continue
            seen.add(st.index)
            stmp = strm.child("STMP")
            st.stmps.append(int.from_bytes(stmp.data[:8], "big") if stmp is not None else None)
            st.has_stmp |= stmp is not None
            cnt = 0
            if key is not None:
                items = [c for c in strm.children or [] if c.key == key]
                if st.grouped:
                    for it in items:
                        st.samples.append(Sample(None, it.size, it.repeat, it.data, it))
                        cnt += 1
                else:
                    for it in items:
                        for k in range(it.repeat):
                            st.samples.append(Sample(None, it.size, 1, it.data[k * it.size:(k + 1) * it.size]))
                        cnt += it.repeat
            st.counts.append(cnt)
        for st in streams:
            if st.index not in seen:
                st.counts.append(0); st.stmps.append(None)
    n_pay = len(payloads)
    # STMP units per frame, measured on the per-frame candidates (firmware clocks differ: true µs, or 1e6 per payload)
    full_idx = list(range(1, n_pay - 1)) if n_pay > 2 else []
    frame_deltas: list[Fraction] = []
    for st in streams:
        total = sum(st.counts)
        full = [st.counts[i] for i in full_idx] if full_idx else st.counts[:1]
        median = sorted(full)[len(full) // 2] if full else 0
        if st.key and total and abs(total - covered) <= 2 and abs(median - frames_per_payload) <= 1 and st.has_stmp:
            for i in full_idx:
                a, b = st.stmps[i], st.stmps[i + 1] if i + 1 < n_pay else None
                if a is not None and b is not None and st.counts[i]:
                    frame_deltas.append(Fraction(b - a, frames_per_payload))
    step = sorted(frame_deltas)[len(frame_deltas) // 2] if frame_deltas else Fraction(period_us, frames_per_payload)
    for st in streams:
        total = sum(st.counts)
        if st.key is None or total == 0:
            st.cls = "empty"
            continue
        full = [st.counts[i] for i in full_idx] if full_idx else st.counts[:1]
        median = sorted(full)[len(full) // 2] if full else st.counts[0]
        aligned = _stmp_frame_aligned(st, step, frames_per_payload)
        st.stmp_per_frame = step
        if aligned and abs(total - covered) <= 2 and abs(median - frames_per_payload) <= 1:
            st.cls, st.stride = "per_frame", 1
        else:
            st.cls = "timed"
            if aligned:
                for k in range(2, 9):
                    exp = frames_per_payload / k
                    if abs(median - exp) < 1 and abs(total - covered / k) <= 2:
                        st.cls, st.stride = "stride", k
                        break
        if st.cls in ("per_frame", "stride"):
            first = st.counts[0]
            lag = frames_per_payload - st.stride * first
            st.lag_src_frames = lag if 0 <= lag < frames_per_payload else 0
            # +1/-1 policies are measured against the frames the metadata covers (the output covers its own span)
            st.extra_tail = total - (covered // st.stride if st.stride > 1 else covered)
            st.t0 = st.stmps[0]
        else:
            _reconstruct_times(st, period_us)
    return streams


def _window_relative_times(st: Stream, src_durs_ms: list[int]) -> list[Fraction]:
    """tau[k] (µs): the sample's position on the metadata timeline, interpolated inside its SOURCE window by index.
    Reproduces the firmware's own binning exactly when the output window equals the source window."""
    out: list[Fraction] = []
    start = Fraction(0)
    for p, c in enumerate(st.counts):
        dur_us = src_durs_ms[p] * 1000
        for k in range(c):
            out.append(start + Fraction(dur_us * k, c))
        start += dur_us
    return out


def _stmp_frame_aligned(st: Stream, step: Fraction, frames_per_payload: int, tol_us: int = 50) -> bool:
    """True when every payload STMP sits on the stream's frame grid (T0 + n*step, n close to a payload boundary) — or
    when the stream carries no STMP. Audio-clock streams (10 Hz on a 1 s clock) fail this by ~1 ms per payload."""
    vals = [(i, v) for i, v in enumerate(st.stmps) if v is not None]
    if len(vals) < 2 or step <= 0:
        return True
    t0 = vals[0][1]
    for i, v in vals[1:]:
        n = (v - t0) / step
        k = round(n)
        if k <= 0 or abs((v - t0) - k * step) > tol_us:
            return False
        # the first sample of payload i must belong to a frame near the payload's first frame (delivery lag < 1 payload)
        if abs(k - i * frames_per_payload) > frames_per_payload:
            return False
    return True


def _reconstruct_times(st: Stream, period_us: int) -> None:
    """t[k] by linear interpolation between consecutive payload STMPs (the way every GPMF reader does it)."""
    first_idx = []
    acc = 0
    for c in st.counts:
        first_idx.append(acc); acc += c
    n = len(st.counts)
    periods: list[Optional[Fraction]] = [None] * n
    for p in range(n):
        if st.counts[p] == 0:
            continue
        s0 = st.stmps[p]
        nxt = next((q for q in range(p + 1, n) if st.stmps[q] is not None and st.counts[q] > 0), None)
        if s0 is not None and nxt is not None and st.stmps[nxt] is not None:
            periods[p] = Fraction(st.stmps[nxt] - s0, sum(st.counts[p:nxt]))
    last_period = next((periods[p] for p in range(n - 1, -1, -1) if periods[p] is not None), None)
    for p in range(n):
        c = st.counts[p]
        if c == 0:
            continue
        s0 = st.stmps[p]
        per = periods[p] if periods[p] is not None else last_period
        if s0 is None or per is None:
            # no clock: spread uniformly over the payload window
            s0_f = Fraction(p * period_us)
            per = Fraction(period_us, c)
        else:
            s0_f = Fraction(s0)
        for k in range(c):
            st.samples[first_idx[p] + k].t = s0_f + per * k


def _measure_phase(st: Stream, period_us: int) -> Fraction:
    """Window phase B0 (µs) such that source payload p holds samples with B0 + p*period <= t < B0 + (p+1)*period."""
    los, his = [], []
    acc = 0
    for p, c in enumerate(st.counts):
        if p >= 1 and c > 0 and acc > 0:
            t_first = st.samples[acc].t
            t_prev = st.samples[acc - 1].t
            if t_first is not None and t_prev is not None:
                his.append(t_first - p * period_us)
                los.append(t_prev - p * period_us)
        acc += c
    if not his:
        return Fraction(0)
    lo, hi = max(los), min(his)
    if lo < hi:
        return (lo + hi) / 2
    his_s, los_s = sorted(his), sorted(los)
    return (his_s[len(his_s) // 2] + los_s[len(los_s) // 2]) / 2


# ---- output -------------------------------------------------------------------------------------------------
def _output_durations(src: SourceFile, n_out: int, out_fps: Fraction) -> tuple[list[int], int]:
    """Payload durations (ms) for the output, following the source firmware's rule (partial final payload or not)."""
    gp = src.track("gpmd")
    fpp, period_ms = frames_per_gpmf_payload(out_fps)
    frame_ms_o = Fraction(period_ms, fpp)
    src_durs = [s.duration for s in gp.samples]
    src_period = max(set(src_durs), key=src_durs.count)
    partial_last = src_durs[-1] != src_period
    audio = src.track("audio")
    if partial_last and audio is not None and audio.media_duration * 1000 // audio.timescale == sum(src_durs):
        # newer firmware: the MET track ends with the audio (audio is copied verbatim, so the total is unchanged)
        total = sum(src_durs)
        n_pay = (total + period_ms - 1) // period_ms
        durs = [period_ms] * n_pay
        durs[-1] = total - period_ms * (n_pay - 1)
        return durs, fpp
    if partial_last:
        n_src = src.video.sample_count
        frame_ms_s = Fraction(src_period) / round(src_period * src.video_frame_rate() / 1000)
        total_src = sum(src_durs)
        x0 = round(Fraction(total_src) / frame_ms_s) - n_src
        lagfix = next((x for x in sorted(range(x0 - 2, x0 + 3), key=lambda v: abs(v - x0)) if int((n_src + x) * frame_ms_s) == total_src), 1)
        # the offset is a time quantity of the firmware (stop-event drain), so convert it to output frames
        lag_out = round(Fraction(lagfix) / src.video_frame_rate() * out_fps)
        total = int((n_out + lag_out) * frame_ms_o)
        n_pay = (total + period_ms - 1) // period_ms
        durs = [period_ms] * n_pay
        durs[-1] = total - period_ms * (n_pay - 1)
    else:
        total_video_ms = Fraction(n_out, 1) * frame_ms_o
        n_pay = max(1, int(total_video_ms // period_ms))
        durs = [period_ms] * n_pay
    return durs, fpp


def rebuild(src: SourceFile, plan: FramePlan, out_fps: Fraction, drop_imu: bool, drop_gps: bool,
            reference: Optional[SourceFile] = None, log=print) -> tuple[list[bytes], list[int]]:
    gp = src.track("gpmd")
    raw = src.read_samples(gp)
    payloads = [gpmf.parse(b) for b in raw]
    if not payloads or not payloads[0] or payloads[0][0].key != b"DEVC":
        raise ValueError("gpmd payloads do not start with DEVC")
    src_durs = [s.duration for s in gp.samples]
    src_period_ms = max(set(src_durs), key=src_durs.count)
    src_period_us = src_period_ms * 1000
    fps_s, fps_o = plan.src_fps, plan.out_fps
    f_s = round(Fraction(src_period_ms, 1000) * fps_s)          # frames per source payload (50 / 30)
    n_devices = max(len(p) for p in payloads)
    durs, f_o = _output_durations(src, plan.out_frames, out_fps)
    per_device = [_rebuild_device(src, plan, out_fps, payloads, d, src_durs, src_period_ms, src_period_us, f_s, durs, f_o,
                                  drop_imu, drop_gps, log) for d in range(n_devices)]
    out_payloads = [b"".join(dev[j] for dev in per_device if j < len(dev)) for j in range(len(durs))]
    return out_payloads, durs


def _rebuild_device(src: SourceFile, plan: FramePlan, out_fps: Fraction, payloads: list[list[gpmf.KLV]], device: int,
                    src_durs: list[int], src_period_ms: int, src_period_us: int, f_s: int, durs: list[int], f_o: int,
                    drop_imu: bool, drop_gps: bool, log) -> list[bytes]:
    fps_s, fps_o = plan.src_fps, plan.out_fps
    covered = min(plan.src_frames, round(sum(src_durs) * fps_s / 1000))
    streams = analyze(payloads, plan.src_frames, f_s, src_period_us, fps_s, device, covered_frames=covered)
    period_o_ms = durs[0]
    period_o_us = period_o_ms * 1000
    frame_us_o = Fraction(period_o_us, f_o)
    n_pay = len(durs)
    n_out = plan.out_frames
    frame_map = plan.frame_map
    drop_keys = (IMU_KEYS if drop_imu else set()) | (GPS_KEYS if drop_gps else set())

    def lag_out(st: Stream) -> int:
        return round(Fraction(st.lag_src_frames) / fps_s * fps_o)

    # --- per-stream output sample lists: list of per-payload lists of Sample ---
    per_stream_out: dict[int, list[list[Sample]]] = {}
    per_stream_stmp: dict[int, list[Optional[int]]] = {}
    for st in streams:
        buckets: list[list[Sample]] = [[] for _ in range(n_pay)]
        stmps: list[Optional[int]] = [None] * n_pay
        if st.cls in ("per_frame", "stride"):
            T = len(st.samples)
            k = st.stride
            lag = lag_out(st)
            n_target = n_out // k if k > 1 else n_out
            n_target += st.extra_tail if st.extra_tail > 0 else 0
            if st.extra_tail < 0:
                n_target = max(0, n_target + st.extra_tail)
            first_frame: list[Optional[int]] = [None] * n_pay
            for m in range(n_target):
                out_frame = m * k
                if out_frame < n_out:
                    src_frame = frame_map[out_frame]
                    idx = min(T - 1, round(src_frame / k) if k > 1 else src_frame)
                else:
                    idx = T - 1
                j = min(n_pay - 1, (out_frame + lag) // f_o)
                buckets[j].append(st.samples[idx])
                if first_frame[j] is None:
                    first_frame[j] = out_frame
            if st.has_stmp and st.t0 is not None:
                # STMP step per output frame in the firmware's own units: (units per source ms) * output window / frames
                units_per_ms = (st.stmp_per_frame * f_s / src_period_ms) if st.stmp_per_frame else Fraction(1000)
                step_o = units_per_ms * period_o_ms / f_o
                for j in range(n_pay):
                    ff = first_frame[j]
                    stmps[j] = st.t0 + int(ff * step_o) if ff is not None else st.t0 + int(j * units_per_ms * period_o_ms)
        elif st.cls == "timed":
            taus = _window_relative_times(st, src_durs)
            bounds = []
            acc = 0
            for d in durs:
                bounds.append(acc); acc += d * 1000
            total_us = acc
            for s, tau in zip(st.samples, taus):
                if tau >= total_us:
                    continue
                j = min(n_pay - 1, int(tau // period_o_us))
                buckets[j].append(s)
            if st.has_stmp:
                for j in range(n_pay):
                    if buckets[j] and buckets[j][0].t is not None:
                        stmps[j] = int(round(buckets[j][0].t))
                    elif buckets[j]:
                        stmps[j] = None
                    else:
                        prev = next((stmps[q] for q in range(j - 1, -1, -1) if stmps[q] is not None), 0)
                        stmps[j] = prev + period_o_us
        per_stream_out[st.index] = buckets
        per_stream_stmp[st.index] = stmps

    # --- assemble payloads ---
    out_payloads: list[bytes] = []
    cum: dict[int, int] = {st.index: 0 for st in streams}
    n_src_pay = len(payloads)
    for j in range(n_pay):
        t_first_ms = Fraction(j * period_o_ms)
        p_src = min(n_src_pay - 1, int(t_first_ms // src_period_ms))
        if device >= len(payloads[p_src]):
            out_payloads.append(b"")
            continue
        devc_src = payloads[p_src][device]
        new_children: list[gpmf.KLV] = []
        strm_i = 0
        for c in devc_src.children or []:
            if c.key != b"STRM":
                new_children.append(c)
                continue
            key, _g = _strm_data_key(c)
            st = next((s for s in streams if s.key == key and s.name == _stream_name(c)), None)
            strm_i += 1
            if st is None:
                new_children.append(c)
                continue
            if key is not None and key in drop_keys:
                continue
            if st.cls == "empty":
                new_children.append(c)
                continue
            samples = per_stream_out[st.index][j]
            cum[st.index] += len(samples)
            items: list[gpmf.KLV] = []
            data_written = False
            for it in c.children or []:
                if it.key == b"STMP":
                    v = per_stream_stmp[st.index][j]
                    items.append(gpmf.KLV(b"STMP", it.type, it.size, 1, (v if v is not None else 0).to_bytes(8, "big")))
                elif it.key == b"TSMP":
                    items.append(gpmf.KLV(b"TSMP", it.type, it.size, 1, cum[st.index].to_bytes(4, "big")))
                elif it.key == key:
                    if data_written:
                        continue
                    data_written = True
                    if st.grouped:
                        for s in samples:
                            items.append(gpmf.KLV(key, s.item.type, s.item.size, s.item.repeat, s.item.data, None, b""))
                        if not samples:
                            items.append(gpmf.KLV(key, it.type, it.size, 0, b""))
                    else:
                        data = b"".join(s.data for s in samples)
                        items.append(gpmf.KLV(key, it.type, it.size, len(samples), data))
                else:
                    items.append(it)
            new_children.append(gpmf.make_nested(b"STRM", items))
        devc = gpmf.make_nested(b"DEVC", new_children)
        out_payloads.append(gpmf.serialize([devc]))
    kinds = {}
    for st in streams:
        kinds.setdefault(st.cls, []).append((st.key or b"?").decode("latin1"))
    log(f"gpmf: device {device}: {len(streams)} streams {kinds}; {n_pay} payloads x {period_o_ms} ms (last {durs[-1]} ms)"
        + (f"; dropped {sorted(k.decode() for k in drop_keys & {s.key for s in streams if s.key})}" if drop_keys else ""))
    return out_payloads


def patch_global_settings_fps(udta_gpmf: bytes, out_fps: Fraction, orientation_dropped: bool, log=print) -> bytes:
    """Rewrite VFPS (if present) to the output rate in place, keeping the byte length; ORDP->'N' when orientation dropped."""
    klvs, trailing = gpmf.parse_with_trailing(udta_gpmf)
    changed = []
    for devc in klvs:
        for it in devc.children or []:
            if it.key == b"VFPS" and it.repeat == 2 and it.size == 4:
                it.data = out_fps.numerator.to_bytes(4, "big") + out_fps.denominator.to_bytes(4, "big")
                changed.append("VFPS")
            elif it.key == b"ORDP" and orientation_dropped and it.type == ord("c"):
                it.data = b"N".ljust(len(it.data), b"\x00")
                changed.append("ORDP")
    if not changed:
        return udta_gpmf
    body = gpmf.serialize(klvs) + trailing
    if len(body) != len(udta_gpmf):
        log("gpmf: Global Settings re-serialisation changed length; leaving udta GPMF untouched")
        return udta_gpmf
    log(f"gpmf: udta Global Settings patched: {changed}")
    return body
