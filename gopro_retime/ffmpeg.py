"""Thin wrappers around ffmpeg / ffprobe."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from typing import Optional

FFMPEG = os.environ.get("GOPRO_RETIME_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("GOPRO_RETIME_FFPROBE", "ffprobe")


class ToolError(RuntimeError):
    pass


def check_tools() -> None:
    for t in (FFMPEG, FFPROBE):
        if shutil.which(t) is None:
            raise ToolError(f"required tool not found on PATH: {t}")


def run(cmd: list[str], log=None, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    if log:
        log("$ " + " ".join(_q(c) for c in cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    if check and p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace") if p.stderr else ""
        raise ToolError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{err[-4000:]}")
    return p


def _q(s: str) -> str:
    if any(ch in s for ch in " '\"\\,;()[]"):
        return "'" + s.replace("'", "'\\''") + "'"
    return s


def ffprobe_json(path: str, extra: Optional[list[str]] = None) -> dict:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams"] + (extra or []) + [path]
    p = run(cmd)
    return json.loads(p.stdout.decode("utf-8", "replace"))


def ffprobe_frames(path: str, stream: str = "v:0", entries: str = "frame=pict_type,key_frame,pkt_size,pts,pkt_dts") -> list[dict]:
    cmd = [FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries", entries, "-print_format", "json", path]
    p = run(cmd)
    return json.loads(p.stdout.decode("utf-8", "replace")).get("frames", [])


def decode_md5(path: str, stream: str = "v:0", extra_in: Optional[list[str]] = None, vf: Optional[str] = None,
               pix_fmt: Optional[str] = None, log=None) -> str:
    """MD5 of the fully decoded raw video (all frames), used to prove two streams decode identically."""
    cmd = [FFMPEG, "-v", "error", "-nostdin"] + (extra_in or []) + ["-i", path, "-map", f"0:{stream}", "-an", "-sn", "-dn"]
    if vf:
        cmd += ["-vf", vf]
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    if log:
        log("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h = hashlib.md5()
    nbytes = 0
    assert p.stdout is not None
    while True:
        chunk = p.stdout.read(1 << 20)
        if not chunk:
            break
        h.update(chunk)
        nbytes += len(chunk)
    err = p.stderr.read() if p.stderr else b""
    rc = p.wait()
    if rc != 0:
        raise ToolError(f"decode failed: {err.decode('utf-8', 'replace')[-2000:]}")
    return f"{h.hexdigest()}:{nbytes}"


def decode_frame_md5s(path: str, stream: str = "v:0", extra_in: Optional[list[str]] = None, vf: Optional[str] = None) -> list[str]:
    """Per-frame MD5 list via the framemd5 muxer."""
    cmd = [FFMPEG, "-v", "error", "-nostdin"] + (extra_in or []) + ["-i", path, "-map", f"0:{stream}", "-an", "-sn", "-dn"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-fps_mode", "passthrough", "-f", "framemd5", "-"]
    p = run(cmd)
    out = []
    for line in p.stdout.decode().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        out.append(line.split(",")[-1].strip())
    return out
