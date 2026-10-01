"""Command line interface."""
from __future__ import annotations

import argparse
import sys

from .pipeline import Options, run


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gopro-retime",
                                description="Convert a GoPro recording to another frame rate while reproducing the camera's native file structure.")
    p.add_argument("src", help="source GoPro MP4 (e.g. 50 fps)")
    p.add_argument("out", help="output MP4")
    p.add_argument("--fps", default="29.97", help="output frame rate: 29.97 (GoPro NTSC '30', default), 30, 25, 24000/1001, ...")
    p.add_argument("--imu", choices=["drop", "keep"], default="drop", help="drop (default) or keep IMU streams (ACCL/GYRO/GRAV/CORI/IORI) in the metadata track")
    p.add_argument("--gps", choices=["keep", "drop"], default="keep")
    p.add_argument("--gpmf", choices=["rebuild", "drop"], default="rebuild", help="rebuild (default) the GoPro MET track for the new timeline, or drop it entirely")
    p.add_argument("--reference", help="a NATIVE recording at the target frame rate from the same camera: used as the template for frame-rate-dependent conventions")
    p.add_argument("--bitrate", type=int, help="target video bitrate in bps (default: derived from the source)")
    p.add_argument("--maxrate", type=int)
    p.add_argument("--bufsize", type=int)
    p.add_argument("--gop", type=int, help="keyframe interval in frames (default: from reference or source)")
    p.add_argument("--preset", default="slow", help="x264/x265 preset (speed/quality)")
    p.add_argument("--two-pass", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--no-external-tools", action="store_true", help="skip mediainfo/exiftool comparisons in the report")
    p.add_argument("--no-transplant", action="store_true", help="do not rewrite the encoder's parameter sets/slice headers to the camera's")
    p.add_argument("--keep-temp", action="store_true")
    p.add_argument("--workdir")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("-x", "--encoder-param", action="append", default=[], metavar="KEY=VALUE",
                   help="extra x264/x265 parameter (repeatable), e.g. -x psy-rd=1.0 -x rc-lookahead=40")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    opts = Options(src=a.src, out=a.out, fps=a.fps, imu=a.imu, gps=a.gps, gpmf=a.gpmf, reference=a.reference,
                   bitrate=a.bitrate, maxrate=a.maxrate, bufsize=a.bufsize, gop=a.gop, preset=a.preset, two_pass=a.two_pass,
                   verify=not a.no_verify, external_tools=not a.no_external_tools, keep_temp=a.keep_temp, workdir=a.workdir,
                   no_transplant=a.no_transplant, threads=a.threads, encoder_params=a.encoder_param)
    log = (lambda s: None) if a.quiet else (lambda s: print(s, file=sys.stderr, flush=True))
    res = run(opts, log=log)
    for n in res.notes:
        print(f"note: {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
