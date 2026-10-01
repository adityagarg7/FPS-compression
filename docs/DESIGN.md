# Design notes and verified GoPro file conventions

This document records what the tool relies on and where each fact comes from. "Verified" means checked byte for
byte on the ten firmware-written GoPro files shipped with gpmf-parser (HERO5/6/7/8, MAX, Fusion, Karma) or on the
other real artefacts listed below; "reported" means quoted from GoPro documentation or community dumps that could not
be re-verified; "inferred" means a rule extrapolated from the evidence. Anything the tool can measure on the source
file it measures instead of assuming.

Evidence available during development: the gpmf-parser sample files; a raw HERO11 Black PAL metadata track; an
ffprobe dump of a 2021 GoPro 1080p59.94 file; an exiftool dump of a HERO9 4K 23.976 HEVC file; the HERO12 firmware
image strings (muxer library "gopro-lib-quicktime64"); GoPro Labs documentation; the gpmf-parser specification.
No complete HERO11/12 MP4 was obtainable from inside the build environment.

## Container (verified)

* `ftyp` 20 bytes: `mp41`, minor 0x20131018, compatible `mp41`. Layout `ftyp mdat moov`, moov last, no top-level
  `free`. `mdat` gap-free: every byte belongs to exactly one sample.
* `moov` children: `mvhd udta iods trak(video) trak(audio) trak(tmcd) trak(gpmd) trak(fdsc)`; track ids 1..5,
  `next_track_id 6`. A Bluetooth-mic recording may add a "GoPro BT AAC" audio track (firmware strings).
* All `mvhd/tkhd/mdhd` version 0, creation == modification time (camera RTC written as if UTC), identical in every box.
* `tkhd` flags 0xF, alternate_group 0, audio volume 0x0064, tmcd width = video width x 16 (fixed point).
* Track child order: video `tkhd tref edts mdia`; audio `tkhd tref mdia`; others `tkhd mdia`. `tref/tmcd` → track 3
  on video and audio. `edts/elst` only on video: one entry, segment = video duration, media_time 0, rate 1.0.
* `mdhd` language 0. `hdlr`: `mhlr` + handler + 12 zero bytes + Pascal string name (`\x0bGoPro AVC  ` with two
  trailing spaces; `GoPro H.265` for HEVC; `GoPro AAC  `, `GoPro TCD  `, `GoPro MET  `, `GoPro SOS  `), no NUL.
* `dinf/dref`: one `alis` entry, flags 1. `vmhd/smhd/gmhd` as in the samples; tmcd `gmhd` has `gmin + tmcd/tcmi`
  (Helvetica, size 10); gpmd/fdsc `gmhd` has `gmin + gpmd|fdsc`.
* `stbl`: `stsd stts stsc stsz stco stss` (video) / `stsd stts stsc stsz stco` (others). `stsc` is always the single
  entry (1,1,1): every sample is its own chunk. `stsz` always a full table. No `ctts`, `sdtp`, `sgpd`, `btrt`, `pasp`.
* Video `avc1`/`hvc1`: 72 dpi, frame_count 1, compressorname Pascal `GoPro AVC encoder` / `GoPro H.265 encoder`,
  depth 24, pre_defined -1, children `colr` (nclx 1/1/1, full_range **0** although the SPS says full range) then
  `avcC`/`hvcC`.
* Audio `mp4a`: version 0, 2 channels, samplesize **24**, compression_id -2, 48000 Hz; `esds` with 3-byte expandable
  lengths, max/avg bitrate fields 48000, AudioSpecificConfig `11 90 00 00 00`. Real AAC rate ≈ 189 kbps, 1024-sample
  frames, first two frames are 11-byte silence.
* `udta` children: `©xyz(30)` or `free(30)` (GPS fix or not), `FIRM(23) LENS(56) CAME(24) SETT(20) MUID(40) HMMT(412)
  BCID(44) GUMI(24) GPMF(25608, fixed size, zero padded) free(132)`.

## Timing (verified unless noted)

| rate | video/tmcd/fdsc/mvhd timescale | frame duration | source |
|---|---|---|---|
| 29.97 | 90000 | 3003 | 7 sample files |
| 23.976 | 24000 | 1001 | 3 sample files + HERO9 dump |
| 59.94 | 60000 | 1001 | 2021 ffprobe dump |
| 25 | 90000 | 3600 | reported (untrunc issues) |
| 50 | 90000 | 1800 | inferred (same rule) |

Rule implemented: 90000 whenever the frame duration is an integer number of 90 kHz ticks, else the nominal
numerator with 1001-tick frames.

* `mvhd.duration = max(video tkhd, audio tkhd)` (gpmd ignored even when longer). `tkhd = mdhd x mvhd_ts / mdhd_ts`.
* tmcd: `stsd` flags 0x2 (24 h, non-drop), timescale/frame duration = video, `numberOfFrames = floor(fps)`
  (29 at 29.97, 23 at 23.976, 59 at 59.94). Sample = `floor((seconds since local midnight + ms/1000) x fps)` using the
  RTC fields stored in the SOS header (within jitter of the real files, exact on HD6+/MAX in most cases).
* GoPro MET payloads: 1001 ms (30 frames) at 29.97, 1001 ms (24 frames) at 23.976, 1001 ms (60 frames) at 59.94;
  **PAL 25 fps: 1040 ms = 26 frames** (verified on the HERO11 track). The tool measures the source period and
  frames-per-payload directly (`stts` + frame rate). HD8+ firmware writes a partial final payload:
  `total_ms = floor((n_frames + x) x frame_ms)` with x solved on the source (hero8: 1; MAX 360: -11).
* Audio ends 0–22 ms after the video on HD8+ files (audio frames are copied as they are).

## mdat interleave (verified on 7/10 files; hero7 and karma firmware anomalies excepted)

`V0` first, the single `tmcd` sample second, then audio and video merged by decode time with audio first on exact
ties. MET payload k is written immediately before the first video frame whose dts ≥ window_end_k + latency;
latency measured on the source (HD8/MAX: ~117 ms = 4 frames at 29.97, 3 at 23.976; HD5/6: 0). The final partial
payload is written right after the last audio frame (HD8+). Every media sample is immediately preceded by its SOS
descriptor.

## GoPro SOS track (verified, regenerates byte-exactly on all 10 files)

* `fdsc` samples: #0 = `GPRO` header struct (firmware-specific layout; copied; only the video timescale field is
  patched, located by the little-endian pair `(1, timescale)`); #1 = descriptor of the first video frame, type 3,
  extended with `u32 len + NAL` blocks for each parameter set in fixed-size buffers (learned from the source);
  then 16-byte descriptors `"GP" type flags size duration X` (type 0 video, 4 audio, 5 tmcd, 6 gpmd; video flags 1
  sync / 3 non-sync; tmcd duration = video frame duration; gpmd duration = payload duration; X = video frame duration
  on HD6+). `stts (n, 0)`, `tkhd/mdhd` durations = video's.
* The header's tail block holds the unix creation time, seconds since midnight and milliseconds (RTC) used for the
  timecode.

## GoPro MET rebuild (verified by reproducing the real payloads of hero8/MAX/Fusion/hero5/hero6 from themselves)

Streams are classified by measurement: per-frame (count = frames per window, STMP deltas frame-locked), every-k-th
frame, grouped (several items per STRM: FACE/SCEN/HUES/DISP), or time-based. Per-frame streams take the kept frames'
samples with the source's delivery lag converted to output frames; stride streams take the nearest sample; time-based
streams keep their fractional position inside their source window and are re-binned into the output windows (exact
identity when windows coincide). STMP = stream T0 + floor(frame x frame_us) for frame-locked streams, interpolated
time for time-based ones; TSMP cumulative. Sticky items (STNM, SIUN, SCAL, TYPE, MTRX, ORIN, ORIO, TMPC, GPSU, VPTS…)
are copied from the overlapping source payload. Nested sizes use ssize 1 / repeat bytes; pads zero.
`--imu drop` removes ACCL, GYRO, MAGN, GRAV, CORI, IORI; `--gps drop` removes GPS5/GPS9 and their sticky items.
`VFPS` in the udta Global Settings is rewritten in place (same length); `ORDP` set to N when orientation is dropped.

## Video stream (verified on the H.264 samples; HEVC specifics must come from the source file)

* Samples: AUD (primary_pic_type 0 for I, 1 for P) + one slice; no SEI, no in-band parameter sets, no B-frames,
  closed GOP, all keyframes IDR. SPS carries NAL + VCL HRD with the nominal mode bit rate (`bit_rate_value` rounded
  down to the HRD scale), VUI full range 1, colour 1/1/1, timing 1001/60000 (H.264 field rate), fixed_frame_rate 1,
  no bitstream_restriction. nal_ref_idc 1 everywhere. poc type 0 with 4-bit lsb = 2 x frame_num; idr_pic_id
  alternates.
* The tool derives the encoder configuration from the source parameter sets (decode-affecting tools), test-encodes
  a tiny clip to calibrate hidden encoder behaviour (x264 shifts chroma_qp_index_offset with psy-rd; x265 forces
  cu_qp_delta on with VBV, …), encodes, then transplants the camera's VPS/SPS/PPS (timing + HRD patched) and
  rewrites every slice header. The result is accepted only if its decode is bit-identical to the encoder's stream.
* Nominal bit rates per mode (HERO11, reported identical on HERO12) are in `modes.py`; the HRD and the rate target of
  the output use the 29.97 variant of the source's mode with the same Standard/High setting.

## Known limits

Signal-level differences (cadence, shutter, double compression, rate-control statistics) are inherent to any
re-encode. Mode-dependent values that only a native recording at the target rate can supply (SETT bits, SROT and
other sensor-mode keys, exact GOP, HRD values of the 30 fps mode) default to the source's or to the tables above;
pass `--reference` to take them from a real recording. x265 cannot produce tiles, PCM, or `cabac_init_flag=1`
slices; if a camera stream uses them the transplant is refused and the encoder's own parameter sets are kept (the
report says so).
