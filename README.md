# gopro-retime

Convert a GoPro recording (e.g. HERO12 Black, 50 fps PAL) to another frame rate (default 29.97 fps, the camera's
NTSC "30") while reproducing, byte for byte wherever possible, the file the camera itself would have written at
that rate.

The video must be re-encoded (50 → 29.97 is a 5:3 ratio; frames are dropped by nearest-timestamp decimation,
nothing is blended or interpolated). Everything else is either copied verbatim from the camera file or
regenerated with the camera's own conventions, which the tool measures on the source file rather than hard-codes:

| part of the file | what the tool does |
|---|---|
| `ftyp`, `udta` (FIRM, LENS, CAME, SETT, MUID, HMMT, BCID, GUMI, GPMF settings, `free` slots), `iods`, handler names, `dinf/alis`, `vmhd/smhd/gmhd`, sample entries (`colr`, `esds`, 24-bit `mp4a`, …), creation times | copied byte for byte from the source `moov` |
| box order, `stsc (1,1,1)` one-sample-per-chunk layout, `stss`, elst, timescales, duration relations (mvhd = max(video, audio), tkhd scaling, fdsc = video) | reproduced by the muxer; verified byte-identical on an identity remux of a real file |
| `mdat` interleave | the firmware's writer rule: V0, tmcd, then audio/video merged by decode time (audio first on ties), each GoPro MET payload written a fixed latency after its window (measured on the source), final payload after the last audio frame |
| **GoPro SOS** (`fdsc`) recovery track | regenerated: header copied (timescale field patched), descriptor per media sample (type/flags/size/duration/X, conventions learned from the source), sample #1 carries the new VPS/SPS/PPS in the source's buffer layout |
| **GoPro MET** (`gpmd`) telemetry | rebuilt for the new frame grid: per-frame streams (SHUT, ISOE, CORI, IORI, GRAV, FACE, MSKP…) pick the kept frames, every-Nth-frame streams (WBAL, WRGB, UNIF) pick the nearest sample, time-based streams (ACCL, GYRO, GPS, audio/scene streams) are re-binned into 1001 ms windows keeping each stream's own phase and delivery lag; TSMP/STMP recomputed; sticky items copied; VFPS patched |
| timecode (`tmcd`) | sample description rewritten (timescale, frame duration, `numberOfFrames = floor(fps)` like the camera), start value recomputed from the camera's RTC fields in the SOS header |
| audio | AAC frames copied verbatim (no re-encode) |
| video parameter sets | the camera's own VPS/SPS/PPS are **transplanted** into the new stream (VUI timing and HRD rate patched); x264/x265 are configured so every decode-affecting tool matches, verified by a calibration encode; every slice header is re-serialised under the camera's parameter sets (POC type/width, frame_num, reference counts, QP compensation, RPS signalling, …); SEI/filler removed; AUD kept; losslessness proven by decoding both streams and comparing MD5s |
| bit rate | the nominal GoPro Standard/High value of the target mode (table for HERO11/12), written into the SPS HRD like the camera does; GOP kept constant in seconds |

Then a forensic self-check compares the output with the source (and the reference, if given) on ~90 dimensions:
box tree, every non-table box, sample-entry bytes, handler names, interleave rule, SOS layout, parameter-set fields
(ffmpeg `trace_headers`), slice-header conventions, ffprobe/mediainfo/exiftool tag diffs, encoder fingerprints,
and the decode-identity proof.

## Requirements

* Python ≥ 3.10, `ffmpeg`/`ffprobe` with `libx264` and `libx265` (any recent build; tested with ffmpeg 6.1)
* optional, for the forensic report: `mediainfo`, `exiftool`

```
pip install -e .
```

## Usage

```
gopro-retime GX010042.MP4 GX010042_30.MP4                 # 50 fps -> 29.97 (GoPro "30"), IMU streams dropped
gopro-retime src.MP4 out.MP4 --imu keep                   # keep ACCL/GYRO/CORI/IORI/GRAV (most native-looking)
gopro-retime src.MP4 out.MP4 --reference GX010099.MP4     # native 30 fps recording from the same camera as template
gopro-retime src.MP4 out.MP4 --fps 30 --bitrate 60000000  # exact 30.000 fps (only native on cameras with Labs ALLI/24HZ), explicit rate
gopro-retime src.MP4 out.MP4 --preset slower --two-pass   # slower, better rate control
gopro-retime src.MP4 out.MP4 -x rc-lookahead=40           # extra x264/x265 parameters (calibration re-checks them)
```

`gopro-retime-inspect FILE.MP4` prints everything the converter learns from a recording (codec and parameter sets,
GOP, writer conventions, SOS layout, telemetry stream classes, udta settings) — run it on a real HERO12 file first.

Options: `--fps` (29.97 default, 30, 25, 24000/1001, …), `--imu drop|keep`, `--gps keep|drop`, `--gpmf rebuild|drop`,
`--reference FILE`, `--bitrate/--maxrate/--bufsize`, `--gop`, `--preset`, `--two-pass`, `--no-verify`,
`--no-external-tools`, `--no-transplant`, `--keep-temp`, `--workdir`, `--threads`, `-x KEY=VALUE`.

The report printed at the end lists every check as PASS/WARN/FAIL plus notes about anything that could not be
matched (e.g. a decode-affecting tool the encoder cannot reproduce).

### Use a reference recording whenever you can

A native recording at the target frame rate from the **same camera and mode** (`--reference`) is the template for
everything that depends on the frame rate but cannot be derived from the 50 fps file: the `SETT` bits, mode-dependent
Global Settings keys (SROT, …), the SOS header's mode fields, GOP length, HRD bit rate, metadata write latency.
Without it the tool uses rules verified on other GoPro files and documents every assumption in its notes.

## What is and is not achievable

Container, metadata and stream-syntax level: everything a tool reads (ffprobe, MediaInfo, ExifTool, mp4 dumpers,
GPMF parsers, GoPro's own apps) is reproduced; there is no "Lavf"/"x264"/"x265" fingerprint anywhere, no SEI, the
parameter sets are the camera's.

Signal level, which no re-encoder can hide:

* **Motion cadence**: 50 → 29.97 decimation keeps frames at intervals of 2,1,2,2,1,… source frames (40/20 ms)
  but presents them at 33.4 ms. Motion is slightly uneven compared with native 29.97 capture.
* **Motion blur**: frames were exposed with the 50 fps shutter (e.g. 1/100 s), native 29.97 capture would use
  ~1/60 s. The per-frame exposure values in the metadata (SHUT) are carried over truthfully.
* **Double compression**: the pixels were quantised twice. At GoPro's bit rates the second quantisation is fine,
  but DCT-histogram style detectors can in principle see it.
* **Rate-control statistics**: x264/x265 do not choose QPs the way GoPro's hardware encoder does.
* **Metadata**: with `--imu drop` (default, as requested) the MET track has no ACCL/GYRO/CORI/IORI/GRAV streams; a
  native HERO12 file always has them. `--imu keep` carries them over re-binned. Likewise `--gpmf drop` removes a
  track every native file has.
* **Unknowns without a reference**: SETT bits, SROT and similar mode-dependent values are copied from the 50 fps
  source; if the camera writes different values at 30 fps, only `--reference` can supply them.
* **Identity atoms**: MUID/GUMI (media unique ids) and the HiLight tags are kept from the source, so the output is
  linked to the original if both files are ever compared. The output's modification time is set to the recording's
  end like the camera does.
* **HRD SEI**: Ambarella-era GoPro streams (HERO5/Fusion/Karma) carry buffering-period/picture-timing SEI in every
  access unit; GP1 cameras (HERO6-8, MAX) write none. The tool never writes SEI; if the source has some, the report
  flags the difference.
* **Sidecars**: a camera writes `GL01xxxx.LRV` and `GX01xxxx.THM` next to every MP4; the tool does not create them.

## Development

```
python3 -m pytest -q          # ~50 tests incl. byte-exact round trips on real GoPro files and two end-to-end conversions
```

The test files are the real camera recordings shipped with GoPro's gpmf-parser repository (`GOPRO_SAMPLES` env var
points at the `samples/` directory). `tests/make_fixture.py` and `tests/make_hevc_fixture.py` turn one of them into
50 fps H.264/HEVC sources with consistent metadata, SOS and interleave.

Package layout: `mp4box` (lossless box tree), `model` (tracks/samples), `plan` (frame map), `encode`
(ffmpeg), `h26x/` (symmetric H.264/HEVC syntax coders, derivation, calibration, transplant), `samples`, `gpmf`
+ `gpmf_rebuild`, `sos`, `interleave`, `reference`, `modes`, `mux`, `verify`, `pipeline`, `cli`.
