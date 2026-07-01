# Mosaic tool instructions
```
Usage: mosaic-tool create-mosaic [OPTIONS]

Options:
  --result-dir DIRECTORY          Directory in which all the intended UWS
                                  results of the job will be deposited.
                                  Defaults to '.', the current working
                                  directory. The tool will create a symlink
                                  'result' in that directory pointing to the
                                  'principal' result  [default: .]
  --err-dir DIRECTORY             Directory in which an error document for the
                                  job, if any, will be deposited by the tool.
                                  Defaults to “.”, the current working
                                  directory.  [default: .]
  --runid TEXT                    Allows specifying a UWS 'runid', essentially
                                  a job name, which may be used in log
                                  messages or other metadata. Defaults to a
                                  short string indicating the specific science
                                  tool being run (e.g., 'mosaic') followed by
                                  an ISO-8601 timestamp, e.g.,
                                  'mosaic-20250307T172132'. This is not the
                                  same as the UWS 'Job Identifier' passed via
                                  --jobid. [optional]
  --jobid TEXT                    Allows the UWS system to pass in the opaque
                                  UWS 'Job Identifier' by which the job is
                                  referenceable in the REST API. Required; no
                                  default is supported. This is not the same
                                  as the UWS 'Run ID' which is set by the user
                                  and passed to the tool via the --runid
                                  option.  [required]
  --tmp-dir DIRECTORY             Directory available for use as temporary
                                  scratch space for the job. Defaults to “.”,
                                  the current working directory. When
                                  executing in the UWS system, tmp-dir should
                                  never be the same as result-dir, unless it's
                                  deliberately intended for temporary files to
                                  be returned to the user as results.
                                  [default: .]
  --log-dir DIRECTORY             Directory into which any log files from the
                                  job will be deposited. Defaults to “.”, the
                                  current working directory. When executing in
                                  the UWS system, log-dir should never be the
                                  same as result-dir, unless it's deliberately
                                  intended for any logs to be returned to the
                                  user as results.
  --fake [normal|error-short|error-detail|error-signal]
                                  Run a simulated job. Modes: "normal" =
                                  simulate successful run, "error-short" =
                                  create errorSummary.yaml, "error-detail" =
                                  create errorSummary.yaml and symlink "error"
                                  to actual error file, "error-signal" = exit
                                  with SIGINT. [optional flag]
  --center TEXT                   Center of a rectangular mosaic in TAN
                                  projection. Values are in ICRS degrees, with
                                  the usual con straints: ra in [0, 360) and
                                  dec in [-90, 90]. The center is placed in
                                  the center of a pixel, so the mosaic will
                                  always have an odd number of rows and
                                  columns. Example: --center=6.2001,-14.8314.
                                  [required]
  --size TEXT                     Sizes of the row and column “diameters” of
                                  the mosaic. Values are in relative degrees
                                  in the rotated frame controlled by the
                                  --rotation option. The precise definition,
                                  which matters more near the poles, and for
                                  larger sizes, is that the limits of the
                                  image are determined as follows: go out
                                  along the central diameters of the mosaic in
                                  the row/x (or column/y) directions from the
                                  requested center, in the TAN projection, one
                                  pixel at a time, until the outer edge of a
                                  pixel is more than half of the requested
                                  size away from the center. This does not in
                                  general yield the same result as “divide the
                                  relative size by the pixel size”. The
                                  specified values are, in general, not equal
                                  to the angular path lengths of the sides of
                                  the image, nor the great-circle distance
                                  between adjacent corners. No default.
                                  Provisionally: maximum size is 5 degrees.
                                  Format: xsize,ysize. Example:
                                  --size=0.05,0.1.  [required]
  --rotation FLOAT                Position angle of the mosaic relative to
                                  equatorial north. Values are in degrees,
                                  with positive values corresponding to a
                                  rotation east of north. Specifically, the
                                  position angle controls the angle from the
                                  north vector at the center of the mosaic to
                                  the vector along the column direction from
                                  that point in the direction of increasing
                                  row number. Default is 0. Allowed range is
                                  [-360,360]. Example: --rotation=-45
                                  [default: 0.0]
  --pixel-scale FLOAT             Pixel scale of the produced mosaic at its
                                  center, reflected directly in CDELT1/2 in
                                  the result. Values are in arcseconds.
                                  Default is 6.15 arcseconds. Allowed range is
                                  [6.15,TBD]. (Super-resolution is not an
                                  option for at least the initial release, if
                                  ever; maximum scale will be set once
                                  performance can be tested.)  [default: 6.15]
  --channel-range TEXT            Range of central wavelengths of standard
                                  SPHEREx spectral channels to include in the
                                  result. Values are in microns and must be in
                                  range [0.75,1] . Default is the 102 standard
                                  channels. Incompatible with the --user-
                                  channels option. V1 of the tool will not
                                  provide access to the “partial channels” at
                                  the short and long wavelength ends of the
                                  detectors. (This might be enabled in future
                                  versions for the two partial channels at the
                                  extrema of the whole SPHEREx bandpass,
                                  and/or for the partial channels for a single
                                  detector if only one detector is chosen via
                                  --detector; or this capability may never be
                                  exposed in the IRSA UI, but only
                                  programmatically.)
  --user-channels TEXT            'Top-hat' spectral channels to be used to
                                  construct the planes of the mosaic result.
                                  Values are in microns. Pairs of consecutive
                                  values in the list define channels. No
                                  default; must be specified explicitly.Values
                                  must be in range [0.75,1]. Specifying just a
                                  single channel is allowed; this still
                                  produces cube extensions (NAXIS=3), just
                                  with NAXIS3=1. Incompatible with the
                                  --channel-range option; overrides the normal
                                  “all-standard-channels” default that applies
                                  when --channel-range is not specified. TBD
                                  whether channels will be allowed to overlap.
                                  There is a good science case to do so, and
                                  it does not make the SSDC implementation
                                  harder, we think. V1 of the tool will not
                                  provide access to the “partial channels” at
                                  the short and long wavelength ends of the
                                  detectors (TBC). (This might be enabled in
                                  future versions for the two partial channels
                                  at the extrema of the whole SPHEREx
                                  bandpass, and/or for the partial channels
                                  for a single detector if only one detector
                                  is chosen via --detector; or this capability
                                  may never be exposed in the IRSA UI, but
                                  only programmatically.)
  --time TEXT                     Time restriction. Values are in days, as MJD
                                  values, following the IVOA DALI and SIAv2
                                  standards. Restricts the selection of Level
                                  2 LVF images used as input to the mosaic.
                                  Note that application of a time restriction
                                  may well interfere with obtaining the full
                                  requested spectral coverage. The tool will
                                  not warn about this before execution. The
                                  min value may be given as -Inf and the max
                                  value may be given as +Inf in order to
                                  specify the absence of a lower or upper
                                  bound on the time interval, respectively.
                                  Defaults to no restriction.
  --detector TEXT                 Detector label: D1, D2, D3, D4, D5, or D6.
                                  Restricts inputs to come from the named
                                  detector(s). Not intended to be available
                                  through the IRSA UI; mainly useful in
                                  certain testing situations. Will in general
                                  interfere with obtaining the full requested
                                  spectral coverage. Defaults to no
                                  restriction. Selecting no detectors is
                                  treated as equivalent to selecting all 6.
  --verbose / --no-verbose        Turns on additional verbosity in a job's
                                  logging; should not affect the output
                                  visible to the user. Defaults to --no-
                                  verbose. Translated within the science tools
                                  to logging control options.  [default: no-
                                  verbose]
```


# Environment variables
The following environment variables should be set before running `mosaic-tool create-mosaic`
- SPXTOOL_BUTLER - Butler datastore root
- SPXTOOL_NCORES: Maximum number of cores that the script is allowed to use for its processing. The
  tool is allowed to take advantage of this either by spawning subprocesses or by multithreading
- SPXMOSAIC_INSTRUMENT - Instrument name: `spx` or `simu`. Defaults to `spx`.
- SPXMOSAIC_L2_DATASET_TYPE: Dataset type for level2 images. Defaults to `level2`.
- SPXMOSAIC_INPUT_COLL - Input collections that include level2 images usable for mosaicking. Defaults to `IRSA-qr2-visible`.
- SPXMOSAIC_CALIBRATION_COLL - Input collections that include calibrations used by the tool. Defaults
to `IRSA-qr2-tools`.
- SPXMOSAIC_QGRAPH_TIMEOUT - Qraph (execution plan) build timeout in seconds. Defaults to 1 hour.
- SPXMOSAIC_PIPELINE_TIMEOUT - Pipeline execution timeout in seconds. Defaults to 12 hours.
- SPXMOSAIC_SAMPLE_RESULT - (Not visible to IRSA), path to sample delivery in `--fake` mode.

# Error codes:
- 11 → Butler does not exist
- 12 → Butler exists but there is no level2 data in it.
- 13 → No level2 image matches for the requested mosaic volume.
- 14 → Subprocess executing one of the pipelines failed.
- 19 → Failed to write data into their designated final locations.