__all__ = ['ChannelMosaic', 'write_mosaic_cube']
import numpy as np
from dataclasses import dataclass
from typing import Optional
import os
import subprocess
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from spx_data import ChannelInfo
from spx_data.fits_flag_io import write_flag_headers, read_flag_headers_as_bitmask
import logging

_BUNIT = {
    'IMAGE': ('MJy/sr', 'Data units'),
    'NHITS': ('count',  'Hit count'),
    'FLAGS': ('',       'Bit flags'),
}


def _add_flag_headers(header, propagated_flags):
    """Add flag definition headers in standard MP or MSK  format.

    Follows the convention understood by Firefly.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header to update.
    propagated_flags : int
        Bitwise OR of propagated Flag values.

    """
    # Use consolidated flag header I/O
    write_flag_headers(header, propagated_flags)


@dataclass
class ChannelMosaic:
    """Result of mosaic creation for a single channel.

    Attributes
    ----------
    data : numpy.ndarray
        2D mosaic data with units, shape (ny, nx).
    nhits : numpy.ndarray
        2D hit count map, shape (ny, nx).
    flags : numpy.ndarray
        2D array of propagated flags (int32), shape (ny, nx).
        Contains bitwise OR of flags to be propagated from
        contributing input images to the final mosaic.
    wcs : astropy.wcs.WCS
        World Coordinate System for the mosaic.
    channel_info : ChannelInfo
        Spectral channel metadata (detector, subchannel, wavelength, R, bandwidth).
    propagated_flags : int, optional
        Bitwise OR of Flag values that were propagated to the flags array.
        Used to populate mask plane headers in FITS output.
    extra_header : astropy.io.fits.Header, optional
        Extra cards to include in the output IMAGE header (e.g. the HiPS
        ORDER/NPIX tile identification).

    Examples
    --------
    After building a mosaic::
        mosaic = builder.build_channel(l2_files, array=1, subchannel=8)
        print(f"Wavelength: {mosaic.channel_info.wavelength:.3f}")
        print(f"Bandwidth: {mosaic.channel_info.effective_bandwidth:.4f}")
        print(f"Valid pixels: {np.sum(~np.isnan(mosaic.data.value))}")
        mosaic.write('output.fits')

    """

    data: np.ndarray
    nhits: np.ndarray
    flags: np.ndarray
    wcs: WCS
    channel_info: ChannelInfo
    propagated_flags: Optional[int] = None
    extra_header: Optional[fits.Header] = None

    def write(self, filepath: str, overwrite: bool = True) -> None:
        """Write mosaic to FITS file.

        Creates a FITS file with:
        - Primary HDU: Mosaic data (IMAGE)
        - Extension 1: Hit count map (NHITS)
        - Extension 2: Propagated flags (FLAGS)

        Parameters
        ----------
        filepath : str
            Output file path.
        overwrite : bool, optional
            Whether to overwrite existing file. Default is True.

        Examples
        --------
        >>> mosaic.write('mosaic_array1_ch08.fits')

        """
        header = self.wcs.to_header()
        info = self.channel_info

        # Extra caller-provided cards (e.g. HiPS ORDER/NPIX)
        if self.extra_header is not None:
            header.update(self.extra_header)

        # Add metadata
        header['BUNIT'] = _BUNIT['IMAGE']
        header['DETECTOR'] = (info.detector, 'Detector number')
        is_interpolated = info.subchannel == -1
        if is_interpolated:
            header['SUBCHAN'] = (info.subchannel, 'Subchannel (-1 = interpolated)')
        else:
            header['SUBCHAN'] = (info.subchannel, 'Subchannel number')
        header['WAVELEN'] = (info.wavelength.to(u.um).value, '[micron] Wavelength')

        # Add wavelength range if available
        if info.wavelength_min is not None:
            if is_interpolated:
                header['WLMIN'] = (info.wavelength_min.to(u.um).value,
                                   '[micron] Min wavelength (contributing channels)')
            else:
                header['WLMIN'] = (info.wavelength_min.to(u.um).value, '[micron] Min wavelength')
        if info.wavelength_max is not None:
            if is_interpolated:
                header['WLMAX'] = (info.wavelength_max.to(u.um).value,
                                   '[micron] Max wavelength (contributing channels)')
            else:
                header['WLMAX'] = (info.wavelength_max.to(u.um).value, '[micron] Max wavelength')

        # Add resolving power and bandwidth if available
        if info.R is not None:
            if is_interpolated:
                header['R'] = (info.R, 'Effective resolving power')
            else:
                header['R'] = (info.R, 'Resolving power')
        if info.R_std is not None:
            if is_interpolated:
                header['R_STD'] = (info.R_std, 'Std dev of effective R')
            else:
                header['R_STD'] = (info.R_std, 'Std dev of resolving power')
        eff_bw = info.effective_bandwidth
        if eff_bw is not None:
            header['BANDWID'] = (eff_bw.to(u.um).value, '[micron] Bandwidth')

        # Statistics
        valid_mask = ~np.isnan(self.data.value)
        header['NVALID'] = (int(np.sum(valid_mask)), 'Number of valid pixels')
        if is_interpolated:
            header['MAXHITS'] = (float(np.nanmax(self.nhits)),
                                 'Max contributing channels per pixel')
        else:
            header['MAXHITS'] = (float(np.nanmax(self.nhits)), 'Maximum hits per pixel')

        # Create HDUs
        # Data values without units for FITS
        data_values = self.data.value if hasattr(self.data, 'value') else self.data
        hdu_sci = fits.PrimaryHDU(data_values, header=header)
        hdu_sci.name = 'IMAGE'

        hdu_nhits = fits.ImageHDU(self.nhits, name='NHITS')
        hdu_nhits.header['BUNIT'] = _BUNIT['NHITS']

        # Create FLAGS HDU
        hdu_flags = fits.ImageHDU(self.flags, name='FLAGS')
        hdu_flags.header['BUNIT'] = _BUNIT['FLAGS']

        # Add comments explaining how NHITS and FLAGS were calculated
        is_interpolated = info.subchannel == -1
        if is_interpolated:
            hdu_nhits.header.add_comment(
                'Number of spectral channels with valid data at each pixel')
            hdu_nhits.header.add_comment(
                'Used for spectral interpolation to target wavelength')
            hdu_flags.header.add_comment(
                'Bitwise OR of flags from all contributing spectral channels')
            hdu_flags.header.add_comment(
                'Channels contribute if they have valid (non-NaN) data at pixel')
        else:
            hdu_nhits.header.add_comment(
                'Number of input images contributing to each mosaic pixel')
            hdu_flags.header.add_comment(
                'Bitwise OR of flags from all contributing input images')

        _add_flag_headers(hdu_flags.header, self.propagated_flags)

        hdul = fits.HDUList([hdu_sci, hdu_nhits, hdu_flags])
        hdul.writeto(filepath, overwrite=overwrite)

    @property
    def valid_fraction(self) -> float:
        """Fraction of pixels with valid (non-NaN) data."""
        data_values = self.data.value if hasattr(self.data, 'value') else self.data
        valid = ~np.isnan(data_values)
        return np.sum(valid) / valid.size

    @property
    def channel_id(self) -> tuple[int, int]:
        """Return (detector, subchannel) tuple."""
        return self.channel_info.detector, self.channel_info.subchannel


def build_wcs_wave_table(wavelengths, bandwidths=None):
    """Build a FITS binary table HDU describing wavelength planes and bandwidths.

    This helper constructs the ``WCS-WAVE`` table used by the spectral WCS
    metadata. Each spectral plane is assigned:

    - a plane index (1-based),
    - a wavelength value,
    - a bandwidth value.

    The FITS table is written with 2D column cells of shape (N, 1) because
    the downstream WCS convention expects vector-valued columns even when
    there is only a single vector per row. Astropy allows a 1D special case,
    but we standardize on explicit 2D arrays for consistency and TDIM
    metadata correctness.

    Parameters
    ----------
    wavelengths : sequence of float
        Wavelength values for spectral planes. Length N defines the number
        of planes.
    bandwidths : sequence of float, optional
        Per-plane bandwidth values in microns. If None, fake values are
        used for backward compatibility.

    Returns
    -------
    astropy.io.fits.BinTableHDU
        Binary table HDU named ``"WCS-WAVE"`` with one row and three columns:

        - ``PLANE`` : int, shape (N)
            1-based spectral plane indices.
        - ``WAVELENGTHS`` : float, shape (N, 1)
            Wavelength values.
        - ``BANDWIDTHS`` : float, shape (N, 1)
            Per-plane bandwidth values.

    """
    # Number of spectral planes
    N = len(wavelengths)

    if bandwidths is not None:
        bandwidth_values = np.asarray(bandwidths, dtype=np.float32)
    else:
        # Fallback: fake bandwidth values for backward compatibility
        bandwidth_values = np.round(np.linspace(0.01, N * 0.01, N), 3).astype(
            np.float32
        )

    # Convert to explicit (N, 1) arrays so FITS columns become vector-valued
    # cells with consistent TDIM metadata.
    values_arr = np.asarray(wavelengths, dtype=np.float32).reshape(N, 1)
    bwidth_arr = bandwidth_values.astype(np.float32).reshape(N, 1)

    # FITS plane indices are 1-based
    plane_index = np.arange(1, N + 1, dtype=np.int32)

    # Build single-row table with vector columns
    tbl = Table()
    tbl.add_column(Column(plane_index[np.newaxis, ...], name="PLANE"))
    tbl.add_column(Column(values_arr[np.newaxis, ...], name="WAVELENGTHS"))
    tbl.add_column(Column(bwidth_arr[np.newaxis, ...], name="BANDWIDTHS"))

    return fits.BinTableHDU(data=tbl, name="WCS-WAVE")


def _get_mosaic_metadata(filepath: str) -> dict:
    """Read only the headers of a mosaic file to extract metadata for sorting.
    Avoids reading the data arrays into memory.
    """
    with fits.open(filepath) as hdul:
        header = hdul['IMAGE'].header
        flags_header = hdul['FLAGS'].header

        detector = int(header['DETECTOR'])
        subchannel = int(header['SUBCHAN'])
        wavelength = float(header['WAVELEN'])

        # Use NaN for missing optional values to match standard cube behavior
        wavelength_min = float(header['WLMIN']) if 'WLMIN' in header else np.nan
        wavelength_max = float(header['WLMAX']) if 'WLMAX' in header else np.nan
        R = float(header['R']) if 'R' in header else np.nan
        R_std = float(header['R_STD']) if 'R_STD' in header else np.nan
        bandwidth = float(header['BANDWID']) if 'BANDWID' in header else np.nan

        # Reconstruct propagated_flags from mask plane headers
        propagated_flags = read_flag_headers_as_bitmask(flags_header)

        # Preserve extra non-WCS cards (e.g. HiPS ORDER/NPIX) that
        # WCS.to_header() would otherwise drop from the cube header
        extra_header = fits.Header()
        for key in ('ORDER', 'NPIX'):
            if key in header:
                extra_header[key] = (header[key], header.comments[key])

        wcs = WCS(header)
        shape = hdul['IMAGE'].shape

        return {
            'path': filepath,
            'detector': detector,
            'subchannel': subchannel,
            'wavelength': wavelength,
            'wavelength_min': wavelength_min,
            'wavelength_max': wavelength_max,
            'R': R,
            'R_std': R_std,
            'bandwidth': bandwidth,
            'propagated_flags': propagated_flags,
            'extra_header': extra_header if len(extra_header) > 0 else None,
            'wcs': wcs,
            'shape': shape
        }


def apply_spectral_wcs_headers(header):
    """Populate FITS WCS keywords for tabular spectral coordinates.

    This function modifies a FITS header in place to define two additional
    tabular WCS axes using the FITS WCS ``-TAB`` algorithm:

    - Axis 3: wavelength
    - Axis 4: bandwidth

    Both axes use lookup tables stored in a binary table extension named
    ``WCS-WAVE``. The coordinate values are read from table columns and
    indexed by the image plane number.

    This configuration is commonly used for multichannel spectral
    cubes where wavelength and bandwidth are not linear functions of pixel
    coordinates.

    Notes
    -----
    - The image itself is assumed to have NAXIS = 3 (e.g., RA, DEC, PLANE),
      but the WCS has 4 axes. Therefore, ``WCSAXES`` is set to 4.
    - The spectral axes are defined using the FITS WCS TAB convention
      (Greisen et al.).
    - Axis coupling is defined so that both spectral coordinates depend on
      the image plane axis (axis 3).
    - The referenced binary table extension must contain columns:
        * WAVELENGTHS
        * BANDWIDTHS
        * PLANE

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header to update. Modified in place.

    Returns
    -------
    header : astropy.io.fits.Header
        The same header instance, updated with spectral WCS keywords.

    Examples
    --------
    >>> hdr = fits.Header()
    >>> hdr = apply_spectral_wcs_headers(hdr)
    >>> hdr["CTYPE3"]
    'WAVE-TAB'

    """
    # Axis 3 — wavelength TAB axis
    header['CTYPE3'] = ('WAVE-TAB', 'Coordinate via lookup table')
    header['CNAME3'] = ('WAVELENGTH', 'Spectral axis: wavelength')
    header['CUNIT3'] = ('um', 'Microns')
    header['CRPIX3'] = (1, 'TAB reference pixel (convention)')
    header['CDELT3'] = (1, 'TAB increment (convention)')
    header['CRVAL3'] = (1, 'TAB reference value (convention)')
    header['PS3_0'] = ('WCS-WAVE', 'Extension with lookup table')
    header['PS3_1'] = ('WAVELENGTHS', 'Column with wavelength values')
    header['PS3_2'] = ('PLANE', 'Column with plane indices')
    header['PV3_3'] = (1, 'Index of wavelength in the WAVELENGTHS array')

    # Axis 4 — bandwidth TAB axis
    header['CTYPE4'] = ('WAVE-TAB', 'Coordinate via lookup table')
    header['CNAME4'] = ('BANDWIDTH', 'Spectral axis: bandwidth')
    header['CUNIT4'] = ('um', 'Microns')
    header['CRPIX4'] = (1, 'TAB reference pixel (convention)')
    header['CDELT4'] = (1, 'TAB increment (convention)')
    header['CRVAL4'] = (1, 'TAB reference value (convention)')
    header['PS4_0'] = ('WCS-WAVE', 'Extension with lookup table')
    header['PS4_1'] = ('BANDWIDTHS', 'Column with bandwidth values')
    header['PS4_2'] = ('PLANE', 'Column with plane indices')
    header['PV4_3'] = (1, 'Index of bandwidth in BANDWIDTHS array')

    # Axis coupling
    header['PC3_3'] = (1.0, 'Wavelength depends on image axis 3')
    header['PC4_3'] = (1.0, 'Bandwidth depends on image axis 3')

    # PC4_4 is explicitly zero because no independent pixel axis 4 exists.
    # Included for clarity — ignored by WCS linear transform.
    header['PC4_4'] = (0.0, 'No independent 4th pixel axis')

    # Image has 3 axes but WCS defines 4 world axes
    # Must set WCSAXES
    header['WCSAXES'] = 4

    return header


def _build_extension_header(extname, base_header, shape, dtype, bunit, flag_propagated=None):
    """Pure header construction — no I/O, no data."""
    # Let Astropy generate the correct header for this dtype
    dummy = fits.ImageHDU(data=np.zeros((0, 0, 0), dtype=dtype), name=extname)
    # Apply base header keywords
    for card in base_header.cards:
        # Exclude placeholder keywords
        if card.keyword not in dummy.header and card.keyword not in \
                ['BUNIT', 'EXTNAME', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3', 'BITPIX']:
            dummy.header.append(card)
    # Set BUNIT (the main difference in derived headers between HDUs)
    dummy.header['BUNIT'] = bunit
    # Force the NAXIS headers to match the true 3D shape, not the dummy (0,0,0) shape
    dummy.header['NAXIS1'] = shape[2]
    dummy.header['NAXIS2'] = shape[1]
    dummy.header['NAXIS3'] = shape[0]
    if flag_propagated is not None:
        _add_flag_headers(dummy.header, flag_propagated)
    return dummy.header


def _stream_extension_data(f, header, data_generator):
    """Write a pre-built header and stream data slice-by-slice."""
    # Write the FITS header (exactly padded to 2880 bytes as per standard
    f.write(header.tostring(padding=True).encode('ascii'))
    # Determine the native FITS Big-Endian type (-32=float, 32=int)
    fits_dtype = '>f4' if header['BITPIX'] == -32 else '>i4'
    bytes_written = 0
    for chunk in data_generator():
        b = chunk.astype(fits_dtype).tobytes()
        f.write(b)
        bytes_written += len(b)
    # Pad data to FITS block size (2880 bytes)
    pad_bytes = (2880 - (bytes_written % 2880)) % 2880
    if pad_bytes:
        f.write(b'\x00' * pad_bytes)


def write_mosaic_cube(
        mosaic_paths: list[str],
        filepath: str,
        overwrite: bool = True,
        compress: bool = True
) -> str:
    """Write multiple channel mosaics to a single FITS cube file.

    Creates a multi-extension FITS file with:
    - Extension 0 (PRIMARY): Empty with basic header including VERSION
    - Extension 1 (IMAGE): 3D data cube (nchannels, ny, nx) in MJy/sr
    - Extension 2 (NHITS): 3D hit count cube (nchannels, ny, nx)
    - Extension 3 (FLAGS): 3D flags cube (nchannels, ny, nx) as int32
    - Extension 4 (SPECTRAL_CHANNELS): Binary table with per-channel spectral metadata
      (DETECTOR, SUBCHAN, WAVELENGTH, WL_MIN, WL_MAX, R, R_STD, BANDWIDTH)
    - Extension 5 (WCS-WAVE): Lookup table for spectral WCS

    Mosaics are sorted by wavelength before stacking.

    Parameters
    ----------
    mosaics : list of ChannelMosaic
        List of channel mosaics to combine. All must have the same
        spatial shape and WCS.
    filepath : str
        Output file path.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is True.
    compress : bool, optional
        Whether to compress output with gzip. If True and filepath does
        not end with '.gz', '.gz' is appended. Default is False.

    Returns
    -------
    str
        Actual filepath written (may have '.gz' appended if compress=True).

    Raises
    ------
    ValueError
        If mosaics list is empty or mosaics have inconsistent shapes.

    Examples
    --------
    >>> mosaics = builder.build_all(l2_files)
    >>> write_mosaic_cube(mosaics, 'mosaic_cube.fits')

    """
    try:
        from spx_apps.mosaic_tool import __version__
    except ImportError:
        __version__ = "Unknown"

    if not mosaic_paths:
        raise ValueError("Cannot write empty mosaic list")

    logging.info("Start writing mosaic cube")

    if os.path.exists(filepath):
        if overwrite:
            os.remove(filepath)
        else:
            raise FileExistsError(f"{filepath} already exists.")

    mosaic_paths = sorted(mosaic_paths)
    first_meta = _get_mosaic_metadata(mosaic_paths[0])
    shape = first_meta['shape']
    n_channels = len(mosaic_paths)
    ny, nx = shape

    wavelengths, wl_mins, wl_maxs = [], [], []
    R_values, R_stds, bandwidths = [], [], []
    arrays, subchannels = [], []

    # Fast Pass 0: Extract Metadata
    for i, path in enumerate(mosaic_paths):
        meta = first_meta if i == 0 else _get_mosaic_metadata(path)
        if meta['shape'] != shape:
            raise ValueError(f"Inconsistent mosaic shapes: {shape} vs {meta['shape']} in {path}")

        arrays.append(meta['detector'])
        subchannels.append(meta['subchannel'])
        wavelengths.append(meta['wavelength'])
        wl_mins.append(meta['wavelength_min'])
        wl_maxs.append(meta['wavelength_max'])
        R_values.append(meta['R'])
        R_stds.append(meta['R_std'])
        bandwidths.append(meta['bandwidth'])

    # Format Spectral WCS Header Template
    header = first_meta['wcs'].to_header()
    if first_meta['extra_header'] is not None:
        header.update(first_meta['extra_header'])
    header = apply_spectral_wcs_headers(header)
    header['NCHAN'] = (n_channels, 'Number of spectral channels')
    header['WLMIN'] = (wavelengths[0], 'Minimum wavelength [micron]')
    header['WLMAX'] = (wavelengths[-1], 'Maximum wavelength [micron]')

    # Generators to stream individual 2D slices straight from the disk
    def sci_gen():
        for p in mosaic_paths:
            with fits.open(p) as h:
                yield h['IMAGE'].data

    def nhits_gen():
        for p in mosaic_paths:
            with fits.open(p) as h:
                yield h['NHITS'].data

    def flags_gen():
        for p in mosaic_paths:
            with fits.open(p) as h:
                yield h['FLAGS'].data

    # --- MAIN FITS WRITING STREAM ---
    cube_shape = (n_channels, ny, nx)
    with open(filepath, 'wb') as f:
        # 1. PRIMARY HDU
        primary_header = fits.Header()
        primary_header.append(('SIMPLE', True, 'conforms to FITS standard'))
        primary_header.append(('BITPIX', 8, 'array data type'))
        primary_header.append(('NAXIS', 0, 'number of array dimensions'))
        primary_header.append(('EXTEND', True))
        primary_header.append(('VERSION', __version__, 'Mosaic tool version'))
        # Fixed padding argument here as well
        f.write(primary_header.tostring(padding=True).encode('ascii'))

        # 2. IMAGE, NHITS, FLAGS HDUs
        for extname, dtype, gen, bunit, flags in [
            ('IMAGE', np.float32, sci_gen,   _BUNIT['IMAGE'], None),
            ('NHITS', np.int32,   nhits_gen, _BUNIT['NHITS'], None),
            ('FLAGS', np.int32,   flags_gen, _BUNIT['FLAGS'], first_meta['propagated_flags']),
        ]:
            hdr = _build_extension_header(extname, header, cube_shape, dtype, bunit, flags)
            _stream_extension_data(f, hdr, gen)

    # --- APPEND ASTROPY TABLES ---
    # Round values to match precision rules from make_spectral_channels_table.py:
    # - WAVELENGTH, WL_MIN, WL_MAX, BANDWIDTH: 3 decimal places
    # - R: 0 decimal places (integer)
    # - R_STD: 1 decimal place
    wavelengths_rounded = np.round(wavelengths, 3)
    wl_mins_rounded = np.round(wl_mins, 3)
    wl_maxs_rounded = np.round(wl_maxs, 3)
    bandwidths_rounded = np.round(bandwidths, 3)
    R_rounded = np.round(R_values, 0)
    R_stds_rounded = np.round(R_stds, 1)

    col_array = fits.Column(name='DETECTOR', format='I', array=arrays)
    col_subchan = fits.Column(name='SUBCHAN', format='I', array=subchannels)
    col_wave = fits.Column(name='WAVELENGTH', format='E', unit='um', disp='F10.3', array=wavelengths_rounded)
    col_wl_min = fits.Column(name='WL_MIN', format='E', unit='um', disp='F10.3', array=wl_mins_rounded)
    col_wl_max = fits.Column(name='WL_MAX', format='E', unit='um', disp='F10.3', array=wl_maxs_rounded)
    col_bw = fits.Column(name='BANDWIDTH', format='E', unit='um', disp='F10.3', array=bandwidths_rounded)
    col_r = fits.Column(name='R', format='E', disp='F10.0', array=R_rounded)
    col_r_std = fits.Column(name='R_STD', format='E', disp='F10.1', array=R_stds_rounded)
    hdu_spectral_bins = fits.BinTableHDU.from_columns(
        [col_array, col_subchan, col_wave, col_wl_min, col_wl_max, col_r, col_r_std, col_bw],
        name='SPECTRAL_CHANNELS'
    )

    bw_for_wcs = bandwidths if not all(np.isnan(bandwidths)) else None
    wave_tab_hdu = build_wcs_wave_table(wavelengths, bandwidths=bw_for_wcs)

    # Use Astropy's native 'append' mode
    # to safely write the binary tables to our output file
    with fits.open(filepath, mode='append') as hdul:
        hdul.append(hdu_spectral_bins)
        hdul.append(wave_tab_hdu)

    if compress:
        result = subprocess.run(
            ['pigz', '-p', os.getenv("SPXTOOL_NCORES", "6"), filepath],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pigz compression failed for {filepath}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return filepath + ".gz"
    return filepath
