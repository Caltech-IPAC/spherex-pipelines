__all__ = ["get_magnitudes", "get_mag_cols", "CatalogConstants", "mag_to_flux", "EmptySourceCatalogError"]

# Standard library imports

# Related third party imports
import numpy as np
from astropy.table import Table, Column
from astropy import units as u

# Local application/library specific imports
from spx_butler.instrument import from_dichroicpair
from spx_utils.image_catalog import SourcePipelineClass
from spx_utils.utils import get_data_file


# Lazily populated cache for bandpass transmission curves.
# Structure: {array_id: (wavelengths, transmission)}
_BANDPASS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _get_bandpass(array_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Get bandpass transmission curve for a detector array.

    Lazily populates the cache on first call using SPHEREx_InstrumentSimulator.

    Parameters
    ----------
    array_id : int
        Detector array ID (1-6).

    Returns
    -------
    wavelengths, transmission : tuple[np.ndarray, np.ndarray]
        Wavelength and transmission arrays for the bandpass.
    """
    if array_id not in _BANDPASS_CACHE:
        import SPHEREx_InstrumentSimulator as SPinst
        bandpass_model = SPinst.Tabular_Bandpass()
        for det in range(1, 7):
            wls, trans = bandpass_model(1000, 1000, array=det, sparse=True)
            _BANDPASS_CACHE[det] = (wls, trans)

    return _BANDPASS_CACHE[array_id]


class CatalogConstants:
    """Constants for use by Query Reference Catalog.

    Attributes
    ----------
    color_corr_m : `float`
        Slope for color correction linear function parameter.
    color_corr_b : `float`
        Y-intercept for color correction linear function parameter.
    dimmest_flux : `float`
        Flux that yields magnitude of 19.
    mag_zero_point : `float`
        Zero point for magnitude estimation.

    """

    color_corr_m = 0.9507674098450911
    color_corr_b = 0.1990664933284922
    dimmest_flux = 91.2010839
    mag_zero_pnt = 23.9


class AstrometricPSFError(ValueError):
    """Custom exception to indicate astrometric/PSF source errors."""

    pass


class EmptySourceCatalogError(ValueError):
    """Custom exception to indicate empty source catalog errors."""

    pass


def get_magnitudes(dichroicpair: int,
                   tbl: Table,
                   keep: np.ndarray,
                   bandinfo_path: str = None) -> tuple[Column, Column]:
    """Add magnitude columns to reference catalog for dichroic pair detectors.

    Parameters
    ----------
    dichroicpair : `int`
        Dichroic pair id.
    tbl : `Table`
        Reference catalog for image.
    keep : `~numpy.ndarray`
        Boolean array specifying catalog sources within bounding box for image.
    bandinfo_path : `str`, optional
        Option to give a direct path for band info, default is None.

    Returns
    -------
    mag_swir, mag_mwir : `tuple[Column, Column]`
        Magnitude columns for detector pair.

    """
    """Adds magnitude columns to reference catalog for dichroic pair detectors."""

    # Load bandpass info
    bandinfo_path = get_data_file("data/SPHEREx_RefCat.BandsInfo_reduced.fits", package="spx_utils")

    bandinfo = Table.read(bandinfo_path, format="fits")
    bandinfo.sort("wavelength")
    column_names = tbl.colnames

    # Remove unavailable bands
    rows_to_remove = [i for i, b in enumerate(bandinfo["band"]) if b not in column_names]
    bandinfo.remove_rows(rows_to_remove)
    bands = list(bandinfo["band"])
    lambda_mean = np.array(bandinfo["wavelength"])  # microns

    # Stack fluxes [N_sources, N_bands]
    flux_table = np.hstack([tbl[k][:, np.newaxis] for k in bands])

    # Detector IDs from dichroicpair
    swir_det = dichroicpair  # 1, 2, or 3
    mwir_det = dichroicpair + 3  # 4, 5, or 6

    # Transmission curves (cached after first call)
    wls_swir, trans_swir = _get_bandpass(swir_det)
    wls_mwir, trans_mwir = _get_bandpass(mwir_det)

    swir = []
    mwir = []

    # Get bandpass transmission curves
    for idx in np.where(keep)[0]:
        fluxes = flux_table[idx]
        good_data = (fluxes > 0) & (fluxes < 1e20)

        if np.any(good_data):
            spec_swir = np.interp(wls_swir, lambda_mean[good_data], fluxes[good_data])
            flux_swir = np.sum(trans_swir * spec_swir, axis=-1) / np.sum(trans_swir, axis=-1)
            spec_mwir = np.interp(wls_mwir, lambda_mean[good_data], fluxes[good_data])
            flux_mwir = np.sum(trans_mwir * spec_mwir, axis=-1) / np.sum(trans_mwir, axis=-1)
            swir.append(flux_swir)
            mwir.append(flux_mwir)
        else:
            swir.append(20.)
            mwir.append(20.)

    swir_mags = (swir*u.uJy).to(u.ABmag)
    mwir_mags = (mwir*u.uJy).to(u.ABmag)
    return swir_mags, mwir_mags


def get_mag_cols(dichroicpair: int, tbl: Table, keep: np.ndarray) -> tuple[Column, Column]:
    """Add magnitude columns to reference catalog for dichroic pair detectors.

    Parameters
    ----------
    dichroicpair : `int`
        Dichroic pair id.
    tbl : `Table`
        Reference catalog for image.
    keep : `~numpy.ndarray`
        Boolean array specifying catalog sources within bounding box for image.

    Returns
    -------
    mag_swir, mag_mwir : `tuple[Column, Column]`
        Magnitude columns for detector pair.

    """
    ps_z = tbl['PS1_z'].data[keep]
    ws_1 = tbl['WISE_W1'].data[keep]
    ws_2 = tbl['WISE_W2'].data[keep]
    gaia = tbl['Gaia_RP'].data[keep]
    spx_cls_flgs = tbl['spherex_class'].data[keep]

    # Make sure that if PanStarrs and Gaia are not available,
    # then the source must not be an astrometric or psf calibrator.
    no_gaia_ps = (ps_z < 0) & (gaia < 0)
    msk_ast = spx_cls_flgs & SourcePipelineClass.ASTROMETRIC == SourcePipelineClass.ASTROMETRIC
    msk_psf = spx_cls_flgs & SourcePipelineClass.PSF == SourcePipelineClass.PSF

    invalid_ast_sources = np.logical_not(no_gaia_ps & msk_ast)
    invalid_psf_sources = np.logical_not(no_gaia_ps & msk_psf)

    if not np.all(invalid_ast_sources):
        raise AstrometricPSFError(format_astrometric_psf_error(invalid_ast_sources, tbl, keep))
    if not np.all(invalid_psf_sources):
        raise AstrometricPSFError(format_astrometric_psf_error(invalid_psf_sources, tbl, keep))

    # Estimated magnitude for dichroic pairs.
    match dichroicpair:
        # In detectors 1 and 2 if Panstarrs not available use Gaia_RP.
        # Dichroic pair 1: detectors 1 and 4.
        case 1:
            # Detector 1 flux ~ PS1_z
            ps_z[ps_z < 0] = color_correction(gaia[np.where(ps_z < 0)])
            # ps_z[ps_z < 0] = gaia[np.where(ps_z < 0)]
            mag_det1 = flux_to_mag(ps_z)
            # Detector 4 flux ~ WISE_W1
            mag_det4 = flux_to_mag(ws_1)
            # Magnitudes for pair.
            magnitudes = [mag_det1, mag_det4]
        # Dichroic pair 2: detectors 2 and 5.
        case 2:
            # Detector 2 flux ~ 2.1*((PS1_z+WISE_W1)/2 - 20)
            ps_z[ps_z < 0] = color_correction(gaia[np.where(ps_z < 0)])
            # ps_z[ps_z < 0] = gaia[np.where(ps_z < 0)]
            flux_det2 = 2.1 * ((ps_z + ws_1) / 2. - 20.)
            mag_det2 = flux_to_mag(flux_det2)
            # Detectpr 5 flux ~ 1.01*(WISE_W1-15)
            flux_det5 = 1.01 * (ws_1 - 15.)
            mag_det5 = flux_to_mag(flux_det5)
            # Magnitudes for pair.
            magnitudes = [mag_det2, mag_det5]
        # Dichroic pair 3: detectors 3 and 6.
        case 3:
            # Detector 3 flux ~ 2.06*(WISE_W1 - 15)
            flux_det3 = 2.06 * (ws_1 - 15.)
            mag_det3 = flux_to_mag(flux_det3)
            # Array 6 flux ~ WISE_W2
            mag_det6 = flux_to_mag(ws_2)
            # Magnitudes for pair.
            magnitudes = [mag_det3, mag_det6]

    # Get detector ids from dichroic pair id.
    swir, mwir = from_dichroicpair(dichroicpair)
    # Dichroic pair to columns.
    mag_swir = Column(magnitudes[from_dichroicpair(dichroicpair).index(swir)],
                      name=f'mag_det{swir}', dtype=np.double, unit='')
    mag_mwir = Column(magnitudes[from_dichroicpair(dichroicpair).index(mwir)],
                      name=f'mag_det{mwir}', dtype=np.double, unit='')
    return mag_swir, mag_mwir


def format_astrometric_psf_error(
    invalid_sources: np.ndarray, tbl: Table, keep: np.ndarray
) -> str:
    """Generate an error message containing the RA, Dec, SPHEREx RefID,
    and LegacySurvey UID for all sources.

    Parameters
    ----------
    invalid_sources : `np.ndarray`
        Boolean array that flags invalid sources lacking Gaia or PanSTARRS data.
    tbl : `Table`
        Reference catalog for image.
    keep : `~numpy.ndarray`
        Boolean array specifying catlog sources within bounding box for image.

    Returns
    -------
    error_message : `str`
        Error message containing information about source.

    """
    no_gaia_ps_ra = tbl["ra"].data[keep][~invalid_sources]
    no_gaia_ps_dec = tbl["dec"].data[keep][~invalid_sources]
    no_gaia_ps_spx_ref_id = tbl["SPHERExRefID"].data[keep][~invalid_sources]
    no_gaia_ps_legacy_id = tbl["LegacySurvey_uid"].data[keep][~invalid_sources]

    source_ids = [
        f"RA: {ra}, Dec: {dec}, SPHEREx RefID: {spx_id}, LecagySurvey UID: {legacy_id}"
        for ra, dec, spx_id, legacy_id in zip(
            no_gaia_ps_ra,
            no_gaia_ps_dec,
            no_gaia_ps_spx_ref_id,
            no_gaia_ps_legacy_id,
        )
    ]

    error_message = (
        "There are sources flagged as ASTROMETRIC with no PanStarrs or Gaia data:\n"
        + "\n".join(source_ids)
    )

    return error_message


def flux_to_mag(flux: np.array) -> np.array:
    """Flux unit is microJansky, to get the magnitude use -2.5*log10(flux)+23.9.

    Parameter
    ---------
    flux : `np.array`
        Input flux in uJ.

    Returns
    -------
    magnitude : `np.array`
        Estimated magnitude.

    """
    flux[flux < 0] = CatalogConstants.dimmest_flux  # uJ that will yield 19 ABmag.
    magnitude = -2.5 * np.log10(flux) + CatalogConstants.mag_zero_pnt
    return magnitude


def color_correction(gaia_flux: np.array) -> np.array:
    """Color correction applied to Gaia_RP (0.6 um) for estimation of PS1_z (0.9 um).

    Parameter
    ---------
    gaia_flux : `np.array`
        Input Gaia flux in uJ.

    Returns
    -------
    gaia_flux : `np.array`
        Color corrected Gaia flux.

    """
    m = CatalogConstants.color_corr_m
    b = CatalogConstants.color_corr_b
    gaia_flux[gaia_flux > 0] = 10**(m * np.log10(gaia_flux[gaia_flux > 0]) + b)
    return gaia_flux


def mag_to_flux(mag_ab):
    """Convert AB magnitude to flux in Janskys (Jy).

    The conversion formula is:
    F_Jy = 3631 * 10^(-0.4 * mag_AB)
    where 3631 Jy is the zero-point flux for the AB system.

    Parameters
    ----------
    mag_ab : float or np.ndarray
        AB magnitude(s).

    Returns
    -------
    float or np.ndarray
        Flux in Janskys (Jy).

    """
    flux_jy = (mag_ab * u.ABmag).to(u.Jy).value
    return flux_jy
