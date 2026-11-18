__all__ = ["compute_tilt_function", "tilt_correct", "construct_tilt_template"]

# Standard library imports
import logging

# Related third party imports
from astropy.coordinates import SkyCoord, Angle, get_body
from astropy.time import Time
import astropy.units as u
import datetime
import numpy as np
import pandas as pd

# Local application/library specific imports


# MPI communicator object
_mpi_comm = None


def _get_mpi_comm():
    """Get MPI communicator object."""
    global _mpi_comm
    if _mpi_comm is None:
        from mpi4py import MPI
        _mpi_comm = MPI.COMM_WORLD
    return _mpi_comm


def get_eclons_from_Sun(obspos: SkyCoord, obstimes: list[Time], unit='deg'):
    """Compute the absolute ecliptic longitude difference between an observation
       position and the Sun at specific times.

    Parameters
    ----------
    obspos : `astropy.coordinates.SkyCoord`
        Array or list of sky coordinates representing the positions of the
        observations in the 'geocentrictrueecliptic' frame.

    obstimes : `list` of `astropy.time.Time`
        List of observation times corresponding to each position in `obspos`.
        The times are used to compute the position of the Sun at the given
        observation time.

    unit : `str`, optional
        The unit for the returned ecliptic longitude differences. Default is
        'deg' for degrees.

    Returns
    -------
    np.ndarray
        Array of absolute ecliptic longitude differences (wrapped to [-180, 180] deg)
        between the observation positions and the Sun at each observation time.

    """
    eclons_from_sun = []

    # Loop through each observation time and position
    for i, obstime in enumerate(obstimes):
        sun_pos = get_body('sun', obstime).geocentrictrueecliptic
        # Compute the ecliptic longitude difference
        eclon_from_sun = Angle(obspos[i].lon.deg - sun_pos.lon.deg, unit='deg')
        eclons_from_sun.append(eclon_from_sun.wrap_at(180*u.deg).to(unit))

    # Convert eclons_from_sun to a numpy array of degrees
    eclons_from_sun_array = np.array([eclon.deg for eclon in eclons_from_sun])
    return eclons_from_sun_array


def meshgrid(npix: int, dnpix: int = None):
    """Generate a 2D mesh grid of pixel indices.

    Parameters
    ----------
    npix : `int`
        Number of pixels along one dimension.
    dnpix : `int`, optional
        Number of pixels along the other dimension. Defaults to `npix`.

    Returns
    -------
    X, Y : `np.ndarray`
        2D arrays representing the x and y coordinates of the grid.

    Raises
    ------
    ValueError
        If `npix` or `dnpix` are not positive integers.

    """
    if not isinstance(npix, int) or npix <= 0:
        raise ValueError("`npix` must be a positive integer.")

    dnpix = dnpix if dnpix is not None else npix  # Use `npix` if `dnpix` is not provided

    if not isinstance(dnpix, int) or dnpix <= 0:
        raise ValueError("`dnpix` must be a positive integer if provided.")

    # Generate the pixel grid
    x = np.arange(npix)
    y = np.arange(dnpix)

    return np.meshgrid(x, y)


def lstsq(Y: np.ndarray, A: np.ndarray, Cinv: np.ndarray, return_err=False):
    """Return the least-squares solution to a linear matrix equation,
       given data Y, design matrix A, and inverse covariance Cinv.

    Parameters
    ----------
    Y : `np.ndarray`
        Data vector.
    A : `np.ndarray`
        Design matrix.
    Cinv : `np.ndarray`
        Inverse covariance matrix or weights.
    return_err : `bool`, optional
        If True, return the covariance matrix of the fit.

    Returns
    -------
    `np.ndarray`
        Least-squares solution.
    `np.ndarray`, optional
        Covariance matrix of the fit (if `return_err=True`).

    """
    if len(Cinv.shape) == 1:
        A_weighted = A * np.sqrt(Cinv[:, None])
        Y_weighted = Y * np.sqrt(Cinv)
    else:
        A_weighted = Cinv @ A
        Y_weighted = Cinv @ Y

    # Regularization term to avoid singularity
    regularization = np.eye(A_weighted.shape[1]) * 1e-8

    # Perform least-squares fit
    res, _, _, _ = np.linalg.lstsq(A_weighted, Y_weighted, rcond=None)

    if return_err:
        covariance_matrix = np.linalg.inv(A_weighted.T @ A_weighted + regularization)
        return res, covariance_matrix
    else:
        return res


def compute_tilt_function(images: list[np.ma.MaskedArray],
                          latitudes: np.ndarray,
                          longitudes: np.ndarray,
                          obstimes: np.ndarray,
                          nlatbins=6,
                          max_elat=90,
                          extra_fit=False,
                          symmetric=True,
                          min_eclon=0,
                          max_eclon=180,
                          ninterp=101,
                          zero_at_poles=False,
                          force_tilt_dir=False):
    """Compute the tilt function across latitude bins based on the mean signals.

    Parameters
    ----------
    images : `list[np.ma.MaskedArray]`
        A list of masked arrays, where each array represents an image.
    latitudes : `np.ndarray`
        Array of latitudes corresponding to the observation positions in degrees.
    longitudes : `np.ndarray`
        Array of longitudes corresponding to the observation positions in degrees.
    obstimes : `np.ndarray`
        Array of observation times as astropy Time objects.
    nlatbins : `int`, optional
        Number of latitude bins for dividing the sky. Default is 6.
    max_elat : `float`, optional
        Maximum absolute latitude in degrees for binning. Default is 90.
    extra_fit : `bool`, optional
        If True, perform an additional fit to the best-fit tilt in each bin.
        Default is False.
    symmetric : `bool`, optional
        If True, enforce symmetry about the ecliptic plane. Default is True.
    min_eclon : `float`, optional
        Minimum ecliptic longitude relative to the Sun, in degrees,
        for considering a point.
        Default is 0.
    max_eclon : `float`, optional
        Maximum ecliptic longitude relative to the Sun, in degrees,
        for considering a point.
        Default is 180.
    ninterp : `int`, optional
        Number of interpolation points for the final tilt function.
        Default is 101.
    zero_at_poles : `bool`, optional
        If True, enforce the tilt function to be zero at the ecliptic poles.
        Default is False.
    force_tilt_dir : `bool`, optional
        If True, force all tilt values to be zero or negative. Default is False.

    Returns
    -------
    results_dict : `dict`
        A dictionary containing the following keys:
        - 'lats': Interpolated latitudes.
        - 'tilt_function': Tilt function values.
        - 'latedges': Latitude bin edges.
        - 'latavg': Average latitude in each bin.
        - 'tilts': Best-fit tilt values in each bin.
        - 'tilts_std': Standard deviations of the best-fit tilts.
        - 'i_latbins_obscrd': Latitude bin indices for each observation.
        - 'bin_idns': Observation indices in each bin.
        - 'eclons_from_Sun_latbins': Ecliptic longitudes from the Sun
           for each latitude bin.
        - 'avg_signals_latbins': Mean signals for each latitude bin.

    """
    comm = _get_mpi_comm()

    rank = comm.Get_rank()
    size = comm.Get_size()

    if symmetric is True:
        latedges = np.linspace(0, max_elat, nlatbins + 1)
    else:
        latedges = np.linspace(-max_elat, max_elat, nlatbins+1)

    # Average latitude in each bin, from the input latedges
    nlatbins = len(latedges) - 1

    # Create SkyCoord objects for the observation positions
    coords = SkyCoord(longitudes, latitudes, frame='icrs', unit='deg')
    # Transform the coordinates to the 'geocentrictrueecliptic' frame
    obscoords = coords.transform_to('geocentrictrueecliptic')

    # Compute ecliptic longitudes from the Sun for all images
    eclons_from_sun = get_eclons_from_Sun(obscoords, obstimes, unit='deg')

    # 1. Divide the pointings into sky coordinate bins
    if symmetric:
        assert np.all(latedges >= 0), "input latedges must be nonzero if symmetric is True"
        i_latbins_obscrd = pd.cut(np.abs(obscoords.lat.deg), bins=latedges, labels=False)
    else:
        i_latbins_obscrd = pd.cut(obscoords.lat.deg, bins=latedges, labels=False)

    # Divide longitudes into lat bins and wrap at 180 degrees
    eclons_from_sun_latbins = [
        eclons_from_sun[i_latbins_obscrd == i] for i in range(nlatbins)
    ]
    if rank == 0:
        logging.info(f"{nlatbins} ecliptic latitude bins between "
                     f"{min(latedges):.1f}-{max(latedges):.1f}deg, "
                     f"fitting absolute ecliptic longitudes {min_eclon:.1f}"
                     f"-{max_eclon:.1f}deg", flush=True)

    # Assign a slope m to each bin via a least-squares fit
    if rank == 0:
        jobs = np.array_split(np.arange(nlatbins), size)
    else:
        jobs = None

    comm.Barrier()
    # scatter jobs across cores
    jobs = comm.scatter(jobs, root=0)

    # where to store results
    tilts = []
    bs = []
    tilts_std = []
    avg_signals = []
    bin_idns = []

    # in each latitude bin,
    # perform a linear least-squares fit of the mean signal in each exposure
    #   to the distance from the Sun
    for ilatbin in jobs:
        idx = ((np.abs(eclons_from_sun_latbins[ilatbin]) >= min_eclon)
               & (np.abs(eclons_from_sun_latbins[ilatbin]) <= max_eclon))

        # Filter images and coordinates for the current latitude bin
        valid_indices = np.where(i_latbins_obscrd == ilatbin)[0]
        valid_idx = valid_indices[idx]
        images_thisbin = [images[i] for i in valid_idx]
        eclons_to_fit = eclons_from_sun_latbins[ilatbin][idx]

        logging.info(f"Rank {rank:02d} {datetime.datetime.now().strftime('%H:%M:%S')}: "
                     f"Getting {len(images_thisbin)} average signals in elatbin {ilatbin}", flush=True)

        # Get the average signal in each image
        avg_signals_thisbin = np.array([np.nanmean(image) for image in images_thisbin])

        # Mean-normalize within the bin itself: individually above and below the plane
        idx_pos = (obscoords.lat.deg[i_latbins_obscrd == ilatbin][idx] >= 0)
        assert sum(idx_pos) + sum(~idx_pos) == len(avg_signals_thisbin)

        if sum(idx_pos) > 0:
            # Make dimensionless to fit relative signals
            avg_signals_thisbin[idx_pos] /= np.nanmean(avg_signals_thisbin[idx_pos])
        if sum(~idx_pos) > 0:
            avg_signals_thisbin[~idx_pos] /= np.nanmean(avg_signals_thisbin[~idx_pos])

        avg_signals.append(avg_signals_thisbin)
        bin_idns.append(np.arange(len(images))[i_latbins_obscrd == ilatbin][idx])

        # Design matrix: fitting a slope and intercept
        # Absolute ecliptic longitude from the Sun
        A = np.column_stack((np.ones_like(eclons_to_fit), np.abs(eclons_to_fit)))

        # Covariance matrix: weight each exposure equally
        Cinv = np.ones_like(eclons_to_fit)

        # Perform the fit
        logging.info(f"Rank {rank:02d} {datetime.datetime.now().strftime('%H:%M:%S')}: "
                     f"Performing least-squares fit...", flush=True)
        (b, tilt), bestfit_Cinv = lstsq(avg_signals_thisbin, A, Cinv, return_err=True)

        # Uncertainties
        b_std, tilt_std = np.sqrt(np.diag(np.linalg.inv(bestfit_Cinv)))
        tilts.append(tilt)
        bs.append(b)
        tilts_std.append(tilt_std)

    comm.Barrier()  # Wait until all ranks finish their jobs

    # Gather results to rank 0
    tilts = comm.gather(tilts, root=0)
    bs = comm.gather(bs, root=0)
    tilts_std = comm.gather(tilts_std, root=0)
    avg_signals = comm.gather(avg_signals, root=0)
    bin_idns = comm.gather(bin_idns, root=0)

    if rank == 0:
        # Flatten lists to 1D
        tilts = [x for sublist in tilts for x in sublist]
        bs = [x for sublist in bs for x in sublist]
        tilts_std = [x for sublist in tilts_std for x in sublist]
        avg_signals = [x for sublist in avg_signals for x in sublist]
        bin_idns = [x for sublist in bin_idns for x in sublist]

        # Average latitude in each input bin
        latavg = 0.5 * (latedges[1:] + latedges[:-1])

        results_dict = dict(latedges=latedges, latavg=latavg,
                            symmetric=symmetric, min_eclon=min_eclon, max_eclon=max_eclon,
                            zero_at_poles=zero_at_poles, force_tilt_dir=force_tilt_dir,
                            tilts=tilts, tilts_std=tilts_std,
                            i_latbins_obscrd=i_latbins_obscrd, bin_idns=bin_idns,
                            eclons_from_Sun_latbins=eclons_from_sun_latbins, avg_signals_latbins=avg_signals,
                            bs=bs)

        # Optionally perform an extra linear fit to the best-fit tilt in each bin
        if extra_fit:
            Y = tilts
            A = np.column_stack((np.ones_like(latavg), latavg))
            Cinv = np.ones_like(latavg)

            if symmetric:
                if zero_at_poles:
                    logging.info("Adding zero tilt at ecliptic poles")
                    Y = np.append(Y, 0.)
                    A = np.vstack([A, [1., 90.]])
                    Cinv = np.append(Cinv, 1.)

                (b, m), bestfit_Cinv = lstsq(Y, A, Cinv, return_err=True)
                tilt_function = latavg * m + b
            else:
                idx_pos = (latavg >= 0)
                (b_pos, m_pos), bestfit_Cinv_pos = lstsq(Y[idx_pos],
                                                         A[idx_pos],
                                                         Cinv[idx_pos],
                                                         return_err=True)
                tilt_function_pos = latavg[idx_pos] * m_pos + b_pos

                idx_neg = (latavg < 0)
                (b_neg, m_neg), bestfit_Cinv_neg = lstsq(Y[idx_neg],
                                                         A[idx_neg],
                                                         Cinv[idx_neg],
                                                         return_err=True)
                tilt_function_neg = latavg[idx_neg] * m_neg + b_neg

                tilt_function = np.concatenate([tilt_function_neg, tilt_function_pos])
        else:
            tilt_function = tilts

        lats = np.linspace(min(latedges), max(latedges), ninterp)
        tilt_function = np.interp(lats, latavg, tilt_function)

        if force_tilt_dir:
            logging.info(f"Forcing tilt direction: setting {sum(tilt_function > 0)} "
                         f"tilts to zero", flush=True)
            tilt_function[tilt_function > 0] = 0.

        logging.info(f"Tilts: {tilt_function}")

        results_dict['lats'] = lats
        results_dict['tilt_function'] = tilt_function

    else:
        results_dict = None

    return comm.bcast(results_dict, root=0)


def fit_tilt_function(latavg: np.ndarray,
                      tilts: np.ndarray,
                      Cinv: np.ndarray,
                      symmetric: bool,
                      zero_at_poles: bool):
    """Fit a linear function to the best-fit tilts in each latitude bin.

    Parameters
    ----------
    latavg : `np.ndarray`
        Array of average latitudes for each bin.
    tilts : `np.ndarray`
        Best-fit tilt values for each latitude bin.
    Cinv : `np.ndarray`
        Inverse covariance matrix used for least-squares fitting.
    symmetric : `bool`
        If True, enforce symmetry about the ecliptic plane.
    zero_at_poles : `bool`
        If True, enforce the tilt function to be zero at the poles.

    Returns
    -------
    tilt_function : np.ndarray
        Fitted tilt function across the latitude range.

    """
    A = np.column_stack((np.ones_like(latavg), latavg))
    Y = tilts

    if symmetric and zero_at_poles:
        Y = np.append(Y, 0)
        A = np.vstack([A, [1., 90.]])
        Cinv = np.append(Cinv, 1.)

    b, m = lstsq(Y, A, Cinv)
    tilt_function = latavg * m + b
    return tilt_function


def tilt_correct(images: list[np.ma.MaskedArray],
                 latitudes,
                 longitudes,
                 obstimes,
                 wcss,
                 tilt_function,
                 exclude_poles=True,
                 save_uncorrected=False,
                 save_poles=True):
    """Apply tilt correction based on ecliptic coordinates and a tilt function.

    Parameters
    ----------
    images : `list[np.ma.MaskedArray]`
        List of masked arrays, where each array represents an image for tilt correction.
    latitudes : `np.ndarray`
        Array of ICRS latitudes corresponding to the observation positions (degrees).
    longitudes : `np.ndarray`
        Array of ICRS longitudes corresponding to the observation positions (degrees).
    obstimes : `np.ndarray`
        Array of observation times as `astropy.time.Time` objects.
    wcss : `list`
        List of WCS (World Coordinate System) objects, corresponding to each image.
    tilt_function : `dict`
        Dictionary containing tilt function information. It includes:
        - 'lats': The latitude bins used for binning observations.
        - 'tilt_function': The tilt correction function for each bin.
        - 'min_eclon': Minimum ecliptic longitude from the Sun (degrees).
        - 'max_eclon': Maximum ecliptic longitude from the Sun (degrees).
        - 'symmetric': Boolean indicating if the tilt function is symmetric
           about the ecliptic plane.
    exclude_poles : `bool`, optional
        If True, exclude images near the ecliptic poles from tilt correction.
        Default is True.
    save_uncorrected : `bool`, optional
        If True, save the original (uncorrected) images for cases where
        the tilt correction is not applied. Default is False.
    save_poles : `bool`, optional
        If True, save images that hit the ecliptic poles
        (even when tilt correction is not applied). Default is True.

    Returns
    -------
    corrected_images : `list[np.ma.MaskedArray]`
        List of tilt-corrected masked images. For images where no correction
        was applied, the original image is returned (if `save_uncorrected=True`).

    """
    comm = _get_mpi_comm()

    # MPI rank and size
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Unpack tilt function
    lats, tilts = tilt_function['lats'], tilt_function['tilt_function']
    min_eclon, max_eclon = tilt_function['min_eclon'], tilt_function['max_eclon']

    if rank == 0:
        logging.info(f"Using tilt function evaluated at {len(tilts)} values", flush=True)
        logging.info(f"Fitting ecliptic longitudes {min_eclon}-{max_eclon}deg", flush=True)

    # Create SkyCoord objects for the observation positions
    coords = SkyCoord(longitudes, latitudes, frame='icrs', unit='deg')
    # Transform the coordinates to the 'geocentrictrueecliptic' frame
    obscoords = coords.transform_to('geocentrictrueecliptic')

    # Compute ecliptic longitudes from the Sun for all images
    eclons_from_sun = get_eclons_from_Sun(obscoords, obstimes, unit='deg')

    idx_eclon = (np.abs(eclons_from_sun) >= min_eclon) & (np.abs(eclons_from_sun) <= max_eclon)

    # Assign a bin index to each observation
    obscrd_lats = np.abs(obscoords.lat.deg) if tilt_function['symmetric'] else obscoords.lat.deg
    i_latbins_obscrd = pd.cut(obscrd_lats, bins=lats, labels=False)
    idx_elat = ~np.isnan(i_latbins_obscrd)
    assert len(i_latbins_obscrd) == len(obscrd_lats) == len(images)

    # Determine which pointings to correct for solar elongation tilt
    idx_to_correct = idx_eclon & idx_elat

    if rank == 0:
        poles_str = " (except where exposures hit an ecliptic pole)" if exclude_poles else ""
        logging.info(f"{len(images)} total images, correcting {sum(idx_to_correct)} {poles_str} "
                     f"\n\t{sum(idx_elat)} within the input latitude bins "
                     f" ({min(lats):.1f}-{max(lats):.1f}deg)"
                     f"\n\t{sum(idx_eclon)} within the input absolute longitude range from Sun "
                     f"({min_eclon:.1f}-{max_eclon:.1f}deg)", flush=True)

    # Index to determine which exposures to generate
    idx_to_generate = np.arange(len(images))

    if rank == 0:
        jobs = np.array_split(idx_to_generate, size)
        logging.info(f"Correcting {len(idx_to_generate)} "
                     f"input exposures by this tilt function...", flush=True)
    else:
        jobs = None

    comm.Barrier()
    # Scatter jobs across cores
    jobs = comm.scatter(jobs, root=0)

    # Create a list to store the processed images
    processed_images = [None] * len(images)
    corrected_images = [None] * len(images)

    for idx in jobs:
        image = images[idx].data
        wcs = wcss[idx]

        # Skip correction if the image falls outside of
        # ecliptic latitude bins or longitude range
        if idx_to_correct[idx] is False:
            if save_uncorrected is True:
                processed_images[idx] = image
            continue

        if exclude_poles is True:
            xp, yp = meshgrid(image.shape[0])
            # change mode to 'all' when distortion is present.
            lats = SkyCoord.from_pixel(xp, yp, wcs=wcss[idx]).geocentrictrueecliptic.lat

            if np.allclose(lats.ravel()[np.abs(np.abs(lats)-90.*u.deg).argmin()].deg, 90):
                if save_poles is True:
                    processed_images[idx] = image
                continue

        # Construct the tilt template and correct the image
        tilt_template = construct_tilt_template(image,
                                                wcs,
                                                obstimes[idx],
                                                tilts[int(i_latbins_obscrd[idx])])
        tiltcorrected_image = np.ma.array(np.divide(image, tilt_template,
                                          where=(~np.isnan(tilt_template) & (tilt_template != 0)),
                                          out=np.full_like(image, np.nan)),
                                          mask=np.isnan(image))

        # Store the processed image in memory
        corrected_images[idx] = tiltcorrected_image

    comm.Barrier()  # Wait until all ranks finish their jobs

    return corrected_images


def construct_tilt_template(image, wcs, obstime, tilt):
    """Construct a tilt template object and tilt function."""
    # pixel indices
    xp, yp = meshgrid(image.shape[0])

    # get the sky coordinate of each pixel
    sc = SkyCoord.from_pixel(xp, yp, wcs=wcs)

    # the raw longitudes
    lons = Angle(sc.geocentrictrueecliptic.lon.deg, unit='deg')
    # get the Sun's longitude, and wrap the distance
    # to the Sun at 180 to avoid discontinuities
    sun_pos = get_body('sun', obstime).geocentrictrueecliptic
    lons = np.abs(Angle(lons - sun_pos.lon, unit='deg').wrap_at(180*u.deg))
    # shift to get longitudes relative to the center of the exposure
    lons -= np.mean(lons)

    # construct tilt template: the mean (or intercept) is unity
    tilt_template = lons.deg * tilt + 1  # dimensionless (fractional change)

    return tilt_template
