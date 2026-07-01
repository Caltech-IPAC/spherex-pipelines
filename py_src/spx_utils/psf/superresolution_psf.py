__all__ = ["SuperresolutionPSF", "bin_ranges"]

# Standard library imports
import time
import warnings

# Related third party imports
from astropy.convolution import convolve
from astropy.wcs import Sip
from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy.ndimage import gaussian_filter
from photutils.profiles import RadialProfile
from scipy.ndimage import binary_dilation

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_utils.psf.psf_parameters import SuperresolutionPSF
from spx_utils.psf.input_map import Sources
from spx_utils.psf.deconvolution import TurboRLD, DeconvolutionIterateVirtual
from spx_utils.psf.stack import StackerVirtual
from spx_utils.psf.regrid import psf_grid
from spx_utils import ImageCatalogIterable
from spx_utils import SourcePipelineClass


def bin_ranges(bin_x: int,
               bin_y: int,
               grid_size: int = 20400,
               oversample: int = 10) -> tuple[list, list, list, list, list, list]:
    """Calculate bin ranges based on number of total PSF zones (aka bins).

    Parameters
    ----------
    bin_x : `int`
        Total number of PSF zones in x.
    bin_y : `int`
        Total number of PSF zones in y.

    Returns
    -------
    x_min : `list`
        Minimum x for all PSF zone bins.
    x_max : `list`
        Maximum x for all PSF zone bins.
    y_min : `list`
        Minimum y for all PSF zone bins.
    y_max : `list`
        Maximum y for all PSF zone bins.
    binedge_x : `list`
        Bin edge x values.
    binedge_y : `list`
        Bin edge y values.
    """

    gridsize = grid_size//oversample

    binwidth_x = gridsize // bin_x
    binwidth_y = gridsize // bin_y
    y1 = 0
    y2 = 0
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    binedge_x = []
    binedge_y = []
    for j in range(0, bin_y):
        binedge_y.append(y1)
        y2 += binwidth_y
        x1 = x2 = 0
        for k in range(0, bin_x):
            if j == 0:
                binedge_x.append(x1)
            x_min.append(x1)
            y_min.append(y1)
            x2 += binwidth_x
            x_max.append(x2)
            y_max.append(y2)
            x1 = x2
        if j == 0:
            binedge_x.append(x1)
        y1 = y2
    binedge_y.append(y1)

    if bin_x != bin_y:
        if len(binedge_x)-1 == bin_y or len(binedge_y)-1 == bin_x:
            raise ValueError("ValueError: Incorrect mapping of x and y zones. Are they switched?")

    return x_min, x_max, y_min, y_max, binedge_x, binedge_y


def convolve_psf_with_jitter(psf_avg: np.ndarray,
                             jdk: np.ndarray,
                             group_sizes: list,
                             output_shape=(121, 101, 101),
                             smooth_jdk: bool = False,
                             sigma_jdk=1.5) -> np.ndarray:
    """
    Convolve psf_avg bins with the appropriate jitter_psf bins based on group_sizes.

    Parameters
    ----------
    psf_avg : `~numpy.ndarray`
        Average PSFs (shape: [bins, size, size]).
    jdk : `~numpy.ndarray`
        Jitter difference kernels (shape: [bins, size, size]).
    group_sizes : list
        Mapping of PSF zones to jitter PSF groups.
    output_shape : tuple, optional
        Shape of the output effective PSFs, by default (121, 101, 101).

    Returns
    -------
    eff_psf : `~numpy.ndarray`
        Effective PSFs (shape: output_shape).
    """
    # Initialize array for effective PSFs
    eff_psf = np.zeros(output_shape)

    # Iterate through the group sizes and
    # map each psf_avg bin to the appropriate jitter_psf
    for group_index, group in enumerate(group_sizes):
        # Get the jitter kernel for this group
        if smooth_jdk:
            # Apply smoothing
            jitter_kernel = gaussian_filter(jdk[group_index], sigma=sigma_jdk)
        else:
            jitter_kernel = jdk[group_index]

        # Convolve each bin in the group with the jitter kernel
        for bin_index in group:
            eff_psf[bin_index] = convolve(
                psf_avg[bin_index],
                jitter_kernel,
                normalize_kernel=False,
                nan_treatment="interpolate",
                boundary="fill"
            )

    return eff_psf


def stack_multiple_images(iterableImgCat: ImageCatalogIterable,
                          sip: Sip,
                          x_min: list,
                          x_max: list,
                          y_min: list,
                          y_max: list,
                          binedge_x: list,
                          binedge_y: list,
                          ibin: int,
                          plt: bool,
                          verbose: bool,
                          config):
    """Stacks superresolution PSF for multiple exposures.

    Parameters
    ----------
    iterableImgCat : `~spx_utils.ImageCatalogIterable`
        Iterable for image and catalog pairs.
    x_min : `list`
        Minimum x for all PSF zone bins.
    x_max : `list`
        Maximum x for all PSF zone bins.
    y_min : `list`
        Minimum y for all PSF zone bins.
    y_max : `list`
        Maximum y for all PSF zone bins.
    binedge_x : `list`
        Bin edge x values.
    binedge_y : `list`
        Bin edge y values.
    ibin : `int`
        PSF zone number.
    plt : `bool`
        Make plots.
    verbose : `bool`
        Processing in verbose mode.
    """
    cat_down_list = []
    abmag_list = []
    good_mags_list = []
    stack_sum = 0
    stack5_sum = 0
    stack50_sum = 0
    stack100_sum = 0
    stack_delete_list = []
    iexp = 0

    for image, catalog in iterableImgCat:

        if verbose:
            print(f"EXPOSURE {iexp}, BIN {ibin}, rldIterations {config.rldIterations}")
        spximg = image

        # Skip this bin if SUR_ERROR exists in it
        bin_slice = spximg.flags[y_min[ibin]:y_max[ibin], x_min[ibin]:x_max[ibin]]
        # if np.any(bin_slice & Flag.SUR_ERROR):  # skips any combination with SUR_ERROR
        if np.any((bin_slice == Flag.SUR_ERROR)):  # skips only if SUR_ERROR alone exists in bin.
            print(f"Skipping bin {ibin} in exposure {iexp}: SUR_ERROR found in zone.")
            continue

        bin_slice = spximg.flags[y_min[ibin]:y_max[ibin], x_min[ibin]:x_max[ibin]]
        bad_mask = (bin_slice & Flag.TRANSIENT) > 0
        frac_bad = np.count_nonzero(bad_mask) / bad_mask.size
        if frac_bad > 0.05:  # reject if >5% of zone pixels are affected
            print("*****************************************************************")
            print(f"Skipping exposure {iexp}, bin {ibin}: {frac_bad*100:.1f}% bad pixels.")
            print("*****************************************************************")
            continue

        zone_data = spximg.data[y_min[ibin]:y_max[ibin], x_min[ibin]:x_max[ibin]]
        if np.isnan(zone_data).mean() > 0.05:
            print(f"Skipping exposure {iexp}, bin {ibin}: contains >5% NaNs.")
            continue
        if np.nanstd(zone_data) < 1e-3:
            print(f"Skipping exposure {iexp}, bin {ibin}: low variance zone (may be flat fill).")
            continue

        try:
            if spximg.header["FINAST"] != 0:
                print(f'FINAST = {spximg.header["FINAST"] }')
                continue
        except Exception as e:
            print(f"{e}: no FINAST flag")
            # TODO continue after replace E2E integration test with real data
            # continue  # no finast flag means something is wrong with image
        try:
            sps_flags = spximg.header.get("SPS_FLAGS", "")
            if "saa" in [f.strip().lower() for f in sps_flags.split(",")]:
                print(f"skipping obs in saa: {spximg.header['OBSID']}D{spximg.getArrayId()}")
                continue
        except Exception as e:
            print(f"{e}: no SPS_FLAGS in header.")
            # TODO continue after replace E2E integration test with real data
            # continue

        # Exclude images in galactic plane;
        crpix = spximg.wcs.wcs.crpix
        coord = spximg.wcs.pixel_to_world(crpix[0], crpix[1])
        b = coord.transform_to('galactic').b.value
        if abs(b) < config.b:
            print(f'Applying |b| < {config.b} cut.')
            continue

        # number of transient flags
        try:
            if spximg.header["L1 N_TRANSIENT"] > 100000:
                continue
        except Exception as e:
            print(f"{e}: no N_TRANSIENT flag")
            # TODO continue after replace E2E integration test with real data
            # continue

        cat = catalog
        # Image data, get only once (not for each bin).
        grid_down = spximg.data.copy()
        bad_flags = Flag.SUR_ERROR | Flag.TRANSIENT | Flag.NONLINEAR | Flag.SUR_MISMATCH
        bad_pixel_mask = (spximg.flags & bad_flags) > 0
        dilated_mask = binary_dilation(bad_pixel_mask, structure=np.ones((25, 25)))  # 20×20 box

        grid_down[dilated_mask] = np.nan
        # Add 5-pixel border mask
        border_width = 10
        grid_down[:border_width, :] = np.nan
        grid_down[-border_width:, :] = np.nan
        grid_down[:, :border_width] = np.nan
        grid_down[:, -border_width:] = np.nan

        # Masks.
        starmask = np.zeros(spximg.flags.shape, dtype=np.bool_)
        statmask = np.zeros(spximg.flags.shape, dtype=np.bool_)

        # if plt is True and ibin == 0:
        #     from spx_utils.psf.psf_plots import plot_zones
        #     plot_zones(spximg=grid_down, binedge_x=binedge_x, binedge_y=binedge_y)
        det_id = spximg.getArrayId()

        # Make stack for each bin, and update stack for each exposure.
        # Initialization for selecting sources in PSF zone.
        sources = Sources(config=config, spximg=spximg, cat=cat, det_id=det_id,
                          x_min=x_min[ibin], x_max=x_max[ibin], y_min=y_min[ibin], y_max=y_max[ibin])
        # Good catalog sources, it's a table.
        cat_down = sources.cat_coords()  # need to keep
        cat_down_list.append(cat_down)
        # Get magnitude.
        abmag = sources.vmag()
        abmag_list.append(abmag)
        # Apply magnitude restriction.
        good_mags = sources.magnitude(abmag=abmag)
        good_mags_list.append(good_mags)
        iexp += 1
        super_psf_exp =\
            SuperresolutionPSF(source_coord_down=cat_down,
                               abmag=abmag,
                               good_mags=good_mags,
                               )
        super_psf_exp.rldIterations = config.rldIterations
        super_psf_exp.rldIterations_decon = config.rldIterations_decon
        super_psf_exp.stackDeconIterations = config.stackDeconIterations
        super_psf_exp.damper_factor = config.damper_factor
        super_psf_exp.factor_up = config.factor_up
        super_psf_exp.factor_down = config.factor_down

        # Do stacking.
        (stack, _, stack_delete, _, _,
            stack5, stack50, stack100, _) = StackerVirtual(grid_down,
                                                           cat_down,
                                                           abmag,
                                                           starmask,
                                                           statmask,
                                                           super_psf_exp,
                                                           verbose)
        stack_sum += np.nan_to_num(stack, nan=0.0)
        stack5_sum += np.nan_to_num(stack5, nan=0.0)
        stack50_sum += np.nan_to_num(stack50, nan=0.0)
        stack100_sum += np.nan_to_num(stack100, nan=0.0)
        stack_delete_list.append(stack_delete)

    num_images = iexp
    print(f"number of images in stack = {num_images}")
    # Averages for stacks.
    stack = stack_sum/num_images
    stack5 = stack5_sum/num_images
    stack50 = stack50_sum/num_images
    stack100 = stack100_sum/num_images
    # Needed complete set of data for deconvolution.
    stack_delete = np.concatenate(stack_delete_list)
    all_cat_down = np.concatenate(cat_down_list)
    all_abmag = np.concatenate(abmag_list)
    all_good_mags = tuple(np.concatenate(subarr) for subarr in zip(*good_mags_list))
    num_sources = sources.num_sources(all_abmag)
    super_psf =\
        SuperresolutionPSF(source_coord_down=all_cat_down, abmag=all_abmag, good_mags=all_good_mags)

    return (
        stack,
        stack5,
        stack50,
        stack100,
        super_psf,
        stack_delete,
        all_cat_down,
        num_sources,
    )


def stack_psf(spximg: SPHERExImage | None,
              cat: Table | None,
              iterableImgCat: ImageCatalogIterable | None,
              imageHandles,
              catalogHandles,
              sip: Sip,
              multi_exprs: bool,
              bins_x: int,
              bins_y: int,
              bins: int,
              ibin: int,
              plt: bool,
              verbose: bool,
              array_id: int,
              config) -> tuple[np.ndarray, SuperresolutionPSF]:
    """Performs PSF stacking per bin.

    Parameters
    ----------
    spximg : `spx_data.SPHERExImage` or `None`
        Input SPHEREx image.
    cat : `~astropy.table.Table` | `None`
        Input source reference catalog.
    iterableImgCat : `~spx_utils.ImageCatalogIterable`
        Image with corresponding catalog.
    multi_exprs : `bool`
        Flag for multiple exposures.
    bins_x : `int`
        Number of PSF zones in x-direction.
    bins_y : `int`
        Number of PSF zones in y-direction.
    bins : `int`
        Total number of PSF zones.
    ibin : `int`
        Current PSF zone being processed.
    plt : `bool`
        Flag to make plots.
    verbose : `bool`
        Flag to output processing details.

    Returns
    -------
    stack, super_psf, stack_delete, cat_down :
        `tuple[np.ndarray, SuperresolutionPSF, np.ndarry, np.ndarray]`
        Superresolution stack PSF and needed arrays for deconvolution and IBP.
    """
    if bins_x * bins_y != bins:
        raise ValueError(f"x bins {bins_x} and y bins {bins_y} "
                         f"does not equal total bins {bins}")
    # if plt is True:
    #     from spx_utils.psf.psf_plots import plot_mult_stacks
    x_min, x_max, y_min, y_max, binedge_x, binedge_y = bin_ranges(bins_x, bins_y)

    if multi_exprs is True:
        iterableImgCat = ImageCatalogIterable(
            imageHandles,
            catalogHandles,
            srcPipeFlag=SourcePipelineClass.PSF,
            addPixelCoords=True,
            pixelCoordsWithDistortion=True,
            filterSourcesOutsideImage=True,
        )

        stack, stack5, stack50, stack100, super_psf, stack_delete, all_cat_down, num_sources =\
            stack_multiple_images(iterableImgCat, sip, x_min, x_max, y_min, y_max,
                                  binedge_x, binedge_y, ibin, plt, verbose, config)
    # Single Exposure.
    else:
        # Image data.
        all_grid_down = spximg.data
        # grid_down[spximg.flags >= 1] = np.nan
        # Masks.
        all_starmask = np.zeros(spximg.flags.shape, dtype=np.bool_)
        all_statmask = np.zeros(spximg.flags.shape, dtype=np.bool_)
        # if plt is True and ibin == 0:
        #     plot_zones(spximg=all_grid_down, binedge_x=binedge_x, binedge_y=binedge_y)
        det_id = spximg.getArrayId()

        # Initialization for selecting sources in PSF zone.
        sources = Sources(config=config, spximg=spximg, cat=cat, det_id=det_id,
                          x_min=x_min[ibin], x_max=x_max[ibin], y_min=y_min[ibin], y_max=y_max[ibin])
        # Good catalog sources, it's a table.
        all_cat_down = sources.cat_coords()
        # Get magnitude.
        all_abmag = sources.vmag()

        # Apply magnitude restriction.
        all_good_mags = sources.magnitude(abmag=all_abmag)

        super_psf =\
            SuperresolutionPSF(source_coord_down=all_cat_down, abmag=all_abmag, good_mags=all_good_mags)

        (stack, _, stack_delete, _, _,
            stack5, stack50, stack100, _) = StackerVirtual(all_grid_down,
                                                           all_cat_down,
                                                           all_abmag,
                                                           all_starmask,
                                                           all_statmask,
                                                           super_psf,
                                                           verbose)
    # # Plot stacks
    # if plt is True:
    #     plot_mult_stacks(stack, stack5, stack50, stack100, num_sources,
    #                      stack_delete, config.plot_path, ibin, array_id)

    return stack, super_psf, stack_delete, all_cat_down


def deconvolve_psf(stack,
                   s,
                   stack_delete,
                   cat_down,
                   plt: bool,
                   verbose: bool,
                   plot_path: str,
                   ibin: int,
                   array_id: int,
                   obsid: str = "") -> tuple[np.ndarray, SuperresolutionPSF]:
    """Performs PSF analysis.

    Parameters
    ----------
    stack : `~numpy.ndarray`
        Superresolution stacked PSF.
    s : `SuperresolutionPSF`
        Object to keep track of PSF zone data and corresponding catalog.
    stack_delete : `~numpy.ndarray`
        Stack elements to delete.
    cat_down : `~numpy.ndarray`
        Source reference catalog.
    plt : `bool`
        Make plots.
    verbose : `bool`
        Print verbosity.

    Returns
    -------
    deconvolved_stack_bins, super_psf : tuple[np.ndarray, SuperresolutionPSF]
        Deconvolved stack for each PSF zone.
        Superresolution psf object for shared variables.
    """
    # preprocess stack
    stack[stack < 0] = 0

    if plt is True:
        from spx_utils.psf.psf_plots import plot_decon_set

    # ==============================
    # =======P_gird aka P_pix=======
    # ==============================
    psf_pix, psf_pix_padded = psf_grid(s.factor_up)

    # ==============================
    # ====Prep for Deconvolution====
    # ==============================
    # Damping adjustment, calculate damping array.
    if (s.halfStampSizeUp_tru == s.halfStampSizeUp):
        stackSize = np.int64(s.halfStampSizeUp * 2)
        damper = np.zeros((stackSize, stackSize), dtype=np.int64())
        # Get a meshgrid
        xx, yy = np.meshgrid(np.arange(0, stackSize), np.arange(0, stackSize))
        # Get the distance from the center
        distFromCenter = np.sqrt((xx - stackSize // 2) ** 2 + (yy - stackSize // 2) ** 2)
        damper[distFromCenter >= (stackSize // 2) * s.damper_factor] = 1
        # Implement damping at X% of the radius or more
        damper = damper * 5
    else:
        stackSize_tru = np.int64(s.halfStampSizeUp_tru * 2)
        damper = np.zeros((stackSize_tru, stackSize_tru), dtype=np.int64())
        xx, yy = np.meshgrid(np.arange(0, stackSize_tru), np.arange(0, stackSize_tru))
        distFromCenter = np.sqrt((xx - stackSize_tru // 2) ** 2
                                 + (yy - stackSize_tru // 2) ** 2)
        damper[distFromCenter >= (stackSize_tru // 2) * s.damper_factor] = 1
        damper = damper * 5

    # ==============================
    # =====RLD PSF Deconvolution====
    # ==============================
    tdec = time.time()
    if verbose is True:
        print('Deconvolution...')
    # RLS dfunction - options for damping,
    # weighting, readout error, and subsampling
    Deconvolved_Stack = np.zeros((s.stackSize, s.stackSize))
    if (s.halfStampSizeUp_tru == s.halfStampSizeUp):
        Deconvolved_Stack = TurboRLD(stack, psf_pix, s.rldIterations, damper, verbose,
                                     weight=0, readout=0)

    else:
        # Only RLD the halfStampSizeUp_tru size for stability
        Deconvolved_Stack[
            s.halfStampSizeUp - s.halfStampSizeUp_tru:
                s.halfStampSizeUp + s.halfStampSizeUp_tru,
            s.halfStampSizeUp - s.halfStampSizeUp_tru:
                s.halfStampSizeUp + s.halfStampSizeUp_tru] =\
            TurboRLD(
                stack[s.halfStampSizeUp - s.halfStampSizeUp_tru: s.halfStampSizeUp
                      + s.halfStampSizeUp_tru, s.halfStampSizeUp
                      - s.halfStampSizeUp_tru: s.halfStampSizeUp
                      + s.halfStampSizeUp_tru],
                psf_pix,
                s.rldIterations,
                damper,
                verbose,
                weight=0,
                readout=0)

    # If not using IBP, apply circular mask now
    # Only apply circmask if the decon isn't used in the IBP later
    if ((s.FLG_circmask == 1) & (s.FLG_stackDeconIterateEnable != 1)):
        xgrid, ygrid = np.meshgrid(np.linspace(0, s.stackSize - 1, s.stackSize),
                                   np.linspace(0, s.stackSize - 1, s.stackSize))
        circmask = ((xgrid - (s.stackSize / 2) - 1) ** 2
                    + (ygrid - (s.stackSize / 2) - 1) ** 2) >= (s.circrad) ** 2
        Deconvolved_Stack = Deconvolved_Stack * ~circmask

    # If normalization desired, apply
    if ((s.FLG_norm >= 3) | (s.FLG_norm == -1)):
        denom = np.sum(Deconvolved_Stack)
        if denom == 0:
            denom = np.sum(Deconvolved_Stack) + 10e-12
            warnings.warn("Deconvolved stack sum is zero. Cannot average. Check stacking.")
        Deconvolved_Stack = Deconvolved_Stack * np.sum(stack) / denom

    # Deconvolved stack plot
    rld_decon = np.copy(Deconvolved_Stack)

    # ==============================
    # ===========IBP PSF============
    # ==============================
    # If IBP is enabled, perform.
    if ((s.FLG_stackDeconIterateEnable == 1)):
        Deconvolved_Stack, Deconvolved_Stack_finalIteration = DeconvolutionIterateVirtual(
            stack, Deconvolved_Stack, psf_pix, stack_delete,
            damper, verbose, cat_down, s)

    # In case of IBP loss of structure -
    # due to testing a poor case with extremely low number of sources -
    # reduce iterations to get a real IBP
    if np.sum(Deconvolved_Stack[s.halfStampSizeUp - 10: s.halfStampSizeUp + 10,
                                s.halfStampSizeUp - 10: s.halfStampSizeUp + 10]) == 0.0:
        s.FLG_deconRerun = 1   # Set flag to 1 b/c rerun occured

        # Force normalization during the decon process
        # This is preferrable to adjusting decon iterative parameters -
        # as it just forces sums to be consistent
        s.FLG_normAdj = 2
        # Same as above (converts fp64 eps of E-16 to E-12)
        stackDeconIterateFillerPower_epsFixAltMode = 10000
        s.stackDeconIterateFillerPower = stackDeconIterateFillerPower_epsFixAltMode
        if verbose is True:
            print('TO WHOM IT MAY CONCERN, please:\n'
                  + 'IBP Numerical instability detected,'
                  + 'rerunning with enhanced normalization FLG_norm='
                  + str(s.FLG_normAdj) + ', eps adjusted x'
                  + str(stackDeconIterateFillerPower_epsFixAltMode)
                  + ' from fp64 eps')
        # Redo IBP with new iteration numbers
        Deconvolved_Stack, Deconvolved_Stack_finalIteration =\
            DeconvolutionIterateVirtual(stack, Deconvolved_Stack, psf_pix, stack_delete,
                                        damper, verbose, cat_down, s)
    if verbose is True:
        print('Decon IBP (' + str(s.rldIterations_decon) + ', '
              + str(Deconvolved_Stack_finalIteration) + ' iters) sum: '
              + str(np.sum(Deconvolved_Stack)) + '   FLG_norm = '
              + str(s.FLG_norm)
              + '(2 means this decon was norm''d, 1 means only error-adj stack was norm''d)')

    # If using circular mask, apply now
    if s.FLG_circmask == 1:
        xgrid, ygrid = np.meshgrid(np.linspace(0, s.stackSize - 1, s.stackSize),
                                   np.linspace(0, s.stackSize - 1, s.stackSize))
        circmask = ((xgrid - (s.stackSize / 2) - 1) ** 2
                    + (ygrid - (s.stackSize / 2) - 1) ** 2) >= (s.circrad) ** 2
        Deconvolved_Stack = Deconvolved_Stack * ~circmask

    denom = np.sum(Deconvolved_Stack)
    if denom == 0:
        denom += 10e-12
        print("Deconvolved stack sum is zero. Cannot average. Check stacking.")

    print(f"np.sum(stack) {np.sum(stack)}")
    print(np.nansum(stack))
    print(f"denom {denom}")
    print(f"stack.shape = {stack.shape}")

    Deconvolved_Stack = (Deconvolved_Stack * np.sum(stack) / denom)

    # Compute FWHM_RLD
    ac = [rld_decon.shape[0] // 2, rld_decon.shape[1] // 2]
    radii = np.arange(0.1, 30, 1)
    pro = RadialProfile(rld_decon, ac, radii)
    fwhm_rld_pix = pro.gaussian_fwhm
    fwhm_rld_arcsec = fwhm_rld_pix * 6.15 / s.factor_up
    print(f"[Bin {ibin:03}] FWHM_RLD = {fwhm_rld_pix:.2f} px = {fwhm_rld_arcsec:.2f} arcsec")

    # Compute FWHM_IBP from the deconvolved stack
    aper_center = [Deconvolved_Stack.shape[0] // 2, Deconvolved_Stack.shape[1] // 2]
    radii = np.arange(0.1, 30, 1)
    profile = RadialProfile(Deconvolved_Stack, aper_center, radii)
    fwhm_pix = profile.gaussian_fwhm
    fwhm_arcsec = fwhm_pix * 6.15 / s.factor_up  # 6.15 arcsec/px nominal scale
    print(f"[Bin {ibin:03}] FWHM_IBP = {fwhm_pix:.2f} px = {fwhm_arcsec:.2f} arcsec")

    # ==============================
    # ==========More Plots==========
    # ==============================
    if plt is True:
        plot_decon_set(stack, Deconvolved_Stack, s.factor_up, psf_pix_padded, rld_decon,
                       plot_path, ibin, array_id, fwhm_arcsec, fwhm_rld_arcsec)

    # ==============================
    # =========Save as FITS=========
    # ==============================
    FITSname = (plot_path+f'/PSF_exp{1}_band{1}_bin{1}_factorUp{s.factor_up}.fits')
    if plt is True:
        fits.PrimaryHDU(Deconvolved_Stack).writeto(FITSname, overwrite=True, checksum=True)

    dectime = time.time() - tdec
    if verbose is True:
        print('Decon Time:\t' + str(dectime))
        print(f"Deconvolved_Stack.shape = {Deconvolved_Stack.shape}")

    # return deconvolved stack and superresolution psf parameters
    return Deconvolved_Stack, rld_decon
