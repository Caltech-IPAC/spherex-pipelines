import numpy as np
from numba import prange
from photutils.psf.matching import (
    create_matching_kernel, TopHatWindow, CosineBellWindow, TukeyWindow,
    SplitCosineBellWindow, HanningWindow)


def photutils_jdk(decon_mission_stack: np.ndarray,
                  psf_lib: np.ndarray,
                  halfstampsize: float,
                  bins: int,
                  trim_pixels: int,
                  window: str,
                  alpha: float,
                  beta: float,
                  verbose=False,
                  plt=False):
    """Determine the jitter difference kernel (JDK) using photutils.

    This function computes the JDK by creating jitter point spread functions (PSFs)
    for a mission stack and a given PSF library. The method involves matching kernels
    with optional trimming and windowing.

    Parameters
    ----------
    decon_mission_stack : `np.ndarray`
        The deconvolved mission stack containing PSFs for different jitter bins.
    psf_lib : `np.ndarray`
        The PSF library, where the average PSF will be computed.
    halfstampsize : float
        The half-size of the jitter PSF stamp,
        (total size will be `(2 * halfstampsize + 1)`).
    bins : `int`
        The number of jitter bins in the mission stack.
    trim_pixels : `int`
        The number of pixels to trim from the edges of the jitter PSF.
        Set to 0 to avoid trimming.
    window : `str`
        The type of window function to apply when creating the matching kernel.
        Options include:
            - `""` (no window)
            - `"tophat"`
            - `"cosine"`
            - `"hanning"`
            - `"splitcosine"`
            - `"tukey"`
    alpha : float
        Parameter for the window function, such as the fraction of the window
        for tapering (depends on the specific window type).
    beta : float
        Secondary parameter for the `splitcosine` window type.
    verbose : bool, optional
        If `True`, prints additional debug information about intermediate shapes
        and processing steps.
    plt : bool, optional
        If `True`, plots the resulting jitter PSFs using a helper plotting function.

    Returns
    -------
    jdk : `np.ndarray`
        The computed jitter difference kernels, with a shape `(bins, ...)`, where the
        exact dimensions depend on the input parameters and trimming.

    Notes
    -----
    - The function uses different window functions for creating matching kernels.
    - The resulting kernels can optionally be trimmed to remove edge artifacts.
    - If `plt` is enabled, the PSFs are plotted, useful for visual verification.

    """
    if window == "":
        win = None
    elif window == "tophat":
        win = TopHatWindow(alpha)
    elif window == "cosine":
        win = CosineBellWindow(alpha)
    elif window == "hanning":
        win = HanningWindow()
    elif window == "splitcosine":
        win = SplitCosineBellWindow(alpha, beta)
    elif window == "tukey":
        win = TukeyWindow(alpha)

    # Create array for jitter PSF
    jdks = []
    jitter_psf = np.zeros((halfstampsize*2+1, halfstampsize*2+1))
    # psf_avg = np.mean(psf_lib.data, axis=0)
    if verbose is True:
        print(f"psf_avg {psf_lib.shape}")
        print(f"decon {decon_mission_stack[0].shape}")

    # Split into 9 groups as evenly as possible
    num_groups = 9
    psf_lib_zones = psf_lib.shape[0]
    group_sizes = np.array_split(np.arange(psf_lib_zones), num_groups)
    if verbose:
        # Print how the 121 bins are divided into 9 groups
        print("Bin group sizes:")
        for i, group in enumerate(group_sizes):
            print(f"PSFzone {i + 1}: {len(group)} bins")

    rebinned_psf_avg = np.array([psf_lib[group].mean(axis=0) for group in group_sizes])

    for j in prange(0, bins):
        # rebin psf_avg from 11x11 to 3x3
        jitter_psf = create_matching_kernel(rebinned_psf_avg[j], decon_mission_stack[j], window=win)
        if trim_pixels > 0:
            trimmed_psf = jitter_psf[trim_pixels:-trim_pixels, trim_pixels:-trim_pixels]
            jdks.append(trimmed_psf)
        else:
            jdks.append(jitter_psf)
    jdk = np.array(jdks)

    if verbose is True:
        print(f"jitter_psf = {jitter_psf.shape}")
        print(f"jdk.shape = {jdk.shape}")

    return jdk, group_sizes
