__all__ = ["TurboRLD", "rld", "DeconvolutionIterateVirtual"]

import logging
import numpy as np
from numba import jit, prange
import time
from spx_utils.psf.regrid import RegridderZenMats
from scipy.signal import convolve2d
# from scipy.signal import fftconvolve
# from astropy.convolution import convolve_fft
from spx_utils.psf.psf_parameters import SuperresolutionPSF
from scipy import ndimage
from skimage.transform import downscale_local_mean


def computeOptimalPhot(source, aperture, spacing, spherex_pixel_size, nmc, nomc):
    '''
    Computes optimal photometry of a source given an aperture.

    Parameters
    ----------
    source : np.ndarray
        Source image, e.g., a PSF.
    aperture : np.ndarray
        Aperture shape, e.g., median PSF.
    spacing : float
        Pixel spacing (mm). Often set to 1.0 for convenience.
    spherex_pixel_size : float
        Pixel size in arcsec/mm.
    nmc : int
        Number of Monte Carlo samplings.
    nomc : bool
        If True, disables Monte Carlo sampling (sets nmc to 1).

    Returns
    -------
    np.ndarray
        Optimal photometry values for each MC trial.
    '''
    pixel_scale = spacing * spherex_pixel_size

    source = source / np.nansum(source)
    aperture = aperture / np.nansum(aperture)

    if nomc:
        nmc = 1

    results = []
    for _ in range(nmc):
        xo = (np.random.random() - 0.5) * 6.15 / pixel_scale
        yo = (np.random.random() - 0.5) * 6.15 / pixel_scale
        if nomc:
            xo, yo = 0.0, 0.0

        factor = int(6.15 // pixel_scale)
        source_dsc = downscale_local_mean(ndimage.shift(source, (xo, yo)), (factor, factor))
        aperture_dsc = downscale_local_mean(ndimage.shift(aperture, (xo, yo)), (factor, factor))

        opt_phot_kernel = aperture_dsc / np.sum(aperture_dsc**2)
        results.append(np.sum(opt_phot_kernel.flatten() * source_dsc.flatten()))

    return np.asarray(results)


def rld(d, p, niter, verbose):
    """Richardson-Lucy Deconvolution

    Parameters
    ----------
    d = data
    p = PSF
    niter = number of iterations
    verbose=True will compute an estimate of the width of the image based on the rms

    Notes
    -----
    This function performs Richardson Lucy deconvolution
    in order to remove the pixel PSF.
    From Brendan Crill.
    """
    u_t = d
    p_mirror = p[::-1, ::-1]
    for iiter in range(0, niter):
        c = convolve2d(u_t, p, mode='same')
        if c == 0:
            c += 1e-12
        # c = fftconvolve(u_t, p, mode='same') #faster version
        # c = convolve_fft(u_t, p)
        u_t1 = u_t * convolve2d(d / c, p_mirror, mode='same')
        # u_t1 = u_t * fftconvolve(d / c, p_mirror, mode='same') #faster version
        # u_t1 = u_t * convolve_fft(d / c, p_mirror)
        u_t = u_t1
    return u_t


def TurboRLD(stack, psf_pix, rldIterations, damper, v, weight=0, readout=0):
    """
    Inputs: image (stack), pixel point-spread function, number of RLD iterations,
    damping factor, weight factor, readout error, subsample factor,
    machine precision adjustment factor

    Deconvolves an image with a given point-spread function.

    Ported from deconvlucy in MATLAB

    # From https://people.csail.mit.edu/dgreensp/may/deconvlucy.html:
    Performing basic R-L as described above converges exceedingly slowly and
    will not match the quality of Matlab's deconvlucy. Matlab uses an
    acceleration technique which essentially extends each step by a small factor.
    The Matlab source code references the following papers:

    "Acceleration of iterative image restoration algorithms, by D.S.C. Biggs
    and M. Andrews, Applied Optics, Vol. 36, No. 8, 1997.

    "Deconvolutions of Hubble Space Telescope Images and Spectra",
    R.J. Hanisch, R.L. White, and R.L. Gilliland. in "Deconvolution of Images
    and Spectra", Ed. P.A. Jansson, 2nd ed., Academic Press, CA, 1997.

    The acceleration technique is described in the first paper.
    Matlab implements the algorithm from sections 2C-D of this paper,
    and it is not a hard algorithm to implement yourself.
    """
    s = SuperresolutionPSF
    s.pixelSize = 6.15
    weight = 0
    factor_subsample = 1
    epsAdjust = 1

    # For each bin, perform Richardson Lucy deconvolution to remove pixel effects
    if (
        s.FLG_readout == 1
    ):  # If using readout adjustment, find minimum negative value in stack
        readout = -1 * stack.min()
    else:
        readout = 0

    # Factor_subsample is used if the psf_pix is on a grid that is X times finer
    # than the stack grid. 1 means they're on the same grid resolution
    if weight == 0:
        # 0 means we need a weight
        weight = np.ones(
            stack.shape
        )  # Set as an option - size of stack, 1's by default

    stackSize = stack.shape[0]  # Get the stack size, it's always square
    eps = (
        np.finfo(np.float64).eps * epsAdjust
    )  # Get float64 machine precision (called epsilon by everyone)

    # ----- This code accomplishes what MATLAB's psf2otf does -----
    # First up, convert the PSF to an OTF (Point Spread Function
    # -> Optical Transfer Function)
    # The PSF is sized based on the upscale factor
    # The OTF is sized based on the stack size
    # FFT is used to make the conversion
    pad_up = 0  # No padding above psf_pix
    pad_down = (
        stackSize - psf_pix.shape[0]
    )  # Pad the bottom by the difference in the y dir
    pad_left = 0  # No padding to the left of psf_pix
    pad_right = (
        stackSize - psf_pix.shape[1]
    )  # Pad the right by the difference in the x dir
    psf_pix_padded = np.pad(
        psf_pix,
        ((pad_up, pad_down), (pad_left, pad_right)),
        "constant",
        constant_values=0,
    )  # Pad so psf_pix is now the size of [stackSize,stackSize]

    psf_pix_padded = np.roll(
        psf_pix_padded, -psf_pix.shape[0] // 2 + 1, axis=0
    )  # Roll psf_pix_padded so that the center is at [0,0]
    psf_pix_padded = np.roll(
        psf_pix_padded, -psf_pix.shape[1] // 2 + 1, axis=1
    )  # Finish the roll on the other axis (+1 cause python stuff probably)

    otf_pix = np.fft.fft2(psf_pix_padded)  # Calculate the otf

    numOps = 0  # Counter for the estimated number of operations doing the FFT for oft_pix took
    for j in range(0, psf_pix.ndim):
        numFFTs = psf_pix.size / psf_pix.shape[j]  # Estimate the number of FFTs
        numOps = numOps + psf_pix.shape[j] * np.log2(psf_pix.shape[j]) * numFFTs

    # Throw out the imaginary portion of oft if it's within round-off error scaled by
    # the number of operations the FFT took
    if np.max(np.abs(np.imag(otf_pix))) / np.max(np.abs(otf_pix)) <= numOps * eps:
        otf_pix = np.real(otf_pix)

    # ----- This code accomplishes what MATLAB's deconvlucy does -----
    # This code is an accelerated Richardson-Lucy deconvolution

    # Create indexes for image according to the sampling rate
    indexesX = np.repeat(np.arange(0, stack.shape[1]), factor_subsample)
    indexesY = np.repeat(np.arange(0, stack.shape[0]), factor_subsample)
    indexesXgrid, indexesYgrid = np.meshgrid(
        indexesX, indexesY
    )  # The meshgrid is needed for the indexing
    # The indexing essentially will transform a 100x100 stack to a 200x200 stack with
    # a factor_subsample of 2 where 1 pixel is now in 4 places

    stackWeighted = weight * (
        stack + readout
    )  # Weight the stack, add in the readout error
    # if stackWeighted.min() < 0:
    # neg = np.where(stackWeighted < 0)
    stackWeighted[
        stackWeighted < 0
    ] = 0  # And all negatives are converted to 0 (stability enforced!)
    decon = stack[
        indexesYgrid, indexesXgrid
    ]  # Create the decon based on the psf_pix grid size (often 1, but can be finer)
    scale = np.real(np.fft.ifft2(np.conj(otf_pix) * np.fft.fft2(weight))) + np.sqrt(
        eps
    )  # Pre-calc this scale factor
    damper22 = (damper**2) / 2  # Pre-calc this damper value

    if factor_subsample == 1:
        vec = []
        num = []
    else:
        # Prep a vec to help with reshaping
        vec = np.zeros(2 * stack.ndim, dtype=np.int64)
        vec[np.arange(1, 2 * stack.ndim, 2) - 1] = factor_subsample
        vec[np.arange(2, 2 * stack.ndim + 2, 2) - 1] = stack.shape
        num = np.flip(np.arange(1, 2 * stack.ndim, 2) - 1)

    # Prep for the iteration of the accel'd RLD
    deconOld = np.zeros(decon.shape)  # Prep the previous decon as an array of zeros
    stabMem = np.zeros(
        (decon.size, 2)
    )  # Prep the stability enforcer memory array (for adjusting the stability enforcer on each new iteration)
    stabForce = 0  # Prep the stability enforcer
    for iters in range(0, rldIterations):
        # Prep the stability enforcement
        if iters > 1:
            stabForce = np.dot(stabMem[:, 0], stabMem[:, 1]) / (
                np.dot(stabMem[:, 1], stabMem[:, 1]) + eps
            )
            # Enforce stability limits on stabForce
            if stabForce > 1:
                stabForce = 1  # Keep from going above 1
            if stabForce < 0:
                stabForce = 0  # Keep from going below 0

        # Apply the stability enforcement
        deconStabed = decon + stabForce * (
            decon - deconOld
        )  # Stability enforce applied to the current decon iteration
        deconStabed[
            deconStabed < 0
        ] = 0  # And all negatives are converted to 0 (stability enforced!)

        # ----- This code accomplishes what MATLAB's corelucy does -----
        # This is an "Accelerated Damped Lucy-Richardson Operator"
        deconBlurred = np.real(np.fft.ifft2(otf_pix * np.fft.fft2(deconStabed)))

        # 1. Resampling if needed
        if (
            factor_subsample != 1
        ):  # Bin ReBlurred back to the sizeI for non-singleton dims
            # !!!!!This is super untested!!!!!
            # 1.Reshape so that the-to-binned dimension separates into two
            # dimensions, with one of them consisting of elements of a single bin.
            deconBlurred = np.reshape(deconBlurred, vec)

            # 2. Bin (==calculate mean) along the first of the-to-binned dimension,
            # that dimension consists of the bin elements. Reshape to get rid of
            # new appeared singleton.
            for j in range(0, num[0]):
                vec[j] = []
                deconBlurred = np.reshape(np.mean(deconBlurred, j), vec)

        # 2. An Estimate for the next step
        deconBlurred = deconBlurred + readout
        deconBlurred[deconBlurred == 0] = eps
        stabEst = stackWeighted / deconBlurred + eps

        # 3. Damping if needed
        if np.all(damper22 == 0) == 1:
            # No Damping
            stabEstScaled = stabEst[indexesYgrid, indexesXgrid]
        else:
            # Damping of the image relative to DAMPAR22 = (N*sigma)^2
            np.seterr(divide='ignore')
            gm = 10
            g = (
                stackWeighted * np.log(stabEst) + deconBlurred - stackWeighted
            ) / damper22
            g[g > 1] = 1
            G = (g ** (gm - 1)) * (gm - (gm - 1) * g)
            ImRatio = 1 + G[indexesYgrid, indexesXgrid] * (
                stabEst[indexesYgrid, indexesXgrid] - 1
            )
            stabEstScaled = ImRatio

        CC = np.fft.fft2(stabEstScaled)

        deconOld = np.copy(decon)  # Record the previous decon

        # Photometric convergence check
        # if iters > 0:
        #     try:
        #         # Compute photometry for previous and current PSFs
        #         phot_prev = computeOptimalPhot(deconOld, deconOld, spacing=1.0,
        #                                        spherex_pixel_size=s.pixelSize,
        #                                        nmc=1, nomc=True).mean()
        #         phot_curr = computeOptimalPhot(decon, deconOld, spacing=1.0,
        #                                        spherex_pixel_size=s.pixelSize,
        #                                        nmc=1, nomc=True).mean()
        #         phot_ratio = phot_curr / phot_prev if phot_prev != 0 else float('nan')
        #         print(f"[TurboRLD Iter {iters+1}]
        #         OptimalPhot convergence ratio = {phot_ratio:.6f}")
        #     except Exception as e:
        #         print(f"[TurboRLD Iter {iters+1}] OptimalPhot check failed: {e}")

        decon = (
            deconStabed * np.real(np.fft.ifft2(np.conj(otf_pix) * CC)) / scale
        )  # Calculate the new decon
        decon[
            decon < 0
        ] = 0  # And all negatives are converted to 0 (stability enforced!)
        stabMem = np.array(
            (decon.T.flatten() - deconStabed.T.flatten(), stabMem[:, 0])
        ).T

    if v is True:
        print(
            "Decon ("
            + str(s.rldIterations)
            + ") sum: "
            + str(np.sum(decon))
            + "   FLG_norm = "
            + str(s.FLG_norm)
            + "(3 or -1 means this 1st decon was norm"
            "d)"
        )
    # Returns: deconvolved image

    return decon


# Initialize jit
@jit(nopython=True, nogil=True, parallel=True, fastmath=False, cache=True)
def RegridderShift_Numba(inArray, shiftXY):
    """
    Inputs: inArray
    - input array of any size, shiftXY
    - (shiftX, shiftY) decimal shift
    - can be any size, any decimal, no integers required
    This function regrids an array by a decimal shift through matrix math
    Accelerated by the Numba package removes support for the FLG_origSize
    Shifts for integers are not supported,
    but no technical limitation prevents future support
    """
    inSize = inArray.shape  # Get the size of the in array
    shiftX = shiftXY[0]  # Get the X shift
    shiftXpix = np.int64(np.ceil(np.abs(shiftX)))  # Number of pixels to shift in X by
    if shiftXpix == 0:
        shiftXdir = np.int64(0)  # 0 for zero
    else:
        shiftXdir = np.int64(
            np.round(shiftX / np.abs(shiftX))
        )  # 1 for positive, -1 for negative
        if shiftXdir == -1:  # Fix so negative shift code isn't directly needed
            shiftX = np.remainder(
                shiftX, 1
            )  # Get just the decimal portion (if a shift of 2.2 occured, it'll do 2.2-2=0.2)
            if shiftX < 0:
                shiftX = 1 + shiftX  # Fix it up so it's the positive mirror
            # END IF
        else:
            shiftX = np.remainder(
                shiftX, 1
            )  # Get just the decimal portion (if a shift of 2.2 occured, it'll do 2.2-2=0.2)
        # END IF
    # END IF
    shiftY = shiftXY[1]  # Get the Y shift
    # Fix for wrong direction Y shift (was shifting down for +values)
    # shiftY = -shiftY # Fixes it
    shiftYpix = np.int64(np.ceil(np.abs(shiftY)))  # Number of pixels to shift in X by
    if shiftYpix == 0:
        shiftYdir = np.int64(0)  # 0 for zero
    else:
        shiftYdir = np.int64(
            np.round(shiftY / np.abs(shiftY))
        )  # 1 for positive, -1 for negative
        if shiftYdir == -1:  # Fix so negative shift code isn't directly needed
            shiftY = np.remainder(
                shiftY, 1
            )  # Get just the decimal portion (if a shift of 2.2 occured, it'll do 2.2-2=0.2)
            if shiftY < 0:
                shiftY = 1 + shiftY  # Fix it up so it's the positive mirror
            # END IF
        else:
            shiftY = np.remainder(
                shiftY, 1
            )  # Get just the decimal portion (if a shift of 2.2 occured, it'll do 2.2-2=0.2)
        # END IF
    # END IF

    outSize = (
        int(inSize[0] + np.abs(shiftYdir)),
        int(inSize[1] + np.abs(shiftXdir)),
    )  # Makes Numba's nopython happy

    # Create the matrix to shift the X
    if shiftX != 0:
        # # This creates a matrix that combines two values in X
        # # shiftMatX = np.diag( np.tile((1-shiftX) , outSize[1]), k = 0 )
        # Diagonal of (1-shiftX)
        # shiftMatX = np.diag( np.ones( (outSize[1],) )*(1-shiftX), k = 0 )
        # Offset diagonal of shiftX
        # # shiftMatX = shiftMatX + np.diag( np.tile(shiftX , inSize[1]), k = 1  )
        # shiftMatX = shiftMatX + np.diag( np.ones( (inSize[1],) )*shiftX, k = 1  )
        # shiftMatX = shiftMatX[0:inSize[1],:]  # Cut off the bottom row
        # Numba did not like diag method, build with loop instead
        shiftMatX = np.zeros((inSize[1], outSize[1]))  # Preallocate
        for j in range(0, inSize[1]):
            shiftMatX[j, j] = 1 - shiftX  # Put (1-shiftX) on the diagonal
            shiftMatX[j, j + 1] = shiftX  # Put shiftX on the diagonal
        # END FOR j
    else:
        shiftMatX = np.diag(
            np.ones((inSize[1],)), k=0
        )  # I matrix so nothing changes, since shiftX == 0
    # END IF
    shiftMatX = np.ascontiguousarray(
        shiftMatX
    )  # Put in the right order to make numpy happy to do it fast

    # Create the matrix to shift the Y
    if shiftY != 0:
        # This creates a matrix that combines two values in Y - just how the shift does
        # Diagonal of shiftY
        # # shiftMatY = np.diag( np.tile(shiftY , outSize[0]), k = 0)
        # shiftMatY = np.diag( np.ones( (outSize[0],) )*(shiftY), k=0)
        # Offset diagonal of (1-shiftY)
        # # shiftMatY = shiftMatY + np.diag( np.tile((1-shiftY) , inSize[0]), k = -1 )
        # shiftMatY = shiftMatY + np.diag( np.ones( (inSize[0],) )*(1-shiftY), k = -1  )
        # shiftMatY = shiftMatY[:,0:inSize[0]]  # Cut off the right column
        # Numba did not like diag method, build with loop instead
        shiftMatY = np.zeros((outSize[0], inSize[0]))  # Preallocate
        for j in range(0, inSize[0]):
            shiftMatY[j, j] = shiftY  # Put shiftX on the diagonal
            shiftMatY[j + 1, j] = 1 - shiftY  # Put (1-shiftX) on the diagonal
        # END FOR j
    else:
        shiftMatY = np.diag(
            np.ones((inSize[0],)), k=0
        )  # I matrix so nothing changes, since shiftY == 0
    # END IF
    shiftMatY = np.ascontiguousarray(
        shiftMatY
    )  # Put in the right order to make numpy happy to do it fast

    outArray = shiftMatY @ inArray @ shiftMatX  # Do the shift calc in one shot
    # Returns: shifted array
    return outArray  # Return it


def DeconvolutionIterateVirtual(stack, stackDecon, psf_pix, stack_deleteArray, damper, v,
                                sourceCoordDown, s):
    """
    This function deconvolves the pixel PSF from a stacked PSF via
    Richardson Lucy Deconvolution, enhanced using Iterative Back-Projection
    This version of Deconvolution does stacking 'virtually' by avoiding making an
    entire array and is faster than the non-virtual version
    Inputs: stacked PSF, initial decon, pixel PSF, number of RLD iterations,
    stack size, stack memory limit,
    number of sources being used in stack, rounded source coords, source coord deltas,
    source coords,
    array for sources deleted due to closeness, regridding factor, half width of stamps,
    number of IBP iterations, iterations to wait to use weighting,
    zeros to negative power,
    stack size check, stack center check, switch to use basic RLD, damping factor,
    switch to use normalization, switch to use readout error,
    switch to use damping, switch to restrict percent of sources used, index of source,
    restrict percent, low-res source coords, low-res source coords for stacking,
    low-res rounded source coords, low-res rounded source coords for stacking,
    low-res half width of stamps, rounded low-res half width of stamps,
    low-res half width of sources
    high-res half width of stamps, switch to delete sources too close,
    switch to mask based on distance, distance to delete, distance to mask,
    masking radius
    """
    weight2 = 1  # Prep the weighting

    if s.FLG_damper == 1:  # Prep the damper (used in TurboRLD)
        damp = damper
    else:
        damp = 0

    # Preallocate
    stackOld = np.copy(stack)
    stackDeconIterate = np.copy(stackDecon)
    FLG_skipLoop = 0  # Prime this flag
    fingerPrintPrev = np.zeros(s.stackSize)  # Prep this
    fingerPrintDeltaMean = 0  # Start this
    finalIteration = (
        s.stackDeconIterations  # Prep this # in case all iterations are finished
    )

    # ----Masking Prep----
    # if( FLG_starDistMask == 2 ):
    # dist_maskRadius = np.sqrt( 2 * (halfSourceSizeDown_tru ** 2))
    # Radius to check around the source that needs to be masked out
    maskPts_x, maskPts_y = np.meshgrid(
        np.arange(0, s.halfStampSizeDown * 2), np.arange(0, s.halfStampSizeDown * 2)
    )  # Get a meshgrid of the virtualArrayDown
    dist_maxPower = (
        2 * (s.sourceCoordUp.max()) ** 2
    )  # Get the max for the dist and then make it a "distance" w/o the sqrt

    if s.FLG_source_perc_restrict == 0:
        blacklist = stack_deleteArray  # Use this instead, straight from stacker and
        # it's exactly what stacker used to remove sources
        if v is True:
            print(
                "Num sources pre-blacklist: "
                + str(s.numSources_stacker)
                + " | post-blacklist: "
                + str(np.int64(s.numSources_stacker - np.sum(blacklist)))
            )
        numSources_stacker = np.int64(
            s.numSources_stacker - np.sum(blacklist)
        )  # Reduce the number of sources by the amount blacklisted
        sourceCoordRoundUp_stacker = np.delete(
            s.sourceCoordRoundUp_stacker, np.where(blacklist == 1)[0], axis=0
        )  # Delete the coords so we don't apply them
        sourceCoordDeltaUp_stacker = np.delete(
            s.sourceCoordDeltaUp_stacker, np.where(blacklist == 1)[0], axis=0
        )  # Delete the coords so we don't apply them

    if (numSources_stacker * s.stackSize * s.stackSize * 8) / 1024**3 < s.stackMemLimit:
        FLG_useStackMat = 1  # Turn on flag
        stackIterateMat = np.empty(
            (numSources_stacker, s.stackSize, s.stackSize)
        )  # Prep place to hold all source shots for
    else:
        FLG_useStackMat = 0  # Turn off flag if memory usage would be too large
        stackIterate = np.zeros(s.stackSize)  # Prep stack

    # Prep constants used in the virtual stacking
    virtualArrayUpSize = np.array(
        (s.halfStampSizeUp * 2 + 1 + s.factor_up, s.halfStampSizeUp * 2 + 1 + s.factor_up)
    )  # Use the original stamp zone PLUS factor b/c the subpixelshift can be up to factor-1
    virtualArrayUpSize = np.int64(
        np.ceil((virtualArrayUpSize / s.factor_up) / 2) * s.factor_up * 2 + 1
    )  # Forces whatever number we get to be divisible by factor AND by 2
    virtualArrayDownSize = np.int64(
        virtualArrayUpSize / s.factor_up
    )  # Should always be an int

    virtualXbase = virtualArrayUpSize[1] // 2  # Start off in the middle-ish
    virtualYbase = virtualArrayUpSize[0] // 2  # Start off in the middle-ish

    virtualXwidth = np.int64(
        np.array((np.floor(s.stackSize / 2), np.ceil(s.stackSize / 2)))
    )  # Calculate the widths -/+ the center
    virtualYwidth = np.int64(
        np.array((np.floor(s.stackSize / 2), np.ceil(s.stackSize / 2)))
    )  # Calculate the widths -/+ the center

    maskMat = np.zeros(
        (s.halfStampSizeDown * 2, s.halfStampSizeDown * 2), dtype=bool
    )  # Prep a 0 maskMat - will keep multiple masks required by nearby stars to be applied at once
    eps32 = np.finfo(np.float32).eps  # Get this ahead of time
    yMatrixDown, xMatrixDown = RegridderZenMats(
        virtualArrayUpSize, virtualArrayDownSize
    )  # Pre-calculate the matricies so don't have to do it constantly
    yMatrixUp, xMatrixUp = RegridderZenMats(
        virtualArrayDownSize, virtualArrayUpSize
    )  # Pre-calculate the matricies so don't have to do it constantly
    yMatrixDownUp = yMatrixUp @ yMatrixDown  # Combine them to do it at once
    xMatrixDownUp = xMatrixDown @ xMatrixUp  # Combine them to do it at once
    yMatrixOP, xMatrixOP = RegridderZenMats(
        np.int64(np.array((s.halfStampSizeDown * 2, s.halfStampSizeDown * 2))),
        np.int64(np.array((s.stackSize, s.stackSize))),
    )  # Pre-calculate the matricies so don't have to do it constantly
    # yMatrixOP32 = np.float32(yMatrixOP) # 32 bit version
    # xMatrixOP32 = np.float32(xMatrixOP)
    for irate in range(0, s.stackDeconIterations):
        if FLG_skipLoop == 0:
            # Goal here is to enable way bigger upscale factor b/c don't need a
            # gigantic array to hold it - each source gets its own lil world
            tic = time.time()

            if FLG_useStackMat == 1:

                stackIterateMat = DeconvolutionIterateVirtual_Celerity_stackMat(
                    stackDeconIterate,
                    virtualArrayUpSize,
                    stackIterateMat,
                    s.factor_up,
                    s.halfStampSizeDown,
                    xMatrixDownUp,
                    yMatrixDownUp,
                    xMatrixOP,
                    yMatrixOP,
                    sourceCoordRoundUp_stacker,
                    sourceCoordDeltaUp_stacker,
                    sourceCoordDown,
                    s.sourceCoordDown_stacker,
                    s.sourceCoordRoundDown,
                    s.sourceCoordRoundDown_stacker,
                    numSources_stacker,
                    virtualXbase,
                    virtualYbase,
                    virtualXwidth,
                    virtualYwidth,
                    eps32,
                    maskMat,
                    s.FLG_starDistDelete,
                    s.FLG_starDistMask,
                    dist_maxPower,
                    s.dist_delete,
                    s.dist_mask,
                    s.dist_maskRadius,
                    maskPts_x,
                    maskPts_y,
                )

            else:
                stackIterate = np.zeros(s.stackSize)  # Prep stack
                stackIterate = DeconvolutionIterateVirtual_Celerity_stackMat(
                    stackDeconIterate,
                    virtualArrayUpSize,
                    stackIterate,
                    s.factor_up,
                    s.halfStampSizeDown,
                    xMatrixDownUp,
                    yMatrixDownUp,
                    xMatrixOP,
                    yMatrixOP,
                    sourceCoordRoundUp_stacker,
                    sourceCoordDeltaUp_stacker,
                    sourceCoordDown,
                    s.sourceCoordDown_stacker,
                    s.sourceCoordRoundDown,
                    s.sourceCoordRoundDown_stacker,
                    numSources_stacker,
                    virtualXbase,
                    virtualYbase,
                    virtualXwidth,
                    virtualYwidth,
                    eps32,
                    maskMat,
                    s.FLG_starDistDelete,
                    s.FLG_starDistMask,
                    dist_maxPower,
                    s.dist_delete,
                    s.dist_mask,
                    s.dist_maskRadius,
                    maskPts_x,
                    maskPts_y,
                )

            if (FLG_useStackMat == 1) & (
                s.FLG_starDistMask != 3
            ):  # If memory allows, do this
                # stack = np.sum(stackMat,axis=0)/numSources_stacker # Sum them up,
                # then normalize by the number summed
                # (should be identical to original source)
                stackIterate = np.mean(
                    stackIterateMat, axis=0
                )  # Direct mean (no nan's can occur here) - nanmean is slower than mean
            elif (FLG_useStackMat == 1) & (
                s.FLG_starDistMask == 3
            ):  # If memory allows, do this
                stackIterate = np.nanmean(
                    stackIterateMat, axis=0
                )  # Direct mean (nan's can occur b/c masking is on)
            else:
                stackIterate = stackIterate / numSources_stacker  # Mimic a mean here

            if irate > s.weightIterationEngage:  # Give the mean time to settle
                weight2 = (
                    weight2 + 1
                )  # Ramp the weight to speed up the error reduction
                # (can go too far - that's what below is for)
            stackIterateError = (
                stackIterate - stack
            )  # Get the difference between the iterate stack and the orig stack

            fingerPrint = stackIterateError
            fingerPrint = fingerPrint / np.max(np.abs(fingerPrint))  # Normalize
            fingerPrintDelta = np.abs(fingerPrint - fingerPrintPrev)
            fingerPrintDeltaSum = np.sum(fingerPrintDelta)
            fingerPrintDeltaMean = fingerPrintDeltaMean * (
                irate / (irate + 1)
            ) + fingerPrintDeltaSum / (
                irate + 1
            )  # Running mean
            if fingerPrintDeltaSum > fingerPrintDeltaMean:
                weight2 = 1  # Reset weight
            fingerPrintPrev = np.copy(fingerPrint)  # Record old one

            stackOld = (
                stackOld + -weight2 * stackIterateError
            )  # Update stack old to be current for this line and the next - then it's old again
            if s.FLG_norm >= 1:
                stackOld = (
                    stackOld * np.sum(stack) / np.sum(stackOld)
                )  # Force sum to stay the same
            if s.FLG_basicRLD == 1:
                if s.halfStampSizeUp_tru == s.halfStampSizeUp:
                    stackDeconIterate = rld(
                        stackOld, psf_pix, niter=s.rldIterations_decon, verbose=False
                    )  # Decon it
                else:
                    stackDeconIterate = np.zeros(
                        (s.stackSize, s.stackSize)
                    )  # Clear out b/c only covering center area w/ a smaller "true" stack size
                    stackDeconIterate[
                        s.halfStampSizeUp
                        - s.halfStampSizeUp_tru: s.halfStampSizeUp
                        + s.halfStampSizeUp_tru,
                        s.halfStampSizeUp
                        - s.halfStampSizeUp_tru: s.halfStampSizeUp
                        + s.halfStampSizeUp_tru,
                    ] = rld(
                        stackOld[
                            s.halfStampSizeUp
                            - s.halfStampSizeUp_tru: s.halfStampSizeUp
                            + s.halfStampSizeUp_tru,
                            s.halfStampSizeUp
                            - s.halfStampSizeUp_tru: s.halfStampSizeUp
                            + s.halfStampSizeUp_tru,
                        ],
                        psf_pix,
                        niter=s.rldIterations_decon,
                        verbose=False,
                    )  # decon it
                stackDeconIterate[stackDeconIterate < 0] = (
                    10**s.stackDeconIterateFillerPower
                )  # Minimizes "walk" because there can't be negative numbers,
                # but the RLD can cause negative numbers if things are close to
                # not deconvolving due to computational error
            else:
                if s.FLG_readout == 1:
                    readout = -1 * stackOld.min()
                else:
                    readout = 0

                if s.halfStampSizeUp_tru == s.halfStampSizeUp:
                    if s.stackDeconIterateFillerPower < 0:
                        stackDeconIterate = TurboRLD(
                            stackOld,
                            psf_pix,
                            s.rldIterations_decon,
                            damp,
                            v,
                            weight=0,
                            readout=readout,
                        )
                    else:
                        stackDeconIterate = TurboRLD(
                            stackOld,
                            psf_pix,
                            s.rldIterations_decon,
                            damp,
                            v,
                            weight=0,
                            readout=readout,
                            epsAdjust=s.stackDeconIterateFillerPower,
                        )
                else:
                    stackDeconIterate = np.zeros(
                        (s.stackSize, s.stackSize)
                    )  # Clear out b/c only covering center area w/ a smaller "true" stack size
                    if s.stackDeconIterateFillerPower < 0:
                        stackDeconIterate[
                            s.halfStampSizeUp
                            - s.halfStampSizeUp_tru: s.halfStampSizeUp
                            + s.halfStampSizeUp_tru,
                            s.halfStampSizeUp
                            - s.halfStampSizeUp_tru: s.halfStampSizeUp
                            + s.halfStampSizeUp_tru,
                        ] = TurboRLD(
                            stackOld[
                                s.halfStampSizeUp
                                - s.halfStampSizeUp_tru: s.halfStampSizeUp
                                + s.halfStampSizeUp_tru,
                                s.halfStampSizeUp
                                - s.halfStampSizeUp_tru: s.halfStampSizeUp
                                + s.halfStampSizeUp_tru,
                            ],
                            psf_pix,
                            s.rldIterations_decon,
                            damp,
                            v,
                            weight=0,
                            readout=readout,
                        )
                    else:
                        stackDeconIterate[
                            s.halfStampSizeUp
                            - s.halfStampSizeUp_tru: s.halfStampSizeUp
                            + s.halfStampSizeUp_tru,
                            s.halfStampSizeUp
                            - s.halfStampSizeUp_tru: s.halfStampSizeUp
                            + s.halfStampSizeUp_tru,
                        ] = TurboRLD(
                            stackOld[
                                s.halfStampSizeUp
                                - s.halfStampSizeUp_tru: s.halfStampSizeUp
                                + s.halfStampSizeUp_tru,
                                s.halfStampSizeUp
                                - s.halfStampSizeUp_tru: s.halfStampSizeUp
                                + s.halfStampSizeUp_tru,
                            ],
                            psf_pix,
                            s.rldIterations_decon,
                            damp,
                            v,
                            weight=0,
                            readout=readout,
                            # epsAdjust=s.stackDeconIterateFillerPower,
                        )
            if s.FLG_norm >= 2:
                denom = np.sum(stackDeconIterate)
                print(f"np.sum(stack) = {np.sum(stack)}")
                print(f"DENOM = {denom}")
                if np.sum(stackDeconIterate):
                    denom += 1e-12
                    if v is True:
                        logging.info("Divide by zero, deconvolved stack is zero.")

                stackDeconIterate = (
                    stackDeconIterate * np.sum(stack) / denom
                )  # Force sum to stay the same as the stack

            toc = time.time() - tic
            if v is True:
                print(
                    "Decon Iterate #"
                    + str(irate + 1)
                    + " time to run: "
                    + str(np.round(toc, 4))
                    + " sec"
                )

    # Returns: deconvolution, final iteration number
    return stackDeconIterate, finalIteration


# Initialize jit
@jit(nopython=True, nogil=True, parallel=True, fastmath=False, cache=True)
def DeconvolutionIterateVirtual_Celerity_stackMat(
    stackDeconIterate,
    virtualArrayUpSize,
    stackIterateMat,
    factorUp,
    halfStampSizeDown,
    xMatrixDownUp,
    yMatrixDownUp,
    xMatrixOP,
    yMatrixOP,
    sourceCoordRoundUp_stacker,
    sourceCoordDeltaUp_stacker,
    sourceCoordDown,
    sourceCoordDown_stacker,
    sourceCoordRoundDown,
    sourceCoordRoundDown_stacker,
    numSources_stacker,
    virtualXbase,
    virtualYbase,
    virtualXwidth,
    virtualYwidth,
    eps32,
    maskMat,
    FLG_starDistDelete,
    FLG_starDistMask,
    dist_maxPower,
    dist_delete,
    dist_mask,
    dist_maskRadius,
    maskPts_x,
    maskPts_y,
):
    """
    Inputs: decon array, virtual array upscale size, array that holds the stack,
    upscale factor, half stamp size downscaled, x matrix that downscales and upscales,
    y matrix that downscales and upscales
    x matrix that upscales, y matrix that upscales,
    upscaled rounded source coords for the stacker,
    upscaled decimal portion of the source coords for the stacker,
    all downscaled source coords, downscaled source coords for the stacker,
    rounded downscaled source coords, rounded downscaled source coords for the stacker,
    number of source for the stacker, x center for the virtual array,
    y center for the virtual array, x width for the virtual array,
    y width for the virtual array,
    machine precision value for float-32 numbers,
    array that holds the mask for a virtual array cutout,
    flag for deleting stars based on distance, flag for masking stars based on distance,
    maximum distance value, distance to delete stars, distance to mask stars,
    radius to apply a mask over a star,
    x points for a meshgrid of the downscaled virtual array for masking,
    y points for a meshgrid of the downscaled virtual array for masking,
    """
    for i in prange(
        0, numSources_stacker
    ):  # Place a source in the virtual right spot, scale it down, scale it up,
        # and capture it without needing to replicate
        # where the source was in the original array
        virtualArray = np.zeros((virtualArrayUpSize[0], virtualArrayUpSize[1]))
        virtualX = virtualXbase + np.remainder(
            sourceCoordRoundUp_stacker[i, 0], factorUp
        )  # Adjust as needed to the right spot to mimic the downscale/upscale shift
        virtualY = virtualYbase + np.remainder(
            sourceCoordRoundUp_stacker[i, 1], factorUp
        )

        if (sourceCoordDeltaUp_stacker[i, 0] > 0) & (sourceCoordDeltaUp_stacker[i, 1] > 0):
            virtualArray[
                virtualY
                - virtualYwidth[0]
                - np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 1]))): virtualY
                + virtualYwidth[1],
                virtualX
                - virtualXwidth[0]: virtualX
                + virtualXwidth[1]
                + np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 0]))),
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source

        elif (sourceCoordDeltaUp_stacker[i, 0] > 0) & (sourceCoordDeltaUp_stacker[i, 1] < 0):
            virtualArray[
                virtualY
                - virtualYwidth[0]: virtualY
                + virtualYwidth[1]
                + np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 1]))),
                virtualX
                - virtualXwidth[0]: virtualX
                + virtualXwidth[1]
                + np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 0]))),
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        elif (sourceCoordDeltaUp_stacker[i, 0] < 0) & (sourceCoordDeltaUp_stacker[i, 1] > 0):
            virtualArray[
                virtualY
                - virtualYwidth[0]
                - np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 1]))): virtualY
                + virtualYwidth[1],
                virtualX
                - virtualXwidth[0]
                - np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 0]))): virtualX
                + virtualXwidth[1],
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        elif (sourceCoordDeltaUp_stacker[i, 0] < 0) & (sourceCoordDeltaUp_stacker[i, 1] < 0):
            virtualArray[
                virtualY
                - virtualYwidth[0]: virtualY
                + virtualYwidth[1]
                + np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 1]))),
                virtualX
                - virtualXwidth[0]
                - np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 0]))): virtualX
                + virtualXwidth[1],
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        elif (sourceCoordDeltaUp_stacker[i, 0] > 0) & (sourceCoordDeltaUp_stacker[i, 1] == 0):
            virtualArray[
                virtualY - virtualYwidth[0]: virtualY + virtualYwidth[1],
                virtualX
                - virtualXwidth[0]: virtualX
                + virtualXwidth[1]
                + np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 0]))),
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        elif (sourceCoordDeltaUp_stacker[i, 0] < 0) & (sourceCoordDeltaUp_stacker[i, 1] == 0):
            virtualArray[
                virtualY - virtualYwidth[0]: virtualY + virtualYwidth[1],
                virtualX
                - virtualXwidth[0]
                - np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 0]))): virtualX
                + virtualXwidth[1],
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        elif (sourceCoordDeltaUp_stacker[i, 0] == 0) & (sourceCoordDeltaUp_stacker[i, 1] > 0):
            virtualArray[
                virtualY
                - virtualYwidth[0]
                - np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 1]))): virtualY
                + virtualYwidth[1],
                virtualX - virtualXwidth[0]: virtualX + virtualXwidth[1],
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        elif (sourceCoordDeltaUp_stacker[i, 0] == 0) & (sourceCoordDeltaUp_stacker[i, 1] < 0):
            virtualArray[
                virtualY
                - virtualYwidth[0]: virtualY
                + virtualYwidth[1]
                + np.int64(np.ceil(np.abs(sourceCoordDeltaUp_stacker[i, 1]))),
                virtualX - virtualXwidth[0]: virtualX + virtualXwidth[1],
            ] = RegridderShift_Numba(
                stackDeconIterate,
                (sourceCoordDeltaUp_stacker[i, 0], sourceCoordDeltaUp_stacker[i, 1]),
            )  # Write in that source
        else:  # if deltaX == 0 and deltaY == 0, no shift
            virtualArray[
                virtualY - virtualYwidth[0]: virtualY + virtualYwidth[1],
                virtualX - virtualXwidth[0]: virtualX + virtualXwidth[1],
            ] = stackDeconIterate  # Place

        virtualArrayUp = (
            yMatrixDownUp @ virtualArray @ xMatrixDownUp
        )  # One shot upscale and downscale - is such power possible? (it is, it seems)

        virtualArrayUp = virtualArrayUp[
            virtualY - virtualYwidth[0]: virtualY + virtualYwidth[1],
            virtualX - virtualXwidth[0]: virtualX + virtualXwidth[1],
        ]  # Pull out source

        stackIterateMat[i, :, :] = virtualArrayUp  # Save

    # Returns: matrix of all sources
    return stackIterateMat
