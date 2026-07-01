__all__ = ["RegridderZenMats", "RegridderZenAccel", "psf_grid"]

import numpy as np
from spx_utils.psf.psf_parameters import PSFConstants


def RegridderZenMats(inSize, outSize):
    """Regridding via zen matrix."""
    # inSize = inArray.shape # Get the size of the in array
    xFact = inSize[0]/outSize[0]  # Get the y resize factor
    yFact = inSize[1]/outSize[1]  # Get the x resize factor
    # outArray = np.zeros(outSize) # Create output array

    # ========== Y Matrix Creation (Uses all x variables, naming flipped) ==========
    xRange = np.arange(0, inSize[0] + xFact, xFact)  # Get a range of pixel slices to make
    xRangeAdj = np.copy(xRange)  # Copy to avoid python variable issues
    for i in range(0, xRangeAdj.size):  # Run through all of the x ranges and adjust as needed
        if (xRangeAdj[i] == int(xRangeAdj[i])):  # Catch true integers
            # Force any int values to floor to the previous integer for counting reasons
            xRangeAdj[i] = xRangeAdj[i] - .1

    if (inSize[0] != outSize[0]):
        yMatrix = np.zeros((outSize[0], inSize[0]))  # Preallocate
        for i in range(0, outSize[0]):
            # Get the pixels involved
            xPixelsInvolved = np.arange(int(xRangeAdj[i]), int(xRangeAdj[i+1]) + 1, 1)
            xPixelPercent = np.ones(xPixelsInvolved.size)  # Get the percentages involved
            # If size is one, then need the difference
            # between the xRange[1] and xRange[0] to cover the divide
            if (xPixelsInvolved.size == 1):
                # Sets the only value to be the difference between the percentages
                xPixelPercent[0] = xRange[i+1] - xRange[i]
            else:
                if (i == 0):  # If i == 0 it means that xRange[0] = 0, so special +1 added
                    # Sets the first value to be the first percentage
                    xPixelPercent[0] = np.ceil(xRange[i]) + 1 - xRange[i]
                else:
                    # Sets the first value to be the first percentage
                    xPixelPercent[0] = np.ceil(xRange[i]) - xRange[i]
                # Sets the last value to be the last percentage
                xPixelPercent[-1] = xRange[i+1] - int(xRangeAdj[i+1])
                # Middle numbers are 1 since full pixels are taken,
                # and if there's only 1 pixel involved the last percentage
                # (one that matters )
            for j in range(0, xPixelsInvolved.size):
                # Put in the pixel percent
                # (the extra magic is using xPixelsInvolved as an index)
                yMatrix[i, xPixelsInvolved[j]] = xPixelPercent[j]

    else:
        yMatrix = np.diag(np.tile(1, inSize[0]), k=0)  # I matrix so nothing changes, since shiftX == 0

    # ========== X Matrix Creation (Uses all y variables, naming flipped) ==========
    yRange = np.arange(0, inSize[1] + yFact, yFact)  # Get a range of pixel slices to make
    yRangeAdj = np.copy(yRange)  # Copy to avoid python variable issues
    for i in range(0, yRangeAdj.size):  # Run through all of the y ranges and adjust as needed
        if (yRangeAdj[i] == int(yRangeAdj[i])):  # Catch true integers
            # Force any int values to floor to the previous integer for counting reasons
            yRangeAdj[i] = yRangeAdj[i] - .1

    if (inSize[1] != outSize[1]):
        xMatrix = np.zeros((inSize[1], outSize[1]))  # Preallocate
        for i in range(0, outSize[1]):
            # Get the pixels involved
            yPixelsInvolved = np.arange(int(yRangeAdj[i]), int(yRangeAdj[i+1])+1, 1)
            yPixelPercent = np.ones(yPixelsInvolved.size)  # Get the percentages involved
            # If size is one, then need the difference
            # between the yRange[1] and yRange[0] to cover the divide
            if (yPixelsInvolved.size == 1):
                # Sets the only value to be the difference between the percentages
                yPixelPercent[0] = yRange[i+1] - yRange[i]
            else:
                if (i == 0):  # If j == 0 it means that yRange[0] = 0, so special +1 added
                    # Sets the first value to be the first percentage
                    yPixelPercent[0] = np.ceil(yRange[i]) + 1 - yRange[i]
                else:
                    # Sets the first value to be the first percentage
                    yPixelPercent[0] = np.ceil(yRange[i]) - yRange[i]
                # Sets the last value to be the last percentage
                yPixelPercent[-1] = yRange[i+1] - int(yRangeAdj[i+1])
                # Middle numbers are 1 since full pixels are taken,
                # and if there's only 1 pixel involved the last percentage
                # (one that matters )
            for j in range(0, yPixelsInvolved.size):
                # Put in the pixel percent
                # (the extra magic is using yPixelsInvolved as an index)
                xMatrix[yPixelsInvolved[j], i] = yPixelPercent[j]

    else:
        xMatrix = np.diag(np.tile(1, inSize[1]), k=0)  # I matrix so nothing changes, since shiftX == 0
    return yMatrix, xMatrix


def RegridderZenAccel(inArray, yMatrix, xMatrix):
    """Accellerator for regridding via zen matrix method."""
    outArray = np.matmul(np.matmul(yMatrix, inArray), xMatrix)  # One shot calculate the resize
    # Returns: Resized array
    return outArray


def psf_grid(factor_up):
    """PSF grid function, aka PSF pix, based on Yun-Ting Cheng's code."""
    p = PSFConstants
    # Calculate the needed size directly, always will be odd.
    sourceSizeY_psf_pix = factor_up * 2 - 1
    sourceSizeX_psf_pix = factor_up * 2 - 1

    xx, yy = np.meshgrid(np.arange(sourceSizeX_psf_pix), np.arange(sourceSizeY_psf_pix))
    xx = abs(xx - ((sourceSizeX_psf_pix) / 2 - .5))
    yy = abs(yy - ((sourceSizeY_psf_pix) / 2 - .5))
    psf_pix = (factor_up - xx) * (factor_up - yy)
    psf_pix[(xx >= (factor_up + .5))] = 0
    psf_pix[(yy >= (factor_up + .5))] = 0
    psf_pix = psf_pix / np.sum(psf_pix)

    # Pad to shape of stack
    pad_up = p.stackSize - psf_pix.shape[0]
    pad_down = p.stackSize - psf_pix.shape[0]  # Pad the bottom by the difference in the y dir
    pad_left = p.stackSize - psf_pix.shape[1]
    pad_right = p.stackSize - psf_pix.shape[1]  # Pad the right by the difference in the x dir
    psf_pix_padded = np.pad(psf_pix, ((int(np.floor(pad_up / 2)), int(np.ceil(pad_down / 2))),
                                      (int(np.floor(pad_left / 2)), int(np.ceil(pad_right / 2)))),
                            'constant',
                            constant_values=0)  # Pad so psf_pix is now the size of [stackSize,stackSize]
    return psf_pix, psf_pix_padded
