__all__ = ["SuperresolutionPSF", "PSFConstants"]

# Standard library imports
from dataclasses import dataclass

# Related third party imports

import numpy as np
import random

# Local application/library specific imports


@dataclass(frozen=True)
class PSFConstants:
    """Shared constants for PSF modules."""

    factor_up = 10
    factor_down = 10
    factorDiff10 = factor_up / 10  # ratio of factorUp to reference 10 factorDown
    factorDiff = np.int64(factor_up/factor_down)
    halfSourceSize = int(25 * factorDiff10)
    halfStampSize = 25
    oggrid = 20400  # 20400
    oggridY = 2040 * factor_down  # 2040
    oggridX = 2040 * factor_down
    gridSize = np.array([oggridY, oggridX], dtype=np.int64)
    gridSizeDown = np.int64(np.round(gridSize / factor_down))
    gridSizeUp = np.int64(gridSizeDown*factor_up)  # used for superresolution stacking
    gridSizeX = gridSizeDown[1]
    sourceSize = np.int64(2 * halfSourceSize)
    halfSourceSizeDown = np.int64(np.ceil(halfSourceSize / factor_up))
    halfStampSizeUp = int(50 * factorDiff10)  # 50 must be cleanly divisible by factorUp
    stackSize = np.int64(2*halfStampSizeUp+1)


class SuperresolutionPSF(PSFConstants):
    """Mutable shared variables for PSF modules."""

    c = PSFConstants
    # Only decon cares about excess blank space - so these are used then
    halfSourceSize_truL = [int(25 * c.factorDiff10), int(25 * c.factorDiff10)]  # 25
    exp = 1
    bands = 1
    gridDown = np.zeros((exp, bands, c.gridSizeDown[0], c.gridSizeDown[1]))
    stackMemLimit = 1000  # GB, size to limit stackMat to
    #  - based off (numSources[e][b]*stackSize[0]*stackSize[1]*8)/1024**3
    # Calculate low-res half stamp size
    halfStampSizeDown = np.int64(np.ceil(c.halfStampSizeUp/c.factor_up))
    # Calculate low-res source size (for masking out sources that are too close)
    halfSourceSizeDown = np.int64(np.ceil(c.halfSourceSize/c.factor_up))
    # only decon cares about excess blank space - so these are used then
    halfStampSizeUp_truL = [int(40 * c.factorDiff10),
                            int(40 * c.factorDiff10)]
    halfSourceSize_truL = [int(25 * c.factorDiff10),
                           int(25 * c.factorDiff10)]
    halfStampSizeUp_tru = halfStampSizeUp_truL[0]  # Detector 1
    halfSourceSize_tru = halfSourceSize_truL[0]  # Detecotr 1
    # Calculate low-res stamp and source estimated thumbnails sizes
    halfStampSizeDown_tru = np.int64(np.ceil(halfStampSizeUp_tru / c.factor_up))
    halfSourceSizeDown_tru = np.int64(np.ceil(halfSourceSize_tru / c.factor_up))
    FLG_useStarCount = 3
    flux_norm = [[0 for y in range(1)] for x in range(exp)]
    FLG_noise = 1
    # Restrict number of sources in stack & decon to X% via representative sample
    FLG_source_perc_restrict = 0  # 0 = don't restrict number of sources, 1 = restrict number of sources
    res_perc = 5  # restrict to X% of total available sources
    # seed to create consistent data
    startSeed = 1774
    np.random.seed(seed=startSeed)  # prime numpy random numbers
    random.seed(a=startSeed)  # prime python random numbers
    FLG_opticalPSFoverride = 0
    FLG_directMag = 1
    FLG_starDistDelete = 1
    FLG_starDistMask = 2
    FLG_starDistMask_mult = [100, 5]
    FLG_FWHMopts = 1  # 1=cbe, 2=mev

    # Set type of normalization of source fluxes to be used
    # when creating stack ofsources, and how normalization is carried out
    # 0 = no source flux normalization,
    # 1 = normalize all source fluxes to self-defined value (1000),
    # 2 = normalizes source fluxes comparatively
    FLG_useStarCount_Norm = 2
    FLG_useStarCount_Norm_mult = [100, 2.5]
    # Apply the magnitude conversion for source comparison by magnitudes, not fluxes
    if (FLG_directMag == 0):
        FLG_useStarCount_Norm_mult = FLG_useStarCount_Norm_mult[0]  # pull out the flux number
        FLG_starDistMask_mult = FLG_starDistMask_mult[0]  # pull out the flux number
    else:
        # conversion for relative flux/magnitude would be (m1-m2) = -2.5*np.log10(f1/f2)
        #  where m1-m2 is mag difference [1] and f1/f2 is flux multiplier [0]
        FLG_useStarCount_Norm_mult = FLG_useStarCount_Norm_mult[1]  # pull out the magnitude
        FLG_starDistMask_mult = FLG_starDistMask_mult[1]  # pull out the magnitude

    FLG_noiseFloor_mult = 40  # normalize flux to N*detected noise floor

    noiseFloor = [[0 for y in range(1)] for x in range(exp)]
    source = np.zeros((exp, bands, c.sourceSize, c.sourceSize))
    FLG_stackerStampChecker = 0
    # pixelrms appears to be average of delta nu I_nu (shx1) and (dpx1),
    # but values don't quite match (nW/m^2 sr)
    pixelrms = 58.5  # Band 1
    sourceCoordFlux_stacker = [[0 for y in range(1)] for x in range(exp)]
    # 0 = stack may have edge roll issues,
    # 1 = adds correction for this (1 is preferred version)
    FLG_correctStack = 1

    # Define pixel distances between opposing sources used to determine
    # when sources can be masked, and when they must be deleted from the stacking list
    # (8.48 for 25 source size (rounded down to 3 and used in math above)
    dist_deleteL = [8.48, 9.48]
    # (11.31 for 50 stack/25 source size
    # (rounded down to 5/rounded down to 3 and used in math above))
    dist_maskL = [11.31, 11.31]
    # (4.24 for 25 source size (rounded down to 3 and used in math above))
    dist_maskRadiusL = [4.24, 5.0]

    dist_delete = dist_deleteL[0]  # get 1st one for band 1/2/3
    dist_mask = dist_maskL[0]  # get 1st one for band 1/2/3
    dist_maskRadius = dist_maskRadiusL[0]  # get 1st one for band 1/2/3
    # Use restricted range of AB magnitudes from total available sources
    FLG_mag_range = 1  # 0 = don't restrict mag range, 1 = restrict mag range
    mag_range_upper = 11  # upper limit of magnitude restriction range - 11 is optimal
    mag_range_lower = 15  # lower limit of magnitude restriction range - 15 is optimal
    FLG_readout = 0  # turn off for elliptical gaussian psf
    FLG_damper = 1

    FLG_basicRLD = 0
    # RLD and IBP settings for when noise added to data
    # NEED TO CHANGE DEPENDING ON BAND
    # for noise:
    # band 1: 200, band 2: 200, band 3: 200,
    # band 4: 100, band 5: 50, band 6: 10
    rldIterations = 3
    # band 1: 300, band 2: 400, band 3: 480,
    # band 4: 100, band 5: 90, band 6: 30
    rldIterations_decon = 3
    # band 1: 40, band 2: 50, band 3: 50,
    # band 4: 25, band 5: 20, band 6: 20]
    # number of iterations to run the whole stack decon iterate loop
    stackDeconIterations = 2
    # look again at distance and magnitude cuts,
    # try maybe looser for elliptical gaussian psf
    # number of iterations to wait until the weight parameter is used
    # to ramp up the error reduction
    weightIterationEngage = 1
    stackDeconIterateFillerPower = -12  # negative power to turn 0's into - rld doesn't like pure 0's
    # 1 puts a circular mask around decon and ibp setting all pixels > radius to 0,
    # normalizing sum afterward
    FLG_circmask = 0  # remove circmask for elliptical gaussian psf
    # NEED TO CHANGE DEPENDING ON BAND
    # circrad = 9 * c.factorDiff10  # * 2 for elliptical gaussian psf
    # !!! NEED TO CHANGE DEPENDING ON BAND - CORRECT VALUES LISTED TO RIGHT !!!#
    # 1st # is used for bands 1/2/3, 2nd # is for bands 4/5/6
    # [band 1: 31, band 2: 29, band 3: 32.5, band 4: 37, band 5: 35, band 6: 34]
    # # radius beyond which pixels will be set to 0 if FLG_circmask == 1
    circrad = 7 * c.factorDiff10
    # band 1: 0.13, band 2: 0.115, band 3: 0.11,
    # band 4: 0.4, band 5: 0.45, band 6: 0.3
    # half of stack size * damper_factor = radius for damping
    damper_factor = 0.13
    FLG_stackDeconIterateEnable = 1
    FLG_norm = 1
    # FLG_norm:
    # 0 = no norm in the decon,
    # 1 = error-adjusted stack is normalized to the original stack,
    # 2 = both error-adjusted stack & decon iteration are normalized to the orig stack,
    # 3 = 1st decon is normalized as well as everything in Decon IBP
    # -1 = only 1st decon is normalized

    abmag_stacker: np.ndarray
    sourceCoordFlux_stacker: np.ndarray
    numSources_stacker: np.ndarray
    sourceCoordDown_stacker: np.ndarray
    sourceCoordRoundDown_stacker: np.ndarray
    sourceCoordUp_stacker: np.ndarray
    sourceCoordRoundUp_stacker: np.ndarray
    sourceCoordDeltaUp_stacker: np.ndarray
    sourceCoordRoundDown: np.ndarray
    sourceCoord: np.ndarray
    sourceCoordRound: np.ndarray
    sourceCoordDelta: np.ndarray
    sourceCoordUp: np.ndarray
    sourceCoordRoundUp: np.ndarray
    sourceCoordDeltaUp: np.ndarray

    def __init__(self, **kwargs):
        self.sourceCoordDown = kwargs['source_coord_down']
        self.abmag = kwargs['abmag']
        self.good_mags = kwargs['good_mags']
        # Create sourceCoordRound/sourceCoordDelta from sourceCoord
        # make sourceCoord to start (high-res version)
        SuperresolutionPSF.sourceCoord = self.sourceCoordDown * SuperresolutionPSF.factor_up
        # make a consistent set of rounded coordinates
        SuperresolutionPSF.sourceCoordRound = np.int64(np.round(SuperresolutionPSF.sourceCoord))
        # get the decimals
        SuperresolutionPSF.sourceCoordDelta = (SuperresolutionPSF.sourceCoord
                                               - SuperresolutionPSF.sourceCoordRound)
        # set sourceCoordUp to be same as sourceCoord
        SuperresolutionPSF.sourceCoordUp = SuperresolutionPSF.sourceCoord  # same because starting at low-res
        # same because starting at low-res
        SuperresolutionPSF.sourceCoordRoundUp = SuperresolutionPSF.sourceCoordRound
        # same because starting at low-res
        SuperresolutionPSF.sourceCoordDeltaUp = SuperresolutionPSF.sourceCoordDelta
        SuperresolutionPSF.sourceCoordRoundDown = np.int64(np.round(self.sourceCoordDown))
        SuperresolutionPSF.abmag_stacker = self.abmag[self.good_mags]
        SuperresolutionPSF.sourceCoordFlux_stacker = self.abmag[self.good_mags]
        SuperresolutionPSF.numSources_stacker = SuperresolutionPSF.abmag_stacker.size
        SuperresolutionPSF.sourceCoordDown_stacker = self.sourceCoordDown[self.good_mags]
        SuperresolutionPSF.sourceCoordRoundDown_stacker =\
            SuperresolutionPSF.sourceCoordRoundDown[self.good_mags]
        SuperresolutionPSF.sourceCoordUp_stacker = SuperresolutionPSF.sourceCoordUp[self.good_mags]
        SuperresolutionPSF.sourceCoordRoundUp_stacker = SuperresolutionPSF.sourceCoordRoundUp[self.good_mags]
        SuperresolutionPSF.sourceCoordDeltaUp_stacker = SuperresolutionPSF.sourceCoordDeltaUp[self.good_mags]
