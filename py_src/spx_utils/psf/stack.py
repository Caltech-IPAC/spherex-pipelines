__all__ = ["StackerVirtual"]

from spx_utils.psf.psf_parameters import SuperresolutionPSF
import numpy as np
import random
from spx_utils.psf.regrid import RegridderZenAccel, RegridderZenMats
import psutil
import platform
import logging
# from scipy.ndimage import center_of_mass, shift


# Please remember to comment out logging level changes applied during development
# logging.basicConfig(level=logging.ERROR,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def StackerVirtual(gridDown,  # spximg.data
                   sourceCoordDown,  # cat_down
                   sourceCoordFlux_est,  # abmag
                   starmask,
                   statmask,
                   s,  # super_psf
                   v):  # verbose
    """Virtual stacking method for superresolution PSF."""

    if s.numSources_stacker == 0:
        # Return empty arrays or None values as placeholders to match
        # expected output format
        # log warning to keep track of where no sources are found
        print("NO SOURCES IN STACK !")
        empty_stack = np.empty((s.stackSize, s.stackSize)) * np.nan  # Match expected stack shape
        return (empty_stack, [], np.array([], dtype=bool), None, None,
                empty_stack, empty_stack, empty_stack, [])

    if v is True:
        buf = 'OG Grid Size = '+str(s.gridSize[0])+'x'+str(s.gridSize[1])+'\t| Factor Diff = '\
            + str(s.factorDiff)+' (fUp/fDown)' + '\nDownsize Grid Size = '+str(s.gridSizeDown[0])\
            + 'x'+str(s.gridSizeDown[0])+'\t| Factor Down = '+str(s.factor_down)\
            + '\nUpsize Grid Size = '+str(s.gridSizeUp[0])+'x'+str(s.gridSizeUp[0])\
            + '\t| Factor Up = '+str(s.factor_up)
        print(buf)

    # sourceCoordFlux = abmag here.
    if v is True:
        print('-------------------')
        print('Getting source fluxes from sourceCoordFlux and estimating noise parameters...')

    # Estimate noise for pre-made masks
    noiseEst_mean = np.nanmean(gridDown[np.logical_not(starmask | statmask)])  # Calculate mean
    noiseEst_std = np.nanstd(gridDown[np.logical_not(starmask | statmask)])  # Calculate std
    noiseFloor = (noiseEst_mean + noiseEst_std)  # Estimate the noise floor

    # Calculate the flux for normalization wrt noise floor
    flux_norm = noiseFloor * SuperresolutionPSF.FLG_noiseFloor_mult / 2
    if (SuperresolutionPSF.FLG_useStarCount_Norm == 1):  # Override above for static all-flux
        flux_norm = 1000  # Just set it to the defined flux

    sumtrack = []

    if (((s.numSources_stacker*s.stackSize*s.stackSize*8)/1024**3
         < s.stackMemLimit) | (s.stackMemLimit == -1)):
        FLG_useStackMat = 1  # Turn on flag
        if v is True:
            print("Using FLG_useStackMat...")
        # Prep place to hold all source shots for
        np.seterr(invalid='ignore')
        stackMat = np.empty((s.numSources_stacker, s.stackSize, s.stackSize))*np.nan
    else:
        FLG_useStackMat = 0  # Turn off flag if memory usage would be too large
        if v is True:
            print("Memory usage would be too large. Not using FLG_useStackMat...")
        stack = np.zeros((s.stackSize, s.stackSize))  # Prep
    np.seterr(invalid='ignore')
    stack5 = np.empty((s.stackSize, s.stackSize))*np.nan  # NaN it so things don't error out
    stack50 = np.empty((s.stackSize, s.stackSize))*np.nan  # NaN it so things don't error out
    stack100 = np.empty((s.stackSize, s.stackSize))*np.nan  # NaN it so things don't error out

    # ----prep for removing sources that are too close (low-res way) -----
    # Array for deleting a stamp that has sources too close to it
    deleteArray = np.zeros(s.numSources_stacker, dtype=bool)
    # Get a meshgrid of the virtualArrayDown
    maskPts_x, maskPts_y = np.meshgrid(np.arange(0, s.halfStampSizeDown * 2),
                                       np.arange(0, s.halfStampSizeDown * 2))
    # Get the max for the dist and then make it a "distance" w/o the sqrt
    dist_maxPower = 2 * s.gridSizeDown[0]*s.gridSizeDown[1]*s.factor_up**2
    dist_delete = s.dist_delete**2  # Undo sqrt used when making this distance
    dist_maskSqrt = s.dist_mask  # Keep the sqrt for other things
    dist_mask = s.dist_mask**2  # Undo sqrt used when making this distance
    dist_maskRadius = s.dist_maskRadius**2  # Undo sqrt used when making this distance

    counter_OK = 0
    counter_masked = 0
    counter_fluxOK = 0
    counter_fluxDelete = 0
    counter_fluxMasked = 0
    counter_deleteClose = 0
    counter_deleteBelowNoise = 0

    # Calculate edge offset as 10% of stack size
    # (for the band around the edge of a stamp to estimate per-stamp noise)
    edgeOffset = np.int64(s.stackSize*.10)

    eps32 = np.finfo(np.float32).eps
    yMatrixOP, xMatrixOP = RegridderZenMats(np.int64(np.array((s.halfStampSizeDown*2,
                                                               s.halfStampSizeDown*2))),  # 10x10
                                            np.int64(np.array((s.stackSize, s.stackSize))))  # 100x100

    # The +1,+1 and +factorUp,+factorUp are to capture an extra row and column
    # of pixels that are needed during rolling to be absolutely accurate to the
    # real grid
    yMatrix, xMatrix = RegridderZenMats(np.int64(np.array((s.halfStampSizeDown*2+1,
                                                           s.halfStampSizeDown*2+1))),  # 11x11
                                        np.int64(np.array((s.stackSize+s.factor_up,
                                                           s.stackSize+s.factor_up))))  # 110x110
    yMatrix32 = np.float32(yMatrix)  # Get a 32-bit version
    xMatrix32 = np.float32(xMatrix)  # Get a 32-bit version
    yMatrixOP32 = np.float32(yMatrixOP)  # Get a 32-bit version
    xMatrixOP32 = np.float32(xMatrixOP)  # Get a 32-bit version

    # Parallel prep
    # Many stamps are rejected so the compute load is light
    # use hyperthreads if available.
    # Consider the following 3 cases must be handled on any number of sources and cores.
    # 1. More threads than sources.
    # 2. More sources than threads.
    # 3. Perfect and imperfect division of sources among threads.
    # Minimum prevents having more threads than sources.
    numThreads = min(psutil.cpu_count(logical=True), s.numSources_stacker)
    # Calculate how many are needed.
    numSources_perThread = np.int64(np.ceil(s.numSources_stacker/numThreads))
    numSources_perThread_indexr = np.arange(0, s.numSources_stacker + numSources_perThread,
                                            numSources_perThread)

    if (numSources_perThread_indexr[-1] != s.numSources_stacker):
        # Make sure it ends with total number of sources
        # (if #s didn't work perfectly, which is likely)
        numSources_perThread_indexr[-1] = s.numSources_stacker

    # Last one isn't full size often, precalc
    numSources_perThread_length = np.diff(numSources_perThread_indexr)
    numThreads = len(numSources_perThread_length)

    # Prepare the list of arguments for each thread.
    parallelList = [None for i in range(0, numThreads)]  # One list per thread
    for i in range(0, numThreads):
        try:
            # Insert a huge amount of variables
            parallelList[i] = (numSources_perThread_length[i], s.stackSize,
                               FLG_useStackMat, sourceCoordFlux_est,
                               sourceCoordDown, s.sourceCoordRoundDown, s.sourceCoordDown_stacker[
                               numSources_perThread_indexr[i]:numSources_perThread_indexr[i+1], :],
                               s.sourceCoordRoundDown_stacker[
                               numSources_perThread_indexr[i]:numSources_perThread_indexr[i+1], :],
                               s.sourceCoordUp_stacker[
                               numSources_perThread_indexr[i]:numSources_perThread_indexr[i+1], :],
                               s.sourceCoordRoundUp_stacker[
                               numSources_perThread_indexr[i]: numSources_perThread_indexr[i+1], :],
                               gridDown, s.gridSizeDown, s.halfStampSizeDown, s.factor_up,
                               s.halfStampSizeUp, s.halfStampSizeUp_tru,
                               s.FLG_useStarCount, s.FLG_useStarCount_Norm, s.FLG_useStarCount_Norm_mult,
                               s.FLG_directMag, s.FLG_opticalPSFoverride, s.FLG_correctStack,
                               s.FLG_noise, edgeOffset, s.sourceSize,
                               s.sourceCoordFlux_stacker[
                               numSources_perThread_indexr[i]:numSources_perThread_indexr[i+1]],
                               s.pixelrms, flux_norm,
                               s.FLG_starDistDelete, s.FLG_starDistMask,
                               dist_maxPower, dist_delete, dist_mask, dist_maskSqrt, s.FLG_starDistMask_mult,
                               dist_maskRadius, maskPts_x, maskPts_y,
                               xMatrix, yMatrix, xMatrix32, yMatrix32, xMatrixOP32, yMatrixOP32, eps32,
                               s.FLG_stackerStampChecker)
        except IndexError as e:
            logging.error(f"{e}:"
                          "numThreads: %s, Lengths - numSources_perThread_length: %s, "
                          "numSources_perThread_indexr: %s, s.sourceCoordDown_stacker: %s",
                          numThreads, len(numSources_perThread_length), len(numSources_perThread_indexr),
                          s.sourceCoordDown_stacker.shape[0])
            raise

    # Call parallel function
    if platform.system() == "k":  # Making a  pool caused error on Mac pyharm so not using
        from multiprocessing import Pool
        pool = Pool(numThreads)  # Make a pool

        import time
        tic = time.time()
        retzObj = []
        for listBit in parallelList:
            retzObj.append(pool.apply_async(_StackerVirtualAccel_stamper, args=listBit))
        pool.close()
        pool.join()
        retz = [None] * numThreads
        for i in numThreads:
            retz[i] = retzObj[i].get()
        if v is True:
            print('Time to run parallel stacker: '+str(time.time()-tic)+' sec, '+str(np.round((
                time.time()-tic)/60, 3))+' min')
    else:
        # Windows does not play nicely with multiprocessing
        # and no "main function" shennanigans, use joblib instead if so
        from joblib import Parallel, delayed

        import time
        tic = time.time()
        if (s.FLG_stackerStampChecker == 0):
            # parallel for speed
            retz = Parallel(n_jobs=numThreads)(
                delayed(_StackerVirtualAccel_stamper)(*listBit) for listBit in parallelList)
        else:
            # sequential for debug
            retz = [None for i in range(0, numThreads)]
            offset = 0
            for i in range(0, numThreads):
                if v is True:
                    print('===Set #'+str(i+1)+', offset to match true index = '+str(offset)+'===')
                retz[i] = _StackerVirtualAccel_stamper(*parallelList[i])
                offset += numSources_perThread_length[i]
        if v is True:
            print('Time to run parallel stacker: '+str(time.time()-tic)+' sec, '+str(np.round((
                time.time()-tic)/60, 3))+' min')
        # Unpack the list of returns
        # Count backwards to limit memory usage
        # (pops last as it goes, list of tuples so only option)
        for i in range(len(retz)-1, -1, -1):
            # ---sumtrack 3---
            sumtrack.extend(retz[i][3])  # Add to sumtrack

            # ---deleteArray 2---
            # Insert the deleteArray
            deleteArray[numSources_perThread_indexr[i]:numSources_perThread_indexr[i+1]] = retz[i][2]

            # ---counters 1---
            counter_OK += retz[i][1]['OK']
            counter_masked += retz[i][1]['masked']
            counter_fluxOK += retz[i][1]['flux OK']
            counter_fluxMasked += retz[i][1]['flux masked']
            counter_fluxDelete += retz[i][1]['flux delete']
            counter_deleteClose += retz[i][1]['delete close']
            counter_deleteBelowNoise += retz[i][1]['delete below noise']

            # ---stack 0---
            if (FLG_useStackMat == 1):
                # Insert the stacks
                stackMat[numSources_perThread_indexr[i]:numSources_perThread_indexr[i+1], :, :] = retz[i][0]
            else:
                stack += retz[i][0]  # Add on the stacks

            # ---control memory---
            retz.pop(-1)  # Control memory (pop entire list)

    if (FLG_useStackMat == 1):  # If memory allows, do this
        logging.info(f"Total sources: {s.numSources_stacker}")
        logging.info(f"Deleted sources: {np.sum(deleteArray)}")
        usable_sources = s.numSources_stacker - np.sum(deleteArray)
        logging.info(f"Usable sources: {usable_sources}")

        # Log a warning if all sources are deleted
        if usable_sources == 0:
            logging.warning("All sources were deleted. Stack matrices may be empty!")

        stackMat = np.delete(stackMat, np.where(deleteArray == 1)[0], axis=0)

        if s.FLG_source_perc_restrict == 1:  # Restricts number of sources.
            oglen = len(stackMat)
            resnumsource = int(np.round(oglen*s.res_perc/100))
            random.seed(s.startSeed)
            select_idx = random.sample(range(oglen), resnumsource)
            stackMat = stackMat[select_idx, :, :]
        else:
            select_idx = None
            resnumsource = None

        stack = np.nanmean(stackMat, axis=0)  # Direct mean
        if np.isnan(stack).all():
            logging.warning("Computed stack contains only NaN values (empty stack).")

        if ((~deleteArray).sum() >= 5):
            stack5 = np.nanmean(stackMat[0:5, :, :], axis=0)  # Direct mean
            if np.isnan(stack5).all():
                logging.warning("Computed stack5 contains only NaN values (empty stack).")

        if ((~deleteArray).sum() >= 50):
            stack50 = np.nanmean(stackMat[0:50, :, :], axis=0)  # Direct mean
            if np.isnan(stack50).all():
                logging.warning("Computed stack50 contains only NaN values (empty stack).")

        if ((~deleteArray).sum() >= 100):
            stack100 = np.nanmean(stackMat[0:100, :, :], axis=0)  # Direct mean
            if np.isnan(stack100).all():
                logging.warning("Computed stack100 contains only NaN values (empty stack).")

    else:
        stack = stack/(s.numSources_stacker-np.nansum(deleteArray))  # Mimic a mean here

        if s.FLG_source_perc_restrict == 1:
            oglen = len(stackMat)
            resnumsource = np.int(np.round(oglen*s.res_perc/100))
            random.seed(s.startSeed)
            select_idx = random.sample(range(oglen), resnumsource)
            raise ValueError("Source percent restriction is on"
                             f"but stackMemLimit memory limit {s.stackMemLimit} GB"
                             " was reached. Non-stackMat method no support source restrictions.")
        else:
            select_idx = None
            resnumsource = None
    if v is True:
        print(f'Stacker Stats - Usable sources: {np.int64(s.numSources_stacker-np.sum(deleteArray))} |'
              f'Total sources: {s.numSources_stacker} |'
              f'%used: {((s.numSources_stacker-np.sum(deleteArray)) / s.numSources_stacker*100)}')
    strReport = 'Bonus Stacker Stats - '  # Prep string, add stuff to it
    if ((s.FLG_starDistDelete == 0) & (s.FLG_starDistMask == 0)):
        strReport = f'{strReport} no sources deleted or masked based on distance, total sources: {counter_OK}'
    elif ((s.FLG_starDistDelete == 0) & (s.FLG_starDistMask > 0)):
        strReport = f'{strReport} Rogue sources: {counter_OK} | Masked sources: {counter_masked}'
    elif ((s.FLG_starDistDelete == 1) & (s.FLG_starDistMask > 0)):
        strReport = f'{strReport} Rogue sources: {counter_OK} | Masked sources: {counter_masked} |'\
            f'Flux OK sources: {counter_fluxOK} | Flux OK & Masked sources: {counter_fluxMasked} | '\
            f'Flux deleted sources: {counter_fluxDelete}'
    elif ((s.FLG_starDistDelete == 1) & (s.FLG_starDistMask == 0)):
        strReport = f'{strReport} Rogue sources: {counter_OK} | Flux OK sources: {counter_fluxOK} |'\
            f' Flux deleted sources: {counter_fluxDelete}'
    elif ((s.FLG_starDistDelete == 2) & (s.FLG_starDistMask > 0)):
        strReport = f'{strReport} Rogue sources: {counter_OK} | Masked sources: {counter_masked} |'\
            f' Deleted too close sources: {counter_deleteClose}'
    elif ((s.FLG_starDistDelete == 2) & (s.FLG_starDistMask == 0)):
        strReport = f'{strReport} Rogue sources: {counter_OK} |'\
            f' Deleted too close sources: {counter_deleteClose}'
    if ((s.FLG_useStarCount >= 1) & (s.FLG_useStarCount_Norm == 2)):
        strReport = f'{strReport} | Below noise floor deleted: {counter_deleteBelowNoise}'
    if v is True:
        print(strReport)  # Report whatever string was made

    return (stack, sumtrack, deleteArray, resnumsource, select_idx,
            stack5, stack50, stack100, stackMat)


def _StackerVirtualAccel_stamper(numSources_stacker, stackSize, FLG_useStackMat,
                                 sourceCoordFlux_est, sourceCoordDown, sourceCoordRoundDown,
                                 sourceCoordDown_stacker, sourceCoordRoundDown_stacker,
                                 sourceCoordUp_stacker, sourceCoordRoundUp_stacker,
                                 gridDown, gridSizeDown, halfStampSizeDown, factorUp,
                                 halfStampSizeUp, halfStampSizeUp_tru,
                                 FLG_useStarCount, FLG_useStarCount_Norm, FLG_useStarCount_Norm_mult,
                                 FLG_directMag, FLG_opticalPSFoverride, FLG_correctStack,
                                 FLG_noise, edgeOffset, sourceSize, sourceCoordFlux_stacker, pixelrms,
                                 flux_norm,
                                 FLG_starDistDelete, FLG_starDistMask,
                                 dist_maxPower, dist_delete, dist_mask, dist_maskSqrt, FLG_starDistMask_mult,
                                 dist_maskRadius, maskPts_x, maskPts_y,
                                 xMatrix, yMatrix, xMatrix32, yMatrix32, xMatrixOP32, yMatrixOP32, eps32,
                                 FLG_stackerStampChecker):
    """Parallelization for virtual stamper function."""

    # print(f"gridDown in _StackerVirtualAccel_stamper {gridDown.shape}") (2048, 2048)

    # Array for deleting a stamp that has sources too close to it
    deleteArray = np.zeros(numSources_stacker, dtype=bool)
    maskMat = np.zeros((halfStampSizeDown*2, halfStampSizeDown*2), dtype=bool)
    padVals = np.zeros(4, dtype=np.int64)  # Prep it
    sumtrack = []  # Prep it
    if FLG_useStackMat == 1:
        np.seterr(invalid='ignore')
        stackMat = np.empty((numSources_stacker, stackSize, stackSize)) * np.nan
    else:
        stack = np.zeros((stackSize, stackSize))  # Prep
    counter_OK = 0
    counter_masked = 0
    counter_fluxOK = 0
    counter_fluxDelete = 0
    counter_fluxMasked = 0
    counter_deleteClose = 0
    for i in range(0, numSources_stacker):
        # ----- First check for deletion/masking -----
        if ((FLG_starDistDelete > 0) | (FLG_starDistMask > 0)):
            # Mask out sources that are in the virtualArrayDown before upscaling
            # sqrt removed for speed, but add them to make sense when debugging!!!
            kNear = (sourceCoordDown[:, 0] <= (sourceCoordDown_stacker[i, 0]+dist_maskSqrt)) & \
                (sourceCoordDown[:, 0] >= (sourceCoordDown_stacker[i, 0]-dist_maskSqrt)) & \
                (sourceCoordDown[:, 1] <= (sourceCoordDown_stacker[i, 1]+dist_maskSqrt)) & \
                (sourceCoordDown[:, 1] >= (sourceCoordDown_stacker[i, 1]-dist_maskSqrt))
            # Try to get rid of things that aren't needed (do not calculate if far)
            dist = ((sourceCoordDown[kNear, 0] - sourceCoordDown_stacker[i, 0]) ** 2
                    + (sourceCoordDown[kNear, 1] - sourceCoordDown_stacker[i, 1]) ** 2)
            # NOTE WIll be more than one when we concatenate exposures.
            # if (np.sum(dist == 0) > 1):
            # raise ValueError(f"ERROR: {np.sum(dist == 0)} stars with a distance of 0,
            # only should be 1.")

            # Get the index of the current star/galaxy wrt the whole source coord array
            distSelf_Index = np.where(dist == 0)[0][0]
            dist[distSelf_Index] = dist_maxPower  # Sets it so it won't find itself
            if (FLG_starDistDelete > 0):
                distInDelete = dist_delete >= dist  # Calculate once
            if (FLG_starDistMask > 0):
                distInMask = dist_mask >= dist  # Calculate once
            if ((FLG_starDistDelete > 0) & (np.any(distInDelete) == 1)):
                if (FLG_starDistDelete == 1):
                    # Check source flux for the various sources that are too close
                    # Get the flux of the self
                    distSelf_flux = sourceCoordFlux_est[kNear][distSelf_Index]
                    # Get the flux of the too close stars
                    distComp_flux = sourceCoordFlux_est[kNear][np.where(distInDelete)]
                    if (((FLG_directMag == 0)
                            & (np.all(distSelf_flux > (distComp_flux*FLG_useStarCount_Norm_mult)) == 0))
                            | ((FLG_directMag == 1)
                               & (np.all(distSelf_flux < (distComp_flux-FLG_useStarCount_Norm_mult)) == 0))):
                        # Sources are too close and current source is too dim
                        # remove entire stamp
                        deleteArray[i] = True  # Set it to delete
                        # Increment counter deleted
                        # star was too dim compared to close stars
                        counter_fluxDelete = 1
                        # delete_reason = 'too close, flux check on'
                    elif ((FLG_starDistMask > 0)
                            & (((FLG_directMag == 0)
                                & (np.all(distSelf_flux
                                          > (distComp_flux*FLG_useStarCount_Norm_mult)) == 1))
                               | ((FLG_directMag == 1)
                                  & (np.all(distSelf_flux
                                            < (distComp_flux-FLG_useStarCount_Norm_mult)) == 1)))):
                        # If above was true - then the current source is
                        # xFLG_useStarCount_Norm_mult brighter than others,
                        # and will be kept
                        # (only mask if DistMask is ON (>0))
                        # Mask the rogue source(s) out -
                        # ignore the delete ones since they're inconsequential
                        # Get where it happens
                        mask_where = np.where((distInMask) & np.logical_not(distInDelete))[0]
                        if ((FLG_starDistMask == 1) | (FLG_starDistMask == 3)):
                            # Get the flux of the too close stars
                            distCompMask_flux = sourceCoordFlux_est[kNear][mask_where]
                            # Also limit compared by flux
                            if (FLG_directMag == 0):
                                # Use this to keep which stars to mask
                                # (throw out low-mag ones)
                                kKeep = distSelf_flux < (distCompMask_flux * FLG_starDistMask_mult)
                            elif (FLG_directMag == 1):
                                # Use this to keep which stars to mask
                                # (throw out low-mag ones)
                                kKeep = distSelf_flux > (distCompMask_flux - FLG_starDistMask_mult)
                            # Keep only the ones that are bright enough to mask
                            mask_where = mask_where[kKeep]
                        # Run through them all and mask
                        for j in range(0, mask_where.size):
                            # Get x delta+halfStampSizeDown; to make it the right coords
                            maskSource_x = (sourceCoordRoundDown[kNear][mask_where[j], 0]
                                            - sourceCoordRoundDown_stacker[i, 0] + halfStampSizeDown)
                            # Get y delta+halfStampSizeDown; to make it the right coords
                            maskSource_y = (sourceCoordRoundDown[kNear][mask_where[j], 1]
                                            - sourceCoordRoundDown_stacker[i, 1] + halfStampSizeDown)
                            maskDist = (maskSource_x - maskPts_x) ** 2 + (maskSource_y - maskPts_y) ** 2
                            # Make a mask matrix, use it after rolling
                            maskMat = np.add(maskMat, (maskDist <= dist_maskRadius))
                        if (mask_where.size > 0):
                            counter_fluxMasked = 1  # Increment flux OK masked counter
                        else:
                            counter_fluxOK = 1  # Increment flux OK (no masking was needed)
                    elif ((FLG_starDistMask == 0)
                          & (((FLG_directMag == 0)
                              & (np.all(distSelf_flux
                                        > (distComp_flux*FLG_useStarCount_Norm_mult)) == 1))
                             | ((FLG_directMag == 1)
                                & (np.all(distSelf_flux
                                          < (distComp_flux-FLG_useStarCount_Norm_mult)) == 1)))):
                        counter_fluxOK = 1  # Increment flux OK (masking is off)
                elif (FLG_starDistDelete == 2):
                    # Sources are too close and flux is disabled, remove entire stamp
                    deleteArray[i] = True  # Set it to delete
                    counter_deleteClose = 1  # Increment too close and flux is off counter
                    # delete_reason = 'too close, flux check off'
            elif (((FLG_starDistDelete > 0)
                   & (FLG_starDistMask > 0)
                   & (np.any(distInDelete) == 0)
                   & (np.any(distInMask) == 1))
                  | ((FLG_starDistDelete == 0)
                     & (FLG_starDistMask > 0)
                     & (np.any(distInMask) == 1))):
                # Two cases lead to only masking:
                # [1] DistDelete is ON (>0) and DistMask is ON (>0)
                # and nothing was in range to delete - but something is in range to mask
                # OR [2] DistDelete is OFF (==0) and DistMask is ON (>0)
                # and something is in range to mask
                # Mask the rogue source(s) out
                mask_where = np.where(distInMask)[0]  # Get where it happens
                if ((FLG_starDistMask == 1) | (FLG_starDistMask == 3)):
                    # Get the flux of the self
                    distSelf_flux = sourceCoordFlux_est[kNear][distSelf_Index]
                    # Get the flux of the too close stars
                    distCompMask_flux = sourceCoordFlux_est[kNear][mask_where]
                    # Also limit compared by flux
                    if (FLG_directMag == 0):
                        # Use this to keep which stars to mask (throw out low-mag ones)
                        kKeep = distSelf_flux < (distCompMask_flux * FLG_starDistMask_mult)
                    elif (FLG_directMag == 1):
                        # Use this to keep which stars to mask (throw out low-mag ones)
                        kKeep = distSelf_flux > (distCompMask_flux - FLG_starDistMask_mult)
                    mask_where = mask_where[kKeep]  # Keep only the ones needed
                # Run through them all and mask
                for j in range(0, mask_where.size):
                    # Get x delta+halfStampSizeDown to make it the right coords
                    maskSource_x = (sourceCoordRoundDown[kNear][mask_where[j], 0]
                                    - sourceCoordRoundDown_stacker[i, 0] + halfStampSizeDown)
                    # Get y delta+halfStampSizeDown to make it the right coords
                    maskSource_y = (sourceCoordRoundDown[kNear][mask_where[j], 1]
                                    - sourceCoordRoundDown_stacker[i, 1] + halfStampSizeDown)
                    maskDist = (maskSource_x - maskPts_x)**2 + (maskSource_y - maskPts_y)**2
                    # Make a mask matrix, use it after rolling
                    maskMat = np.add(maskMat, (maskDist <= dist_maskRadius))
                counter_masked = 1  # Increment masked counter
            else:
                counter_OK = 1  # Increment OK counter, cause nothing was too close
        else:
            counter_OK = 1  # Increment OK counter, cause nothing was too close

        # ----- Only do the stacking work if the source isn't destined to be NaN'd out
        if ((deleteArray[i] == 0) | (FLG_stackerStampChecker >= 1)):
            if (FLG_opticalPSFoverride != 1):
                # Get the x rescale factor (remainder after dividing by factor)
                xpAdj = np.remainder(sourceCoordRoundUp_stacker[i, 0], factorUp)
                # Get the y rescale factor (remainder after dividing by factor)
                ypAdj = np.remainder(sourceCoordRoundUp_stacker[i, 1], factorUp)

                # This is some subpixel shift magic found by reverse engineering
                # For a factorUp of 10
                # 0  1  2  3  4    5 6 7 8 9 <- remainders you can get
                # 0 -1 -2 -3 -4 +/-5 4 3 2 1 <- pixel shifts calculated
                #                      from those above (middle is weird!)
                if (xpAdj < factorUp//2):
                    xpAdj = -1 * xpAdj  # Negate to enforce accuracy
                elif (xpAdj > factorUp//2):
                    xpAdj = factorUp - xpAdj  # Revert so 9->1 etc. for a factorUp of 10
                # If source coord is divisible by 2 is the deciding factor
                elif ((xpAdj == factorUp//2)
                      & (np.mod(sourceCoordRoundUp_stacker[i, 0]
                                // factorUp, 2) == 0)
                      & (np.round(sourceCoordUp_stacker[i, 0]/factorUp)
                         == sourceCoordRoundUp_stacker[i, 0]//factorUp)):
                    xpAdj = -1 * xpAdj  # This time the middle value is negated
                # If source coord is divisible by 2 is the deciding factor
                elif ((xpAdj == factorUp//2)
                      & (np.mod(sourceCoordRoundUp_stacker[i, 0]//factorUp, 2) != 0)
                      & (np.round(sourceCoordUp_stacker[i, 0]/factorUp)
                         == sourceCoordRoundUp_stacker[i, 0]//factorUp)):
                    xpAdj = -1 * xpAdj  # This time the middle value is negated
                if (ypAdj < factorUp//2):
                    ypAdj = -1 * ypAdj  # Negate to enforce accuracy
                elif (ypAdj > factorUp//2):
                    ypAdj = factorUp - ypAdj  # Revert so 9->1 etc. for a factorUp of 10
                # If source coord is divisible by 2 is the deciding factor
                elif ((ypAdj == factorUp//2)
                      & (np.mod(sourceCoordRoundUp_stacker[i, 1]//factorUp, 2) == 0)
                      & (np.round(sourceCoordUp_stacker[i, 1]/factorUp)
                         == sourceCoordRoundUp_stacker[i, 1]//factorUp)):
                    ypAdj = -1 * ypAdj  # This time the middle value is negated
                # If source coord is divisible by 2 is the deciding factor
                elif ((ypAdj == factorUp//2)
                      & (np.mod(sourceCoordRoundUp_stacker[i, 1]//factorUp, 2) != 0)
                      & (np.round(sourceCoordUp_stacker[i, 1]/factorUp)
                         == sourceCoordRoundUp_stacker[i, 1]//factorUp)):
                    ypAdj = -1 * ypAdj  # This time the middle value is negated

                # The np.int64(xAdj<0) stuff is to get an extra row&column of pixels
                # for rolling later after virtualArrayUp is made (the np.roll call) -
                # w/o the extra row&column it rolls edge data
                # that's not fully real-world
                # CUT OUT
                if (FLG_correctStack == 1):  # Correction Enabled.
                    xp = [np.int64(sourceCoordRoundDown_stacker[i, 0]
                                   - halfStampSizeDown - np.int64(xpAdj >= 0)),
                          np.int64(sourceCoordRoundDown_stacker[i, 0]
                                   + halfStampSizeDown + np.int64(xpAdj < 0))]
                    yp = [np.int64(sourceCoordRoundDown_stacker[i, 1]
                                   - halfStampSizeDown - np.int64(ypAdj >= 0)),
                          np.int64(sourceCoordRoundDown_stacker[i, 1]
                                   + halfStampSizeDown + np.int64(ypAdj < 0))]
                else:  # Stack may have edge roll issues.
                    xp = [np.int64(sourceCoordRoundDown_stacker[i, 0]
                                   - halfStampSizeDown),
                          np.int64(sourceCoordRoundDown_stacker[i, 0]
                                   + halfStampSizeDown)]
                    yp = [np.int64(sourceCoordRoundDown_stacker[i, 1]
                                   - halfStampSizeDown),
                          np.int64(sourceCoordRoundDown_stacker[i, 1]
                                   + halfStampSizeDown)]

                # NOTE Here account for bin boundary selection.

                padVals[:] = 0  # Reset padVals to 0 but don't remake it with np.zeros (speeds it up)
                if (xp[0] < 0):
                    padVals[2] = -xp[0]  # Record how much
                    xp[0] = 0  # Set so no errors
                if (xp[1] > gridSizeDown[1]):
                    padVals[3] = xp[1]-gridSizeDown[1]  # Record how much
                    xp[1] = gridSizeDown[1]
                if (yp[0] < 0):
                    padVals[0] = -yp[0]  # Record how much
                    yp[0] = 0  # Set so no errors
                if (yp[1] > gridSizeDown[0]):
                    padVals[1] = yp[1]-gridSizeDown[0]  # Record how much
                    yp[1] = gridSizeDown[0]

                # stackVar
                virtualArrayDown = np.copy(gridDown[yp[0]:yp[1], xp[0]:xp[1]])  # Get stamp

                sumtrack.append(np.sum(virtualArrayDown))

                if (np.sum(padVals) != 0):  # Pad if there's pad values to pad with
                    virtualArrayDown = np.pad(virtualArrayDown,
                                              ((padVals[0], padVals[1]),
                                               (padVals[2], padVals[3])),
                                              'constant', constant_values=0)

                nanlocs = np.isnan(virtualArrayDown)  # Get the NaN locations
                virtualArrayDown[nanlocs] = 0  # 0 NaNs for the reconstruction
                virtualArrayUp = RegridderZenAccel(virtualArrayDown, yMatrix, xMatrix)
                nanlocsUp = RegridderZenAccel(np.float32(nanlocs), yMatrix32, xMatrix32)
                virtualArrayUp[nanlocsUp > eps32] = np.nan  # Re-NaN the bits that are to be NaN'd
                # Enforces the subpixel shift needed for the magic
                virtualArrayUp = np.roll(virtualArrayUp, (xpAdj, ypAdj), axis=(1, 0))
                # # Compute centroid of virtualArrayUp (ignore NaNs)
                # if not np.all(np.isnan(virtualArrayUp)):
                # com = center_of_mass(np.nan_to_num(virtualArrayUp, nan=0.0))
                # center = [virtualArrayUp.shape[0] // 2, virtualArrayUp.shape[1] // 2]
                #     dy = center[0] - com[0]
                #     dx = center[1] - com[1]

                #     # Shift image to align centroid to center
                #     virtualArrayUp = shift(virtualArrayUp,
                # shift=(dy, dx), order=1, mode='constant', cval=np.nan)

                if (FLG_correctStack == 1):
                    # Removes the extra 10
                    # pixels that were kept to keep the edges correct
                    virtualArrayUp =\
                        virtualArrayUp[np.int64((ypAdj >= 0))*factorUp:
                                       stackSize+factorUp-np.int64((ypAdj < 0))*factorUp,
                                       np.int64((xpAdj >= 0))*factorUp:stackSize
                                       + factorUp-np.int64((xpAdj < 0))*factorUp]
                # Now apply the mask
                if (maskMat.sum() > 0):  # If a maskMat needs to be applied
                    # Scale up to fine grid
                    # (will not be logical anymore, 1's will be split into fractions,
                    # 0's will stay 0 <- key)
                    maskMatUp = RegridderZenAccel(np.float32(maskMat), yMatrixOP32, xMatrixOP32)
                    virtualArrayUp[maskMatUp > eps32] = np.nan  # Apply the mask
                    maskMat = maskMat & False  # Reset to all 0's

            if (halfStampSizeUp == halfStampSizeUp_tru):
                arraySum = np.nansum(virtualArrayUp)  # Sum whole thing
            else:
                # Sum only tru bit
                arraySum = np.nansum(virtualArrayUp[np.int64(np.floor(virtualArrayUp.shape[0]/2)
                                                             - halfStampSizeUp_tru):
                                                                 np.int64(np.ceil(virtualArrayUp.shape[0]/2)
                                                                          + halfStampSizeUp_tru),
                                                                 np.int64(np.floor(virtualArrayUp.shape[0]/2)
                                                                          - halfStampSizeUp_tru):
                                                                              np.int64(np.ceil(
                                                                                  virtualArrayUp.shape[0]/2)
                                                                                  + halfStampSizeUp_tru)])
            if (arraySum != 0):
                if ((FLG_useStarCount >= 1) & (FLG_useStarCount_Norm == 1)):
                    # Normalize so all fluxes have equal weighting
                    virtualArrayUp = virtualArrayUp*(flux_norm/arraySum)
                elif ((FLG_useStarCount >= 1) & (FLG_useStarCount_Norm == 2)):
                    if (np.nansum(virtualArrayUp) >= flux_norm):
                        # Normalize so all fluxes have equal weighting
                        virtualArrayUp = virtualArrayUp * (flux_norm / arraySum)
                    else:
                        # Normalize so all fluxes have equal weighting
                        virtualArrayUp = virtualArrayUp * (flux_norm / arraySum)

            else:
                deleteArray[i] = True  # Set it to delete (already all NaNs per nansum returning 0)
                # delete_reason = 'all NaNs after masking'

            # Subtract estimate of background from stamp
            if (FLG_noise == 1):
                p1 = virtualArrayUp[:, 0:edgeOffset]
                s1 = virtualArrayUp[:, 0:edgeOffset].size
                p2 = virtualArrayUp[:, stackSize-edgeOffset:stackSize]
                s2 = virtualArrayUp[:, stackSize-edgeOffset:stackSize].size
                p3 = virtualArrayUp[0:edgeOffset, edgeOffset:stackSize-edgeOffset]
                s3 = virtualArrayUp[0:edgeOffset, edgeOffset:stackSize-edgeOffset].size
                p4 = virtualArrayUp[stackSize-edgeOffset:stackSize, edgeOffset:stackSize-edgeOffset]
                s4 = virtualArrayUp[stackSize-edgeOffset:stackSize, edgeOffset:stackSize-edgeOffset].size

                # Subtract background per pixel, remove NaNs from pixel total
                numNoisePix = (s1 + s2 + s3 + s4 - np.isnan(p1).sum()
                               - np.isnan(p2).sum() - np.isnan(p3).sum() - np.isnan(p4).sum())
                if (numNoisePix > 0):
                    # Background is sum of outermost 10 pixels
                    # (10% of outer most pixels - 10 for a stackSize of 100)
                    mbar = np.nansum(p1) + np.nansum(p2) + np.nansum(p3) + np.nansum(p4)
                    virtualArrayUp -= mbar / numNoisePix
                else:
                    # nan all since noise can't be estimated (for now - revisit)
                    virtualArrayUp = virtualArrayUp * np.nan
                    deleteArray[i] = True  # Set it to delete
                    # delete_reason = 'edges all NaNs, noise can\'t be estimated'

            if (FLG_useStackMat == 1):  # If memory allows, do this
                stackMat[i, :, :] = virtualArrayUp  # Set it
            else:
                k = np.isnan(virtualArrayUp)  # Provide NaN protection
                virtualArrayUp[k] = 0  # 0 the NaNs
                stack += virtualArrayUp  # Avoid a big matrix to hold every stack

        if ((FLG_stackerStampChecker == 1)
                | ((FLG_stackerStampChecker == 2) & (deleteArray[i] is False))):
            # Try to get rid of things that aren't needed
            # (no need to calculate if super far away)
            kNear = (sourceCoordDown[:, 0] <= (sourceCoordDown_stacker[i, 0]+dist_maskSqrt)) & \
                    (sourceCoordDown[:, 0] >= (sourceCoordDown_stacker[i, 0]-dist_maskSqrt)) & \
                    (sourceCoordDown[:, 1] <= (sourceCoordDown_stacker[i, 1]+dist_maskSqrt)) & \
                    (sourceCoordDown[:, 1] >= (sourceCoordDown_stacker[i, 1]-dist_maskSqrt))

    if (FLG_useStackMat == 1):  # If memory allows, do this
        stack = stackMat  # Alias for ease
    counters = {
        'OK': counter_OK,
        'masked': counter_masked,
        'flux OK': counter_fluxOK,
        'flux delete': counter_fluxDelete,
        'flux masked': counter_fluxMasked,
        'delete close': counter_deleteClose,
        'delete below noise': 0,  # Unused but fix later
    }  # A bunch of counters for how things went

    return stack, counters, deleteArray, sumtrack
