__all__ = ["BSHConfig", "BSHFlagger", "apply_bsh_flags_to_spx"]


# Standard library imports
import copy

# Related third party imports
# from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy.ndimage import convolve
from scipy import ndimage
from skimage.morphology import disk, binary_dilation, skeletonize
from lsst.pex.config import Config, Field

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_utils.l1_utils import active_area_range

HaloMethod = Literal["border_expand", "dilation", "smooth_skeleton"]


class BSHConfig(Config):
    """Configuration for Bloom, Snowball, and Halo (BSH) flagging."""
    max_trans_frac = Field(
        doc="Maximum allowed transient fraction.",
        dtype=float,
        default=0.10
    )
    max_nonoverbloom_loops = Field(
        doc="Maximum number of loops for non-overflow bloom.",
        dtype=int,
        default=5
    )
    max_bloom_loops = Field(
        doc="Maximum number of loops for regular bloom.",
        dtype=int,
        default=15
    )
    satellite_halo_factor = Field(
        doc="Halo amplification for satellite track.",
        dtype=float,
        default=3.14
    )
    extend_radius_small = Field(
        doc="Extend radius (small).",
        dtype=float,
        default=2.0
    )
    extend_radius_large = Field(
        doc="Extend radius (large).",
        dtype=float,
        default=0.5
    )
    min_overflow_d123 = Field(
        doc="Min overflow fluence for detectors 1, 2, 3 [e-].",
        dtype=float,
        default=5.0e4
    )
    min_overflow_d456 = Field(
        doc="Min overflow fluence for detectors 4, 5, 6 [e-].",
        dtype=float,
        default=3.0e4
    )
    border_diff_min = Field(
        doc="Bloom detection - min diff btwn border layers.",
        dtype=float,
        default=2.0
    )
    satellite_halomethod = Field(
        doc="Method for satellite halos: 'border_expand', 'dilation', or 'smooth_skeleton'.",
        dtype=str,
        default="smooth_skeleton"
    )
    save_borderinfo = Field(
        doc="If True, save the bloom border difference info as a 'BLMBDRIN' layer.",
        dtype=bool,
        default=False
    )

    def setDefaults(self):
        """Validate configuration."""
        super().setDefaults()
        allowed_methods = {"border_expand", "dilation", "smooth_skeleton"}
        if self.satellite_halomethod not in allowed_methods:
            raise ValueError(f"satellite_halomethod must be one of {allowed_methods}, "
                             f"not {self.satellite_halomethod}")


class BSHFlagger:
    """SPHERE Bloom / Snowball / Halo detection and flagger.

    Notes
    -----
    det = BSHDetector(spx, cfg)
    det.detect_bloom()
    det.detect_snowballs()
    det.detect_nonoverflow_bloom()
    det.mask_transient_halo()
    flags are set in-place on spx.flags
    diagnostics stored in layers (optional, good for testing)
    """
    def __init__(self, spx: SPHERExImage, cfg: Optional[BSHConfig] = None, verbose: bool = False):
        self.spx = spx
        self.cfg = cfg or BSHConfig()
        self.verbose = verbose

        self.flags = spx.flags.astype(np.int32, copy=False)
        self.data = spx.data  # e/s
        self.header = self.spx.header
        self.shape = self.data.shape

        # Prefer TINT else fall back to TELAPSE if present.
        try:
            telapse = spx.getExposureDuration().sec
        except Exception:
            telapse = float(spx.header.get("TELAPSE", 0.0))
        if telapse <= 0:
            raise RuntimeError("Cannot compute fluence: exposure duration missing/zero.")
        self.itime = telapse
        self.fluence = self.data * telapse  # electrons

        self.transient = spx.getFlagsAsMask(Flag.TRANSIENT)
        self.non_trans = ~self.transient
        self.early = spx.getFlagsAsMask(Flag.SUR_ERROR)
        self.late = ~self.early
        self.over = spx.getFlagsAsMask(Flag.OVERFLOW)
        self.large = self.over
        self.small = ~self.large
        self.non_funct = spx.getFlagsAsMask(Flag.NONFUNC)

        # Get active area mask
        active_mask = np.zeros(self.shape, dtype=bool)
        try:
            row_slice, col_slice = active_area_range(self.spx, include_refs=False)
            active_mask[row_slice, col_slice] = True
            n_active_pixels = np.count_nonzero(active_mask)
            if n_active_pixels == 0:  # Safety check
                n_active_pixels = (2040 * 2040)  # Fallback
                print(f"Warning: Active area mask was empty. Using default {n_active_pixels} pixels.")
        except Exception as e:
            # Fallback in case ReadoutParameters fails
            n_active_pixels = self.shape[0] * self.shape[1]
            print(f"Warning: Could not determine active area, "
                  f"proceeding with full {self.shape} image. Error: {e}")
            active_mask[:, :] = True  # Treat all as active

        # global stats
        self.trans_frac = np.count_nonzero(self.transient & active_mask) / n_active_pixels

        # High transient fraction (used for re-observation decisions)
        self.high_transient = (self.trans_frac > self.cfg.max_trans_frac)
        self.header['HIGHTRNS'] = (
            self.high_transient,
            'T if transient fraction > max_trans_frac (reobs candidate)'
        )

        if self.trans_frac > self.cfg.max_trans_frac:
            self.extend_radius = 0.5
            self.max_snow_loops = 15
            self.force_snow_circle = True
        else:
            self.extend_radius = 2
            self.max_snow_loops = 30  # This can be reached by DOTs that have heavy signal
            self.force_snow_circle = False

        # detector-dependent overflow threshold
        det = spx.getArrayId()
        if det in (4, 5, 6):
            self.min_overflow = self.cfg.min_overflow_d456
        else:
            self.min_overflow = self.cfg.min_overflow_d123

        # Now add non-active pixels to the non_funct mask to exclude them
        # from all subsequent processing.
        self.non_funct |= ~active_mask

        # Use uint8 for the flags
        self.bloom_flag = np.zeros(self.shape, dtype=np.uint8)
        self.snowball_flag = np.zeros(self.shape, dtype=np.uint8)
        self.halo_flag = np.zeros(self.shape, dtype=np.uint8)
        self.considered = np.zeros(self.shape, dtype=bool)

        # direction sets
        self.diag_dirs = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1),
                                   (-1, -1), (-1, 0), (-1, 1)], dtype=int)
        self.edge_dirs = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=int)

        self.count_bloom = 0
        self.count_snowball = 0
        self.count_5bloom = 0

        self.nn_rows = []
        self.nn_cols = []
        self.bloom_rows = []
        self.bloom_cols = []
        self.frac_flagged = 0
        self.n_bloom = None
        self.border_diff_info = np.empty([0, 4])  # center_row, center_col, type, diff

        self.vis = None
        self.vis_center = None
        self.use_weighted_avg_ctr = False
        self.name = f"BSHFlagger[D{det}]"

        # Any pixel.
        self.any_pix = np.ones(self.flags.shape, dtype='bool')

        # If desired any pixel EXCEPT a non-functional one, use
        # ~self.non_funct
        # self.any_pix[np.where(self.non_funct)] = False

        # Update header with information here
        self.header['TRANSFRC'] = (self.trans_frac, 'Transient Fraction')
        self.header['MXTRNSFR'] = (self.cfg.max_trans_frac, 'Max Allowed Transient Fraction')
        self.header['XTENDRAD'] = (self.extend_radius, 'N Pix Past Bloom Edge Radius')
        self.header['MINOVRFL'] = (self.min_overflow, 'e- beyond which overflow expected')
        self.header['WTAVGCTR'] = (self.use_weighted_avg_ctr, 'T if weighted avg ctr used for bloom')
        self.header['MAXOVRLP'] = (self.cfg.max_nonoverbloom_loops,
                                   'Maximum number of loops for non-overflow bloom')
        self.header['MAXBLMLP'] = (self.cfg.max_bloom_loops,
                                   'Maximum number of loops for regular bloom')
        self.header['MAXSNWLP'] = (self.max_snow_loops, 'Maximum number of loops for snowball')
        self.header['SNWCIRCL'] = (self.force_snow_circle, 'T if snowballs are forced circular')
        self.header['SATHALOM'] = (self.cfg.satellite_halomethod,
                                   'Method by which satellite halos are determined.')
        self.header['SATSTRKS'] = (0, 'No Strong Satellite Streaks Detected')
        self.header['SATHALOF'] = (self.cfg.satellite_halo_factor, 'Halo amplification for satellite track')
        self.header['BRDRMIND'] = (self.cfg.border_diff_min, 'Bloom detection - min diff btwn border layers.')

    def _update_bloom_pix(self):
        self.bloom_rows = np.concatenate((self.bloom_rows, self.nn_rows))
        self.bloom_cols = np.concatenate((self.bloom_cols, self.nn_cols))

    @staticmethod
    def _remove_overlapping_indices_BfromA(indexesA, indexesB):
        # Convert to coordinate pairs (N,2)
        coordsA = np.column_stack(indexesA)
        coordsB = np.column_stack(indexesB)

        # Remove overlaps, preserving order
        coordsB_set = set(map(tuple, coordsB))
        mask = [tuple(idx) not in coordsB_set for idx in coordsA]
        uniqueA = coordsA[mask]

        # Convert back to np.where style (rows, cols)
        rows, cols = uniqueA[:, 0], uniqueA[:, 1]
        INDEXESA_unique = (rows, cols)

        return INDEXESA_unique

    def _enlarge_curve_by_smoothed_skeleton(self):
        '''
        # Subroutine to identify SATELLITE HALOs by tracking path and
        # assigning halo radii based upon width along path
        # This version is smoothed so that noise variations in edges have
        # little effect upon the end result.  (RECOMMENDED METHOD)
        '''
        # Step 1: clean up small noise
        mask0 = np.zeros(self.shape, dtype=bool)
        mask0[self.bloom_rows, self.bloom_cols] = True

        # Smooth mask (get rid of pixelation)
        mask = ndimage.gaussian_filter(mask0.astype(float), sigma=2) > 0.5

        # NN convolve original in case smoothing removes necessary detail
        nn_trans = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        mask0 = (convolve(mask0, nn_trans, mode='reflect') > 0)*1

        # Step 2: get skeleton (stable)
        skeleton = skeletonize(mask)

        # Step 3: get smooth distance map
        dist_inside = ndimage.distance_transform_edt(mask)
        dist_inside = ndimage.gaussian_filter(dist_inside, sigma=3)

        # Step 4: distance to skeleton
        dist_to_skel, indices = ndimage.distance_transform_edt(~skeleton, return_indices=True)
        nearest_y, nearest_x = indices
        radius_at_nearest = dist_inside[nearest_y, nearest_x]
        radius_smooth = ndimage.gaussian_filter(radius_at_nearest, sigma=3)

        # Step 5: scale widths smoothly
        # instead of radius_at_nearest
        new_mask = dist_to_skel <= (self.cfg.satellite_halo_factor * radius_smooth)
        (self.halo_rows, self.halo_cols) = np.where((new_mask | mask0))

    def _enlarge_curve_by_dilation(self, approx_halfwidth=None):
        '''
        # Subroutine to assign SATELLITE HALOS by finding the mean width along a track
        # and assigning the same width to the halo all along the track (less than ideal)
        '''
        mask = np.zeros(self.shape, dtype=bool)
        mask[self.bloom_rows, self.bloom_cols] = True

        # estimate avg half-width if not provided
        if approx_halfwidth is None:
            # distance to background inside mask gives local half-width estimate
            dist_inside = ndimage.distance_transform_edt(mask)
            if dist_inside[mask].size == 0:
                avg_half = 0
            else:
                avg_half = np.mean(dist_inside[mask])
        else:
            avg_half = approx_halfwidth

        extra_radius = int(round((self.cfg.satellite_halo_factor) * avg_half))
        # extra_radius = int(round((self.cfg.satellite_halo_factor - 1.0) * avg_half))
        if extra_radius >= 0:
            struct = disk(extra_radius)
            mask = binary_dilation(mask, struct)

        (self.halo_rows, self.halo_cols) = np.where(mask)

    def _spread_search_multi(self, use_dirs, use_vis, check_flags):
        '''
        # This subroutine uses the arrays use_vis and check_flags to find all
        # nearest neighbors using the directions use_dirs.
        # It will then update self.nn_rows and self.nn_cols to contain only the
        # valid nearest neighbors (pixels that are False in use_vis).  It will
        # also update the fraction of those valid nearest neighbors that are
        # flagged in self.frac_flagged.

        # PASS IN EMPTY arrays [], [] for use_vis and check_flags if you want
        # to ignore those.
        '''

        # Define NN Indexes array
        check_rows = np.array([], dtype=int)
        check_cols = np.array([], dtype=int)

        # Get rows / columns of nearest neighbors in edge directions.
        for k in range(len(use_dirs)):
            check_rows = np.concatenate((check_rows, self.nn_rows+use_dirs[k][0]))
            check_cols = np.concatenate((check_cols, self.nn_cols+use_dirs[k][1]))

        # Remove repeated indices.
        all_pairs = np.column_stack([check_rows, check_cols])
        uniq_pairs = np.unique(all_pairs, axis=0)
        indexes = (uniq_pairs[:, 0], uniq_pairs[:, 1])

        # Remove Out of Bounds NNs
        test_oob = np.where(((indexes[0] < 0) | (indexes[0] > self.shape[0]-1)
                             | (indexes[1] < 0) | (indexes[1] > self.shape[1]-1)))
        indexes = (np.delete(indexes[0], test_oob), np.delete(indexes[1], test_oob))

        # INDEXES gives ALL pixels that are not out of bounds.
        # INDICES gives all pixels that are eligible NEXT ROUND candidates.

        if (len(use_vis) == 0) or (len(check_flags) == 0):
            self.nn_rows = indexes[0]
            self.nn_cols = indexes[1]
            self.flag_frac = 0
        else:
            # BELOW gives numbers from which FLAGGING FRACTION can be found.
            outer_edge_pix = self._remove_overlapping_indices_BfromA(indexes,
                                                                     (self.nn_rows, self.nn_cols))

            outer_funct_ind = np.where(~self.non_funct[outer_edge_pix])
            outer_funct_pix = (outer_edge_pix[0][outer_funct_ind],
                               outer_edge_pix[1][outer_funct_ind])

            outer_flagged_ind = np.where(check_flags[outer_funct_pix])
            outer_flagged_pix = (outer_funct_pix[0][outer_flagged_ind],
                                 outer_funct_pix[1][outer_flagged_ind])

            n_outer_funct_pix = len(outer_funct_pix[0])
            n_outer_flagged_pix = len(outer_flagged_pix[0])

            if n_outer_funct_pix == 0:
                self.flag_frac = 0
            else:
                self.flag_frac = n_outer_flagged_pix/n_outer_funct_pix  # len(next_candidates[0])

            # IDENTIFY Next Round candidates:  Are these pixels eligible?  Y/N
            indices = np.where((~use_vis[indexes]) & (check_flags[indexes]))

            # Find next set of NNs to search
            to_check_nn_rows = indexes[0][indices]
            to_check_nn_cols = indexes[1][indices]

            # TRY doing vis here - only with those next pixels to check
            # Properly update: changing the value of a single item and not overwriting.
            use_vis[to_check_nn_rows, to_check_nn_cols] = True

            self.nn_rows = to_check_nn_rows
            self.nn_cols = to_check_nn_cols

    def _grow_halo_border(self, r_max):
        '''
        # Subroutine to assign SATELLITE HALOS by growing the border of the
        # snowball by a uniform number of pixels. This assumes that the track
        # is always the same width.  (less than ideal)
        '''

        self.nn_rows = self.bloom_rows.copy()
        self.nn_cols = self.bloom_cols.copy()

        self.halo_rows = self.bloom_rows.copy()
        self.halo_cols = self.bloom_cols.copy()

        self.considered = np.zeros(self.flags.shape, dtype='bool')
        self.considered[self.nn_rows, self.nn_cols] = True

        n_rounds = int(len(self.bloom_rows)/(np.pi*r_max))  # semi-minor axis if this were an ellipse

        for i in range(n_rounds):
            if len(self.nn_rows) == 0:
                break

            self._spread_search_multi(self.diag_dirs, self.considered, self.any_pix)

            self.halo_rows = np.concatenate((self.halo_rows, self.nn_rows))
            self.halo_cols = np.concatenate((self.halo_cols, self.nn_cols))

    def start_bloom_flag(self):
        '''
        # NOTE: This code sets different values for the different conditions of
        # bloom and snowball detection.  This is used in evaluating accuracy.
        # For IPAC use, set SRC_BLOOM flag for all bloom detections, SNOWBALL
        # flag for detected snowball or high energy cosmic rays.  This version
        # sends snowballs to the snowball code as different conditions applied.
        '''
        # FLAG LAYER COMBOS PARTICULAR TO BLOOM FLAG

        late_transient = (self.transient & self.late & self.small
                          & (self.fluence > self.min_overflow/10))  # 1
        # (with overflow ring this sometimes happens)
        # late_large_trans = (self.transient & self.late & self.large
        #                     & (self.fluence > self.min_overflow/10))  # 3
        early_transient = (self.transient & self.early & self.small
                           & (self.fluence > self.min_overflow/5))  # 5
        overflow_only = (self.large & self.late & self.non_trans
                         & (self.fluence > self.min_overflow))  # 2
        early_large = (self.large & self.early & self.non_trans)  # transient ok 6&7
        early_large_trans = (self.large & self.early & self.transient
                             & (self.fluence > self.min_overflow))  # transient ok 6&7
        # non_bloom = (self.non_trans & self.small)

        # SKIP all pixels marked with non-funct
        self.considered = self.non_funct.copy()

        # FLAGS to consider:
        # blooming = [1, 2, 5, 6, 7] # all of above
        # center_bloom = [2, 6, 7] # early_large | overflow_only

        # Need to consider these without concerning other flags:
        center_bloom_flags = self.over

        # Find center bloom candidates
        # start with early_large, then early_large_trans,
        # then overflow_only for very center.
        indexes_bloom_1 = np.where(early_large)
        indexes_bloom_2 = np.where(early_large_trans)
        indexes_bloom_3 = np.where(overflow_only)

        rows_bloom_ctr = np.concatenate((indexes_bloom_1[0], indexes_bloom_2[0],
                                         indexes_bloom_3[0]))
        cols_bloom_ctr = np.concatenate((indexes_bloom_1[1], indexes_bloom_2[1],
                                         indexes_bloom_3[1]))

        indexes_bloom = (rows_bloom_ctr, cols_bloom_ctr)

        # Set pixels to ignore as "visible"
        # Start with bad pixels.
        self.vis = self.non_funct.copy()
        # self.vis_center = self.vis.copy()

        if self.verbose:
            print(f"{self.name}: Checking {len(indexes_bloom[0])} pixels for bloom.")

        # Iterate through each candidate (wp = which pixel)
        for wp in range(len(indexes_bloom[0])):
            self.vis_center = self.vis.copy()

            # FIND CENTER FIRST

            # START with r, c.
            r = indexes_bloom[0][wp]
            c = indexes_bloom[1][wp]

            # If already considered, skip this pixel.
            if self.vis[r][c] == True: # noqa
                continue

            # Now that we checked it, mark it true so we don't check it again.
            self.vis[r, c] = True

            # Start with a row / col.
            # make an array out of it.
            self.bloom_rows, self.bloom_cols = np.array([r], dtype=int), np.array([c], dtype=int)

            # copy this array to feed into spread_search_multi
            self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

            # Restart Considered Array (especially if previously used) so that each
            # center starts from scratch
            self.considered[:, :] = self.non_funct[:, :]

            # Then run the following code until it returns len(nn_rows) == 0
            # This should then give you the rows and columns present
            # as part of that blooming incident.

            # Track number of times we loop this code to find NNs:
            rounds = 0

            # FIND center bloom flags only.  Find center and force circle.
            while (len(self.nn_rows) > 0):
                # Increment number of rounds
                rounds += 1
                # Find NNs (first time of r, c, other times of NNs)
                # This updates nn_rows, nn_cols and flag_frac.
                self._spread_search_multi(self.diag_dirs, self.vis_center, center_bloom_flags)

                # Add NNs to collection of indexes representing this event.
                self._update_bloom_pix()
                # When no more NNs exist, this will cause nn_rows = []

            # FIND CENTER of grouping
            center_row = np.average(self.bloom_rows)
            center_col = np.average(self.bloom_cols)

            # Start with center grouping
            # FOR CENTER OVERFLOW ONLY -
            # this may happen at the edge of a bloom or anywhere within it.
            # Setting the center based upon the center of the CENTER may not be wise.

            self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

            # Set center detected pixels as 'visible'
            self.vis[(self.bloom_rows, self.bloom_cols)] = True

            # FIND TRANSIENT RING
            # AUGMENT circle around center until late-transient is detected
            # First, add the center to the list of pixels already considered.
            self.considered[(self.bloom_rows, self.bloom_cols)] = True
            # Use this to grow the edge of the bloom.
            edge_rows, edge_cols = np.array([]), np.array([])
            current_r = 0
            rounds = 0

            # FIND all of the pixels in this blooming instance.
            while ((len(self.nn_rows) > 0) & (rounds <= self.cfg.max_bloom_loops)):
                # Increment number of rounds
                rounds += 1

                # Find NNs (first time of r, c, other times of NNs)
                # updates nn_rows, nn_cols, and frac_flagged along with considered.
                # Using non funct so that it returns all functional pixels so we can
                # examine them in the next step.
                self._spread_search_multi(self.diag_dirs, self.considered, ~self.non_funct)

                # GET EDGE PIXELS

                # Are there pixels indicating we have reached the edge of the bloom?
                # If NOT blooming flags (1,5,2) or late_lg_trans (3).
                indices = np.where(~(self.over[(self.nn_rows, self.nn_cols)]
                                     | self.transient[(self.nn_rows, self.nn_cols)]))
                # What do I REALLY want?  Edge of bloom is indicated by
                # NOT 2 and NOT 1.  Not transient and not overflow.  However,
                # late transient can be indicative of edge.
                # If so, add to list so we can truncate.
                if (len(indices[0]) > 0):
                    # Keep track of edge rows and cols
                    (edge_rows, edge_cols) = (np.concatenate((edge_rows, self.nn_rows[indices])),
                                              np.concatenate((edge_cols, self.nn_cols[indices])))

                    # Remove pixels not part of the bloom for the next iteration
                    self.nn_rows, self.nn_cols = \
                        np.delete(self.nn_rows, indices), np.delete(self.nn_cols, indices)

                # Once we have edge rows, define radius to them
                if ((len(edge_rows) > 0) & (current_r == 0)):
                    current_r = (np.sqrt(np.min((edge_rows-center_row)**2 + (edge_cols-center_col)**2))
                                 + self.extend_radius)

                    indices = np.where(np.sqrt((self.nn_rows-center_row)**2
                                               + (self.nn_cols-center_col)**2) > current_r)

                    self.nn_rows, self.nn_cols = \
                        np.delete(self.nn_rows, indices), np.delete(self.nn_cols, indices)

                # Once we have radius, remove pixels out of radius from nns.
                if current_r > 0:
                    indices = np.where(np.sqrt((self.nn_rows-center_row)**2
                                               + (self.nn_cols-center_col)**2) > current_r)

                    self.nn_rows, self.nn_cols = \
                        np.delete(self.nn_rows, indices), np.delete(self.nn_cols, indices)

                # add pixels to blooming
                self._update_bloom_pix()

            # Even though we did this as we went, in case any got through:
            # delete pixels out of radius and save the rest.
            if current_r == 0:
                if self.verbose:
                    print(f"{self.name}: R{center_row:.1f}, C{center_col:.1f}: "
                          "Encountered Loop Termination, discarding pixels with low signal")

                # If this condition is reached, define edge of bloom as first
                # pixel that does not meet minimum signal requirement

                current_pix_notover = self.small[self.bloom_rows, self.bloom_cols]
                current_pix_smallsig = (self.fluence[self.bloom_rows, self.bloom_cols] < self.min_overflow/5)
                indices = np.where(((current_pix_notover) & (current_pix_smallsig)))

                if (len(indices[0]) > 0):
                    # Pixels with low signal found.
                    # Redefine r and get indices to discard.
                    edge_rows = self.bloom_rows[indices]
                    edge_cols = self.bloom_cols[indices]

                    current_r = (np.sqrt(np.min((edge_rows-center_row)**2 + (edge_cols-center_col)**2))
                                 + self.extend_radius)
                    indices = np.where(np.sqrt((self.bloom_rows-center_row)**2
                                               + (self.bloom_cols-center_col)**2) > current_r)

                else:
                    # No pixels with low signal found.
                    # Keep all pixels. (indices has length 0.)
                    if self.verbose:
                        print(f"{self.name}: R{center_row:.1f}, C{center_col:.1f}: "
                              "No low signal pixels found.  Keeping all pixels.")

            else:
                # Find pixels out of radius
                indices = np.where(np.sqrt((self.bloom_rows-center_row)**2
                                           + (self.bloom_cols-center_col)**2) > current_r)

            # Delete selected pixels.
            self.bloom_rows, self.bloom_cols = \
                np.delete(self.bloom_rows, indices), np.delete(self.bloom_cols, indices)

            self.vis[(self.bloom_rows, self.bloom_cols)] = True

            # NEW! FROM HERE??? Get ring followed by ring to verify bloom...?
            self.nn_rows = self.bloom_rows.copy()
            self.nn_cols = self.bloom_cols.copy()

            self.considered = self.non_funct.copy()
            # (self.non_funct | self.transient | self.over)
            self.considered[(self.bloom_rows, self.bloom_cols)] = True

            # Start here.  Set defaults.
            decreasing_border = False
            # If not enough information (particularly with high trans),
            # treat it as a non-decreasing border.
            med_border_1 = 0
            med_border_2 = 0
            save_border_diff = True
            border_diff = 0

            if len(self.nn_rows) == 0:
                if self.verbose:
                    print(f"{self.name}: R{center_row:.1f}, C{center_col:.1f}: "
                          "How did we get nn_rows = 0?")
                    print('ROWS: ', self.nn_rows)
                    print('COLS: ', self.nn_cols)
                continue

            # Get outside border
            self._spread_search_multi(self.diag_dirs, self.considered, ~self.non_funct)
            indices = np.where((~self.early[(self.nn_rows, self.nn_cols)])
                               & (~self.over[(self.nn_rows, self.nn_cols)]))

            # Pixel boarder handling
            if len(self.nn_rows[indices]) > 2:
                # Take median of outside border
                # ignoring bad pixels and transient / over pixels.
                med_border_1 = np.ma.median(self.fluence[(self.nn_rows[indices],
                                                          self.nn_cols[indices])]) / self.itime

                # Next round
                self._spread_search_multi(self.diag_dirs, self.considered, ~self.non_funct)
                indices = np.where((~self.early[(self.nn_rows, self.nn_cols)])
                                   & (~self.over[(self.nn_rows, self.nn_cols)]))

                if len(self.nn_rows[indices]) > 2:
                    # Take median of next outer border
                    # ignoring bad pixels and transient / over pixels.
                    med_border_2 = np.ma.median(self.fluence[(self.nn_rows[indices],
                                                              self.nn_cols[indices])]) / self.itime
                    border_diff = med_border_1 - med_border_2
                    if border_diff > self.cfg.border_diff_min:
                        decreasing_border = True

            if med_border_2 == 0:
                save_border_diff = False

            # Then once we have found all pixels in collection, flag accordingly.
            # Count number of pixels with flagging condition
            # so can set bloom_flag appropriately.
            count5 = np.count_nonzero(early_transient[self.bloom_rows, self.bloom_cols])
            count1 = np.count_nonzero(late_transient[self.bloom_rows, self.bloom_cols])
            count2 = np.count_nonzero(overflow_only[self.bloom_rows, self.bloom_cols])
            count7 = np.count_nonzero(early_large_trans[self.bloom_rows, self.bloom_cols])
            count6 = np.count_nonzero(early_large[self.bloom_rows, self.bloom_cols])

            # CONDITIONS for possible bloom consideration.
            set_bloom_flag = 0

            # cutout_len = 100
            if ((count5 == 0) & (count1 == 0)):
                # No bloom at all.
                None
            elif ((count5 > 0) & (count1 > 0) & (self.trans_frac < self.cfg.max_trans_frac)):
                if save_border_diff:
                    self.border_diff_info = np.vstack((self.border_diff_info,
                                                       np.array([center_row, center_col, 6, border_diff])))

                if ((count6 > 0) & (count7 == 0) & (count2 == 0)):
                    if decreasing_border:
                        set_bloom_flag = 4
                    # else: #(defer to snowball routine since conditions differ)

                # Typical bloom.
                elif ((count6 > 0) | (count7 > 0) | (count2 > 0)):  # any of these with multiple timing
                    if decreasing_border:
                        set_bloom_flag = 1
                    else:
                        set_bloom_flag = 7

            elif (count5 > 0):
                if save_border_diff:
                    self.border_diff_info = np.vstack((self.border_diff_info,
                                                       np.array([center_row, center_col, 5, border_diff])))

                # only early transients
                # or both with transient fraction too high -
                # in this case consider only early trans
                if ((count6 > 0) & (count7 == 0) & (count2 == 0)):
                    # only 5 & 6.  Call snowballs if signal doesn't drop as expected.
                    if decreasing_border:
                        set_bloom_flag = 4
                    # else: #defer to snowball routine
                elif ((count2 > 0) & (count6 == 0) & (count7 == 0)):
                    # only 5 & 2.  Snowball?  But 2 has signal...
                    set_bloom_flag = 3
                elif ((count2 == 0) & (count6 == 0) & (count7 > 0)):
                    # only 5 & 7.  Snowball?  But 7 has signal...
                    set_bloom_flag = 3
                elif ((count2 > 0) | (count6 > 0) | (count7 > 0)):
                    # 2 & 6, 6 & 7, 2 & 7, or 2 & 6 & 7.
                    set_bloom_flag = 2

            elif (count1 > 0):
                if save_border_diff:
                    self.border_diff_info = np.vstack((self.border_diff_info,
                                                       np.array([center_row, center_col, 1, border_diff])))

                # In all these cases we have
                # late transients with early trans or overflow
                # likely bloom
                set_bloom_flag = 6

            # Ignore other conditions - move to snowball flagging.
            if set_bloom_flag > 0:
                # Increment total of blooming occurrences.
                self.count_bloom += 1
                # Add the number of NN rounds to find all these pixels to the list.
                # Set blooming flag number for later diagnostics.
                self.bloom_flag[(self.bloom_rows, self.bloom_cols)] = set_bloom_flag

            # This version of the code moves snowballs to the snowball flagging portion
            # due to differing conditions (such as forcing circles).

        # PURGE looped quantities that will not be needed for the next part:
        # These will likely already be [] due to conditions above.
        self.nn_cols = []
        self.nn_rows = []

        if self.verbose:
            print(f"{self.name}: Found {self.count_bloom} blooming occurrences.")
            print(f"{self.name}: Flagged {np.count_nonzero(self.bloom_flag)} pixels for bloom.")
            print(f"{self.name}: Found {self.count_snowball} snowball or high energy CR occurrences.")

        self.header['NBLMSRCS'] = (self.count_bloom, 'Number of Bloomed Srcs')
        self.header['NBLMPIX'] = (np.count_nonzero(self.bloom_flag), 'Number of Pixels Bloom Flag')
        self.header['NSNOWSRC'] = (self.count_snowball, 'Number of Snowball Srcs')
        self.header['NBLM5SCS'] = (self.count_5bloom, 'Number of 5-Bloomed Srcs')

        self.bloom_flag[self.non_funct] = 0
        # n_bloom only used for diagnostics
        # print('Saturating persistence current should be used
        # for bloomed pixels with flags 2, 5, 6, 7.')

    def get_snowball_flag(self):
        '''
        # Only do this after bloom_flag_layer is complete
        # START from initial snowball_flag layer produced by blooming.
        # Read in last snowball count last_cnt from bloom flag detection algorithm.

        # NOTE: This code sets different values for the different conditions of
        # snowball detection.
        # This is used in evaluating accuracy of detection.
        # For IPAC use, set SNOWBALL flag for all identified snowballs / CRs,
        # snowball_flag code 2 in this function.
        # Set TFLAG_HALO for the halo, snowball_flag code 3 in this function.
        '''

        if self.verbose:
            print(f"{self.name}: Searching for snowballs...")

        # START HERE AND DO THIS THRICE?
        for snow_type in range(4):
            # Define pixels to ignore as "visible".
            # REDEFINE visible each time?  Yes.

            if snow_type == 0:  # EARLY SNOWBALLS / CRs
                self.vis = ((self.bloom_flag > 0) | self.non_funct | (self.snowball_flag > 0))
                snowball_candidates = ((self.transient & ~self.vis & self.early))  # outer ring
                # center high energy group
                center_candidates = ((self.transient & ~self.vis & self.early & self.over))
                center_val = 7
                bloom_val = 5
            # LATE SNOWBALLS / CRs # Bloom can have 3, but not traditionally.
            elif snow_type == 1:  # (ignore bloom)
                self.vis = (self.non_funct | (self.snowball_flag > 0) | (self.bloom_flag > 0))
                snowball_candidates = ((self.transient & ~self.vis & ~self.early))  # outer ring
                # center high energy group
                center_candidates = ((self.transient & ~self.vis & self.over & ~self.early))
                center_val = 3
                bloom_val = 1
            elif snow_type == 2:  # 2 -> 5 w/o signal in 2. (VERY early snowballs / CRs)
                self.vis = ((self.bloom_flag > 0) | self.non_funct | (self.snowball_flag > 0))
                # typically only 1 pixel deep...
                snowball_candidates = ((self.transient & ~self.vis & self.early))
                center_candidates = ((~self.transient & ~self.vis & self.over
                                      & ~self.early & (self.fluence < self.min_overflow)))
                center_val = 2  # low signal
                bloom_val = 5
            else:  # 6 -> 5 w/o signal bloom. (VERY early snowballs / CRs)
                self.vis = ((self.bloom_flag > 0) | self.non_funct | (self.snowball_flag > 0))
                # typically only 1 pixel deep...
                snowball_candidates = ((self.transient & ~self.vis & self.early))
                center_candidates = ((~self.transient & ~self.vis & self.over & self.early))
                center_val = 6
                bloom_val = 5

            # Where are the center candidates?  Start with these.
            indexes_bloom = np.where(center_candidates)

            # Last iteration of snow_type, what was the total?
            last_cnt = copy.copy(self.count_snowball)

            # Iterate through each candidate (wp = which pixel)
            for wp in range(len(indexes_bloom[0])):

                r = indexes_bloom[0][wp]
                c = indexes_bloom[1][wp]

                # If already considered, skip this pixel.
                if self.vis[r][c] == True: # noqa
                    continue

                # Now that we checked it, mark it true so we don't check it again.
                self.vis[r, c] = True
                self.vis_center = self.vis.copy()

                # Start with a row / col.
                self.bloom_rows, self.bloom_cols = np.array([r], dtype=int), np.array([c], dtype=int)
                self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

                # Then run the following code until it returns len(nn_rows) == 0
                # This should then give you the rows and columns
                # present as part of that blooming incident.

                # Use only center candidates at first.  With these use diagonal dirs.
                while len(self.nn_rows) > 0:
                    # Find nearest neighbors
                    self._spread_search_multi(self.diag_dirs, self.vis_center, center_candidates)

                    # Add nearest neighbors to list
                    self._update_bloom_pix()

                    # Quit when there are no more nearest neighbors
                    # We use a copy of vis so that when looking for
                    # center candidates we do not rule out snowball
                    # candidates for next round.

                self.vis[self.bloom_rows, self.bloom_cols] = True

                # Consider snowball on whole center.
                self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

                # So each check for pixels surrounding center can be independent.
                self.considered[:, :] = self.non_funct[:, :]
                self.considered[self.nn_rows, self.nn_cols] = True

                # Find center position of snowball.
                center_row = np.average(self.bloom_rows)
                center_col = np.average(self.bloom_cols)

                # Now consider snowball candidates.
                # If encounter
                last_loop = False
                # Only update n_loops after we get OUT of the center...
                n_loops = 0
                current_r = 0

                while len(self.nn_rows) > 0:
                    # Find nearest neighbors of central snowball.
                    # If HIGH TRANSIENT FRACTION, force circle to restrict growth.
                    # If LOW transient fraction, allow whatever shape to occur.
                    # Cap at number of loops specified in class.

                    if self.force_snow_circle:

                        if current_r == 0:

                            # Use diagonal dirs until an edge is found.
                            # Do not restrict which pixels return
                            # as nns until edge located.
                            self._spread_search_multi(self.diag_dirs, self.considered, ~self.non_funct)

                            indices = np.where(~snowball_candidates[self.nn_rows, self.nn_cols])

                            if len(indices[0]) > 0:

                                # Edge found!
                                edge_rows = self.nn_rows[indices]
                                edge_cols = self.nn_cols[indices]

                                # Remove non-snowball edge pixels from snowball.
                                self.nn_rows = np.delete(self.nn_rows, indices)
                                self.nn_cols = np.delete(self.nn_cols, indices)

                                # Find radius
                                current_r = np.sqrt(np.min((edge_rows-center_row)**2
                                                           + (edge_cols-center_col)**2)) + self.extend_radius

                                # Delete pixels out of radius
                                indices = np.where(np.sqrt((self.nn_rows-center_row)**2
                                                           + (self.nn_cols-center_col)**2) > current_r)

                                self.nn_rows = np.delete(self.nn_rows, indices)
                                self.nn_cols = np.delete(self.nn_cols, indices)

                            # Add valid snowball pixels to list.
                            self._update_bloom_pix()

                            n_loops += 1

                        elif current_r > 0:

                            # Use EDGE dirs once edge is found.
                            self._spread_search_multi(self.edge_dirs, self.considered, snowball_candidates)

                            # Delete pixels out of radius
                            if len(self.nn_rows) > 0:
                                indices = np.where(np.sqrt((self.nn_rows-center_row)**2
                                                           + (self.nn_cols-center_col)**2) > current_r)

                                self.nn_rows = np.delete(self.nn_rows, indices)
                                self.nn_cols = np.delete(self.nn_cols, indices)

                            # Add valid snowball pixels to list.
                            self._update_bloom_pix()

                            # If we have been truncated,
                            # stop here - only if we have found our
                            # radius.
                            if last_loop:
                                self.nn_rows = []
                                self.nn_cols = []

                            n_loops += 1

                            # How to deal with high transient fraction
                            # (and runaway loops)?
                            if n_loops > self.max_snow_loops:
                                last_loop = True

                    else:  # USE EXPANDING BORDER

                        self._spread_search_multi(self.diag_dirs, self.considered, snowball_candidates)

                        # If we found any new nns
                        if len(self.nn_rows) > 0:

                            # Add nearest neighbors to list
                            self._update_bloom_pix()

                            # Put all pix back into
                            # nn_rows / nn_cols for expanding border
                            self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

                        # Quit when there are no more nearest neighbors
                        # or if frac_flagged gets too small

                        # If we have been truncated, stop here.
                        if last_loop:
                            self.nn_rows = []
                            self.nn_cols = []

                        # If we are catching less than 50%
                        # in our next border expansion,
                        # do only one more loop.
                        if self.flag_frac <= 0.5:
                            last_loop = True

                        n_loops += 1

                        # How to deal with high transient fraction (and runaway loops)?
                        if n_loops > self.max_snow_loops:
                            last_loop = True

                # Add all detected pixels to vis.
                self.vis[self.bloom_rows, self.bloom_cols] = True

                # Then determine what kind of collection we have
                n_inner = np.count_nonzero(center_candidates[self.bloom_rows, self.bloom_cols])
                n_outer = np.count_nonzero(snowball_candidates[self.bloom_rows, self.bloom_cols]) - n_inner

                # n_inner:
                # # Then most likely CR or snowball.
                # (if we have any neighboring transients)
                if n_outer > 0:

                    # Increment snowball counter.
                    self.count_snowball += 1

                    # Set snowball flag for CRs
                    self.snowball_flag[self.bloom_rows, self.bloom_cols] = snow_type + 1

                    # Apply halo flag to snowball.
                    self._apply_halo_flag()

            if self.verbose:
                print(f"{self.name}: Found {self.count_snowball-last_cnt} "
                      f"snowball occurrences with center flag {center_val} "
                      f"and bloom flag {bloom_val}.")

        if self.verbose:
            print(f"{self.name}: Flagged a total of "
                  f"{np.count_nonzero(self.snowball_flag)} pixels for snowball.")

        # Total snowball pixels
        self.header['NSNOWPIX'] = (np.count_nonzero(self.snowball_flag), 'Number of Pixels Snowball Flag')
        self.snowball_flag[self.non_funct] = 0

        # Snowball timing subtype counts
        # Mapping from snow_type to snowball_flag value:
        #   snow_type = 0 -> snowball_flag == 1  (center_val=7, bloom_val=5)
        #   snow_type = 1 -> snowball_flag == 2  (center_val=3, bloom_val=1)
        #   snow_type = 2 -> snowball_flag == 3  (2 -> 5, very early)
        #   snow_type = 3 -> snowball_flag == 4  (6 -> 5, very early)
        n_snw_7_5 = int(np.count_nonzero(self.snowball_flag == 1))
        n_snw_3_1 = int(np.count_nonzero(self.snowball_flag == 2))
        n_snw_2_5 = int(np.count_nonzero(self.snowball_flag == 3))
        n_snw_6_5 = int(np.count_nonzero(self.snowball_flag == 4))

        self.header['SNW7-5CT'] = (n_snw_7_5, 'Early snowball count (first half of exposure)')
        self.header['SNW3-1CT'] = (n_snw_3_1, 'Late snowball count (second half of exposure)')
        self.header['SNW2-5CT'] = (n_snw_2_5, 'Very early snowball count (frames ~5-7)')
        self.header['SNW6-5CT'] = (n_snw_6_5, 'Very early snowball count (before frame ~10)')

    def _apply_halo_flag(self):
        '''
        Apply halo flag to known bloom rows / bloom cols.
        If suspicious that it is following a satellite track, apply halo appropriately.
        Satellite tracks with large numbers of pixels and linear features are
        included in snowball layer as they cause overflow and transient behavior
        in the same way as snowballs.  They also have halos!
        '''

        # FOR THIS VERSION ONLY USE CIRCLE HALO
        halo_num = 2

        # Find centroid center of snowball pixels
        indexes_center = np.where(self.over[self.bloom_rows, self.bloom_cols])
        row_center = np.average(self.bloom_rows[indexes_center])
        col_center = np.average(self.bloom_cols[indexes_center])

        # Get MAX radius in case of streaks
        r_max = np.sqrt(np.max(self.bloom_rows - row_center)**2
                        + np.max(self.bloom_cols - col_center)**2)

        # Set radius by some combination of n_inner and n_outer and possibly band
        # Sample equation:
        # r_ = 1.3*np.sqrt(len(cr_rows))
        # NEW EQUATION: (GUESSED IT!!!)
        r_ = np.sqrt(2*len(self.bloom_rows))

        if r_max > r_:
            # WE have a SITUATION!  Eccentricity.
            # Safely increment SATSTRKS whether header value is int or (int, str)
            _sat = self.header.get('SATSTRKS', 0)
            _sat_count = _sat[0] if isinstance(_sat, tuple) else int(_sat)
            self.header['SATSTRKS'] = (_sat_count + 1, 'Satellite Streaks Detected')

            # FOR NOW choose in function call.
            if self.cfg.satellite_halomethod == "border_expand":
                self._grow_halo_border(r_max)
            elif self.cfg.satellite_halomethod == "dilation":
                self._enlarge_curve_by_dilation()
            elif self.cfg.satellite_halomethod == "smooth_skeleton":
                self._enlarge_curve_by_smoothed_skeleton()

            # Clamp halo_rows/cols to active + functional region
            valid = ~self.non_funct
            mask_valid = valid[(self.halo_rows, self.halo_cols)]
            self.halo_rows = self.halo_rows[mask_valid]
            self.halo_cols = self.halo_cols[mask_valid]

            self.halo_flag[self.halo_rows, self.halo_cols] += halo_num

        else:
            (n_rows, n_cols) = self.fluence.shape
            # Determine the range of rows / columns this will affect
            min_row = np.max((int(np.ceil(row_center-r_)), 0))
            max_row = np.min((int(np.ceil(row_center+r_))+1, n_rows))
            min_col = np.max((int(np.ceil(col_center-r_)), 0))
            max_col = np.min((int(np.ceil(col_center+r_))+1, n_cols))

            # Generate array of radii the size of the current analysis,
            # do not extend beyond active region.
            test_arr = np.sqrt(((np.arange(min_col, max_col, 1)
                                 - int(col_center)).reshape(1, (max_col-min_col)))**2
                               + (np.arange(min_row, max_row, 1)
                                  - int(row_center)).reshape((max_row-min_row), 1)**2)

            # Flag pixels with halo within the radius
            indexes = np.where((test_arr <= r_))
            self.halo_flag[min_row:max_row, min_col:max_col][indexes] += halo_num

        self.header['NHALOPIX'] = (np.count_nonzero(self.halo_flag), 'Number of Pixels Halo Flag')
        self.halo_flag[self.non_funct] = 0

    def get_nonoverflow_bloom_flag(self):
        '''
        # Only do this after bloom_flag_layer and snowball_flag_layer is complete
        This code identifies blooming that does not include an overflow flag.
        Thus it is only applied to images with reasonable transient fraction
        as otherwise false detections are common.
        # NOTE: This code sets different values for the different conditions
        of bloom and snowball detection.  This is used in validating accuracy.
        # For IPAC use, set SRC_BLOOM flag for all bloom detections
        '''
        # Do not run this if the transient fraction is too high
        if self.trans_frac >= self.cfg.max_trans_frac:
            if self.verbose:
                print(f"{self.name}: Skipping non-overflow bloom check due to high transient fraction "
                      f"({self.trans_frac:.4f} >= {self.cfg.max_trans_frac:.4f}).")
            return

        if self.verbose:
            print(f"{self.name}: Searching for non-overflow bloom candidates...")

        # Set conditions of excluded pixels
        # (fluence < 2000))
        self.vis = (self.bloom_flag > 0) | (self.snowball_flag > 0) | self.non_funct

        # Define collections of flags
        late_transient = (self.transient & ~self.early & ~self.over)
        early_transient = (self.transient & self.early & ~self.over)
        # non_bloom = (~self.transient & ~self.over)

        # DEFINE blooming candidates
        blooming_candidates = (self.transient & ~self.over & ~self.vis
                               & (self.fluence > self.min_overflow/10))
        center_candidates = (self.transient & ~self.over & self.early
                             & (self.fluence > self.min_overflow/2) & ~self.vis)

        # Determine which pixels to start with
        indexes_bloom = np.where(center_candidates)

        # If there aren't any pixels to start, return.
        # This condition is highly unlikely.
        if len(indexes_bloom[0]) == 0:
            if self.verbose:
                print(f"{self.name}: No candidates for source blooming starting with Flag 5.")
            return

        # sort pixels by fluence
        indices = np.argsort(self.fluence[indexes_bloom])[::-1]

        # Sort pixels by indices
        rows_bloom_sorted = np.array([indexes_bloom[0][i] for i in indices])
        cols_bloom_sorted = np.array([indexes_bloom[1][i] for i in indices])

        indexes_bloom = (rows_bloom_sorted, cols_bloom_sorted)

        # Iterate through each candidate (wp = which pixel)
        for wp in range(len(indexes_bloom[0])):

            r = indexes_bloom[0][wp]
            c = indexes_bloom[1][wp]

            # If already considered, skip this pixel.
            if self.vis[r][c] == True: # noqa
                continue

            # Now that we checked it, mark it true so we don't check it again.
            self.vis[r, c] = True

            # Start with a row / col.
            self.bloom_rows, self.bloom_cols = np.array([r], dtype=int), np.array([c], dtype=int)
            self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

            # Then run the following code until it returns len(nn_rows) == 0
            # This should then give you the rows and columns
            # present as part of that blooming incident.

            # Track number of loops
            # (useful for checking if this is following a transient track)
            n_loops = 0
            current_r = 0

            self.vis_center = self.vis.copy()

            while ((len(self.nn_rows) > 0) & (n_loops <= self.cfg.max_nonoverbloom_loops)):
                # Find nearest neighbors
                # Find CENTER first
                self._spread_search_multi(self.diag_dirs, self.vis_center, center_candidates)

                # Add nearest neighbors to list
                self._update_bloom_pix()

                # If flagging fraction is too low,
                # call it quits. (Helps with satellite tracks.)
                if self.flag_frac < 0.2:
                    n_loops = self.cfg.max_nonoverbloom_loops

                # Increment the number of loops
                n_loops += 1

            # Center group found
            self.vis[(self.bloom_rows, self.bloom_cols)] = True
            self.nn_rows, self.nn_cols = self.bloom_rows.copy(), self.bloom_cols.copy()

            # find center_row / center_col weighted by fluence
            center_row = np.sum(self.nn_rows*self.fluence[(self.bloom_rows, self.bloom_cols)]) / \
                np.sum(self.fluence[(self.bloom_rows, self.bloom_cols)])
            center_col = np.sum(self.nn_cols*self.fluence[(self.bloom_rows, self.bloom_cols)]) / \
                np.sum(self.fluence[(self.bloom_rows, self.bloom_cols)])

            n_loops = 0

            while ((len(self.nn_rows) > 0) & (n_loops <= self.cfg.max_nonoverbloom_loops)):
                # Find nearest neighbors
                # Find late-transient ring.
                self._spread_search_multi(self.diag_dirs, self.vis, blooming_candidates)

                # FIND edge of bloom (non_blooms not found with above definition)
                indices = np.where((late_transient[self.nn_rows, self.nn_cols]))

                # IF edge found for the first time, find center and radius to edge.
                if ((len(indices[0]) > 0) & (current_r == 0)):

                    # Define radius to edge as min radius to non-bloom / late transient
                    # Add 1 to include late-transient
                    current_r = \
                        (np.sqrt(np.min((self.nn_rows[indices]-center_row)**2
                                        + (self.nn_cols[indices]-center_col)**2)) + self.extend_radius)
                    # Keep all pixels in this round in the bloom pix.

                # IF edge already found, remove any pixels past the defined radius
                if ((len(indices[0]) > 0) & (current_r > 0)):
                    indices = np.where(np.sqrt((self.nn_rows-center_row)**2
                                               + (self.nn_cols-center_col)**2) > current_r)
                    self.nn_rows = np.delete(self.nn_rows, indices)
                    self.nn_cols = np.delete(self.nn_cols, indices)

                # Add nearest neighbors to list
                self._update_bloom_pix()

                # Increment the number of loops
                n_loops += 1
                # Quit when there are no more nearest neighbors
                # or we have wandered too far.

            # Identify if collection contains both timing of transients
            count5 = np.count_nonzero(early_transient[self.bloom_rows, self.bloom_cols])
            count1 = np.count_nonzero(late_transient[self.bloom_rows, self.bloom_cols])

            # Assign flag if both present
            if ((count5 > 0) & (count1 > 0)):
                self.count_5bloom += 1
                self.bloom_flag[(self.bloom_rows, self.bloom_cols)] = 5

        n_flagged = np.count_nonzero(self.bloom_flag)
        if self.verbose:
            print(f"{self.name}: Found {self.count_5bloom} "
                  "more blooming occurrences.")
            print(f"{self.name}: Flagged a total of {n_flagged} pixels, "
                  f"{n_flagged/(2040*20.4):.2f}% for bloom.")

        self.bloom_flag[self.non_funct] = 0

    def mark_transient_halo(self):
        '''
        # AFTER SNOWBALL and BLOOM layers are completed, find remaining good-pixel
        # transients and flag their nearest neighbors for halo.
        '''
        valid = ~self.non_funct

        transients = (
            self.transient
            & valid
            & (self.snowball_flag == 0)
            & (self.bloom_flag == 0)
        )

        nn_trans = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])

        halo_core = convolve(transients, nn_trans, mode='reflect') > 0

        # Enforce active / functional region only
        halo_core &= valid

        # Use OR with uint8 to avoid dtype upcast/casting errors
        self.halo_flag |= halo_core.astype(np.uint8)

        n_flagged = np.count_nonzero(self.halo_flag)
        if self.verbose:
            print(f"{self.name}: Flagged a total of {n_flagged} pixels, "
                  f"{n_flagged/(2040*20.4):.2f}% for halo.")


def apply_bsh_flags_to_spx(spx: SPHERExImage, bsh: BSHFlagger) -> None:
    """Project BSH masks into spx.flags bits.

    NOTES
    -----
    Goal of this is to preserve the fine flag definitions in bsh_flagger
    while assingning coarse spherex flags to the image.
    """

    # Bloom: any nonzero bloom_flag set BLOOM bit
    bloom_mask = (bsh.bloom_flag > 0)
    if np.any(bloom_mask):
        spx.setFlagsFromMask(bloom_mask, Flag.BLOOM)

    # Snowball: any nonzero snowball_flag set SNOWBALL bit
    snow_mask = (bsh.snowball_flag > 0)
    if np.any(snow_mask):
        spx.setFlagsFromMask(snow_mask, Flag.SNOWBALL)

    # Halo (generic): any nonzero halo_flag set HALO bit
    halo_mask = (bsh.halo_flag > 0)
    if np.any(halo_mask):
        spx.setFlagsFromMask(halo_mask, Flag.HALO)

    # Satellite halo
    if (
        hasattr(bsh, "halo_rows")
        and hasattr(bsh, "halo_cols")
        and len(getattr(bsh, "halo_rows", [])) > 0
    ):
        sat_streaks = (bsh.header.get("SATSTRKS", (0,))[0]
                       if isinstance(bsh.header.get("SATSTRKS"), tuple)
                       else bsh.header.get("SATSTRKS", 0))
        if sat_streaks and sat_streaks > 0:
            # Create a boolean mask for satellite halos
            sat_halo_mask = np.zeros(spx.data.shape, dtype=bool)
            sat_halo_mask[(bsh.halo_rows, bsh.halo_cols)] = True
            # Extra guard: restrict to active + functional region
            if hasattr(bsh, "non_funct"):
                sat_halo_mask[bsh.non_funct] = False
            spx.setFlagsFromMask(sat_halo_mask, Flag.SATELLITE_HALO)
