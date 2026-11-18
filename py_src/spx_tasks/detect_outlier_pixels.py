__all__ = ["DetectOutlierPixelsConfig", "DetectOutlierPixelsTask"]

# Standard library imports
import numpy as np

# Related third party imports
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from scipy.ndimage import median_filter, label, binary_dilation
from scipy.ndimage import find_objects

# Local application/library specific imports
from spx_data import SPHERExImage, Flag, ensure_uncertainty_parentdata
from spx_tasks import add_history
from spx_utils import log_dqa_result
# ==============================================================================


def dqa_check_outlier_pixel_count(spximg: SPHERExImage, tolerance: float = 2.0) -> dict:
    """Check that the number of OUTLIER flagged pixels is within a factor
    of `tolerance` relative to the number of SOURCE, HOT, and COLD pixels.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Processed image containing flag plane.

    tolerance : `float`, optional
        Tolerance factor (default is 2.0).

    Returns
    -------
    result : `dict`
        Dictionary with "result" and "description" for Level2DQA.

    """
    num_outliers = np.sum((spximg.flags & Flag.OUTLIER) > 0)
    num_source = np.sum((spximg.flags & Flag.SOURCE) > 0)
    num_hot = np.sum((spximg.flags & Flag.HOT) > 0)
    num_cold = np.sum((spximg.flags & Flag.COLD) > 0)
    total_bad = num_source + num_hot + num_cold

    if total_bad == 0:
        result = {
            "result": "FAIL",
            "description": "No SOURCE, HOT, or COLD pixels found to validate against."
        }

    upper = total_bad * tolerance

    if num_outliers <= upper:
        result = {
            "result": "PASS",
            "description": f"OUTLIER pixel count ({num_outliers}) is within "
                           f"a factor {tolerance} of SOURCE+HOT+COLD pixels ({total_bad})."
        }
    else:
        result = {
            "result": "FAIL",
            "description": f"OUTLIER pixel count ({num_outliers}) is outside "
                           f"factor {tolerance} of SOURCE+HOT+COLD pixels ({total_bad})."
        }
    log_dqa_result("outliers dqa", result)
    return result


class DetectOutlierPixelsConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure"),
                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after detected outlier pixels.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class DetectOutlierPixelsConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=DetectOutlierPixelsConnections):
    """Configuration for DetectOutlierPixels task."""

    n_channels = Field(
        doc="Number of channels a H2RG detector is divided into.",
        dtype=int,
        default=32
    )

    n_sigma = Field(
        "Variance threshold (n * std) for isolated outlier pixels.",
        dtype=float,
        default=2.
    )

    kernel = Field(
        "Median-filter kernel size.",
        dtype=int,
        default=5
    )

    dqa_tolerance = Field(
        doc="Tolerance for number of flagged OUTLIER pixels.",
        dtype=float,
        default=2.0,
    )

    enable_streak_detection = Field(
        doc="Enable connected-component streak detection.",
        dtype=bool,
        default=True,
    )

    streak_n_sigma = Field(
        doc="Significance threshold for streak seed pixels.",
        dtype=float,
        default=1.2,
    )

    streak_min_pixels = Field(
        doc="Minimum number of connected seed pixels in a streak candidate.",
        dtype=int,
        default=12,
    )

    streak_min_length = Field(
        doc="Minimum projected length [pixels] for a streak candidate.",
        dtype=float,
        default=20.0,
    )

    streak_max_width = Field(
        doc="Maximum robust half-width [pixels] for a streak candidate.",
        dtype=float,
        default=4.0,
    )

    streak_min_aspect = Field(
        doc="Minimum length/width ratio for a streak candidate.",
        dtype=float,
        default=4.0,
    )

    streak_grow_radius = Field(
        doc="Optional binary dilation radius in pixels for accepted streak mask.",
        dtype=int,
        default=1,
    )

    streak_use_transient = Field(
        doc="Allow TRANSIENT-flagged pixels to seed streak detection.",
        dtype=bool,
        default=True,
    )
    streak_min_transient_fraction = Field(
        doc="Minimum fraction of component pixels overlapping TRANSIENT when enabled.",
        dtype=float,
        default=0.2,
    )

    streak_fragment_min_pixels = Field(
        doc="Minimum number of pixels for initial streak fragments.",
        dtype=int,
        default=10,
    )

    streak_fragment_min_length = Field(
        doc="Minimum projected length [pixels] for initial streak fragments.",
        dtype=float,
        default=12.0,
    )

    streak_fragment_max_width = Field(
        doc="Maximum width [pixels] for initial streak fragments.",
        dtype=float,
        default=3.5,
    )

    streak_fragment_min_aspect = Field(
        doc="Minimum aspect ratio for initial streak fragments.",
        dtype=float,
        default=8.0,
    )

    streak_merge_angle_diff = Field(
        doc="Maximum angle difference [deg] for merging streak fragments.",
        dtype=float,
        default=5.0,
    )

    streak_merge_perp_offset = Field(
        doc="Maximum perpendicular offset [pixels] for merging streak fragments.",
        dtype=float,
        default=3.5,
    )

    streak_merge_parallel_gap = Field(
        doc="Maximum gap [pixels] along the streak direction for merging fragments.",
        dtype=float,
        default=80.0,
    )

    streak_dense_width_scale = Field(
        doc="Scale factor multiplying fitted width when rasterizing dense streak masks.",
        dtype=float,
        default=1.5,
    )

    streak_dense_min_half_width = Field(
        doc="Minimum half-width [pixels] for dense streak mask rasterization.",
        dtype=float,
        default=1.,
    )

    streak_dense_min_snr = Field(
        doc="Minimum SNR for pixels included in dense streak mask.",
        dtype=float,
        default=0.5,
    )

    streak_merged_min_length = Field(
        doc="Minimum length [pixels] for merged streak groups to be kept.",
        dtype=float,
        default=60.0,
    )

    streak_dense_use_transient = Field(
        doc="Allow TRANSIENT-flagged pixels to be included in the final dense streak mask.",
        dtype=bool,
        default=False,
    )

    streak_require_transient_fraction = Field(
        doc="Require initial fragments to overlap TRANSIENT when streak_use_transient is enabled.",
        dtype=bool,
        default=False,
    )

class DetectOutlierPixelsTask(pipeBase.PipelineTask):
    """Implementation of L2 Detect Outlier Pixels.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-020

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "detect-outlier-pixels"
    ConfigClass = DetectOutlierPixelsConfig
    full_name = "Detect Outlier Pixels"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-020 v1.4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _bridge_source_holes_along_track(
        self,
        track: dict,
        observed_mask: np.ndarray,
        source_mask: np.ndarray,
        hard_exclude: np.ndarray,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
    ) -> np.ndarray:
        """Fill short gaps along a fitted track
        when the gap is caused by SOURCE pixels.
        """
        center = track["center"]
        direction = track["direction"]
        normal = track["normal"]

        rel_x = xgrid - center[0]
        rel_y = ygrid - center[1]

        t = rel_x * direction[0] + rel_y * direction[1]
        d = rel_x * normal[0] + rel_y * normal[1]

        half_width = max(
            self.config.streak_dense_min_half_width,
            self.config.streak_dense_width_scale * track["width"],
        )

        line_corridor = (
            (t >= track["tmin"])
            & (t <= track["tmax"])
            & (np.abs(d) <= half_width)
            & ~hard_exclude
        )

        observed_on_line = line_corridor & observed_mask

        if not np.any(observed_on_line):
            return observed_mask

        t_obs = t[observed_on_line]
        t_lo = np.min(t_obs)
        t_hi = np.max(t_obs)

        bridge_region = (
            line_corridor
            & source_mask
            & (t >= t_lo)
            & (t <= t_hi)
        )

        return observed_mask | bridge_region

    def _make_good_pixel_mask(self, spximg: SPHERExImage) -> np.ndarray:
        """Mask of pixels allowed to participate in outlier / streak detection."""
        flags = spximg.flags
        good = np.isfinite(spximg.data)

        # Exclude source/bad pixels from both isolated outlier and streak seeding.
        exclude = (
            Flag.SOURCE
            | Flag.HOT
            | Flag.COLD
            | Flag.MISSING_DATA
            | Flag.REFERENCE
            | Flag.NONFUNC
            | Flag.DICHROIC
        )

        good &= (flags & exclude) == 0
        return good

    def _compute_residual_and_snr(
        self, spximg: SPHERExImage, good_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute median-filter residual image and significance image."""
        ensure_uncertainty_parentdata(spximg)

        data = np.asarray(spximg.data, dtype=float)
        variance = np.asarray(spximg.uncertainty.array, dtype=float)

        variance = np.where(np.isfinite(variance) & (variance > 0), variance, np.nan)
        std_map = np.sqrt(variance)

        # Fill masked/invalid pixels before median filtering
        # so they do not affect the filter.
        # Using median of good pixels is simple and stable.
        if np.any(good_mask):
            fill_value = np.nanmedian(data[good_mask])
        else:
            fill_value = 0.0

        data_filled = np.where(np.isfinite(data), data, fill_value)
        median_filtered = median_filter(data_filled, size=self.config.kernel)
        residual = data_filled - median_filtered

        eps = 1.0e-12
        snr = np.zeros_like(residual, dtype=float)
        valid_snr = good_mask & np.isfinite(std_map) & (std_map > eps)
        snr[valid_snr] = residual[valid_snr] / std_map[valid_snr]

        return residual, snr, std_map

    def _detect_pixel_outliers(
        self,
        snr: np.ndarray,
        good_mask: np.ndarray,
    ) -> np.ndarray:
        """Detect isolated positive outlier pixels."""
        return (snr > self.config.n_sigma) & good_mask

    def _measure_component_linearity(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
    ) -> dict | None:
        """Measure line-like geometry for one connected component."""
        xs = xs.astype(float)
        ys = ys.astype(float)
        npix = len(xs)

        if npix < 2:
            return None

        coords = np.column_stack([xs, ys])
        center = coords.mean(axis=0)
        centered = coords - center

        cov = np.cov(centered.T, bias=True)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        direction = evecs[:, 0]
        normal = np.array([-direction[1], direction[0]])

        # Fix sign convention for more stable angle comparisons
        if direction[0] < 0:
            direction = -direction
            normal = -normal

        t = centered @ direction
        d = centered @ normal

        length = float(t.max() - t.min())

        ad = np.abs(d)
        p84 = np.percentile(ad, 84.1)
        p16 = np.percentile(ad, 15.9)
        width = float(max(0.5 * (p84 - p16), 0.0))

        aspect_ratio = np.inf if width <= 1.0e-6 else length / width
        angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))

        return {
            "npix": npix,
            "center": center,
            "direction": direction,
            "normal": normal,
            "tmin": float(t.min()),
            "tmax": float(t.max()),
            "length": length,
            "width": width,
            "aspect_ratio": aspect_ratio,
            "angle_deg": angle_deg,
            "xs": xs,
            "ys": ys,
        }

    @staticmethod
    def _angle_diff_deg(a: float, b: float) -> float:
        """Smallest absolute angle difference in degrees for unoriented lines."""
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _get_xy_grids(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Generate x and y coordinate grids for the input image shape."""
        ny, nx = shape
        y, x = np.indices((ny, nx))
        return x.astype(float), y.astype(float)


    def _find_initial_line_fragments(self, seed: np.ndarray) -> list[dict]:
        """Find initial connected line-like streak fragments."""
        structure = np.ones((3, 3), dtype=np.int8)
        labeled, ncomp = label(seed, structure=structure)

        if ncomp == 0:
            return []

        counts = np.bincount(labeled.ravel())
        objects = find_objects(labeled)

        fragments = []

        for comp_id, slc in enumerate(objects, start=1):
            if slc is None:
                continue

            npix = int(counts[comp_id])
            if npix < self.config.streak_fragment_min_pixels:
                continue

            sub = labeled[slc] == comp_id
            ys, xs = np.nonzero(sub)

            y0 = slc[0].start
            x0 = slc[1].start
            ys = ys + y0
            xs = xs + x0

            info = self._measure_component_linearity(ys, xs)
            if info is None:
                continue

            if info["length"] < self.config.streak_fragment_min_length:
                continue
            if info["width"] > self.config.streak_fragment_max_width:
                continue
            if info["aspect_ratio"] < self.config.streak_fragment_min_aspect:
                continue

            fragments.append(info)

        return fragments

    def _fragments_are_mergeable(
        self,
        f1: dict,
        f2: dict,
    ) -> bool:
        """Return True when two fragments are consistent with the same streak."""
        if self._angle_diff_deg(f1["angle_deg"], f2["angle_deg"]) > self.config.streak_merge_angle_diff:
            return False

        c1 = f1["center"]
        c2 = f2["center"]

        # Perpendicular distance of each center to the other's line
        perp12 = abs(np.dot(c2 - c1, f1["normal"]))
        perp21 = abs(np.dot(c1 - c2, f2["normal"]))
        perp_offset = min(perp12, perp21)

        if perp_offset > self.config.streak_merge_perp_offset:
            return False

        # Parallel gap using average direction
        direction = f1["direction"] + f2["direction"]
        norm = np.hypot(direction[0], direction[1])
        if norm == 0:
            direction = f1["direction"]
        else:
            direction = direction / norm

        t1_center = np.dot(c1, direction)
        t2_center = np.dot(c2, direction)

        half1 = 0.5 * f1["length"]
        half2 = 0.5 * f2["length"]

        i1 = np.array([t1_center - half1, t1_center + half1])
        i2 = np.array([t2_center - half2, t2_center + half2])

        if i1[1] < i2[0]:
            gap = i2[0] - i1[1]
        elif i2[1] < i1[0]:
            gap = i1[0] - i2[1]
        else:
            gap = 0.0

        return gap <= self.config.streak_merge_parallel_gap

    @staticmethod
    def _connected_groups_from_edges(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
        """Find connected groups from undirected edges."""
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in edges:
            union(a, b)

        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        return list(groups.values())

    def _merge_fragment_group(
        self,
        fragments: list[dict],
        group: list[int],
    ) -> dict | None:
        """Combine all pixels from a fragment group and refit one line model."""
        ys_all = np.concatenate([fragments[i]["ys"] for i in group])
        xs_all = np.concatenate([fragments[i]["xs"] for i in group])

        merged = self._measure_component_linearity(ys_all, xs_all)
        if merged is None:
            return None
        merged["group"] = group
        return merged

    def _merge_collinear_fragments(
        self,
        fragments: list[dict],
    ) -> list[dict]:
        """Merge fragments that are collinear and nearby."""
        if not fragments:
            return []

        edges = []
        for i in range(len(fragments)):
            for j in range(i + 1, len(fragments)):
                if self._fragments_are_mergeable(fragments[i], fragments[j]):
                    edges.append((i, j))

        groups = self._connected_groups_from_edges(len(fragments), edges)
        merged_tracks = []

        for group in groups:
            merged = self._merge_fragment_group(fragments, group)
            if merged is None:
                continue
            if merged["length"] < self.config.streak_merged_min_length:
                continue
            merged_tracks.append(merged)

        return merged_tracks

    def _make_dense_track_mask(
        self,
        track: dict,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
    ) -> np.ndarray:
        """Rasterize a fitted track into a dense mask."""
        center = track["center"]
        direction = track["direction"]
        normal = track["normal"]

        rel_x = xgrid - center[0]
        rel_y = ygrid - center[1]

        t = rel_x * direction[0] + rel_y * direction[1]
        d = rel_x * normal[0] + rel_y * normal[1]

        half_width = max(
            self.config.streak_dense_min_half_width,
            self.config.streak_dense_width_scale * track["width"],
        )

        return (
            (t >= track["tmin"])
            & (t <= track["tmax"])
            & (np.abs(d) <= half_width)
        )

    def _detect_streak_mask(
        self,
        spximg: SPHERExImage,
        snr: np.ndarray,
        good_mask: np.ndarray,
        use_transient: bool | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Detect streaks by fragmenting, merging collinear fragments,
        and rasterizing dense masks.
        """
        if use_transient is None:
            use_transient = self.config.streak_use_transient

        if not self.config.enable_streak_detection:
            return np.zeros_like(snr, dtype=bool), []

        transient_mask = (spximg.flags & Flag.TRANSIENT) != 0
        source_mask = (spximg.flags & Flag.SOURCE) != 0

        hard_exclude = (
            (spximg.flags & (
                Flag.HOT
                | Flag.COLD
                | Flag.MISSING_DATA
                | Flag.NONFUNC
                | Flag.DICHROIC
            )) != 0
        )
        if use_transient and not np.any(transient_mask[~hard_exclude]):
            return np.zeros_like(snr, dtype=bool), []



        # Original positive-residual streak seeds.
        positive_seed = (snr > self.config.streak_n_sigma) & good_mask

        # Allow TRANSIENT pixels to seed streak detection.
        # This affects only the streak-detection mask, not spximg.data.
        transient_seed = transient_mask & ~hard_exclude

        if use_transient:
            seed = positive_seed | transient_seed
        else:
            seed = positive_seed

        if np.count_nonzero(seed) < self.config.streak_fragment_min_pixels:
            return np.zeros_like(snr, dtype=bool), []

        # Initial line-like fragments
        fragments = self._find_initial_line_fragments(seed)

        # If transient seeding is enabled, keep only fragments with some
        # transient overlap. This avoids random positive SNR fragments being
        # accepted just because the mode is enabled.
        if use_transient and self.config.streak_require_transient_fraction:
            kept = []
            for frag in fragments:
                frag_mask = np.zeros_like(seed, dtype=bool)
                frag_mask[frag["ys"].astype(int), frag["xs"].astype(int)] = True

                transient_fraction = float(np.mean(transient_mask[frag_mask]))
                if transient_fraction >= self.config.streak_min_transient_fraction:
                    frag["transient_fraction"] = transient_fraction
                    kept.append(frag)

            fragments = kept

        # Merge collinear fragments
        merged_tracks = self._merge_collinear_fragments(fragments)

        streak_mask = np.zeros_like(seed, dtype=bool)
        streak_info = []
        xgrid, ygrid = self._get_xy_grids(seed.shape)

        for track in merged_tracks:
            line_mask = self._make_dense_track_mask(track, xgrid=xgrid, ygrid=ygrid)

            transient_support = line_mask & transient_mask & ~hard_exclude
            n_transient_support = int(np.sum(transient_support))

            # Reject short/small transient-supported tracks.
            # This prevents tiny cosmic-ray clumps from becoming STREAK.
            if use_transient:
                if track["length"] < self.config.streak_merged_min_length:
                    continue

                if n_transient_support < self.config.streak_fragment_min_pixels:
                    continue

            # Allow either positive SNR pixels OR TRANSIENT pixels
            # to count as observed support for the dense streak mask.

            dense_support = snr > self.config.streak_dense_min_snr

            if self.config.streak_dense_use_transient:
                dense_support = dense_support | transient_mask

            observed_dense_mask = (
                line_mask
                & ~hard_exclude
                & dense_support
            )

            bridged_dense_mask = self._bridge_source_holes_along_track(
                track=track,
                observed_mask=observed_dense_mask,
                source_mask=source_mask,
                hard_exclude=hard_exclude,
                xgrid=xgrid,
                ygrid=ygrid,
            )

            dense_mask = bridged_dense_mask & ~hard_exclude

            if self.config.streak_grow_radius > 0:
                grow_structure = np.ones(
                    (
                        2 * self.config.streak_grow_radius + 1,
                        2 * self.config.streak_grow_radius + 1,
                    ),
                    dtype=bool,
                )
                dense_mask = binary_dilation(dense_mask, structure=grow_structure) & ~hard_exclude

            if not np.any(dense_mask):
                continue

            streak_mask |= dense_mask

            info = {
                "npix": int(dense_mask.sum()),
                "length": track["length"],
                "width": track["width"],
                "aspect_ratio": track["aspect_ratio"],
                "angle_deg": track["angle_deg"],
                "n_fragments": len(track.get("group", [])),
            }

            if use_transient:
                info["transient_fraction"] = float(np.mean(transient_mask[dense_mask]))

            streak_info.append(info)

        return streak_mask, streak_info

    @add_history()
    def run(self, spximg: SPHERExImage) -> pipeBase.Struct:
        good_mask = self._make_good_pixel_mask(spximg)
        residual, snr, std_map = self._compute_residual_and_snr(spximg, good_mask)

        # Run streak detection without TRANSIENT support.
        streak_mask_no_transient, streak_info_no_transient = self._detect_streak_mask(
            spximg,
            snr,
            good_mask,
            use_transient=False,
        )

        # Run streak detection with TRANSIENT support.
        streak_mask_with_transient, streak_info_with_transient = self._detect_streak_mask(
            spximg,
            snr,
            good_mask,
            use_transient=True,
        )

        # Union of both modes.
        streak_mask = streak_mask_no_transient | streak_mask_with_transient
        streak_info = streak_info_no_transient + streak_info_with_transient


        source_mask = (spximg.flags & Flag.SOURCE) != 0
        outlier_mask = good_mask | (streak_mask & source_mask)
        # Second, isolated outliers
        outliers = self._detect_pixel_outliers(snr, outlier_mask)

        spximg_out = spximg.copy()
        # Pixels are allowed in include both OUTLIER and STREAK.
        spximg_out.setFlagsFromMask(outliers, Flag.OUTLIER)
        spximg_out.setFlagsFromMask(streak_mask, Flag.STREAK)

        n_outlier = int(np.sum((spximg_out.flags & Flag.OUTLIER) > 0))
        n_streak = int(np.sum((spximg_out.flags & Flag.STREAK) > 0))
        n_tracks = int(len(streak_info))

        spximg_out.header["HIERARCH L2 N_OUTLIER"] = (
            n_outlier,
            "number of flagged outlier pixels"
        )

        spximg_out.header["HIERARCH L2 SATSTRKS_PXS"] = (
            n_streak,
            "Number of flagged satellite streak pixels"
        )

        spximg_out.header["HIERARCH L2 SATSTRKS"] = (
            n_tracks,
            "Number of detected satellite streaks"
        )

        self.log.info("Number of pixels flagged as outliers: %d", n_outlier)
        self.log.info("Number of pixels flagged as streak: %d", n_streak)
        self.log.info("Number of accepted streak components: %d", n_tracks)

        _ = dqa_check_outlier_pixel_count(
            spximg_out,
            tolerance=self.config.dqa_tolerance,
        )

        return pipeBase.Struct(spximg_out=spximg_out)
