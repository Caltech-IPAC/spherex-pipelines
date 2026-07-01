import os
import datetime
import logging
import tempfile
import numpy as np
from typing import Optional, Tuple
from astropy.table import Table
from astropy.time import Time
from lsst.daf.butler import (
    Butler, DatasetRef, Timespan
)
from astropy.wcs import WCS
from spx_utils.tempfile_utils import get_temp_path
from spx_apps.photometry_tool.temp_butler import get_unique_dataset_refs, \
    export_datasets
from spx_apps.mosaic_tool.exceptions import EmptyMosaicQueryError

BUTLER_CONFIG = "resource://spx_butler/configs"
BUTLER_YAML = "butler.yaml"
DIMENSION_YAML = "dimensions.yaml"

# Default defaults based on SPHEREx (approx 6.2") and buffer default of 2
# arcminutes.
DEFAULT_PIXEL_SCALE_DEG = 6.2
DEFAULT_BUFFER_DEG = 2.0


def compute_polygon_wcs_angular(center_ra, center_dec, width_deg, height_deg, rotation_deg,
                                pixel_scale_deg=DEFAULT_PIXEL_SCALE_DEG,
                                buffer_deg=os.getenv("SPXMOSAIC_BUFFER_SIZE", 2)):
    """
    Algorithm:
        1. Find the “ideal” corners of the requested region by assuming tha the
        width and height (5 degrees maximum) are the true angular extents of
        the mosaic footprint on the sky.
        Set a tangent WCS, rotate it and extrapolate the half angle from the boresight,
        and let astropy do the work to determine what the corners of the polygon are.

        2. Given the pixel size requested for the pixels at the center of the image,
        determine the number of pixels needed in the TAN projection to actually cover
        the requested region,
        rounding up to full pixels. This will be on average 1/2 a pixel farther out
        in each direction.
        That's the actual mosaic grid that we're going to fill out.

        3. Expand out from #2 by a configurable margin - by 2 arcminutes as default- to
        construct the query polygon to be used with the Butler.
    """
    pixel_scale_deg = pixel_scale_deg / 3600  # Converting to arcsec
    buffer_deg = buffer_deg / 60  # Converting to arcmin

    # 1. Setup WCS centered on the target
    w = WCS(naxis=2)
    w.wcs.crval = [center_ra, center_dec]
    # Center the target on the *center* of a pixel
    # (e.g., 1.0, 1.0 is center of 1-based pixel)
    w.wcs.crpix = [1, 1]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # We use degree units for the intermediate projection plane
    w.wcs.cdelt = [1.0, 1.0]

    # 2. Apply Rotation
    theta = np.deg2rad(-rotation_deg)
    w.wcs.pc = [[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]

    # --- Step 2: Pixel Grid Logic ---
    deg_to_rad = np.pi / 180.0

    # Calculate "Ideal" projection plane half-lengths (R = tan(theta)) in degrees
    x_ideal_proj_deg = (1.0 / deg_to_rad) * np.tan((width_deg / 2.0) * deg_to_rad)
    y_ideal_proj_deg = (1.0 / deg_to_rad) * np.tan((height_deg / 2.0) * deg_to_rad)

    # Convert to pixels (rounding UP to ensure coverage)
    x_pixels_half = np.ceil(x_ideal_proj_deg / pixel_scale_deg)
    y_pixels_half = np.ceil(y_ideal_proj_deg / pixel_scale_deg)

    # Convert back to projection plane degrees
    # This is our "snapped" grid size
    x_snapped_proj_deg = x_pixels_half * pixel_scale_deg
    y_snapped_proj_deg = y_pixels_half * pixel_scale_deg

    # --- Step 3: Buffer Expansion ---
    # Add the configurable margin (e.g. 2 arcmin) to the projection plane extent
    # Note: Adding degrees directly in projection plane is a safe approximation
    # for buffer
    x_offset = x_snapped_proj_deg + buffer_deg
    y_offset = y_snapped_proj_deg + buffer_deg

    logging.info(f"Polygon Calc: Req {width_deg:.3f}x{height_deg:.3f} -> "
                 f"Snapped {x_snapped_proj_deg*2:.3f}x{y_snapped_proj_deg*2:.3f} -> "
                 f"Buffered {(x_offset*2):.3f}x{(y_offset*2):.3f}")

    # 4. Generate Corners (in projection plane degrees)
    # TL, TR, BR, BL
    corners_proj = np.array([
        [-x_offset, y_offset],
        [x_offset, y_offset],
        [x_offset, -y_offset],
        [-x_offset, -y_offset]
    ])

    # 5. Convert to Sky Coordinates
    ra, dec = w.wcs_pix2world(corners_proj[:, 0], corners_proj[:, 1], 0)

    # 6. Format
    coords_list = []
    for r, d in zip(ra, dec):
        coords_list.append(f"{r:.7f} {d:.7f}")

    polygon_query = f"POLYGON {' '.join(coords_list)}"
    logging.info(polygon_query)
    return polygon_query


def lvf_lookup(
        butler: Butler,
        region: Optional[str] = None,
        time_interval: Optional[tuple[Time, Time]] = None,
        detectors: Optional[list[int]] = None,
        instrument="spx",
        level2_dataset_type="level2",
        collections="*",
        verbose=False
) -> tuple[Table, list[list[DatasetRef]]]:
    """Get coverage table and dataset references."""

    base_where = f"instrument='{instrument}'"
    bind = None

    if time_interval is not None:
        stime: Time = time_interval[0]
        etime: Time = time_interval[1]
        if stime is not None or etime is not None:
            base_where += " AND observation.timespan OVERLAPS ts"
            bind = {"ts": Timespan(begin=stime, end=etime)}

    if detectors is not None:
        detectors_str = ", ".join(str(d) for d in detectors)
        base_where += f" AND detector in ({detectors_str})"

    dataset_refs = []

    # --- Mosaic Tool Path: Region-based query ---
    if region is not None:
        if verbose:
            logging.info(f"Querying region: {region}")

        where = base_where + f" AND exposure.region OVERLAPS REGION('{region}')"

        logging.info(
            f"Querying: WHERE {where} BIND {bind}"
        )

        find_first = "*" not in collections
        level2_datasets = butler.query_datasets(
            level2_dataset_type,
            collections.split(","),
            where=where,
            bind=bind,
            find_first=find_first
        )
        refs = [r for r in level2_datasets]
        dataset_refs.append(refs)

        # Create a dummy coverage table
        coverage_table = Table({"n_img": [len(refs)], "mosaic_region": [region]})
    else:
        # Handle case where no region provided (if applicable) or return empty
        coverage_table = Table({"n_img": [0], "mosaic_region": [""]})

    return coverage_table, dataset_refs


def collect_inputs_for_temp_butler(
        butler_repo: str,
        tmp_dir: str,
        center: Optional[Tuple[float, float]] = None,
        size: Optional[Tuple[float, float]] = None,
        rotation: Optional[float] = 0.0,
        time_interval: Optional[tuple[Time, Time]] = None,
        detectors: Optional[list[int]] = None,
        pixel_scale_deg: float = DEFAULT_PIXEL_SCALE_DEG,
) -> dict:
    """Find and export all necessary inputs from a source Butler to run
    mosaic tool.

    Parameters include pixel_scale_deg and buffer_deg to control polygon generation.
    """
    instrument = os.getenv("SPXPHOTO_INSTRUMENT", "spx")
    if instrument not in ["spx", "simu"]:
        raise ValueError(f"Unsupported instrument: {instrument}")

    lvf_dataset_type_name = os.getenv("SPXPHOTO_L2_DATASET_TYPE", "level2")
    butler = Butler(butler_repo)
    input_coll = os.getenv("SPXPHOTO_INPUT_COLL", "IRSA-qr2-visible")

    region_str = None

    # Logic to handle Mosaic arguments
    logging.info(f"Center {center} Size {size} Rotation {rotation}")
    if center is not None and size is not None:
        center_ra, center_dec = center
        width_deg, height_deg = size

        # Updated polygon generation with pixel snapping and buffer
        region_str = compute_polygon_wcs_angular(
            center_ra, center_dec,
            width_deg, height_deg, rotation,
            pixel_scale_deg=pixel_scale_deg,
        )
        logging.info(f"Computed Mosaic Query Polygon: {region_str}")

    coverage_table, dataset_refs = lvf_lookup(
        butler,
        region=region_str,
        time_interval=time_interval,
        detectors=detectors,
        instrument=instrument,
        level2_dataset_type=lvf_dataset_type_name,
        collections=input_coll,
        verbose=True
    )

    if len(coverage_table) == 0 or coverage_table["n_img"][0] == 0:
        raise EmptyMosaicQueryError(mosaic_region=region_str)

    unique_dataset_refs = get_unique_dataset_refs(dataset_refs)

    timestamp = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
    temp_butler_root = tempfile.mkdtemp(prefix=f"{timestamp}_butler_", dir=tmp_dir)
    export_file = get_temp_path(prefix=f"{timestamp}_export_", suffix=".yaml", dir=tmp_dir)

    export_datasets(butler,
                    dataset_refs=(unique_dataset_refs),
                    export_file=export_file,
                    export_directory=str(temp_butler_root))

    results = {
        "temp_butler_root": str(temp_butler_root),
        "export_file": str(export_file),
        "coverage_table": coverage_table
    }
    return results
