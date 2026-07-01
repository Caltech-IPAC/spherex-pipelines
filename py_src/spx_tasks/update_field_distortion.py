__all__ = ["UpdateFieldDistortionConnections",
           "UpdateFieldDistortionConfig",
           "UpdateFieldDistortionTask"]

# Standard library imports
from pathlib import Path
import copy
import logging

# Related third party imports
from lsst.utils.timer import timeMethod
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field
from lsst.daf.butler import DeferredDatasetHandle
from astropy.wcs import WCS
from astropy import units as u
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from photutils.detection import find_peaks
from astropy.wcs import Sip

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils import SourcePipelineClass, ImageCatalogIterable
from spx_utils.update_distortion_utils import calculate_sip_coefficients, update_wcs_with_sip
from spx_utils.solid_angle_utils import calculate_solid_angle, calculate_median_std_solid_angle
from spx_tasks import add_history
from spx_utils.utils import add_detcoord

# Please remember to comment out logging level changes applied during development
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
def calculate_distortion_map(imageHandles: list[DeferredDatasetHandle],
                             catalogHandles: list[DeferredDatasetHandle],
                             sip: Sip,
                             array_id: int,
                             min_mag: float,
                             max_mag: float,
                             plot: bool = False,
                             logger: logging.Logger = logging.getLogger(__name__)
                             ) -> tuple:
    """Calculates the distortion between detected object positions
       in the images and predicted positions from the catalog
       using WCS with and without SIP.

    Parameters
    ----------
    imageHandles : `list[DeferredDatasetHandle]`
        Handles to the SPHEREx images to process
    catalogHandles : `list[DeferredDatasetHandle]`
        Handles to the catalogs corresponding to each image
    sip : `~astropy.wcs.SIP`
        Initial SIP solution.
    array_id : `int`
        Detector ID number.
    min_mag : `float`
        Minimum magnitude for calibrator source.
    max_mag : `float`
        Maximum magnitude for calibrator source.
    plot : `bool`
        Analysis plots made, default is False.
    logger : `logging.Logger`
        Logger to use.

    Returns
    -------
    `tuple`
    det_xs, det_ys : detected (distorted) sources
    cat_xs_no_sip, cat_ys_no_sip : catalog sources with no SIP
    cat_xs_with_sip, cat_ys_with_sip : catalog sources with SIP
    dist_xs, dist_ys : difference between distorted sources and catalog with no SIP
    res_xs, res_ys : difference between distorted sources and catalog with SIP
    """
    det_xs = []
    det_ys = []
    cat_xs_no_sip = []
    cat_ys_no_sip = []
    cat_xs_with_sip = []
    cat_ys_with_sip = []
    dist_xs = []
    dist_ys = []
    res_xs = []
    res_ys = []

    # Iterate over images
    iterableImgCat = ImageCatalogIterable(imageHandles,
                                          catalogHandles,
                                          srcPipeFlag=SourcePipelineClass.ASTROMETRIC,
                                          addPixelCoords=False,
                                          pixelCoordsWithDistortion=False,
                                          filterSourcesOutsideImage=True
                                          )

    for spximg, cat in iterableImgCat:
        image_data = spximg.data
        # Select calibrator sources from reference catalog
        sel_calibrators = np.where((cat[f"mag_det{array_id}"] > min_mag)
                                   & (cat[f"mag_det{array_id}"] < max_mag)  # magnitude cut
                                   & (cat["dist_nearest"] > 0)  # select isolated sources
                                   )[0]
        catalog = cat[sel_calibrators]
        logger.info(f'len catalog : {len(catalog)}')

        # WCS with SIP
        wcs_with_sip = copy.deepcopy(spximg.wcs)
        wcs_with_sip.sip = sip
        logger.info(wcs_with_sip.to_header(relax=True))

        if catalog['ra'].unit is None:
            catalog['ra'] = catalog['ra'] * u.deg
        if catalog['dec'].unit is None:
            catalog['dec'] = catalog['dec'] * u.deg
        # Convert catalog world coordinates to pixel coordinates
        catalog_coords = SkyCoord(ra=catalog['ra'], dec=catalog['dec'], frame='icrs')
        ra = catalog_coords.ra.deg
        dec = catalog_coords.dec.deg

        # Convert RA and Dec to pixel coordinates
        catalog_pix_coords_with_sip = wcs_with_sip.all_world2pix(ra, dec, 0)

        # WCS without SIP to obtain distorted source position
        wcs_no_sip = copy.deepcopy(spximg.wcs)
        wcs_no_sip.sip = None
        logger.info(wcs_no_sip.to_header(relax=True))

        catalog_pix_coords_no_sip = wcs_no_sip.world_to_pixel(catalog_coords)

        # Detect sources in the image
        peaks = find_peaks(image_data, threshold=np.mean(image_data) + 3*np.std(image_data), box_size=11)
        if peaks is None:
            continue

        used_peaks = set()

        # Match detected peaks to catalog sources
        for (cat_x_sip, cat_y_sip), (cat_x_no_sip, cat_y_no_sip) in zip(zip(catalog_pix_coords_with_sip[0],
                                                                            catalog_pix_coords_with_sip[1]),
                                                                        zip(catalog_pix_coords_no_sip[0],
                                                                            catalog_pix_coords_no_sip[1])):
            distances = np.sqrt((peaks['x_peak'] - cat_x_sip)**2 + (peaks['y_peak'] - cat_y_sip)**2)
            scores = distances / peaks['peak_value']  # Selection by proximity and brightness
            best_index = np.argmin(scores)
            if best_index not in used_peaks:
                used_peaks.add(best_index)
                # Detected sources
                closest_x = peaks['x_peak'][best_index]
                closest_y = peaks['y_peak'][best_index]
                # Matched detected sources to catalog (with and w/o SIP)
                cat_xs_no_sip.append(cat_x_no_sip)
                cat_ys_no_sip.append(cat_y_no_sip)
                cat_xs_with_sip.append(cat_x_sip)
                cat_ys_with_sip.append(cat_y_sip)
                det_xs.append(closest_x)
                det_ys.append(closest_y)
                # distortion is difference between detected peak and catalog with no SIP
                dist_xs.append(closest_x - cat_x_no_sip)
                dist_ys.append(closest_y - cat_y_no_sip)
                # residual is difference between detected peak and catalog with SIP
                res_xs.append(closest_x - cat_x_sip)
                res_ys.append(closest_y - cat_y_sip)

    # Plotting
    if plot is True:
        plt.figure(figsize=(10, 8))
        plt.scatter(det_xs, det_ys, color='blue', label='Detected Sources', alpha=0.9)
        plt.scatter(cat_xs_with_sip, cat_ys_with_sip, color='green',
                    label='Catalog Sources with SIP', alpha=0.5, marker='x')
        plt.scatter(cat_xs_no_sip, cat_ys_no_sip, color='red',
                    label='Catalog Sources without SIP', alpha=0.5, marker='x')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.xlabel('X Pixel Coordinate')
        plt.ylabel('Y Pixel Coordinate')
        plt.title('Detected vs. Catalog Source Positions with/without SIP')
        plt.legend()
        plt.grid(True)
        plt.show()

    return (det_xs, det_ys,
            cat_xs_no_sip, cat_ys_no_sip,
            cat_xs_with_sip, cat_ys_with_sip,
            dist_xs, dist_ys,
            res_xs, res_ys)


def create_residual_map(residuals_x: np.ndarray,
                        residuals_y: np.ndarray,
                        cat_x: np.ndarray,
                        cat_y: np.ndarray,
                        image_shape: tuple,
                        plot: bool):
    """Creates a residual map using displacement vectors.

    Parameters
    ----------
    residuals_x : `np.ndarray`
        Displacement values in the x-direction.
    residuals_y : `np.ndarray`
        Displacement values in the y-direction.
    cat_x : `np.ndarray`
        X-coordinates of the catalog points, must match the length of displacement_x.
    cat_y : `np.ndarray`
        Y-coordinates of the catalog points, must match the length of displacement_y.
    image_shape : `tuple`
        Shape of the image, height, width).
    plot : `bool`
        If True, displays the residual maps using matplotlib.

    Returns
    -------
    `tuple` of (map_dx, map_dy)
        A tuple containing the x and y residual maps, respectively.

    Notes
    -----
    This function interpolates the provided x and y displacement values
    over a regular grid corresponding to the image dimensions.
    The interpolation is done using a 'cubic' method.
    """
    grid_x, grid_y = np.mgrid[0:image_shape[1], 0:image_shape[0]]
    if len(residuals_x) > 3:  # Ensure enough points for interpolation
        method = 'cubic'
    else:
        return None, None

    map_dx = griddata((cat_x, cat_y), residuals_x, (grid_x, grid_y), method=method)
    map_dy = griddata((cat_x, cat_y), residuals_y, (grid_x, grid_y), method=method)

    plot = True
    if plot is True:
        # Determine color scale limits based on a robust method
        vmax_dx = np.percentile(map_dx[~np.isnan(map_dx)], 98)  # 98th percentile
        vmin_dx = np.percentile(map_dx[~np.isnan(map_dx)], 2)   # 2nd percentile

        vmax_dy = np.percentile(map_dy[~np.isnan(map_dy)], 98)
        vmin_dy = np.percentile(map_dy[~np.isnan(map_dy)], 2)

        # Plot residual maps
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(map_dx, origin='lower', cmap='coolwarm', extent=(0, image_shape[1], 0, image_shape[0]),
                   vmin=vmin_dx, vmax=vmax_dx)
        plt.colorbar(label='DX Residuals (pixels)')
        plt.title('DX Residual Map')
        plt.subplot(122)
        plt.imshow(map_dy, origin='lower', cmap='coolwarm', extent=(0, image_shape[1], 0, image_shape[0]),
                   vmin=vmin_dy, vmax=vmax_dy)
        plt.colorbar(label='DY Residuals (pixels)')
        plt.title('DY Residual Map')
        plt.show()

    return map_dx, map_dy


# ==============================================================================


class UpdateFieldDistortionConnections(pipeBase.PipelineTaskConnections,
                                       dimensions={"instrument", "detector"},
                                       defaultTemplates={}):
    imageHandles = cT.Input(
        name="intype",  # default dataset type for input
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "exposure"],
        deferLoad=True,
        multiple=True
    )
    catalogHandles = cT.Input(
        name="cattype",  # default dataset type for input
        doc="Input catalogs.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
        deferLoad=True,
        multiple=True
    )
    sip = cT.Output(
        name="outtype",  # default dataset type for output
        doc="Output distortion as SIP (Simple Imaging Polynomial).",
        storageClass="AstropySIP",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    solid_angle_pixel_map = cT.Output(
        name="solid_angle_pixel_map",
        doc="Output solid angle pixel map.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class UpdateFieldDistortionConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=UpdateFieldDistortionConnections):
    n_edge_pix = Field(
        doc="Number of edge pixels to be ignored.",
        dtype=int,
        default=0)
    cutout_hw = Field(
        doc="Cutout half width. Full width is 2*hw + 1.",
        dtype=int,
        default=5)
    min_n_samples = Field(
        doc="Tolerance for the minimum number of measurements.",
        dtype=int,
        default=25)
    null_src_th = Field(
        doc="Tolerance for number of astrometric calibrators"
            " that could not be extracted as a source.",
        dtype=int,
        default=1000)
    a_order = Field(
        doc="Forward SIP polynomial order for first axis.",
        dtype=int,
        default=3)
    b_order = Field(
        doc="Forward SIP polynomial order for second axis.",
        dtype=int,
        default=3)
    ap_order = Field(
        doc="Backward SIP polynomial order for first axis.",
        dtype=int,
        default=3)
    bp_order = Field(
        doc="Backward SIP polynomial order for second axis.",
        dtype=int,
        default=3)
    suppress_linear_terms = Field(
        doc="Ideally, the linear terms in the forward distortion model should be suppressed."
            " By setting this option to `False`, linear terms can be kept to"
            " check for their strong presence of due to rigid translation or rotations not corrected by WCS."
            " If `True`, the coefficients in ``c_0_0 + c_0_1 * x + c_1_0 * y`` are favored to vanish."
            " Here, ``c`` represent ``A`` or ``B`` coefficients in SIP."
            " This option will not affect ``AP`` and ``BP`` coefficients in the backward model.",
        dtype=bool,
        default=True)


class UpdateFieldDistortionTask(pipeBase.PipelineTask):
    """Get SIP coefficients describing distortion.

    """
    _DefaultName = "update-field-distortion"
    ConfigClass = UpdateFieldDistortionConfig
    full_name = "Update Field Distortion"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-028 v1.3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @timeMethod
    @add_history(fld_name="solid_angle_pixel_map")
    def run(self,
            imageHandles: list[DeferredDatasetHandle],
            catalogHandles: list[DeferredDatasetHandle],
            min_mag: float = 0.,
            max_mag: float = 15.,
            plot: bool = False):
        """Estimate Simple Imaging Polynomial (SIP coefficients).

        Parameters
        ----------
        imageHandles
            Input images: deferred handles for `~spx_data.SPHERExImage`
        catalogHandles
            Input catalogs: deferred handles for `~astropy.table.Table`
        min_mag : `float`
            Minimum magnitude cut for calibrator sources
        max_mag : `float`
            Maximum magnitude cut for calibrator sources
        plot : `bool`
            Output analysis plots, default is False

        Returns
        -------
        result : `lsst.pipe.base.Struct`

        Notes
        -----
        Return result includes the following.
        ``sip`` : `astropy.wcs.Sip`
            Estimated SIP.
        ``array_id`` : `int`
            Detector ID
        ``reduced_chisq1_fwd`` : `float`
            Reduced chi-square statistics from residual for the 1st axis,
            forward transformation
        ``reduced_chisq2_fwd`` : `float`
            Reduced chi-square statistics from residual for the 2nd axis,
            forward transformation
        ``reduced_chisq1_inv`` : `float`
            Reduced chi-square statistics from residual for the 1st axis,
            backward transformation
        ``reduced_chisq2_inv`` : `float`
            Reduced chi-square statistics from residual for the 2nd axis,
            backward transformation
        ``num_images`` : `int`
            Number of images used
        ``resmap_dx`` : `np.ndarray`
            Residual map, dx.
        ``resmap_dy`` : `np.ndarray`
            Residual map, dy.
        ``solid_angle_pixel_map`` : `np.ndarray`
            Solid angle pixel map.
        """
        if len(imageHandles) < 1:
            raise ValueError("No images to estimate SIP")

        # Get basic info
        # NOTE: Detector information is extracted from the first image and
        # the user is responsible for ensuring all images are from the same detector
        spximg0 = imageHandles[0].get()
        array_id = spximg0.getArrayId()
        self.log.info(f"Processing detector {array_id}")

        wcs_sip_init = copy.deepcopy(spximg0.wcs)

        shape = spximg0.shape

        det_xs, det_ys, cat_xs_nosip, cat_ys_nosip, _, _, _, _, _, _ =\
            calculate_distortion_map(imageHandles, catalogHandles, spximg0.wcs.sip, array_id,
                                     min_mag=min_mag, max_mag=max_mag, plot=plot, logger=self.log.logger)

        # Calulate average SIP coefficients to use as initial values in the fit.
        # Foward model: ``poly_x``, ``poly_y``
        # Backward model: ``poly_xp``, ``poly_yp``
        poly_x, poly_y, poly_xp, poly_yp, chi2_x, chi2_y, chi2_xp, chi2_yp =\
            calculate_sip_coefficients(cat_xs_nosip, cat_ys_nosip, det_xs, det_ys, wcs_sip_init)

        wcs_updated = update_wcs_with_sip(poly_x, poly_y, poly_xp, poly_yp, wcs_sip_init)

        # Residual map
        det_xs, det_ys, _, _, _, _, _, _, res_xs, res_ys =\
            calculate_distortion_map(imageHandles, catalogHandles, wcs_updated.sip, array_id,
                                     min_mag=min_mag, max_mag=max_mag, logger=self.log.logger)
        resmap_dx, resmap_dy = create_residual_map(res_xs, res_ys, det_xs, det_ys, shape, plot)

        # Calculate solid angles with SIP solution
        solid_angle = calculate_solid_angle(wcs_updated, array_id, plot)

        solid_angle_pixel_map = SPHERExImage(solid_angle, unit=u.arcsec**2)
        solid_angle_pixel_map.setArrayId(array_id)
        solid_angle_pixel_map.add_provenance()
        add_detcoord(solid_angle_pixel_map)

        # Calculte median and std of solid angle map
        median, std_dev = calculate_median_std_solid_angle(solid_angle)

        return pipeBase.Struct(sip=wcs_updated.sip,
                               wcs_updated=wcs_updated,
                               array_id=array_id,
                               reduced_chisq1_fwd=chi2_x,
                               reduced_chisq2_fwd=chi2_y,
                               reduced_chisq1_inv=chi2_xp,
                               reduced_chisq2_inv=chi2_yp,
                               num_images=len(imageHandles),
                               resmap_dx=resmap_dx,
                               resmap_dy=resmap_dy,
                               solid_angle_pixel_map=solid_angle_pixel_map,
                               median=median,
                               std_dev=std_dev)


def main(data_path):
    """Acts as butler, providing the data and running the task.

    Parameters
    ----------
    data_path : `~pathlib.Path`
        Path to SPHERExImages of the SAME detector.
    """
    from spx_tests.mock_dataset import mockImageCatalogHandles

    img_handles, cat_handles = mockImageCatalogHandles(data_path)

    task = UpdateFieldDistortionTask()
    output = task.run(img_handles, cat_handles)

    wcs = WCS()
    wcs.sip = output.sip


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"\nUsage: {sys.argv[0]} <path_to_fits>")
        sys.exit(1)
    main(Path(sys.argv[1]))
