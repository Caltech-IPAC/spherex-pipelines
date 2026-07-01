"""Tests for sliding window utility functions."""

# Standard library imports
from pathlib import Path
import itertools
import pytest

# Related third party imports
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.table import Column
from astropy.units import Quantity
from lsst.pex.config import Config
import astropy.units as u
import healpy as hp
import numpy as np

# Local application/library specific imports
from spx_data import PSFZoneData, SPHERExImage, spherex_image_reader
from spx_tasks import PerformForcedPhotometryConfig
from spx_utils.photometry.window_utils import (
    midpoint, slide_window, estimate_loc_bkg, run_tractor_window, combine_measurements, combine_fits)
from spx_utils import SourcePipelineClass

test_root = Path(__file__).parent.parent


def get_sample_file(type: str) -> SPHERExImage:
    if type == "point source":
        sample_file = test_root / "data/photometry_point_sources.fits"
    elif type == "galaxy":
        sample_file = test_root / "data/photometry_galaxy.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)


def get_window_shape(window_params: dict) -> tuple[int, int]:
    """Calculate window shape.

    Parameters
    ----------
    window_params : `dict`
        Dictionary of window boundaries.

    Returns
    -------
    `tuple`[`int`, `int`]
        2D window shape.
    """
    x_size = window_params["window_x2"] - window_params["window_x1"]
    y_size = window_params["window_y2"] - window_params["window_y1"]
    return (x_size, y_size)


def check_fields(flux_arr: np.ndarray):
    """Checks for missing photometry measurement fields.

    Parameters
    ----------
    flux_arr : `numpy.ndarray`
        Structured array of photometry results.
    """
    expected_fields = ["SPHERExRefID", "ra", "dec", "x_image", "y_image", "flux", "flux_err", "flux_bkg",
                       "local_bkg_flg", "ext_flg", "flags", "fit_ql", "hips6"]
    for field in expected_fields:
        assert field in flux_arr.dtype.names, f"Field {field} not found in combined photometry array."


def setup_fixture(img, window_params, config: Config, keep_frame_sources: bool):
    """Set up fixtures for testing.

    Parameters
    ----------
    img : `spx_data.SPHERExImage`
        Small LVF image (including catalog of sources to be measured).
    window_params : `dict`[`str`, `int`]
        Keys and values corresponding to window parameters. All X, Y values are
        pixel coordinates on the detector.
    config : `lsst.pex.config.Config`
        Contains fields with local background estimation parameters.
    keep_frame_sources : `bool`
        If `True`, returns photometry for sources in the outer frame of the
        cutout.

    Returns
    -------
    results : `numpy.ndarray` or `dict`
        Returns array of photometry measurements by default. If
        `config`.``return_fit`` is ``True``, returns dictionary with flux
         results and best-fit model/residual image.
    """
    # Set up input parameters
    srccat = img.catalog
    hips_col_name = f"hips{config.hips_order}"
    col = Column(data=hp.ang2pix(2 ** config.hips_order, srccat["ra"], srccat["dec"], nest=True,
                                 lonlat=True),
                 name=hips_col_name, unit="")  # leaving unit None, makes "deg" unit
    srccat.add_column(col)
    cat_coords = SkyCoord(ra=srccat["ra"], dec=srccat["dec"], frame="icrs", unit=u.deg)
    src_xdet, src_ydet = (img.wcs).world_to_pixel(cat_coords)
    psf = PSFZoneData.from_hdu(img.layers["PSF"])
    detector = img.getArrayId()

    # Perform fitting
    results = run_tractor_window(img.data, img.unit, img.wcs, img.uncertainty, img.flags, window_params,
                                 src_xdet, src_ydet, srccat, cat_coords, False, config,
                                 psf, detector, keep_frame_sources)

    return results


class TestWindowUtils():
    """Test sliding window utility functions."""
    base_cutout_config = PerformForcedPhotometryConfig(cut_side_length=5, frame_width=3,
                                                       bkg_side_length=9,
                                                       class_filter=SourcePipelineClass.ALL)
    return_fit_config = PerformForcedPhotometryConfig(cut_side_length=5, frame_width=3,
                                                      bkg_side_length=9, return_fit=True,
                                                      class_filter=SourcePipelineClass.ALL)

    ps_img = spherex_image_reader(get_sample_file("point source"), hdu_layers=["PSF", "CATALOG"])
    ps_windows_local = slide_window(ps_img, base_cutout_config, use_local=True)
    ps_windows_global = slide_window(ps_img, base_cutout_config, use_local=False)

    gal_img = spherex_image_reader(get_sample_file("galaxy"), hdu_layers=["PSF", "CATALOG"])
    gal_windows_global = slide_window(gal_img, base_cutout_config, use_local=False)

    @pytest.fixture(scope="class")
    def tractor_ps_return_fit(self):
        flux_dict = setup_fixture(self.ps_img, self.ps_windows_global[6], self.return_fit_config, False)
        return flux_dict

    @pytest.fixture(scope="class")
    def tractor_ps_keep_frame(self):
        flux_tab = setup_fixture(self.ps_img, self.ps_windows_global[6], self.base_cutout_config, True)
        return flux_tab

    @pytest.fixture(scope="class")
    def tractor_ps_cls_filter(self):
        flux_tab = setup_fixture(self.ps_img, self.ps_windows_global[6], self.cls_filter_config, False)
        return flux_tab

    @pytest.fixture(scope="class")
    def tractor_gal_return_fit(self):
        flux_dict = setup_fixture(self.gal_img, self.gal_windows_global[3], self.return_fit_config, False)
        return flux_dict

    def test_midpoint(self):
        line = [(0, 0), (5, 5)]
        expected = (2.5, 2.5)
        result = midpoint(x1=line[0][0], x2=line[1][0], y1=line[0][1], y2=line[1][1])
        assert result == expected

    def test_slide_window_global(self):
        """Test sliding window for global background case."""
        window = self.ps_windows_global[0]
        nan_vals = [window["bkgd_x1"], window["bkgd_x2"], window["bkgd_y1"], window["bkgd_y2"]]
        nan_msg = "Background estimation window bounds are not NaN for global background subtraction case."
        assert np.all(np.isnan(nan_vals)), nan_msg

        # For cut_side_length=5 and sample image of 15x15, we expect 3 windows
        # per row. Window size will vary based on position within the image.
        corner = self.ps_windows_global[0]
        top_edge = self.ps_windows_global[1]
        left_edge = self.ps_windows_global[3]
        inner = self.ps_windows_global[4]
        assert get_window_shape(corner) == (8, 8), "Sliding window is not calculating corners correctly."
        assert get_window_shape(top_edge) == (11, 8), "Sliding window is not calculating edges correctly."
        assert get_window_shape(left_edge) == (8, 11), "Sliding window is not calculating edges correctly."
        assert get_window_shape(inner) == (11, 11), "Sliding window is not calculating inners correctly."

    def test_slide_window_local(self):
        """Test sliding window for local background case."""
        window = self.ps_windows_local[0]
        finite_vals = [window["bkgd_x1"], window["bkgd_x2"], window["bkgd_y1"], window["bkgd_y2"]]
        finite_msg = "Background estimation window bounds are not finite (+/- inf. and NaN) for local"\
                     " background subtraction case."
        assert np.all(np.isfinite(finite_vals)), finite_msg

    def test_estimate_loc_bkg(self):
        """Test local background estimation."""
        window = self.ps_windows_local[0]
        bkg_cut = estimate_loc_bkg(window, self.ps_img.data, self.ps_img.unit, self.ps_img.wcs,
                                   self.base_cutout_config)

        window_ctr = midpoint(window["window_x1"], window["window_x2"],
                              window["window_y1"], window["window_y2"])
        size = ((window["window_y2"] - window["window_y1"]), (window["window_x2"] - window["window_x1"]))
        img_cut = Cutout2D(Quantity(value=self.ps_img.data, unit=self.ps_img.unit, copy=False),
                           position=window_ctr, size=size, wcs=self.ps_img.wcs)

        pos_msg = "Local background cutout and image cutout do not share the same pixel size or origin."
        units_msg = "Local background cutout and image cutout do not share the same units."
        assert (bkg_cut.wcs.to_header(relax=True) == img_cut.wcs.to_header(relax=True)), pos_msg
        assert bkg_cut.data.unit == img_cut.data.unit, units_msg

    def test_run_tractor_window_return_fit(self, tractor_ps_return_fit, tractor_gal_return_fit):
        """Test running tractor on a single window, returning the fit."""
        results_list = [tractor_ps_return_fit, tractor_gal_return_fit]
        num_srcs = [2, 1]
        x_bounds = [slice(0, 5), slice(-5, None)]
        y_bounds = [slice(0, 5), slice(-5, None)]

        for results, n_srcs, x_lim, y_lim in zip(results_list, num_srcs, x_bounds, y_bounds):
            # Check return type; check # of sources
            assert isinstance(results, dict)
            flux_arr = results["flux_result"]
            assert len(flux_arr) == n_srcs

            # Check for proper fields
            check_fields(flux_arr)

            # Check photometry residual for central cutout
            hdul = results["fit"][1]
            assert len(hdul) == 5  # model, cutout, source list, variance, and chi HDUs
            window_residual = hdul[1].data - hdul[0].data
            cutout_residual = window_residual[x_lim, y_lim]
            cut_median = np.nanmedian(cutout_residual)
            cut_std = np.nanstd(cutout_residual)
            residual_cond = ((cut_median > (0.0 - 2*cut_std))
                             & (cut_median < (0.0 + 2*cut_std)))
            assert residual_cond, "The median of the cutout residual is not within 0 +/- 2*sigma."

    def test_run_tractor_window_keep_frame(self, tractor_ps_keep_frame):
        """Test running tractor on a single window, keeping frame sources."""
        # Check for 3 sources (2 in central cutout, 1 in frame)
        assert len(tractor_ps_keep_frame) == 3

        # Check for proper fields
        check_fields(tractor_ps_keep_frame)

    def test_run_tractor_window_filter(self):
        """Test running tractor on a single window, only proceeding if 1 or
        more sources in the class filter are in the window."""
        cat = self.ps_img.catalog
        img_flags = set(itertools.chain.from_iterable(
            SourcePipelineClass(val).get_names_as_list() for val in cat["spherex_class"]))

        # SNR_CUT: not in image
        snr_config = PerformForcedPhotometryConfig(
            cut_side_length=5, frame_width=3, bkg_side_length=9, class_filter=SourcePipelineClass.SNR_CUT)
        flux_tab = setup_fixture(self.ps_img, self.ps_windows_global[6], snr_config, False)
        assert flux_tab is None
        assert "SNR_CUT" not in img_flags

        # UNDEFINED: in image, but not in this window
        no_flg_config = PerformForcedPhotometryConfig(
            cut_side_length=5, frame_width=3, bkg_side_length=9, class_filter=SourcePipelineClass.UNDEFINED)
        flux_tab = setup_fixture(self.ps_img, self.ps_windows_global[7], no_flg_config, False)
        assert flux_tab is None
        assert "UNDEFINED" in img_flags

        # COMBO FLAG: COSMOLOGY in window, SECONDARY not in image
        combo_config = PerformForcedPhotometryConfig(
            cut_side_length=5, frame_width=3, bkg_side_length=9,
            class_filter=(SourcePipelineClass.COSMOLOGY | SourcePipelineClass.SECONDARY))
        flux_tab = setup_fixture(self.ps_img, self.ps_windows_global[6], combo_config, False)
        assert flux_tab is not None
        assert "COSMOLOGY" in img_flags
        assert "SECONDARY" not in img_flags

    def test_combine_measurements(self, tractor_ps_keep_frame, tractor_ps_return_fit):
        """Test combining photometry measurements for different return
        objects."""
        combined_arr = np.zeros(5, dtype=[("SPHERExRefID", "int64"),
                                          ("ra", "float64"),
                                          ("dec", "float64"),
                                          ("x_image", "float32"),
                                          ("y_image", "float32"),
                                          ("flux", "float32"),
                                          ("flux_err", "float32"),
                                          ("flux_bkg", "float32"),
                                          ("local_bkg_flg", "bool"),
                                          ("ext_flg", "bool"),
                                          ("flags", "int64"),
                                          ("fit_ql", "float32"),
                                          ("hips6", "int32")])
        start = 0
        end = 0
        init_array = np.copy(combined_arr)

        # None return case (empty window; no sources from catalog)
        start, end = combine_measurements(combined_arr, None, start, end)
        np.testing.assert_array_equal(init_array, combined_arr)

        # Array return case (return_fit=False; default mode; adds 3 sources)
        start, end = combine_measurements(combined_arr, tractor_ps_keep_frame, start, end)
        np.testing.assert_array_equal(combined_arr[0:3], tractor_ps_keep_frame)

        # Dictionary return case (return_fit=True; adds 2 sources)
        start, end = combine_measurements(combined_arr, tractor_ps_return_fit, start, end)
        np.testing.assert_array_equal(combined_arr[3:], tractor_ps_return_fit["flux_result"])

    def test_combine_fits(self, tractor_ps_return_fit):
        """Test combining Tractor fit results for different return objects."""
        tractor_fits = {}
        # None return case (empty window; no sources from catalog)
        combined_fits = combine_fits(tractor_fits, None)
        assert len(combined_fits.keys()) == 0

        # Dictionary return case (return_fit=True)
        combined_fits = combine_fits(tractor_fits, tractor_ps_return_fit)
        assert len(combined_fits.keys()) == 1  # check that one window has been added
        assert len(list(combined_fits.values())[0]) == 5  # check that list HDUList has 5 extensions
