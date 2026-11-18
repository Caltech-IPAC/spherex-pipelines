"""Tests for sliding window utility functions."""

# Standard library imports
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock
import contextlib
import itertools
import pytest
import re

# Related third party imports
from astropy.coordinates import SkyCoord
from astropy.table import Column
from lsst.pex.config import Config
from photutils.psf import ImagePSF
import astropy.units as u
import healpy as hp
import numpy as np

# Local application/library specific imports
from spx_butler.defaults import fake_products
from spx_data import PSFZoneData, spherex_image_reader, EPSFModels, ObservationId
from spx_tasks import PerformForcedPhotometryConfig
from spx_utils.photometry.window_utils import (
    WindowInfo, _compute_mask, midpoint, slide_window, run_tractor_window, combine_measurements, combine_fits)
from spx_utils.photometry.tractor_utils import (
    PSFZoneDataTractor, EPSFTractor, _subimg_center, MJysr_to_uJy)
from spx_utils import SourcePipelineClass

test_root = Path(__file__).parent.parent


# Helper functions ============================================================
def get_sample_file(type: str) -> str:
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
    x_size = window_params.window_x2 - window_params.window_x1
    y_size = window_params.window_y2 - window_params.window_y1
    return (x_size, y_size)


def check_fields(flux_arr: np.ndarray):
    """Check for missing photometry measurement fields.

    Parameters
    ----------
    flux_arr : `numpy.ndarray`
        Structured array of photometry results.

    """
    expected_fields = ["SPHERExRefID", "ra", "dec", "x_image", "y_image", "mjd", "lambda", "lambda_width",
                       "flux", "flux_err", "flux_bkg", "ext_flg", "flags", "fit_ql", "deep_flg", "lvf_id",
                       "det_id", "hips6"]
    for i, field in enumerate(expected_fields):
        assert field == flux_arr.dtype.names[i], \
            f"Field '{field}' not found in combined photometry array or at expected column index {i}."


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
    read_cols = list(config.cat_phot_cols) + list(config.pass_thru_cols) if \
        config.merge_srccat else list(config.cat_phot_cols)
    srccat = srccat[read_cols]
    hips_col_name = f"hips{config.hips_order}"
    col = Column(data=hp.ang2pix(2 ** config.hips_order, srccat["ra"], srccat["dec"], nest=True,
                                 lonlat=True),
                 name=hips_col_name, unit="")  # leaving unit None, makes "deg" unit
    srccat.add_column(col)
    cat_coords = SkyCoord(ra=srccat["ra"], dec=srccat["dec"], frame="icrs", unit=u.deg)
    src_xdet, src_ydet = (img.wcs).world_to_pixel(cat_coords)
    detector = img.getArrayId()
    if config.psf_extension == "PSF":
        psf = PSFZoneData.from_hdu(img.layers["PSF"])
    elif config.psf_extension == "EPSF":
        psf = EPSFModels.from_hdu(img.layers["EPSF"])
    special_ids = None
    exp_id_int = ObservationId(img.header["OBSID"]).pointing_id + detector

    # Perform fitting
    results = run_tractor_window(img.data, img.unit, img.wcs, img.uncertainty, img.flags, window_params,
                                 src_xdet, src_ydet, srccat, cat_coords, config, psf, detector,
                                 keep_frame_sources, special_ids, exp_id_int)

    return results


# Fixtures ===================================================================
@pytest.fixture
def psf_obj(request):
    """Fixture to provide a fake PSF or EPSF for testing.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `dict` of
        `str`. Key "psf_type" maps to the type of PSF to create ("psf", "epsf",
        or "None"). Key "shape" maps to a `list` of `int` indicating the shape
        to use for the mock PSF.

    Returns
    -------
    fake_psf : `spx_data.PSFZoneData` or `photutils.psf.ImagePSF`, optional
        The fake PSF or EPSF object.
    """
    param_dict = request.param
    psf_type = param_dict["psf_type"]
    shape = param_dict.get("shape", None)
    if psf_type == "psf":
        fake_psf = fake_products.fake_PSF(detector=1)
        if shape:
            fake_psf = MagicMock(spec=PSFZoneData)
            fake_psf.oversampling = 1  # use test-specified shape
            fake_psf.get_blended_psf.return_value = np.ones(shape)
    elif psf_type == "epsf":
        fake_epsf = EPSFModels.from_hdu(fake_products.fake_epsf(detector=1).layers["EPSF"])
        fake_psf = fake_epsf.model_for_xy(x=1, y=1)
        if shape:
            fake_psf = MagicMock(spec=ImagePSF)
            fake_psf.oversampling = 1  # use test-specified shape
            fake_psf.data = np.ones(shape)
    elif psf_type == "None":
        return None

    return fake_psf


# Test cases ==================================================================
class TestWindowUtils():
    """Test sliding window utility functions."""

    base_cutout_config = PerformForcedPhotometryConfig(cut_side_length=5, frame_width=3,
                                                       class_filter=SourcePipelineClass.ALL)
    return_fit_config = PerformForcedPhotometryConfig(cut_side_length=5, frame_width=3, return_fit=True,
                                                      class_filter=SourcePipelineClass.ALL,
                                                      fit_shapes=False, psf_extension="PSF")

    ps_img = spherex_image_reader(get_sample_file("point source"))
    ps_windows = slide_window(ps_img, base_cutout_config)

    gal_img = spherex_image_reader(get_sample_file("galaxy"))
    gal_windows = slide_window(gal_img, base_cutout_config)

    @pytest.fixture(scope="class")
    def tractor_ps_return_fit(self):
        flux_dict = setup_fixture(self.ps_img, self.ps_windows[6], self.return_fit_config, False)  # PSF
        return flux_dict

    @pytest.fixture(scope="class")
    def tractor_ps_keep_frame(self):
        flux_tab = setup_fixture(self.ps_img, self.ps_windows[6], self.base_cutout_config, True)  # EPSF
        return flux_tab

    @pytest.fixture(scope="class")
    def tractor_gal_return_fit(self):
        flux_dict = setup_fixture(self.gal_img, self.gal_windows[3], self.return_fit_config, False)  # PSF
        return flux_dict

    @pytest.mark.parametrize("src_ximg, src_yimg, x1, x2, y1, y2, expected", [
        (np.array([1, 2, 3], dtype=float), np.array([1, 2, 3], dtype=float), 0, 7, 0, 7,
         np.array([True, True, True])),  # inside window
        (np.array([8, 9, 10], dtype=float), np.array([8, 9, 10], dtype=float), 0, 7, 0, 7,
         np.array([False, False, False]))  # outside window
    ])
    def test_compute_mask(self, src_ximg: np.ndarray, src_yimg: np.ndarray, x1: float, x2: float,
                          y1: float, y2: float, expected: np.ndarray):
        """Test that mask is computed correctly for a window."""
        mask = _compute_mask(src_ximg, src_yimg, x1, x2, y1, y2)
        np.testing.assert_array_equal(mask, expected, err_msg=f"Expected mask {expected}, got {mask}.")

    def test_midpoint(self):
        line = [(0, 0), (5, 5)]
        expected = (2.5, 2.5)
        result = midpoint(x1=line[0][0], x2=line[1][0], y1=line[0][1], y2=line[1][1])
        assert result == expected

    @pytest.mark.parametrize("window_idx, exp_shape, type", [
        # For cut_side_length=5 and sample image of 15x15, we expect 3 windows
        # per row. Window size will vary based on position within the image.
        (0, (8, 8), "corner"), (1, (11, 8), "top edge"), (3, (8, 11), "left edge"), (4, (11, 11), "inner")])
    def test_slide_window(self, window_idx: int, exp_shape: tuple[int, int], type: str):
        """Test sliding window for global background case."""
        assert get_window_shape(self.ps_windows[window_idx]) == exp_shape, \
            f"Sliding window is not calculating {type} correctly."

    def test_run_tractor_window_no_sources(self):
        """Test running tractor on a single window with no sources from the
        catalog.
        """
        # Set up empty window parameters (ensuring no sources fall within)
        outside_params = WindowInfo(3, 3, 20, 25, 20, 25, 17, 28, 17, 28)  # outside the 15x15 image
        flux_tab = setup_fixture(self.ps_img, outside_params, self.base_cutout_config, False)
        assert flux_tab is None, f"Expected no results for window with no sources, got:\n{flux_tab}"

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

    @pytest.mark.parametrize("class_filter, window_idx, flag_img_window", [
        (SourcePipelineClass.SNR_CUT, 6, [("SNR_CUT", False, False)]),  # SNR_CUT in neither image nor window
        (SourcePipelineClass.UNDEFINED, 7, [("UNDEFINED", True, False)]),  # UNDEFINED only in image
        (SourcePipelineClass.COSMOLOGY | SourcePipelineClass.SECONDARY, 6,
         [("COSMOLOGY", True, True), ("SECONDARY", False, True)])  # COSMOLOGY in window/SECONDARY in neither
    ])
    def test_run_tractor_window_class_filter(self, class_filter, window_idx, flag_img_window):
        """Test running tractor on a single window, only proceeding if 1 or
        more sources in the class filter are in the window.
        """
        # Record flags in image and attempt photometry
        cat = self.ps_img.catalog
        img_flags = set(itertools.chain.from_iterable(
            SourcePipelineClass(val).get_names_as_list() for val in cat["spherex_class"]))
        config = PerformForcedPhotometryConfig(cut_side_length=5, frame_width=3, class_filter=class_filter,
                                               pass_thru_cols=["spherex_class"], merge_srccat=True)
        flux_tab = setup_fixture(self.ps_img, self.ps_windows[window_idx], config, False)

        # Check for expected output
        for flg, in_img, in_window in flag_img_window:
            assert (flg in img_flags) == in_img, \
                f"Flag '{flg}' presence in image does not match expected value {in_img}."
        any_in_window = any(in_window for _, _, in_window in flag_img_window)
        assert (flux_tab is not None) == any_in_window, \
            f"Expected {'photometry measurements' if any_in_window else 'no results'} for flag '{flg}'."

    def test_combine_measurements(self, tractor_ps_keep_frame, tractor_ps_return_fit):
        """Test combining photometry measurements for different return
        objects, ensuring indexing is correct.
        """
        combined_arr = np.zeros(5, dtype=[("SPHERExRefID", "int64"),
                                          ("ra", "float64"),
                                          ("dec", "float64"),
                                          ("x_image", "float32"),
                                          ("y_image", "float32"),
                                          ("mjd", "float64"),
                                          ("lambda", "float32"),
                                          ("lambda_width", "float32"),
                                          ("flux", "float32"),
                                          ("flux_err", "float32"),
                                          ("flux_bkg", "float32"),
                                          ("ext_flg", "bool"),
                                          ("flags", "int64"),
                                          ("fit_ql", "float32"),
                                          ("deep_flg", "bool"),
                                          ("lvf_id", "int64"),
                                          ("det_id", "uint8"),
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

    @pytest.mark.parametrize("empty_fits, n_windows, n_hdus", [
        (True, 0, 0),  # `None` return case (empty window; no sources from catalog)
        (False, 1, 5)  # Dictionary return case (return_fit=True; adds 1 window with 5 HDUs)
    ])
    def test_combine_fits(self, empty_fits: bool, n_windows: int, n_hdus: int, tractor_ps_return_fit: dict):
        """Test combining Tractor fit results for different return objects."""
        tractor_fits = {}
        combined_fits = combine_fits(tractor_fits, None if empty_fits else tractor_ps_return_fit)
        assert len(combined_fits.keys()) == n_windows
        if n_windows > 0:
            assert len(list(combined_fits.values())[0]) == n_hdus


class TestTractorUtils():
    """Test Tractor utility functions."""

    @pytest.mark.parametrize("height, width, subimg, exp_ctrs, n_ones, n_zeros", [
        (5, 5, np.ones((3, 3)), [2.0, 2.0], 3**2, 5**2 - 3**2),  # odd-odd pad
        (3, 3, np.ones((5, 5)), [1.0, 1.0], 3**2, 0),  # odd-odd cut
        (6, 6, np.ones((3, 3)), [2.0, 2.0], 3**2, 6**2 - 3**2),  # odd-even pad
        (2, 2, np.ones((5, 5)), [1.0, 1.0], 2**2, 0),  # odd-even cut
        (4, 4, np.ones((2, 2)), [1.0, 1.0], 2**2, 4**2 - 2**2),  # even-even pad
        (2, 2, np.ones((4, 4)), [0.0, 0.0], 2**2, 0),  # even-even cut
        (5, 5, np.ones((2, 2)), [1.0, 1.0], 2**2, 5**2 - 2**2),  # even-odd pad
        (3, 3, np.ones((6, 6)), [1.0, 1.0], 3**2, 0),  # even-odd cut
        (10, 10, np.ones((10, 10)), [4.0, 4.0], 10**2, 0)  # same size
    ])
    def test_subimg_center(self, height: int, width: int, subimg: np.ndarray, exp_ctrs: list, n_ones: int,
                           n_zeros: int):
        """Test that subimage center is calculated correctly.
        """
        embedded, sub_cx, sub_cy = _subimg_center(height, width, subimg)

        # Check centers, output shape and padding/cutting
        np.testing.assert_array_equal([sub_cx, sub_cy], exp_ctrs, err_msg=f"Expected subimage center"
                                      f" {exp_ctrs}, got [{sub_cx}, {sub_cy}].")
        assert embedded.shape == (height, width), \
            f"Expected embedded image shape {(height, width)}, got {embedded.shape}."
        assert n_ones == np.sum(embedded), \
            f"Expected {n_ones} ones in embedded image, got {np.sum(embedded)}."
        assert n_zeros == np.sum(embedded == 0), \
            f"Expected {n_zeros} zeros in embedded image, got {np.sum(embedded == 0)}."

    @pytest.mark.parametrize("flux_unit, pixmap_unit, match", [
        (u.MJy, u.arcsec**2, "flux is not in MJy/sr"),  # invalid flux, valid pixel map
        (u.MJy / u.sr, u.pixel**2, "pixel map is not in arcsec^2"),  # valid flux, invalid pixel map
        (u.MJy / u.sr, u.arcsec**2, None)  # valid flux and pixel map
    ])
    def test_MJysr_to_uJy(self, flux_unit: u.Unit, pixmap_unit: u.Unit, match: str):
        """Test that specific intensity to spectral flux density conversion
        safeguards against invalid units and returns expected units.
        """
        ctx = pytest.raises(AssertionError, match=re.escape(match)) if match else contextlib.nullcontext()
        with ctx:
            flux_uJy = MJysr_to_uJy(
                np.ones(3) * flux_unit, oversample=5, pix_map=np.ones((5, 5)) * pixmap_unit,
                xpos_idx=[0, 1, 2], ypos_idx=[0, 1, 2])
        if not match:
            assert flux_uJy.unit == u.uJy, f"Expected output unit uJy, got {flux_uJy.unit}."

    @pytest.mark.parametrize("psf_obj, radius, ft_size", [
        ({"psf_type": "psf", "shape": [4, 4]}, 2.0, 4),  # size 4, closest power of 2 is 4
        ({"psf_type": "epsf", "shape": [9, 9]}, 4.5, 16)  # size 9, next highest power of 2 is 16
    ], indirect=["psf_obj"])
    @pytest.mark.optional_deps  # psf_obj builds fake_PSF()/fake_epsf(), which need the simulator
    def test_SPHERExTractorPSF(self, psf_obj: None | PSFZoneData | ImagePSF, radius: float, ft_size: int):
        """Test SPHERExTractorPSF class methods."""
        psf = PSFZoneDataTractor(psf=psf_obj, detector=1) if isinstance(psf_obj, PSFZoneData) \
            else EPSFTractor(psf=psf_obj, detector=1)
        assert psf.shape == [radius*2, radius*2], f"Expected shape [{radius*2}, {radius*2}], got {psf.shape}."
        assert psf.getRadius() == radius, f"Expected radius {radius}, got {psf.getRadius()}."
        assert psf.getFourierTransformSize(psf.radius) == ft_size, \
            f"Expected Fourier transform size {ft_size}, got {psf.getFourierTransformSize(psf.radius)}."

    @pytest.mark.parametrize("psf_obj, init_class, match", [
        ({"psf_type": "psf"}, "EPSF", "not a `photutils.psf.ImagePSF` object"),  # invalid input
        ({"psf_type": "epsf"}, "PSFZoneData", "not a `spx_data.psf.PSFZoneData` object"),  # invalid input
        ({"psf_type": "psf"}, "PSFZoneData", None),
        ({"psf_type": "epsf"}, "EPSF", None),
    ], indirect=["psf_obj"])
    @pytest.mark.optional_deps  # psf_obj builds fake_PSF()/fake_epsf(), which need the simulator
    def test_tractor_psf_children(self, psf_obj: PSFZoneData | ImagePSF,
                                  init_class: Literal["PSFZoneData", "ImagePSF"], match: str):
        """Test EPSFTractor and PSFZoneDataTractor classes."""
        # Safeguard against invalid PSF input objects
        ctx = pytest.raises(TypeError, match=re.escape(match)) if match else contextlib.nullcontext()
        with ctx:
            if init_class == "PSFZoneData":
                psf = PSFZoneDataTractor(psf=psf_obj, detector=1)
            elif init_class == "EPSF":
                psf = EPSFTractor(psf=psf_obj, detector=1)

        if match is None:
            # Return expected values for valid inputs
            copy = psf.copy()
            assert (copy is not psf) and (copy.shape == psf.shape), \
                "Expected shared attributes but different objects for PSF copy."
            psf_str = (
                f"{init_class}Tractor: PSF type={type(psf_obj).__name__}, detector={psf.detector},"
                f" psf_shape={psf.shape}, radius={psf.radius}"
                + (", xshift=0, yshift=0" if init_class == "PSFZoneData" else ""))
            assert psf.__repr__() == psf.__str__() == psf_str, "String representation mismatch."

            # Raise errors for unsupported parameters
            with pytest.raises(NotImplementedError, match=re.escape("`radius` not supported yet")):
                psf.getPointSourcePatch(1.0, 1.0, radius=2.0)
            with pytest.raises(NotImplementedError, match=re.escape("`modelMask` not supported yet")):
                psf.getPointSourcePatch(1.0, 1.0, modelMask=np.ones((3, 3), dtype=bool))
