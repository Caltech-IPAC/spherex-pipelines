# Standard library imports
import contextlib
import pytest
import re

# Related third-party imports
import numpy as np
from photutils.psf import ImagePSF

# Local application/library specific imports
from spx_butler.defaults import fake_products
from spx_data.epsf import EPSFModels
from spx_data.spx_param import get_npix_x_active, get_npix_y_active

# The psf_hdu fixture builds fake_PSF()/fake_epsf(), which need
# SPHEREx_InstrumentSimulator (spx_butler/defaults/fake_products.py).
pytestmark = pytest.mark.optional_deps


# Fixtures ====================================================================
@pytest.fixture
def psf_hdu(request):
    hdu_type = request.param
    if hdu_type == "img":
        return fake_products.fake_PSF(detector=1).to_hdu()
    elif hdu_type == "bin":
        return fake_products.fake_epsf(detector=1).layers["EPSF"]
    else:
        raise ValueError(f"Unknown HDU type '{hdu_type}' for test fixture.")


# Test cases ==================================================================
class TestEPSFModels():
    """Test `EPSFModels` class and associated methods."""

    @pytest.mark.parametrize("psf_hdu, ext_name, rm_kws, rm_flds, match", [
        ("img", None, None, None, "Unexpected HDU type for EPSF layer."),
        ("bin", "PSF", None, None, "Expected EXTNAME 'EPSF', got"),
        ("bin", None, ["OVSMPY"], None, "Missing required keyword 'OVSMPY'"),
        ("bin", None, None, ["BINX"], "Missing required field 'BINX'")
    ], indirect=["psf_hdu"])
    def test_epsf_from_hdu(self, psf_hdu: pytest.FixtureRequest, ext_name: str, rm_kws: list[str],
                           rm_flds: list[str], match: str):
        """Test initialization from FITS HDU."""
        if ext_name is not None:
            psf_hdu.header["EXTNAME"] = ext_name
        if rm_kws is not None:
            for kw in rm_kws:
                del psf_hdu.header[kw]
        if rm_flds is not None:
            for fld in rm_flds:
                psf_hdu.data = np.lib.recfunctions.drop_fields(psf_hdu.data, fld)

        with pytest.raises(ValueError, match=re.escape(match)):
            EPSFModels.from_hdu(psf_hdu)

    @pytest.mark.parametrize("psf_hdu", ["bin"], indirect=["psf_hdu"])
    def test_get_models_by_bin(self, psf_hdu: pytest.FixtureRequest):
        """Test model dictionary construction."""
        psf = EPSFModels.from_hdu(psf_hdu)
        assert isinstance(psf.epsf_models_by_bin, dict)
        assert len(psf.epsf_models_by_bin.keys()) == len(psf.data)  # 1 key per row in data table

    @pytest.mark.parametrize("psf_hdu, faulty_data, match", [
        ("bin", True, "Expected .* edges for x-axis bins"),
        ("bin", False, None),
    ], indirect=["psf_hdu"])
    def test_calculate_edges(self, psf_hdu: pytest.FixtureRequest, faulty_data: bool, match: str):
        """Test edge calculation."""
        psf = EPSFModels.from_hdu(psf_hdu)
        if faulty_data:
            psf.data = np.array(
                [(1.0, 0.5, 1.0, 0.5, 0, 0),  # for BINX=0, edges [0.75, 1.25] are correct
                 (1.0, 0.5, 2.0, 0.5, 1, 0)],  # for BINX=1, expect [1.25, 1.75], but trigger [0.75, 1.25]
                dtype=[("XCENTER", "f8"), ("XWIDTH", "f8"), ("YCENTER", "f8"), ("YWIDTH", "f8"),
                       ("BINX", "i4"), ("BINY", "i4")])

        ctx = pytest.raises(ValueError, match=match) if match else contextlib.nullcontext()
        with ctx:
            x_edges, y_edges = psf._calculate_edges()

        if not match:
            assert min(x_edges) == min(y_edges) == -0.5
            assert max(x_edges) == max(y_edges) == max(get_npix_x_active(), get_npix_y_active()) - 0.5

    @pytest.mark.parametrize("psf_hdu, xy, match, exp_bin_xy", [
        ("bin", (-4.999e-1, -4.999e-1), None, (0, 0)),  # near lower-left corner of first bin
        ("bin", (2039.4999, 2039.4999), None, (10, 10)),  # near upper-right corner of last bin
        ("bin", (300.0, 1500.0), None, (1, 8)),  # unequal X/Y to check proper axis handling
        ("bin", (-0.5, 2039.499), "outside the bounds of the EPSF grid", None),  # at X edge
        ("bin", (-4.999e-1, 2039.5), "outside the bounds of the EPSF grid", None),  # at Y edge
    ], indirect=["psf_hdu"])
    def test_model_for_xy(self, psf_hdu: pytest.FixtureRequest, xy: tuple, match: str, exp_bin_xy: tuple):
        """Test model retrieval for given (x, y) coordinates."""
        psf = EPSFModels.from_hdu(psf_hdu)
        ctx = pytest.raises(ValueError, match=re.escape(match)) if match else contextlib.nullcontext()
        with ctx:
            model = psf.model_for_xy(xy[0], xy[1])

        if not match:
            assert isinstance(model, ImagePSF)
            assert all(model.oversampling == psf.oversampling)
            np.testing.assert_array_equal(model.data, psf.epsf_models_by_bin[exp_bin_xy].data)
