import pytest

# Standard library imports
import numpy as np

# Related third party imports
from astropy.io import fits, registry

# Local application/library specific imports
from spx_data import PSFZoneData
from spx_tests import get_sample_psf
from spx_butler.defaults.fake_products import fake_PSF, hasSPHERExSimulator


@pytest.mark.skipif(not hasSPHERExSimulator, reason="SPHEREx Simulator is not installed")
def test_psf(tmp_path):

    for func in [get_sample_psf, fake_PSF]:
        psf0 = func()
        psf0_path = tmp_path / "testcube.fits"
        # Test writing
        psf0.write(psf0_path, overwrite=True)

        # Test getting hdu
        psf0.to_hdu()

        # Test reading in psf file
        with fits.open(psf0_path) as hdul:
            # PSF stored in second hdu
            psf0_read_hdul = PSFZoneData.from_hdu(hdul[1])

        assert np.allclose(psf0.x_centers, psf0_read_hdul.x_centers)
        assert np.allclose(psf0.y_centers, psf0_read_hdul.y_centers)
        assert np.allclose(psf0.oversampling, psf0_read_hdul.oversampling)
        assert np.allclose(psf0.data, psf0_read_hdul.data)
        assert np.allclose(psf0.zone_xy, psf0_read_hdul.zone_xy)

        # Test PSF zone retrieval for a point
        psf_LL = psf0.get_blended_psf(x=-0.5, y=-0.5)  # all 0 corner weights, except LL
        msg_LL = "``get_blended_psf`` result for (-0.5, -0.5) does not exactly match PSF for zone ID 1."
        assert np.array_equal(psf_LL, psf0.data[0]), msg_LL

        psf_UR = psf0.get_blended_psf(x=2039.5, y=2039.5)  # all 0 corner weights, except UR
        msg_UR = "``get_blended_psf`` result for (2039.5, 2039.5) does not exactly match PSF for zone ID 121."
        assert np.array_equal(psf_UR, psf0.data[120]), msg_UR

        middle_psf = psf0.get_blended_psf(x=psf0.x_widths[0], y=psf0.y_widths[0])  # all non-0 corner weights
        middle_msg = (f"``get_blended_psf`` result for ({psf0.x_widths[0]}, {psf0.y_widths[0]}) exactly "
                      f"matches PSF for zone ID 1.")
        assert not np.array_equal(middle_psf, psf0.data[0]), middle_msg

        # Check equality
        assert psf0_read_hdul == psf0, "Read in using ``from_hdu`` does not match."
        assert PSFZoneData.read(psf0_path) == psf0, "Read in using registry does not match."

        assert registry.get_reader("fits", PSFZoneData).__name__ == "spx_psf_reader", (
            "PSFZoneData has wrong registry for read.")
        assert registry.get_writer("fits", PSFZoneData).__name__ == "spx_psf_writer", (
            "PSFZoneData has wrong registry for write.")

    # Test x and y zones with third dimension.
    psf_zone_data = fake_PSF()  # default 11x11 zones.
    assert psf_zone_data.data.shape[0] == len(psf_zone_data.x_centers) * len(psf_zone_data.y_centers)
    # Test with different x and y zone lengths.
    psf_zone_data = fake_PSF(nbr_x_zones=8, nbr_y_zones=3)
    assert psf_zone_data.data.shape[0] == len(psf_zone_data.x_centers) * len(psf_zone_data.y_centers)
    # Check floating point zone center calculation for different (odd and even) lengths.
    x_centers = [128., 383., 638., 893., 1148., 1403., 1658., 1913.]
    y_centers = [340.5, 1020.5, 1700.5]
    assert np.array_equal(psf_zone_data.x_centers, x_centers)
    assert np.array_equal(psf_zone_data.y_centers, y_centers)
