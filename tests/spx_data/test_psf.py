import pytest

# Standard library imports
import numpy as np

# Related third party imports
from astropy.io import fits, registry

# Local application/library specific imports
from spx_data import PSFZoneData, relabel_buggy_psf_header_to_xfast
from spx_tests import get_sample_psf
from spx_butler.defaults.fake_products import fake_PSF, hasSPHERExSimulator


@pytest.mark.skipif(not hasSPHERExSimulator, reason="SPHEREx Simulator is not installed")
def test_psf(tmp_path, caplog):

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

            # Check header float precision is rounded to 6 decimals
            nplane = hdul[1].data.shape[0]
            for k1 in range(1, nplane + 1):
                ix = (k1 - 1) % len(psf0.x_centers)
                iy = (k1 - 1) // len(psf0.x_centers)

                assert round(float(hdul[1].header[f"XCTR_{k1}"]), 6) == round(float(psf0.x_centers[ix]), 6)
                assert round(float(hdul[1].header[f"YCTR_{k1}"]), 6) == round(float(psf0.y_centers[iy]), 6)
                assert round(float(hdul[1].header[f"XWID_{k1}"]), 6) == round(float(psf0.x_widths[ix]), 6)
                assert round(float(hdul[1].header[f"YWID_{k1}"]), 6) == round(float(psf0.y_widths[iy]), 6)

            # Test warning for y-fast-looking header (buggy header case)
            hdu = hdul[1]

            x_centers = psf0.x_centers
            y_centers = psf0.y_centers
            x_widths = psf0.x_widths
            y_widths = psf0.y_widths
            nx = len(x_centers)
            ny = len(y_centers)

            # Rewrite header cards in y-fast order, including comments and widths
            k = 1
            for ix in range(nx):
                for iy in range(ny):
                    hdu.header[f"XCTR_{k}"] = (
                        float(x_centers[ix]),
                        f"X pixel at center of zone ({ix}, {iy})"
                    )
                    hdu.header[f"YCTR_{k}"] = (
                        float(y_centers[iy]),
                        f"Y pixel at center of zone ({ix}, {iy})"
                    )
                    hdu.header[f"XWID_{k}"] = (
                        float(x_widths[ix]),
                        f"[pixel] Zone ({ix}, {iy}) width along X axis"
                    )
                    hdu.header[f"YWID_{k}"] = (
                        float(y_widths[iy]),
                        f"[pixel] Zone ({ix}, {iy}) width along Y axis"
                    )
                    k += 1

            # Ensure log warning is emitted
            caplog.clear()
            with caplog.at_level("WARNING"):
                PSFZoneData.from_hdu(hdu)

            assert any("y-fast" in rec.message for rec in caplog.records)

        # Test optional header repair utility
        buggy_path = tmp_path / "testcube_buggy.fits"
        fixed_path = tmp_path / "testcube_fixed.fits"

        # Write out a deliberately buggy file with y-fast-looking header metadata
        psf0.write(buggy_path, overwrite=True)
        with fits.open(buggy_path, mode="update") as hdul_buggy:
            hdu_buggy = hdul_buggy[1]

            x_centers = psf0.x_centers
            y_centers = psf0.y_centers
            x_widths = psf0.x_widths
            y_widths = psf0.y_widths
            nx = len(x_centers)
            ny = len(y_centers)

            k = 1
            for ix in range(nx):
                for iy in range(ny):
                    hdu_buggy.header[f"XCTR_{k}"] = (
                        float(x_centers[ix]),
                        f"X pixel at center of zone ({ix}, {iy})"
                    )
                    hdu_buggy.header[f"YCTR_{k}"] = (
                        float(y_centers[iy]),
                        f"Y pixel at center of zone ({ix}, {iy})"
                    )
                    hdu_buggy.header[f"XWID_{k}"] = (
                        float(x_widths[ix]),
                        f"[pixel] Zone ({ix}, {iy}) width along X axis"
                    )
                    hdu_buggy.header[f"YWID_{k}"] = (
                        float(y_widths[iy]),
                        f"[pixel] Zone ({ix}, {iy}) width along Y axis"
                    )
                    k += 1

            hdul_buggy.flush()

        # Repair the buggy header
        relabel_buggy_psf_header_to_xfast(buggy_path, fixed_path, overwrite=True)

        # Confirm repaired file keeps data unchanged but fixes header ordering
        with fits.open(buggy_path) as hdul_buggy, fits.open(fixed_path) as hdul_fixed:
            # Data should be identical: header-only fix
            assert np.array_equal(hdul_buggy[1].data, hdul_fixed[1].data), (
                "PSF cube data changed, but relabel_buggy_psf_header_to_xfast "
                "should only modify header metadata."
            )

            zone_xy_buggy = []
            zone_xy_fixed = []
            nplane = hdul_fixed[1].data.shape[0]

            for k1 in range(1, nplane + 1):
                zone_xy_buggy.append((
                    float(hdul_buggy[1].header[f"XCTR_{k1}"]),
                    float(hdul_buggy[1].header[f"YCTR_{k1}"]),
                ))
                zone_xy_fixed.append((
                    float(hdul_fixed[1].header[f"XCTR_{k1}"]),
                    float(hdul_fixed[1].header[f"YCTR_{k1}"]),
                ))

            _, _, _, buggy_order = PSFZoneData._detect_header_zone_order(zone_xy_buggy)
            _, _, _, fixed_order = PSFZoneData._detect_header_zone_order(zone_xy_fixed)

            assert buggy_order == "y-fast"
            assert fixed_order == "x-fast"

            # Header metadata should actually have changed
            assert zone_xy_buggy != zone_xy_fixed, (
                "Expected repaired header zone metadata to differ from buggy header metadata."
            )

            psf_fixed = PSFZoneData.from_hdu(hdul_fixed[1])

        assert psf_fixed == psf0, "Fixed-header PSF does not match original."

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
        middle_msg = (
            f"``get_blended_psf`` result for ({psf0.x_widths[0]}, {psf0.y_widths[0]}) exactly "
            f"matches PSF for zone ID 1."
        )
        assert not np.array_equal(middle_psf, psf0.data[0]), middle_msg

        # Check equality
        assert psf0_read_hdul == psf0, "Read in using ``from_hdu`` does not match."
        assert PSFZoneData.read(psf0_path) == psf0, "Read in using registry does not match."

        assert registry.get_reader("fits", PSFZoneData).__name__ == "spx_psf_reader", (
            "PSFZoneData has wrong registry for read."
        )
        assert registry.get_writer("fits", PSFZoneData).__name__ == "spx_psf_writer", (
            "PSFZoneData has wrong registry for write."
        )

    # Test x and y zones with third dimension.
    psf_zone_data = fake_PSF()  # default 11x11 zones.
    assert psf_zone_data.data.shape[0] == len(psf_zone_data.x_centers) * len(psf_zone_data.y_centers)

    # Test with different x and y zone lengths.
    psf_zone_data = fake_PSF(nbr_x_zones=8, nbr_y_zones=3)
    assert psf_zone_data.data.shape[0] == len(psf_zone_data.x_centers) * len(psf_zone_data.y_centers)

    # Check floating point zone center calculation for different (odd and even) lengths.
    x_centers = [127., 382., 637., 892., 1147., 1402., 1657., 1912.]
    y_centers = [339.5, 1019.5, 1699.5]
    assert np.array_equal(psf_zone_data.x_centers, x_centers)
    assert np.array_equal(psf_zone_data.y_centers, y_centers)
