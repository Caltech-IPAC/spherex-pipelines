# Standard library imports
import logging
import unittest

# Related third party imports
import numpy as np
from astropy.io import fits

# Local application/library specific imports
from spx_data import Flag, ReadoutParameters, ReadNoiseParameters
from spx_tasks import CreateVarianceMapConfig, CreateVarianceMapTask, calculate_poisson_term
from spx_tests import sample_data
from spx_utils import l1_utils


class TestCreateVarianceMap(unittest.TestCase):

    def test_CreateVarianceMap(self):
        config = CreateVarianceMapConfig()
        task = CreateVarianceMapTask(config=config)
        # Get sample Level 1b image.
        spximg = sample_data.get_sample_level1b_img(bad_flags=True)
        # Get sample gain factors for array detector 1.
        gain_factors = sample_data.get_sample_gain_factors(spximg)
        # Read noise map for detector 1 for a single read.
        read_noise = sample_data.get_sample_read_noise_map(spximg)
        # Add TCUT to header, which is required for the new late transient logic.
        spximg.header["TCUT"] = 70
        spximg.header["TSAMP"] = 1.5349
        spximg.header["SURLIM"] = 77
        spximg.header["TRANDLY"] = 7
        spximg.header["SURDLY"] = 3

        # Read noise map for detector 1 for n reads.
        # Run task.
        output = task.run(spximg,
                          gain_factors,
                          read_noise=read_noise)
        # Uncertainty shape should be in l1 format.
        assert output.spximg_out.uncertainty.array.shape == spximg.data.shape
        # Check variance propagation on scalar multiplication.
        spx_mult = output.spximg_out.multiply(3)
        assert np.allclose(spx_mult.uncertainty.array,
                           output.spximg_out.uncertainty.array * 9,
                           equal_nan=True)
        # Check variance propagation on scalar division.
        spx_divd = output.spximg_out.divide(4)
        assert np.allclose(spx_divd.uncertainty.array,
                           output.spximg_out.uncertainty.array / 16,
                           equal_nan=True)

    # A test to verify the updated transient variance calculation.
    def test_transient_cases(self):
        """Test the variance calculation for early and late transients."""
        config = CreateVarianceMapConfig()
        # Use a consistent sample time for easier manual calculation.
        task = CreateVarianceMapTask(config=config)

        spximg = sample_data.get_sample_level1b_img()
        gain_factors = sample_data.get_sample_gain_factors(spximg)
        spximg.header["TCUT"] = 70
        spximg.header["DETECTOR"] = 1
        spximg.header["TSAMP"] = 1
        spximg.header["SURLIM"] = 77
        spximg.header["TRANDLY"] = 7
        spximg.header["SURDLY"] = 3

        # Create a predictable read noise map with the required extensions.
        read_noise_val = 20.0
        rp = ReadoutParameters.from_metadata(spximg.meta)
        noise_shape = (rp.Jactive + rp.jref, rp.Kactive + rp.kref)
        noise_map = np.full(noise_shape, read_noise_val, dtype=np.float32)

        # Create the ReadNoiseParameters object with both required extensions.
        primary_header = fits.Header({'DETECTOR': 1})
        fits_content = [
            fits.PrimaryHDU(header=primary_header),
            fits.ImageHDU(data=noise_map, header=fits.Header({'EXTNAME': ReadNoiseParameters.single})),
            fits.ImageHDU(data=noise_map, header=fits.Header({'EXTNAME': ReadNoiseParameters.full}))
        ]
        read_noise = ReadNoiseParameters(data=fits.HDUList(fits_content))

        # Set specific pixel flags and data for testing
        rp = ReadoutParameters.from_metadata(spximg.meta)
        actv_ref_area = l1_utils.active_area_range(spximg, include_refs=True)

        # Define pixel coordinates within the active+ref area
        early_transient_loc = (10, 10)
        late_transient_loc = (20, 20)

        # Set flags
        spximg.flags[actv_ref_area][early_transient_loc] = Flag.TRANSIENT | Flag.SUR_ERROR
        spximg.flags[actv_ref_area][late_transient_loc] = Flag.TRANSIENT

        # Set data
        flux = 50.0
        spximg.data[actv_ref_area][early_transient_loc] = flux
        spximg.data[actv_ref_area][late_transient_loc] = flux

        # Run task
        output = task.run(spximg, gain_factors, read_noise=read_noise)
        variance_map = output.spximg_out.uncertainty.array[actv_ref_area]

        # --- Calculate expected values ---
        # Shared values
        N_LAB = 71.0
        elec_var_lab = read_noise_val
        t_sample = spximg.header["TSAMP"]

        # 1. Early transient (N=4)
        n_early = float(spximg.meta["TRANDLY"] - spximg.meta["SURDLY"])
        t_intg_early = n_early * t_sample  # Assuming t_sample is defined in your test
        elec_var_early = elec_var_lab * (t_intg_early**2) * (N_LAB / n_early)
        poisson_early = calculate_poisson_term(n=int(n_early))
        expected_var_early = (elec_var_early + poisson_early * flux * t_intg_early) / t_intg_early**2

        # 2. Late transient (N = TCUT - 2 = 68)
        n_late = float(spximg.header["TCUT"] - (spximg.meta["SURDLY"] - 1))
        t_intg_late = n_late * t_sample  # Assuming t_sample is defined in your test
        elec_var_late = elec_var_lab * (t_intg_late**2) * (N_LAB / n_late)
        poisson_late = calculate_poisson_term(n=int(n_late))
        expected_var_late = (elec_var_late + poisson_late * flux * t_intg_late) / t_intg_late**2

        # Assert correctness
        self.assertAlmostEqual(variance_map[early_transient_loc], expected_var_early, places=5)
        self.assertAlmostEqual(variance_map[late_transient_loc], expected_var_late, places=5)

    def test_calculate_poisson_term(self):
        with self.assertRaises(TypeError):
            calculate_poisson_term(n="invalid")
        with self.assertRaises(ValueError):
            calculate_poisson_term(n=1)

    def test_analyze_reference_pixels(self):
        """Test reference pixel analysis for statistical consistency."""
        config = CreateVarianceMapConfig()
        task = CreateVarianceMapTask(config=config)

        # Create a mock SPHEREx image
        spximg = sample_data.get_sample_level1b_img(bad_flags=True)

        # Simulate a variance map with noise
        variance = np.full(spximg.data.shape, 1.0)

        # Set reference pixel flags
        ref_pix_indices = np.random.choice(
            np.arange(spximg.data.size), size=100, replace=False
        )
        spximg.flags.flat[ref_pix_indices] |= Flag.REFERENCE

        # Modify reference pixel values to be slightly off-zero to trigger warnings
        spximg.data.flat[ref_pix_indices] += 0.002  # Slight deviation from zero

        # Capture logs to check warning messages
        with self.assertLogs(task.log.name, level=logging.INFO) as cm:
            task.analyze_reference_pixels(spximg, variance, plot=False)

        # Ensure median pixel check is performed
        self.assertTrue(any("Reference pixels median" in msg for msg in cm.output))

        # Ensure reference pixel standard deviation check is performed
        self.assertTrue(any("Reference pixel std dev" in msg for msg in cm.output))


if __name__ == '__main__':
    unittest.main()
