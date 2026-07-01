# Standard library imports
import unittest

# Related third party imports
from astropy.table import Table
import astropy.units as u
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import UpdatePhotometricCalibrationConfig, UpdatePhotometricCalibrationTask
from spx_butler.defaults.fake_products import initial_distortion


class TestUpdatePhotometricCalibrationTask(unittest.TestCase):

    def setUp(self):
        """Set up necessary data for the test using data."""
        self.calspec_sources = Table(
            {'name': ['10lac', '109vir', '16cygb', '1732526', '1740346', '1743045',
                      '1757132', '18sco', '1802271', '1805292'],
             'ra': [339.8153291666666, 221.5621875, 295.46655416666664,
                    263.21929166666666, 265.1444958333333, 265.7686916666666,
                    269.30513749999994, 243.90529166666667, 270.6132083333333,
                    271.3719791666666],
             'dec': [39.05026944444444, 1.8928833333333333, 50.517525, 71.07864444444444,
                     65.45410277777778, 66.91712777777778,
                     67.061325, -8.36943888888889, 60.72658333333334, 64.46448055555555],
             'spectrum_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             'fnorm': [1.0] * 10})

        self.calspec_spx = Table(
            {'SOURCE_ID': ['1732526'] * 10,
             'IMAGE_ID': [488, 489, 637, 638, 639, 640, 678, 679, 680, 681],
             'X': [578.6580615951193, 578.6602645793346, 578.6748132536159,
                   578.6765787635873, 578.6780523345951, 578.6792227006126,
                   578.6782756832645, 578.6800731997719, 578.681579121012,
                   578.6827821789238],
             'Y': [1990.9768667278736, 1877.209546880451, 1763.4363913914779,
                   1649.6505260443269, 1535.857409709602, 1422.0583850804803,
                   1763.4375283966901, 1649.651658604008, 1535.8585377943568,
                   1422.0595086634203],
             'JD': [2460741.8509367486, 2460741.8523985543, 2460742.1163999364,
                    2460742.117861742, 2460742.1193235475, 2460742.1207853532,
                    2460742.1885836744, 2460742.19004548, 2460742.1915072855,
                    2460742.1929690912],
             'RA': [263.21929166666666] * 10 * u.deg,
             'DEC': [71.07864444444444] * 10 * u.deg,
             'DETECTOR': [1] * 10,
             'WAVELENGTH': [0.745808346431472, 0.7632968872519063, 0.7811972102375814,
                            0.799520365766872, 0.8182753462823886, 0.8374721119637759,
                            0.7811969980731511, 0.7995201495047414, 0.8182751258440343,
                            0.8374718872691409],
             'FLAG': [1, 1, 3, 3, 3, 3, 1, 1, 1, 1],
             'UNIQUE_ID': range(10),
             'FLUX': [28.61748213914264, 28.243972839090862, 27.240284701846914,
                      27.06880722097399, 26.511102784002436, 25.909476682269858,
                      27.201378048455474, 26.927205487693964, 26.640892799950425,
                      26.2024223433711],
             'FLUX_ERR': [0.2244092136992562, 0.21686595539683542, 0.22529672078624435,
                          0.2091622131073686, 0.2076429510571204, 0.203363178718675,
                          0.2087224389419214, 0.2043382129390983, 0.21591839521653136,
                          0.20803026463334157]})

        # Instantiate the task
        config = UpdatePhotometricCalibrationConfig()
        self.task = UpdatePhotometricCalibrationTask(config=config)

    def test_run(self):
        # Fake flats for unit test
        flat = np.ones((2040, 2040))
        flat_field = SPHERExImage(data=flat, unit=u.dimensionless_unscaled)
        flat_field.setArrayId(array_id=1)
        # Initial Distortion table
        initial_distortion_table = initial_distortion()

        # Run the task with some data
        output = self.task.run(calspec_sources=self.calspec_sources,
                               calspec_spx=self.calspec_spx,
                               ff_prev=flat_field,
                               initial_distortion_table=initial_distortion_table,
                               sip=None)

        # Get the output table
        tab = output.tab

        # Ensure the table is returned with predicted fluxes
        self.assertIn('predicted_fluxes', tab.colnames)

        # Check that predicted flux is within errors of the actual flux
        for i, row in enumerate(tab):
            flux = row['FLUX']
            flux_err = row['FLUX_ERR']
            predicted_flux = row['predicted_fluxes']

            # Check if the predicted flux is within the error range of the actual flux
            # Should it be closer in agreemeent?
            self.assertTrue(
                np.abs(predicted_flux - flux) <= (flux_err*3),
                f"Predicted flux {predicted_flux} is not within the error range of {flux} +/- {flux_err}"
            )


if __name__ == '__main__':
    unittest.main()
