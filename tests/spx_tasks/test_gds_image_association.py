# Standard library imports
import unittest
from unittest.mock import MagicMock

# Related third party imports
from astropy.table import Table
from astropy.io.fits import Header
# import numpy as np
# from datetime import datetime, timedelta
# from random import uniform, randint  # choice

# Local application/library specific imports
from spx_tasks.gds_image_association import GdsImageAssociationTask, GdsImageAssociationConfig
from spx_data import SPHERExImage, PlannedObservation
from spx_butler.defaults.fake_products import sband_subset_table, sgt_table


sband_raw = """2017/01/01 00:08:15.826,adstate,0,wait
2017/01/01 00:08:15.826,adraterr,0,clear
2017/01/01 00:08:15.826,adraterrx,0,0.0
2017/01/01 00:08:15.826,admsunvx,0,0.0
2017/01/01 00:08:15.826,adbfaq1,0,0.0
2017/01/01 00:08:15.828,adgydq,2,bad
2017/01/01 00:08:15.828,adst1sang,0,0.0
2017/01/01 00:08:15.828,adcsttms,0,0.0
2017/01/01 00:08:15.828,adrw1sdir,0,0.0
2017/01/01 00:08:15.828,adstcnt,0,0
2017/01/01 00:08:15.828,stmeasstarmin,0,0
2017/01/01 00:08:15.828,adgpsdatst,0,0
2017/01/01 00:08:15.828,tsa1t,0,0
2017/01/01 00:08:15.828,ra_ts_v8_t,0,0
2017/01/01 00:08:16.826,adstate,0,wait
2017/01/01 00:08:16.826,adatterr,0,clear
2017/01/01 00:08:16.826,adraterrx,0,0.0
2017/01/01 00:08:16.826,admsunvx,0,0.0
2017/01/01 00:08:16.826,adbfaq1,0,0.0
2017/01/01 00:08:16.828,adgydq,2,bad
2017/01/01 00:08:16.828,adst1sang,0,0.0
2017/01/01 00:08:16.828,adcsttms,0,0.0
2017/01/01 00:08:16.828,adrw1sdir,0,0.0
2017/01/01 00:08:16.828,adstcnt,0,0
2017/01/01 00:08:16.828,stmeasstarmin,0,0
2017/01/01 00:08:16.828,adgpsdatst,0,0
2017/01/01 00:08:16.828,tsa1t,0,0
2017/01/01 00:08:16.828,ra_ts_v8_t,0,0
2017/01/01 00:08:17.826,adstate,0,wait
2017/01/01 00:08:17.826,adatterr,0,clear
2017/01/01 00:08:17.826,adraterrx,0,0.0
2017/01/01 00:08:17.826,admsunvx,0,0.0
2017/01/01 00:08:17.826,adbfaq1,0,0.0
2017/01/01 00:08:17.828,adgydq,2,bad
2017/01/01 00:08:17.828,adst1sang,0,0.0
2017/01/01 00:08:17.828,adcsttms,0,0.0
2017/01/01 00:08:17.828,adrw1sdir,0,0.0
2017/01/01 00:08:17.828,adstcnt,0,0
2017/01/01 00:08:17.828,stmeasstarmin,0,0
2017/01/01 00:08:17.828,adgpsdatst,0,0
2017/01/01 00:08:17.828,tsa1t,0,0
2017/01/01 00:08:17.828,ra_ts_v8_t,0,0"""


def sband_data_to_table(data):
    """
    Parse the sample sband data into an Astropy Table.

    Parameters
    ----------
    data : `str`
        Multiline string containing the data.

    Returns
    -------
    table : `astropy.table.Table`
        An Astropy Table containing the parsed data.
    """
    # Split the data into lines and process each line
    lines = data.strip().split("\n")

    # Parse the lines into columns
    timestamps = []
    mnemonics = []
    values1 = []
    values2 = []

    for line in lines:
        parts = line.split(",")
        timestamps.append(parts[0].strip())
        mnemonics.append(parts[1].strip())
        values1.append(parts[2].strip())
        values2.append(parts[3].strip())

    # Create the table
    table = Table(
        [timestamps, mnemonics, values1, values2],
        names=("Timestamp", "Mnemonic", "Value1", "Value2"),
    )

    return table


class TestGdsImageAssociationTask(unittest.TestCase):

    def test_run(self):
        # GDS data example
        sband_data = sband_data_to_table(data=sband_raw)
        print(sband_data)
        # List of sband items
        sband_mnemonics = sband_subset_table()
        # Ground track data
        sgt = sgt_table()
        # Create a mock SPHERExImage
        mock_spximg = MagicMock(spec=SPHERExImage)
        mock_spximg.header = Header()
        mock_spximg.header["DATE-AVG"] = "2025-03-27T17:08:47.472"

        # Test combined flags to include both SURVEY and DEEPFIELD
        combined_flags = 0b0000001 | 0b0000010

        # Create a mock PlannedObservation JSON
        planned_observation_data = {
            "ra": "153.07868515243598 deg",
            "dec": "28.562713618791285 deg",
            "pa_eq": "-21.684687729651454 deg",
            "timespan": [1766849941957500000, 1766850058609900000],
            "obs_id": "2025W52_2W_0567_1_4",
            "target_id": 8135,
            "flags": combined_flags,
            "quaternions": {
                "q1": 0.34431247385155067,
                "q2": -0.3773437051276251,
                "q3": -0.8042093815951339,
                "q4": 0.3038221833657866
            },
            "lat_ecl": 16.288000000000004,
            "lon_ecl": 144.6728971962617,
            "pa_ecl": 0.0,
            "zenith": [-6113.994234204474, -374.4457305876063, 3444.001951890787],
            "moon": [381986.0973236787, 18834.07077098724, 18824.84796546748],
            "sun": [14821425.774413032, -134300387.24309325, -58220336.38516569],
            "vel": [-26.52530361322105, -1.4256177563452266, 5.237193739959988],
            "in_sband": False
        }

        # Create PlannedObservation object
        planned_observation = PlannedObservation(**planned_observation_data)

        # Initialize task config and task
        config = GdsImageAssociationConfig()
        config.useSband = True
        config.useSgt = True
        config.useSpk = False
        task = GdsImageAssociationTask(config=config)

        # Run the task
        result = task.run(mock_spximg, planned_observation, sgt, sband_data, None, sband_mnemonics)

        # Check that the output SPHEREx image contains the aggregated GDS table
        self.assertTrue(hasattr(result.spximg_out, 'gds'))
        gds_output_table = result.spximg_out.gds

        # Verify the structure of the output table
        self.assertEqual(gds_output_table.colnames,
                         ['Keyword', 'Description',
                          'First', 'Last', 'Mid',
                          'Min', 'Max', 'Median', 'Mean',
                          'Bitmap', 'FlagOptions', 'Type'])

        # Verify lon, lat, SAA keywords correspond to expected values from
        # fake table
        img_out_hdr = result.spximg_out.header
        self.assertEqual(img_out_hdr['HIERARCH SGT_LAT_MIDPT'], -68.724732)
        self.assertEqual(img_out_hdr['HIERARCH SGT_LON_MIDPT'], 351.684482)
        self.assertEqual(img_out_hdr['HIERARCH SGT_MIN_GCD2SAA'], 16.359751)

    def test_populate_position_velocity_headers(self):
        from astropy.time import Time
        from astropy.table import Table

        # Mock SPHERExImage
        mock_spximg = MagicMock(spec=SPHERExImage)
        mock_spximg.header = Header()

        # ISO timestamp
        mock_spximg.header["DATE-AVG"] = "2025-03-27T17:08:47.472"
        mock_spximg.getExposureStartTime.return_value = Time("2025-03-27T17:08:47.472").jd
        mock_spximg.getExposureEndTime.return_value = Time("2025-03-27T17:08:52.472").jd
        mock_spximg.getExposureDuration.return_value = 5.0 / 86400.0  # 5 seconds in days

        # Minimal SPK table
        spk_table = Table()
        spk_table["time"] = [Time("2025-03-27T17:08:47.472").to_datetime()]
        spk_table["position"] = [[1234.0, 5678.0, 91011.0]]
        spk_table["velocity"] = [[1.1, 2.2, 3.3]]

        # Config and run task
        config = GdsImageAssociationConfig()
        config.useSpk = True
        task = GdsImageAssociationTask(config=config)
        result = task.run(mock_spximg, None, None, None, spk_table, None)

        # Validate headers
        hdr = result.spximg_out.header
        self.assertEqual(hdr["X_SC"], 1234.0)
        self.assertEqual(hdr["Y_SC"], 5678.0)
        self.assertEqual(hdr["Z_SC"], 91011.0)
        self.assertEqual(hdr["VX_SC"], 1.1)
        self.assertEqual(hdr["VY_SC"], 2.2)
        self.assertEqual(hdr["VZ_SC"], 3.3)
        self.assertEqual(hdr["HIERARCH XYZ_SC_SYSTEM"], "GEOCENTER")
        self.assertEqual(hdr["HIERARCH VXYZ_SC_SYSTEM"], "GEOCENTER")
        self.assertEqual(hdr["PLEPHEM"], "DE440")


if __name__ == '__main__':
    unittest.main()
