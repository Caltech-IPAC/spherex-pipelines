import unittest
from astropy.table import Table
from spx_utils.aggregate_gds import AggregateGDS

# Sample GDS raw data for testing
sband_raw = """2017/01/01 00:08:15.826,adstate,0,wait
2017/01/01 00:08:15.826,adraterr,0,clear
2017/01/01 00:08:15.826,adraterrx,0,0.0
2017/01/01 00:08:15.826,admsunvx,0,0.0
2017/01/01 00:08:15.828,adgydq,2,bad
2017/01/01 00:08:15.828,adst1sang,0,0.0
2017/01/01 00:08:15.828,adcsttms,0,0.0
2017/01/01 00:08:15.828,adrw1sdir,0,0.0
2017/01/01 00:08:17.828,stmeasstarmin,0,0
2017/01/01 00:08:17.828,adgpsdatst,0,0
2017/01/01 00:08:17.828,tsa1t,0,0
2017/01/01 00:08:17.828,ra_ts_v8_t,0,0"""


def sband_data_to_table(data):
    lines = data.strip().split("\n")
    timestamps, mnemonics, values1, values2 = [], [], [], []

    for line in lines:
        parts = line.split(",")
        timestamps.append(parts[0].strip())
        mnemonics.append(parts[1].strip())
        values1.append(parts[2].strip())
        values2.append(parts[3].strip())

    return Table(
        [timestamps, mnemonics, values1, values2],
        names=["Timestamp", "Mnemonic", "Value1", "Value2"]
    )


class TestAggregateGDS(unittest.TestCase):

    def setUp(self):
        self.data_table = sband_data_to_table(sband_raw)

        self.numerical_keywords = ["admsunvx", "adst1sang", "adcsttms", "adraterrx"]
        self.flag_keywords = ["adstate", "adgydq"]

        # Include numerical mnemonics too, since compute_aggregates looks up
        # Description/Type from sband_mnemonics
        self.sband_mnemonics = Table(
            [
                ["adstate", "adgydq", "admsunvx", "adst1sang", "adcsttms", "adraterrx"],
                [
                    "attitude state",
                    "gyro data",
                    "sun vector x",
                    "star sensor angle",
                    "coarse sun sensor time",
                    "attitude error x",
                ],
                [
                    "0/Wait 1/Detumble 2/AcqSun 3/Point",
                    "0/Good 1/Bad",
                    "",
                    "",
                    "",
                    "",
                ],
                ["D3", "D2", "F8", "F8", "F8", "F8"],
            ],
            names=["Mnemonic", "Description", "Value/State", "Type"]
        )

        self.aggregate_gds = AggregateGDS(
            data_table=self.data_table,
            numerical_keywords=self.numerical_keywords,
            flag_keywords=self.flag_keywords
        )

    def test_compute_aggregates(self):
        aggregated_table = self.aggregate_gds.compute_aggregates(self.sband_mnemonics)

        self.assertEqual(
            aggregated_table.colnames,
            [
                "Keyword", "Description", "First", "Last",
                "Min", "Max", "Mean", "Median", "Std",
                "Variability", "Type"
            ]
        )

        keyword_row = aggregated_table[aggregated_table["Keyword"] == "adst1sang"]
        self.assertEqual(len(keyword_row), 1)

        # First/Last are strings in the current implementation
        self.assertEqual(keyword_row["First"][0], "0.0")
        self.assertEqual(keyword_row["Last"][0], "0.0")

        self.assertEqual(keyword_row["Min"][0], 0.0)
        self.assertEqual(keyword_row["Max"][0], 0.0)
        self.assertEqual(keyword_row["Mean"][0], 0.0)
        self.assertEqual(keyword_row["Median"][0], 0.0)
        self.assertEqual(keyword_row["Std"][0], 0.0)
        self.assertEqual(keyword_row["Variability"][0], "CONSTANT")
        self.assertEqual(keyword_row["Description"][0], "star sensor angle")
        self.assertEqual(keyword_row["Type"][0], "F8")

    def test_compute_flag_aggregates(self):
        aggregated_flags = self.aggregate_gds.compute_flag_aggregates(self.sband_mnemonics)

        self.assertEqual(
            aggregated_flags.colnames,
            [
                "Keyword", "Description", "First", "Last", "Mid",
                "FlagOptions", "Type", "Bitmap", "Variability"
            ]
        )

        adstate_row = aggregated_flags[aggregated_flags["Keyword"] == "adstate"]
        self.assertEqual(len(adstate_row), 1)
        self.assertEqual(adstate_row["First"][0], "wait")
        self.assertEqual(adstate_row["Last"][0], "wait")
        self.assertEqual(adstate_row["Mid"][0], "wait")
        self.assertIn("0/wait", adstate_row["FlagOptions"][0])
        self.assertIn("3/point", adstate_row["FlagOptions"][0])
        self.assertEqual(adstate_row["Variability"][0], "CONSTANT")

        adgydq_row = aggregated_flags[aggregated_flags["Keyword"] == "adgydq"]
        self.assertEqual(len(adgydq_row), 1)
        self.assertEqual(adgydq_row["First"][0], "bad")
        self.assertEqual(adgydq_row["Last"][0], "bad")
        self.assertEqual(adgydq_row["Mid"][0], "bad")
        self.assertIn("0/good", adgydq_row["FlagOptions"][0])
        self.assertIn("1/bad", adgydq_row["FlagOptions"][0])
        self.assertEqual(adgydq_row["Variability"][0], "CONSTANT")


if __name__ == "__main__":
    unittest.main()
