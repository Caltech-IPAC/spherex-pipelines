import unittest
from unittest.mock import patch
import numpy as np
from pathlib import Path
from spx_utils.psf.psf_plots import (
    plot_mult_stacks,
    plot_decon_set,
    plot_zones,
    plot_psf,
)


class TestPlottingFunctions(unittest.TestCase):
    def setUp(self):
        """Set up sample data for testing."""
        self.stack = np.random.rand(100, 100)
        self.stack5 = np.random.rand(100, 100)
        self.stack50 = np.random.rand(100, 100)
        self.stack100 = np.random.rand(100, 100)
        self.num_sources = 500
        self.stack_delete_array = np.zeros(500)
        self.expr = "example_expr"
        self.band = "example_band"
        self.bin = "example_bin"
        self.plot_path = "plots/"
        self.obsid = "example_obsid"

        self.Deconvolved_Stack = np.random.rand(100, 100)
        self.psf_pix_padded = np.random.rand(100, 100)
        self.rld_decon = np.random.rand(100, 100)

        self.spximg = np.random.rand(2048, 2048)
        self.binedge_x = [512, 1024, 1536]
        self.binedge_y = [512, 1024, 1536]

        self.psf_array = np.random.rand(16, 32, 32)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_mult_stacks(self, mock_savefig, mock_show):
        """Test plot_mult_stacks function."""
        plot_mult_stacks(
            self.stack,
            self.stack5,
            self.stack50,
            self.stack100,
            self.num_sources,
            self.stack_delete_array,
            self.plot_path,
            1,
            1,
        )

        # Ensure plots are saved and shown
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_decon_set(self, mock_savefig, mock_show):
        """Test plot_decon_set function."""
        plot_decon_set(
            self.stack,
            self.Deconvolved_Stack,
            10,
            self.psf_pix_padded,
            self.rld_decon,
            self.plot_path,
            1,
            1,
            1.,
            1.,
            self.obsid,
        )

        # Ensure plots are saved and shown
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_zones(self, mock_savefig, mock_show):
        """Test plot_zones function."""
        plot_zones(self.spximg, self.binedge_x, self.binedge_y, self.plot_path)

        # Ensure plots are saved and shown
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_psf(self, mock_savefig, mock_show):
        """Test plot_psf function."""
        output_dir = Path("output/")
        output_dir.mkdir(exist_ok=True)

        plot_psf(self.psf_array, str(output_dir), self.obsid)

        # Ensure plots are saved and shown
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()


if __name__ == "__main__":
    unittest.main()
