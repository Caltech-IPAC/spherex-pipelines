
__all__ = ["EPSFModels"]

# Standard library imports
from typing import Tuple

# Related third-party imports
from astropy.io import fits
from photutils.psf import ImagePSF
import numpy as np


class EPSFModels():
    """EPSF cube containing multiple EPSF models for different bins across the
    detector.

    Parameters
    ----------
    data: `numpy.record`
        Structured array containing the EPSF models and associated metadata
        for each bin.
    oversampling: `tuple` of `int`
        Tuple specifying the oversampling factor in the y and x directions,
        respectively, for the EPSF models.

    Notes
    -----
    The bin edge calculation and model lookup use 0-based NumPy indexing,
    with detector pixel coordinates ranging from -0.5 to (N - 0.5) for an
    N-pixel-wide axis.
    """

    def __init__(self, data: np.record, oversampling: Tuple[int, int]):
        self.data = data
        self.oversampling = oversampling
        self.epsf_models_by_bin = self._get_models_by_bin()
        self.x_edges, self.y_edges = self._calculate_edges()

    @classmethod
    def from_hdu(cls, hdu: fits.BinTableHDU):
        """Create an EPSF cube from a FITS HDU.

        Parameters
        ----------
        hdu: `astropy.io.fits.BinTableHDU`
            FITS HDU containing the EPSF.

        Returns
        -------
        `spx_data.epsf.EPSFModels`
            EPSF cube containing the EPSF models and lookup information for
            each bin.
        """
        # Validate HDU, header keywords, and binary table fields
        if not isinstance(hdu, fits.BinTableHDU):
            raise ValueError("Unexpected HDU type for EPSF layer.")

        header = hdu.header
        if header.get("EXTNAME") != "EPSF":
            raise ValueError(f"Expected EXTNAME 'EPSF', got {header.get('EXTNAME')}.")

        req_kws = ["OVSMPY", "OVSMPX"]
        for kw in req_kws:
            if kw not in header:
                raise ValueError(f"Missing required keyword '{kw}' in EPSF HDU header.")
        oversampling = (header["OVSMPY"], header["OVSMPX"])  # `ImagePSF` expects (y, x) order

        data = hdu.data
        req_fields = ["BINX", "BINY", "XCENTER", "YCENTER", "XWIDTH", "YWIDTH", "EPSF"]
        for fld in req_fields:
            if fld not in data.dtype.names:
                raise ValueError(f"Missing required field '{fld}' in EPSF HDU data.")

        return cls(data=data, oversampling=oversampling)

    def _get_models_by_bin(self):
        """Create a lookup dictionary mapping bin indices (X, Y) to their
        corresponding EPSF model.

        Returns
        -------
        epsf_models_by_bin : `dict`
            Dictionary mapping tuple keys (BINX, BINY) to
            `photutils.psf.ImagePSF` values.
        """
        # Initialize lookup dictionary
        epsf_models_by_bin = {}
        for row in self.data:
            epsf_models_by_bin[(row["BINX"].item(), row["BINY"].item())] = ImagePSF(
                data=row["EPSF"], oversampling=self.oversampling)
        return epsf_models_by_bin

    def _calculate_edges(self):
        """Derive the bin edges based on the centers and widths without
        assuming uniform bin width or grid shape.

        Returns
        -------
        x_edges, y_edges : `numpy.ndarray`
            Arrays containing the unique zero-based detector pixel bin edges
            for the x and y axes, respectively.
        """
        # Derive the bin edges from the centers and widths
        x_left = self.data["XCENTER"] - self.data["XWIDTH"] / 2.
        x_right = self.data["XCENTER"] + self.data["XWIDTH"] / 2.
        y_bottom = self.data["YCENTER"] - self.data["YWIDTH"] / 2.
        y_top = self.data["YCENTER"] + self.data["YWIDTH"] / 2.
        # NOTE: We round the edges to 2 decimals to avoid duplicates
        #       arising from floating point precision issues.
        x_edges = np.unique(np.concatenate([x_left, x_right]).round(decimals=2), sorted=True)
        y_edges = np.unique(np.concatenate([y_bottom, y_top]).round(decimals=2), sorted=True)

        # Verify number of bin edges based on expected number of bins
        for edges, bins, axis in zip(
            [x_edges, y_edges], [self.data["BINX"], self.data["BINY"]], ["x", "y"]
        ):
            if len(edges) != len(np.unique(bins)) + 1:
                raise ValueError(
                    f"Expected {len(np.unique(bins)) + 1} edges for {axis}-axis bins, got {len(edges)}.")

        return x_edges, y_edges

    def model_for_xy(self, x: float, y: float) -> ImagePSF:
        """Fetch the ePSF model for a given detector pixel position.

        Parameters
        ----------
        x, y : `float`
            Zero-based detector pixel coordinates.

        Returns
        -------
        model : `photutils.psf.ImagePSF`
            The ePSF model for the nearest bin.
        """
        if not (self.x_edges[0] < x < self.x_edges[-1]
                and self.y_edges[0] < y < self.y_edges[-1]):
            raise ValueError(f"Position ({x}, {y}) is outside the bounds of the EPSF grid.")
        x_bin = np.searchsorted(self.x_edges, x, side="right") - 1
        y_bin = np.searchsorted(self.y_edges, y, side="right") - 1
        model = self.epsf_models_by_bin.get((x_bin, y_bin))
        return model
