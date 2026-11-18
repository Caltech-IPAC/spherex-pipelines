__all__ = ["PSFZoneData", "spx_psf_reader", "spx_psf_writer", "relabel_buggy_psf_header_to_xfast"]

# Standard library imports
import copy
import logging
from typing import TypeVar, Union, Any
from pathlib import Path
import re

# Related third party imports
import astropy.units as u
from astropy.io import fits, registry
from astropy import wcs
from astropy.nddata import NDIOMixin
from numpy.typing import ArrayLike
import numba
import numpy as np

from .spx_param import get_pix_scale

# Local application/library specific imports

TPSFZoneData = TypeVar("TPSFZoneData", bound="PSFZoneData")


def centers_to_edges(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Given monotonically increasing centers, return length N+1 edges.
    Works for non-uniform spacing.
    """
    c = np.asarray(centers, dtype=float)
    if c.ndim != 1 or len(c) == 0:
        raise ValueError("centers must be a non-empty 1D array")
    if len(c) == 1:
        return np.array([lo, hi], dtype=float)

    if not np.all(np.diff(c) > 0):
        raise ValueError("centers must be strictly increasing")

    mids = 0.5 * (c[:-1] + c[1:])          # length N-1
    edges = np.empty(len(c) + 1, dtype=float)
    edges[0] = lo
    edges[1:-1] = mids
    edges[-1] = hi
    return edges


def centers_to_widths(centers: np.ndarray, lo: float, hi: float) -> np.ndarray:
    edges = centers_to_edges(centers, lo=lo, hi=hi)
    return np.diff(edges)                  # length N


@numba.njit(fastmath=True)
def manual_dot(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
               w0: float, w1: float, w2: float, w3: float) -> np.ndarray:
    """Manually computes the tensor dot product, summing the product of
    the weight and the PSF over all corners of an inner PSF rectangle.

    Parameters
    ----------
    p0, p1, p2, p3 : `numpy.ndarray`
        Corner PSFs, in order lower left, upper left, lower right, upper right.
    w0, w1, w2, w3 : `float`
        Corner weights, in order lower left, upper left, lower right, upper right.

    Returns
    -------
    blended_psf : `numpy.ndarray`
        Blended/interpolated PSF with shape (N, N).

    """
    j, k = p0.shape
    blended_psf = np.zeros((j, k), dtype=p0.dtype)
    for jj in range(j):
        for kk in range(k):
            blended_psf[jj, kk] = (w0 * p0[jj, kk] + w1 * p1[jj, kk] +
                                   w2 * p2[jj, kk] + w3 * p3[jj, kk])
    return blended_psf


def _hdr_float(val: float, ndec: int = 6) -> float:
    return round(float(val), ndec)


class PSFZoneData(NDIOMixin):
    """PSF normalized to unity integral and centered to create PSF grid.

    This class is used to represent the exposure-averaged effective PSF
    (optical+jitter) calculated from the multiple exposures, and the
    per-exposure effective PSF (optical+jitter).

    PSF is calculated per zone, defined by the x center, y center, x width, and
    y width, where coordinates are in FITS.

    Parameters
    ----------
    data : `~numpy.typing.ArrayLike`
        The PSF grid.
    x_centers : `~numpy.ndarray`
        Array of x coordinates of zone centers in the FITS convention.
    y_centers : `~numpy.ndarray`
        Array of y coordinates of zone centers in the FITS convention.
    oversampling : `float`, optional
        PSF oversampling value, by default 10.

    Notes
    -----
    Using FITS coords in the original image.
    We maybe have to apply the offsets when the coordinates are used elsewhere.

    """

    def __init__(self, data: ArrayLike, x_centers: np.ndarray, y_centers: np.ndarray,
                 oversampling: float = 10) -> None:

        data = np.asarray(data)
        # Convert to native byte order to enable SIMD numba compatibility.
        if not data.dtype.isnative:
            data = data.byteswap().view(data.dtype.newbyteorder())
        self.data: ArrayLike = data
        # Explicitly initialize `<axis>_centers` as arrays in order to use
        # `repeat` method and convert any input lists
        self.x_centers = np.asarray(x_centers, dtype=float)
        self.y_centers = np.asarray(y_centers, dtype=float)
        self.oversampling = int(oversampling)
        # Calculate zone IDs using centers
        self.x_zones = len(x_centers)
        self.y_zones = len(y_centers)
        # Detector bounds in "pixel center" coords (consistent with your quad edges)
        XLO, XHI = -0.5, 2039.5
        YLO, YHI = -0.5, 2039.5
        # edges
        self.x_edges = centers_to_edges(self.x_centers, lo=XLO, hi=XHI)
        self.y_edges = centers_to_edges(self.y_centers, lo=YLO, hi=YHI)
        # Determine the widths using the centers
        self.x_widths = np.diff(self.x_edges)
        self.y_widths = np.diff(self.y_edges)
        # x-fast, then y
        self.zone_xy = np.column_stack((
            np.tile(self.x_centers, self.y_zones),      # x varies fastest
            np.repeat(self.y_centers, self.x_zones)     # y varies slowest
        ))
        # Now header is guaranteed to follow zone_xy’s convention
        self.header = self._get_header()
        # quads
        self.quads = self._get_psf_quads()
        self.zone_idx_map = {
            (x, y): idx for idx, (x, y) in enumerate(self.zone_xy)
        }
        # Checks for the PSF cube.
        if self.data.ndim != 3:
            raise ValueError(
                f"PSF data is not a cube, it has {self.data.ndim} dimension(s)."
            )
        if len(self.x_centers) * len(self.y_centers) != self.data.shape[0]:
            raise ValueError(
                f"First dimension of PSF cube, {self.data.shape[0]}, "
                "does not equal x and y centers multiplied: "
                f"{len(self.x_centers)}, {len(self.y_centers)}."
            )

    def _get_header(self) -> fits.Header:
        """Build the header and WCS for PSF grid.

        Returns
        -------
        `fits.Header`
            Header for PSF grid.

        """
        # Build WCS
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [(self.data[0].shape[0]//2)+1, (self.data[0].shape[1]//2)+1]
        w.wcs.cdelt = np.repeat(get_pix_scale().to(u.arcsec).value / self.oversampling, 2)
        w.wcs.ctype = ["XANG", "YANG"]
        w.wcs.cname = ["Relative offset in detector X from center of PSF, in arcseconds",
                       "Relative offset in detector Y from center of PSF, in arcseconds"]
        w.wcs.cunit = ["arcsec", "arcsec"]
        header = w.to_header()

        # Resolve CDELT comment truncation
        header.comments["CDELT1"] = "[arcsec] Pixel scale"
        header.comments["CDELT2"] = "[arcsec] Pixel scale"
        header.comments["CNAME1"] = ""
        header.comments["CNAME2"] = ""

        # Remove headers not needed
        del header["LON*"]
        del header["LAT*"]

        header["OVERSAMP"] = self.oversampling
        # Zone keywords in exact plane order: i = iy*x_zones + ix
        for i, (xctr, yctr) in enumerate(self.zone_xy):
            zone_id = i + 1
            ix = i % self.x_zones
            iy = i // self.x_zones
            location = f"({ix}, {iy})"
            header[f"XCTR_{zone_id}"] = _hdr_float(xctr), f"X pixel at center of zone {location}"
            header[f"YCTR_{zone_id}"] = _hdr_float(yctr), f"Y pixel at center of zone {location}"
            header[f"XWID_{zone_id}"] = (_hdr_float(self.x_widths[ix]),
                                         f"[pixel] Zone {location} width along X axis")
            header[f"YWID_{zone_id}"] = (_hdr_float(self.y_widths[iy]),
                                         f"[pixel] Zone {location} width along Y axis")

        # PSF is normalized to unity integral
        header["BUNIT"] = "", "Unit of pixel value"
        return header

    def _get_psf_quads(self):
        """Get the coordinate boundaries (along zone centers or edges)
        for each quadrilateral enclosing where 4 PSF zones meet.

        Notes
        -----
        # The zone ID is determined from the image *header*

        1) Definition of a 'quad'
           Example grid with zone IDs
            [7][8][9]
            [4][5][6]
            [1][2][3]

           Example quad
            [4][5]
            [1][2]

        2) Distinguishing between centers and edges
           A quad dictionary has 2 keys. "zone_ctrs" ("zone_edges") outline the
           boundaries of the inner (outer) rectangle within the PSF quad:

           edge ___________ edge
               |     |     |
               | ctr | ctr |
               |_____|_____|
               |     |     |
               | ctr | ctr |
               |_____|_____|
           edge             edge

        3) Using coordinate boundaries
           From the 4-tuple value associated with each quad dictionary key,
           we can determine the corners of a rectangle:

           (xmin, ymax)     (xmax, ymax)
                UL ------------ UR
                 |              |
                 |              |
                LL ------------ LR
           (xmin, ymin)     (xmax, ymin)

        Returns
        -------
        quads : `numpy.ndarray` of `dict`
            An array where each element is a dictionary of quadrilaterals. Keys
            "zone_ctrs" and "zone_edges" map to different coordinate
            boundary values. The coordinates are in order xmin,xmax,ymin,ymax.

        """
        def find_value_coords(grid_xy, position, axis):
            # Find all coordinate sets matching an input position along 1 axis
            if axis == "x":
                match_idx = np.nonzero(grid_xy[:, 0] == position)
            if axis == "y":
                match_idx = np.nonzero(grid_xy[:, 1] == position)

            return grid_xy[match_idx]

        quads = np.empty(shape=((self.x_zones - 1) * (self.y_zones - 1)), dtype=dict)
        quad_idx = 0
        for (ctr_xmin, ctr_ymin) in zip(self.zone_xy[:, 0], self.zone_xy[:, 1]):
            i_xmin = np.nonzero(self.x_centers == ctr_xmin)[0][0]  # find first occurence where true
            i_ymin = np.nonzero(self.y_centers == ctr_ymin)[0][0]

            # If min coordinates belong to last index along an axis
            # (right/top edge of grid), there's no quad
            if (i_xmin == self.x_zones - 1) or (i_ymin == self.y_zones - 1):
                continue

            # Search for a complete quad
            d_ix = 1
            d_iy = 1
            i_xtest = i_xmin + d_ix
            i_ytest = i_ymin + d_iy
            found_quad = False
            while (i_xtest < self.x_zones) and (i_ytest < self.y_zones):
                coords_xtest = find_value_coords(self.zone_xy, self.x_centers[i_xtest], 'x')
                coords_ytest = find_value_coords(self.zone_xy, self.y_centers[i_ytest], 'y')
                # Intersection of the two lists of coordinates
                int_idx = (coords_xtest[:, None] == coords_ytest).all(-1).any(1)  # broadcast to compare rows
                coords_xytest = coords_xtest[int_idx]
                if len(coords_xtest) > 0 and len(coords_ytest) > 0 and len(coords_xytest) > 0:
                    ctr_xmax, ctr_ymax = coords_xytest[0]
                    # Verified presence of LL and UR corners in list of grid
                    # coordinates; check for UL and LR corners
                    if (ctr_xmax, ctr_ymin) in self.zone_xy and (ctr_xmin, ctr_ymax) in self.zone_xy:
                        found_quad = True
                        break

                ''' If none of the above conditions were met, search for
                satisfactory grid points at incrementally larger distances.
                The following sequence stretches the block in each dimension
                while keeping the other dimension as thin as possible, for
                example, (1,1), (2,1), (1,2), (3,1), (1,3), etc. '''
                if d_ix == d_iy:
                    d_ix += 1
                elif d_ix > d_iy:
                    delta = d_ix - d_iy
                    d_ix = copy.deepcopy(d_iy)
                    d_iy += delta
                elif d_iy > d_ix:
                    delta = d_iy - d_ix
                    d_iy = copy.deepcopy(d_ix)
                    d_ix += delta + 1
                i_xtest = i_xmin + d_ix
                i_ytest = i_ymin + d_iy

            if found_quad:
                # Calculate boundaries along edge; use -0.5/2039.5 at margins
                # due to source positions assuming zero-based indices
                edge_xmin = -0.5 if (i_xmin == 0) else ctr_xmin
                edge_ymin = -0.5 if (i_ymin == 0) else ctr_ymin
                edge_xmax = 2039.5 if (i_xtest == self.x_zones - 1) else ctr_xmax
                edge_ymax = 2039.5 if (i_ytest == self.y_zones - 1) else ctr_ymax

                quad = {}
                quad["zone_ctrs"] = (ctr_xmin, ctr_xmax, ctr_ymin, ctr_ymax)
                quad["zone_edges"] = (edge_xmin, edge_xmax, edge_ymin, edge_ymax)
                quads[quad_idx] = quad
                quad_idx += 1

        return quads

    def _get_inner_rectangle(self, x: float, y: float) -> tuple[float]:
        """Get the coordinate boundaries for the inner PSF rectangle
        enclosing the point (x, y).

        Parameters
        ----------
        x, y : `float`
            Source position (in detector coordinates).

        Returns
        -------
        `tuple` of `float`
            Four-tuple of coordinates for the inner PSF rectangle,
            in order (xmin, xmax, ymin, ymax).

        Raises
        ------
        ValueError
            Raised if input positions are outside of all PSF zone edges.

        """
        inner_rect = None
        for quad in self.quads:
            # Use zone edges to avoid excluding query points near image margins
            x1, x2, y1, y2 = quad["zone_edges"]

            if (x1 <= x <= x2) and (y1 <= y <= y2):
                inner_rect = quad["zone_ctrs"]
                return inner_rect
        if inner_rect is None:
            raise ValueError(f"Input positions do not fall within the boundaries of any PSF quads: X={x},"
                             f" Y={y}.")

    def _get_corner_psf(self, rectangle: tuple[float],
                        corner_type: str) -> np.ndarray:
        """Extract one of the four PSFs in an inner PSF rectangle.

        Parameters
        ----------
        rectangle : `tuple` of `float`
            Four-tuple of coordinates for the inner PSF rectangle,
            in order (xmin, xmax, ymin, ymax).
        corner_type : `str`
            The corner for which to extract the PSF estimate, specified as one
            of: {'lower left', 'upper left', 'lower right', 'upper right'}.

        Returns
        -------
        `numpy.ndarray`
            The PSF for the queried corner.

        """
        x1, x2, y1, y2 = rectangle

        if corner_type == "lower left":
            zone_coords = (x1, y1)
        elif corner_type == "upper left":
            zone_coords = (x1, y2)
        elif corner_type == "lower right":
            zone_coords = (x2, y1)
        elif corner_type == "upper right":
            zone_coords = (x2, y2)

        if zone_coords in self.zone_idx_map:
            idx = self.zone_idx_map[zone_coords]
            return self.data[idx]
        else:
            raise ValueError(f"Coordinates {zone_coords} not found in `PSFZoneData.zone_idx_map`. Valid"
                             f"coordinates must correspond to a PSF zone center.")

    def _get_dist_weights(self, x_in: float, y_in: float, rectangle: tuple[float]) -> np.ndarray[float]:
        """Calculate the distance-based weights for PSF interpolation.

        Notes
        -----
        These weights are identical to those of a bilinear interpolation.

        Since we may not have PSF estimates on the boundaries of the detector,
        it is necessary to extrapolate to the margins using PSFs from the
        interior. To avoid numerical pathologies in the extrapolation, use the
        nearest part of the PSF rectangle to determine the PSF for a point on
        the boundary.

        Parameters
        ----------
        x_in, y_in : `float`
            Source position (in detector coordinates).
        rectangle : `tuple` of `float`
            Four-tuple of coordinates for the inner PSF rectangle,
            in order (xmin, xmax, ymin, ymax).

        Returns
        -------
        weights : `np.ndarray` of `float`
            The distance-based weight for each corner, in order (lower left,
            upper left, lower right, upper right).

        """
        x1, x2, y1, y2 = rectangle

        x_eff = max(min(x_in, x2), x1)
        y_eff = max(min(y_in, y2), y1)

        rectangle_area = (x2 - x1) * (y2 - y1)

        weights = np.array([
            (x2 - x_eff) * (y2 - y_eff) / rectangle_area,  # lower left
            (x2 - x_eff) * (y_eff - y1) / rectangle_area,  # upper left
            (x_eff - x1) * (y2 - y_eff) / rectangle_area,  # lower right
            (x_eff - x1) * (y_eff - y1) / rectangle_area   # upper right
        ])

        return weights

    def get_blended_psf(self, x: float, y: float) -> np.ndarray:
        """Retrieve the interpolated PSF for a given source position on the
        detector.

        Notes
        -----
        This method and its helpers are adapted from Ari Cukierman's `get_psf`
        method used in the SPHEREx Sky Simulator. It smoothly varies across the
        detector.

        Parameters
        ----------
        x, y : `float`
            Source position (in detector coordinates).

        Returns
        -------
        psf : `numpy.ndarray`
            The blended PSF.

        """
        psf_rectangle = self._get_inner_rectangle(x, y)

        # Fetch PSFs directly without packing into a new np.array
        p0 = self._get_corner_psf(psf_rectangle, "lower left")
        p1 = self._get_corner_psf(psf_rectangle, "upper left")
        p2 = self._get_corner_psf(psf_rectangle, "lower right")
        p3 = self._get_corner_psf(psf_rectangle, "upper right")

        corner_weights = self._get_dist_weights(x, y, psf_rectangle)
        psf = manual_dot(p0, p1, p2, p3,
                         corner_weights[0], corner_weights[1],
                         corner_weights[2], corner_weights[3])

        weight_sum = np.sum(corner_weights)
        if not (abs(weight_sum - 1.0) <= 1e-3):
            logging.warning(f"Weights sum to {weight_sum:.6f}, expected ~1.0.")

        return psf
    def to_hdu(self, name: str = "PSF-DATA-CUBE") -> fits.ImageHDU:
        """Return the PSF zone data and header as an image HDU.

        Parameters
        ----------
        name : `str`, optional
            Name of the image HDU, by default "PSF-DATA-CUBE".

        Returns
        -------
        `~astropy.io.fits.ImageHDU`
            The image HDU of the PSF zone data and header.

        """
        return fits.ImageHDU(data=self.data.astype(np.float32, copy=False), header=self.header, name=name)

    @staticmethod
    def _detect_header_zone_order(zone_xy: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Inspect header-provided plane sequence and
           classify it as x-fast, y-fast, or unknown.

        This is diagnostic only. It does not imply that the cube data should
        be reordered.
        """
        zone_xy_arr = np.asarray(zone_xy, dtype=float)

        if zone_xy_arr.ndim != 2 or zone_xy_arr.shape[1] != 2:
            raise ValueError(f"zone_xy must have shape (nplane, 2); got {zone_xy_arr.shape}")

        nplane = zone_xy_arr.shape[0]
        if nplane == 0:
            raise ValueError("zone_xy must contain at least one (x, y) pair.")

        unique_rows = np.unique(zone_xy_arr, axis=0)
        if unique_rows.shape[0] != nplane:
            raise ValueError("zone_xy contains duplicate (x, y) zone centers.")

        x_centers = np.array(sorted(set(zone_xy_arr[:, 0])), dtype=float)
        y_centers = np.array(sorted(set(zone_xy_arr[:, 1])), dtype=float)

        nx = len(x_centers)
        ny = len(y_centers)

        if nx * ny != nplane:
            raise ValueError(
                "zone_xy does not define a complete rectangular grid: "
                f"{nx} x {ny} != {nplane}"
            )

        expected_xfast = np.column_stack((
            np.tile(x_centers, ny),
            np.repeat(y_centers, nx),
        ))

        expected_yfast = np.column_stack((
            np.repeat(x_centers, ny),
            np.tile(y_centers, nx),
        ))

        if np.allclose(zone_xy_arr, expected_xfast):
            order = "x-fast"
        elif np.allclose(zone_xy_arr, expected_yfast):
            order = "y-fast"
        else:
            order = "unknown"

        return zone_xy_arr, x_centers, y_centers, order

    @classmethod
    def from_hdu(cls, hdu: fits.ImageHDU) -> TPSFZoneData:
        """Construct a `PSFZoneData` from a FITS ImageHDU.

        Notes
        -----
        The PSF data cube is always interpreted in canonical x-fast order.
        Header zone ordering is inspected only for diagnostics, so that buggy
        files with y-fast-looking headers can still be read without reordering
        the data. (Buggy files are from version <= 6.5.5)
        """
        if not isinstance(hdu, fits.ImageHDU):
            raise ValueError("Unexpected HDU type for PSF layer.")

        data = np.asarray(hdu.data)
        header = hdu.header

        if data.ndim != 3:
            raise ValueError(f"PSF data is not a cube; got shape {data.shape}.")

        if "OVERSAMP" not in header:
            raise ValueError("Missing required header keyword OVERSAMP.")
        oversampling = int(header["OVERSAMP"])

        nplane = data.shape[0]
        zone_xy = []

        for k1 in range(1, nplane + 1):
            xk = f"XCTR_{k1}"
            yk = f"YCTR_{k1}"
            if xk not in header or yk not in header:
                raise ValueError(f"Missing required zone center keywords: {xk} / {yk}")
            zone_xy.append((float(header[xk]), float(header[yk])))

        _, x_centers, y_centers, header_order = cls._detect_header_zone_order(zone_xy)

        if header_order == "y-fast":
            logging.warning(
                "PSF header zone metadata appear to be y-fast, but the data cube "
                "is assumed to remain stored in canonical x-fast order. "
                "No data reordering is performed."
            )
        elif header_order == "unknown":
            logging.warning(
                "PSF header zone metadata are not recognized as x-fast or y-fast. "
                "Proceeding by recovering x/y center axes from header values only. "
                "No data reordering is performed."
            )

        return cls(data, x_centers, y_centers, oversampling)

    @staticmethod
    def _validate_zone_xy_and_recover_axes(zone_xy: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate a plane-ordered zone_xy array and recover x/y center axes.

        Parameters
        ----------
        zone_xy : `ArrayLike`
            Plane-ordered list of (x, y) zone centers.

        Returns
        -------
        zone_xy_arr : `np.ndarray`
            Normalized float array of shape (nplane, 2).
        x_centers : `np.ndarray`
            Sorted unique x centers.
        y_centers : `np.ndarray`
            Sorted unique y centers.

        Raises
        ------
        ValueError
            If zone_xy is malformed, has duplicates, does not define a complete
            rectangular grid, or is not in canonical x-fast order.
        """
        zone_xy_arr = np.asarray(zone_xy, dtype=float)

        if zone_xy_arr.ndim != 2 or zone_xy_arr.shape[1] != 2:
            raise ValueError(
                f"zone_xy must have shape (nplane, 2); got {zone_xy_arr.shape}"
            )

        nplane = zone_xy_arr.shape[0]
        if nplane == 0:
            raise ValueError("zone_xy must contain at least one (x, y) pair.")

        # Reject duplicate zone centers
        unique_rows = np.unique(zone_xy_arr, axis=0)
        if unique_rows.shape[0] != nplane:
            raise ValueError("zone_xy contains duplicate (x, y) zone centers.")

        x_centers = np.array(sorted(set(zone_xy_arr[:, 0])), dtype=float)
        y_centers = np.array(sorted(set(zone_xy_arr[:, 1])), dtype=float)

        if len(x_centers) * len(y_centers) != nplane:
            raise ValueError(
                "zone_xy does not define a complete rectangular grid: "
                f"{len(x_centers)} x {len(y_centers)} != {nplane}"
            )

        return zone_xy_arr, x_centers, y_centers

    @classmethod
    def from_zone_xy(
        cls,
        data: ArrayLike,
        zone_xy: ArrayLike,
        oversampling: int = 10,
    ) -> TPSFZoneData:
        """Construct a `PSFZoneData` from an explicit plane-ordered list of
        (x, y) zone centers.

        Parameters
        ----------
        data : `ArrayLike`
            PSF cube with shape (nplane, ny, nx).
        zone_xy : `ArrayLike`
            Plane-ordered list of (x, y) zone centers. Must define a complete
            rectangular grid in canonical x-fast order.
        oversampling : `int`, optional
            PSF oversampling factor, default is 10.

        Returns
        -------
        `PSFZoneData``
            New instance with x-fast internal convention.
        """
        data_arr = np.asarray(data)
        if data_arr.ndim != 3:
            raise ValueError(f"PSF data is not a cube; got shape {data_arr.shape}.")

        zone_xy_arr, x_centers, y_centers = cls._validate_zone_xy_and_recover_axes(zone_xy)

        if zone_xy_arr.shape[0] != data_arr.shape[0]:
            raise ValueError(
                f"zone_xy has {zone_xy_arr.shape[0]} planes, but data has {data_arr.shape[0]}."
            )

        return cls(data_arr, x_centers, y_centers, oversampling)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, PSFZoneData) and \
            np.allclose(self.x_centers, other.x_centers) and \
            np.allclose(self.y_centers, other.y_centers) and \
            np.allclose(self.x_widths, other.x_widths) and \
            np.allclose(self.y_widths, other.y_widths) and \
            np.allclose(self.data, other.data)

    def __repr__(self) -> str:
        return f"SPHEREx PSF Data Object - {len(self.data)} kernels"


def spx_psf_writer(psf: PSFZoneData, filename: Union[str, Path], **kwargs) -> None:
    """Write PSF zone data object to a file.

    Parameters
    ----------
    psf : `PSFZoneData`
        The instance to write to a file.
    filename : `Union[str, Path]`
        Filename and path to write to.
    **kwargs
        Additional keyword arguments passed to `astropy.io.fits.HDUList.writeto`.

    """
    primary_header = fits.Header()
    primary_header["COMMENT"] = "PSF zone data defined by x and y centers and x and y zones"
    primary_hdu = fits.PrimaryHDU(data=None, header=primary_header)
    psf_hdu = psf.to_hdu()
    hdul = fits.HDUList([primary_hdu, psf_hdu])
    hdul.writeto(filename, **kwargs)


def spx_psf_reader(filename: Union[str, Path], **kwargs) -> PSFZoneData:
    """Read in FITS file and create PSF instance.

    Parameters
    ----------
    filename : `Union[str, Path]`
        Path to FITS file to read in.
    **kwargs
        Additional keyword arguments passed to `astropy.io.fits.open`.

    Returns
    -------
    `PSFZoneData`
        Instance of PSF built from FITS file.

    Notes
    -----
    FITS files that contain scaled data (e.g. unsigned integer images) will
    be scaled and the keywords used to manage scaled data in
    :mod:`astropy.io.fits` are disabled.

    """
    with fits.open(filename, **kwargs) as hdul:
        return PSFZoneData.from_hdu(hdul[1])


_zone_pat = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


def _parse_ixiy_from_comment(comment: str) -> tuple[int, int]:
    m = _zone_pat.search(str(comment))
    if not m:
        raise ValueError(f"Could not parse zone indices from comment: {comment!r}")
    return int(m.group(1)), int(m.group(2))


def _infer_grid_shape_from_header_comments(hdr: fits.Header, nzone: int) -> tuple[int, int]:
    max_ix = -1
    max_iy = -1

    for k1 in range(1, nzone + 1):
        key = f"XCTR_{k1}"
        if key not in hdr:
            raise KeyError(f"Missing required key: {key}")
        ix, iy = _parse_ixiy_from_comment(hdr.comments[key])
        max_ix = max(max_ix, ix)
        max_iy = max(max_iy, iy)

    bins_x = max_ix + 1
    bins_y = max_iy + 1

    if bins_x * bins_y != nzone:
        raise ValueError(
            f"Inconsistent grid inferred from comments: "
            f"bins_x={bins_x}, bins_y={bins_y}, nzone={nzone}"
        )

    return bins_x, bins_y


def _collect_axis_values_by_zone(
    hdr: fits.Header,
    nzone: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    x_center_by_ix = {}
    y_center_by_iy = {}
    x_width_by_ix = {}
    y_width_by_iy = {}

    for k1 in range(1, nzone + 1):
        ix, iy = _parse_ixiy_from_comment(hdr.comments[f"XCTR_{k1}"])

        xck = f"XCTR_{k1}"
        yck = f"YCTR_{k1}"
        xwk = f"XWID_{k1}"
        ywk = f"YWID_{k1}"

        if xck in hdr:
            val = hdr[xck]
            if ix in x_center_by_ix and not np.isclose(x_center_by_ix[ix], val):
                raise ValueError(f"Inconsistent XCTR for ix={ix}: {x_center_by_ix[ix]} vs {val}")
            x_center_by_ix[ix] = val

        if yck in hdr:
            val = hdr[yck]
            if iy in y_center_by_iy and not np.isclose(y_center_by_iy[iy], val):
                raise ValueError(f"Inconsistent YCTR for iy={iy}: {y_center_by_iy[iy]} vs {val}")
            y_center_by_iy[iy] = val

        if xwk in hdr:
            val = hdr[xwk]
            if ix in x_width_by_ix and not np.isclose(x_width_by_ix[ix], val):
                raise ValueError(f"Inconsistent XWID for ix={ix}: {x_width_by_ix[ix]} vs {val}")
            x_width_by_ix[ix] = val

        if ywk in hdr:
            val = hdr[ywk]
            if iy in y_width_by_iy and not np.isclose(y_width_by_iy[iy], val):
                raise ValueError(f"Inconsistent YWID for iy={iy}: {y_width_by_iy[iy]} vs {val}")
            y_width_by_iy[iy] = val

    return x_center_by_ix, y_center_by_iy, x_width_by_ix, y_width_by_iy


def relabel_buggy_psf_header_cards_to_xfast(hdr_in: fits.Header, nzone: int) -> fits.Header:
    """Return a copy of a buggy PSF header with per-plane zone metadata
    rewritten to canonical x-fast ordering.

    Only the per-plane zone metadata cards are rewritten. No image data are
    touched.
    """
    bins_x, bins_y = _infer_grid_shape_from_header_comments(hdr_in, nzone)

    x_center_by_ix, y_center_by_iy, x_width_by_ix, y_width_by_iy = _collect_axis_values_by_zone(
        hdr_in, nzone
    )

    missing = []
    for ix in range(bins_x):
        if ix not in x_center_by_ix:
            missing.append(f"x_center[{ix}]")
        if ix not in x_width_by_ix:
            missing.append(f"x_width[{ix}]")
    for iy in range(bins_y):
        if iy not in y_center_by_iy:
            missing.append(f"y_center[{iy}]")
        if iy not in y_width_by_iy:
            missing.append(f"y_width[{iy}]")

    if missing:
        raise ValueError(f"Missing axis metadata recovered from old header: {missing}")

    hdr_out = hdr_in.copy()

    for k0 in range(nzone):
        ix = k0 % bins_x
        iy = k0 // bins_x
        k1 = k0 + 1

        hdr_out[f"XCTR_{k1}"] = (x_center_by_ix[ix], f"Center of x zone ({ix}, {iy})")
        hdr_out[f"YCTR_{k1}"] = (y_center_by_iy[iy], f"Center of y zone ({ix}, {iy})")
        hdr_out[f"XWID_{k1}"] = (x_width_by_ix[ix], f"Width of x zone ({ix}, {iy})")
        hdr_out[f"YWID_{k1}"] = (y_width_by_iy[iy], f"Width of y zone ({ix}, {iy})")

    hdr_out["HISTORY"] = "Rewrote PSF per-plane zone metadata to x-fast ordering."
    hdr_out["HISTORY"] = "Cube plane data left unchanged."

    return hdr_out


def relabel_buggy_psf_header_to_xfast(
    infn: Union[str, Path],
    outfn: Union[str, Path],
    extname: str = "PSF-DATA-CUBE",
    overwrite: bool = True,
) -> None:
    """Rewrite buggy PSF header metadata in a FITS file to x-fast ordering.

    The cube data are left unchanged.
    """
    with fits.open(infn) as hdul:
        hdu = hdul[extname]
        cube = np.asarray(hdu.data)

        if cube.ndim != 3:
            raise ValueError(f"Expected 3D PSF cube, got shape {cube.shape}")

        hdr_out = relabel_buggy_psf_header_cards_to_xfast(hdu.header, cube.shape[0])
        new_hdu = fits.ImageHDU(data=cube, header=hdr_out, name=hdu.name)

        ext_index = hdul.index_of(extname)
        hdul_out = fits.HDUList()

        for i, old in enumerate(hdul):
            hdul_out.append(new_hdu if i == ext_index else old.copy())

        hdul_out.writeto(outfn, overwrite=overwrite)


# Register read/write methods for PSFZoneData
with registry.delay_doc_updates(PSFZoneData):
    registry.register_reader(data_format="fits", data_class=PSFZoneData,
                             function=spx_psf_reader)
    registry.register_writer(data_format="fits", data_class=PSFZoneData,
                             function=spx_psf_writer)
    registry.register_identifier(data_format="fits", data_class=PSFZoneData,
                                 identifier=fits.connect.is_fits)
