"""Utility classes and functions for performing Tractor fitting. Adapted
from SPHEREx-Sky-Simulator/py/SPHEREx_SkySimulator/tractor_utils.py and
SPHEREx-Sky-Simulator/py/SPHEREx_SkySimulator/quickcatalog.py.
"""
__all__ = ["tractor_fit_spherex", "MJysr_to_uJy"]

# Standard library imports
from typing import Tuple
from abc import ABC, abstractmethod

# Related third party imports
from astropy.io.fits import HDUList, PrimaryHDU, ImageHDU, BinTableHDU, Header
from astropy.nddata import Cutout2D
from astropy.table import Table
from cv2 import INTER_AREA, resize
from numpy.typing import ArrayLike
from photutils.psf import ImagePSF
from regions import CirclePixelRegion, PixCoord
from tractor import (
    ConstantSky, Flux, GalaxyShape, LinearPhotoCal, NullWCS, PixPos, PointSource)
from tractor.psf import lanczos_shift_image
from tractor.sersic import SersicGalaxy, SersicIndex
from tractor.spherex_utils import getUnitFluxModelPatchSPHEREx
import astropy.units as u
import numpy as np
import tractor

# Local application/ library specific imports
from spx_data import Flag, PhotometryFlag, PSFZoneData, spx_param

tractor.galaxy.ProfileGalaxy.getUnitFluxModelPatch = \
    getUnitFluxModelPatchSPHEREx  # override to support arbitrary PSF oversampling

# =============================================================================
# Abstracted Tractor PSF classes
def _subimg_center(height: int, width: int, subimg: ArrayLike) -> Tuple[np.ndarray, int, int]:
    """Embed the given subimage into an image with the desired shape, using
    padding or cutting as necessary.

    Parameters
    ----------
    height, width : `int`
        Shape of the output image to embed the subimage into.
    subimg : array-like
        Subimage to embed.

    Returns
    -------
    pad : `numpy.ndarray`
        The subimage embedded into an image of shape (`height`, `width`).
    subimg_cx, subimg_cy : `int`
        The x- and y-coordinates, respectively, of the center of the
        subimage in the embedded image.

    Notes
    -----
    Return pixel coordinates follow the FITS standard with 0-based indices.
    """
    subimg_h, subimg_w = subimg.shape

    # Pad/cut along y-axis
    if height >= subimg_h:
        pad_y0 = (height - subimg_h) // 2
        subimg_cy = pad_y0 + np.floor((subimg_h - 1.0) / 2)
    else:
        pad_y0 = 0
        cut = (subimg_h - height) // 2
        subimg = subimg[cut: cut + height, :]
        subimg_cy = np.floor((subimg_h - 1.0) / 2) - cut

    # Pad/cut along x-axis
    if width >= subimg_w:
        pad_x0 = (width - subimg_w) // 2
        subimg_cx = pad_x0 + np.floor((subimg_w - 1.0) / 2)
    else:
        pad_x0 = 0
        cut = (subimg_w - width) // 2
        subimg = subimg[:, cut: cut + width]
        subimg_cx = np.floor((subimg_w - 1.0) / 2) - cut

    # Embed the subimage
    subimg_h, subimg_w = subimg.shape
    pad = np.zeros((height, width), subimg.dtype)
    pad[pad_y0: pad_y0 + subimg_h, pad_x0: pad_x0 + subimg_w] = subimg

    return pad, subimg_cx, subimg_cy


class SPHERExTractorPSF(tractor.ducks.Params, tractor.ducks.ImageCalibration, ABC):
    """Base class representing SPHEREx PSF models in Tractor.

    Parameters
    ----------
    psf : `photutils.psf.ImagePSF` | `spx_data.psf.PSFZoneData`
        Instantiated SPHEREx PSF object.
    detector : {1, 2, 3, 4, 5, 6}
        Detector/array ID.
    psf_shape : `list` of `int`
        PSF shape in native pixels.
    radius : `float`
        PSF radius in native pixels.
    """

    @abstractmethod
    def __init__(self, psf: PSFZoneData | ImagePSF, detector: int, psf_shape: list[int], radius: float):
        if not (isinstance(psf, PSFZoneData) or isinstance(psf, ImagePSF)):
            raise TypeError("Input parameter `psf` not a `spx_data.psf.PSFZoneData` or"
                            " `photutils.psf.ImagePSF` object")
        self._psf = psf
        self.detector = int(detector)
        self.psf_shape = psf_shape
        self.radius = radius

    def __repr__(self):
        """Return direct string representation of the PSF object."""
        return self.__str__()

    def __str__(self):
        """Return human-readable string representation of the PSF object."""
        return (
            f"{self.__class__.__name__}: PSF type={type(self._psf).__name__}, detector={self.detector}, "
            f"psf_shape={self.psf_shape}, radius={self.radius}"
        )

    @property
    def shape(self) -> list[int]:
        """Return the PSF shape (height, width) in native (detector) pixels."""
        return self.psf_shape

    @abstractmethod
    def copy(self):
        """Return a shallow copy of this class with the same parameters."""
        pass

    def getRadius(self):
        """Return the PSF radius in native (detector) pixels."""
        return self.radius

    def getFourierTransformSize(self, radius: float) -> int:
        """Calculate the next highest power-of-two size."""
        size = 2 ** int(np.ceil(np.log2(2 * radius)))
        return size

    def _padInImage(self, height: int, width: int, px: float, py: float,
                    img: ArrayLike = None) -> Tuple[np.ndarray, int, int]:
        """Embed PSF image into shape (`height`, `width`).

        Parameters
        ----------
        height, width : `int`
            Embedded image shape.
        px, py : `float`
            Pixel coordinates.
        img : array-like, optional
            PSF image to embed. Default is None.

        Returns
        -------
        pad : `numpy.ndarray`
            The embedded PSF image.
        subimg_cx, subimg_cy : `int`
            The x- and y-coordinates, respectively, of the center of the
            subimage in the embedded image..

        """
        if img is None:
            point_source_patch = self.getPointSourcePatch(px, py)
            subimg = point_source_patch.patch
        else:
            subimg = img.copy()

        return _subimg_center(height, width, subimg)

    def getFourierTransform(self, px: float, py: float, radius: float) -> Tuple[
        np.ndarray, Tuple[float, float], Tuple[int, int], Tuple[np.ndarray, np.ndarray]
    ]:
        """Calculate the Fourier transform of the PSF.

        Parameters
        ----------
        px, py : `float`
            x- and y-axis position of the source, respectively, in
            `tractor.Image`/SPHEREx window cutout pixel coordinates.
        radius : `float`
            Radius for Fourier transform size calculation.

        Returns
        -------
        p : `numpy.ndarray`
            2-D Fast Fourier Transform of the PSF.
        (cx, cy) : `tuple` of `float`
            The x- and y-pixel coordinates of the PSF center in the PSF
            subimage.
        (imh, imw) : `tuple` of `int`
            Shape of the padded PSF image (height, width).
        (v, w) : `tuple` of `numpy.ndarray`
            Discrete Fourier Transform sample frequencies for width- (v) and
            height-long (w) windows.

        """
        # Get padded PSF image
        size = self.getFourierTransformSize(radius)
        pad, cx, cy = self._padInImage(size, size, px, py)

        # Compute 2-D FFT and DFT sample frequencies
        p = np.fft.rfft2(pad)
        p = p.astype(np.complex64)
        pad_height, pad_width = pad.shape
        v = np.fft.rfftfreq(pad_width)
        w = np.fft.fftfreq(pad_height)

        return p, (cx, cy), (pad_height, pad_width), (v, w)

    @abstractmethod
    def getPointSourcePatch(self, px: float, py: float, minval=0, radius=None, modelMask=None,
                            **kwargs) -> tractor.patch.Patch:
        """Return a patch of PSF of a point source at the given pixel
        coordinates.
        """
        pass

    @abstractmethod
    def _padInImageHiRes(self, height: int, width: int, _px: float = None, _py: float = None,
                         img=None) -> Tuple[np.ndarray, int, int]:
        """Embed high-resolution PSF image into shape (`height`, `width`)."""
        pass

    def getFourierTransformHiRes(self, px: float, py: float, radius: float) -> Tuple[
            np.ndarray, Tuple[float, float], Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        """Calculate the Fourier transform of the high-resolution PSF.

        Parameters
        ----------
        px, py : `float`
            x- and y-axis position of the source, respectively, in
            `tractor.Image`/SPHEREx window cutout pixel coordinates.
        radius : `float`
            Radius for Fourier transform size calculation.

        Returns
        -------
        P : `numpy.ndarray`
            2-D Fast Fourier Transform of the high-resolution PSF.
        (cx, cy) : `tuple` of `float`
            The x- and y-pixel coordinates of the high-resolution PSF center in
            the high-resolution PSF subimage.
        (imh, imw) : `tuple` of `int`
            Shape of the padded high-resolution PSF image (height, width).
        (v, w) : `tuple` of `numpy.ndarray`
            Discrete Fourier Transform sample frequencies for width- (v) and
            height-long (w) windows.

        """
        # Get padded high-resolution PSF image
        oversample = self._psf.oversampling
        if not np.isscalar(oversample):
            if oversample[0] != oversample[1]:
                raise NotImplementedError("Non-square ePSF oversampling not yet supported")
            oversample = oversample[0]
        size = self.getFourierTransformSize(oversample * radius)  # native radius, must scale up
        pad, cx, cy = self._padInImageHiRes(size, size, px, py)

        # Compute 2-D FFT and DFT sample frequencies
        P = np.fft.rfft2(pad).astype(np.complex64)
        pad_height, pad_width = pad.shape
        v = np.fft.rfftfreq(pad_width)
        w = np.fft.fftfreq(pad_height)

        return P, (cx, cy), (pad_height, pad_width), (v, w)


class EPSFTractor(SPHERExTractorPSF):
    """Tractor PSF object to handle ePSF models for SPHEREx.

    Parameters
    ----------
    psf : `photutils.psf.ImagePSF`
        Effective PSF (ePSF) model for the closest zone to the cutout center.
    detector : {1, 2, 3, 4, 5, 6}
        SPHEREx detector/array ID.
    """

    def __init__(self, psf: ImagePSF, detector: int):
        if not isinstance(psf, ImagePSF):
            raise TypeError("Input parameter `psf` not a `photutils.psf.ImagePSF` object")

        # Derive the shape/radius of the PSF taking into account oversampling
        if np.isscalar(psf.oversampling):
            psf_shape = [
                int(x / psf.oversampling) for x in psf.data.shape
            ]
        else:
            psf_shape = [
                int(x / o) for x, o in zip(psf.data.shape, psf.oversampling)  # in (y, x) order
            ]
        radius = max(psf_shape) * 0.5

        super().__init__(psf=psf, detector=detector, psf_shape=psf_shape, radius=radius)

    def copy(self):
        return self.__class__(psf=self._psf, detector=self.detector)

    def render_epsf_stamp_centered(self, epsf: ImagePSF, x0: float, y0: float, stamp_size: int = 11,
                                   unit_flux: bool = True) -> Tuple[np.ndarray, int, int]:
        """
        Render an ePSF model into a stamp centered on requested coordinates.

        Parameters
        ----------
        epsf : `photutils.psf.ImagePSF`
            Effective PSF (ePSF) model; must support the `evaluate` method.
        x0, y0 : `float`
            Source centroid in image pixel coordinates (pixel-center
            convention).
        stamp_size : `int`
            Side length of the returned square stamp in pixels. Default is 11.
        unit_flux : `bool`
            If True, renormalize the stamp so the sum equals 1. Default is
            True.

        Returns
        -------
        psf_stamp : `numpy.ndarray`
            Rendered PSF stamp.
        x_ll, y_ll : `int`
            Integer lower-left pixel index of the stamp in the image coordinate
            system. That is, psf_stamp[0,0] corresponds to image pixel (y_ll,
            x_ll).
        """
        # Make grid of pixel-center coordinates
        half = (stamp_size - 1) / 2.0
        x_ll = int(np.floor(x0 - half))
        y_ll = int(np.floor(y0 - half))
        x_centers = x_ll + np.arange(stamp_size)
        y_centers = y_ll + np.arange(stamp_size)
        Y, X = np.meshgrid(y_centers, x_centers, indexing="ij")

        psf_stamp = epsf.evaluate(X, Y, flux=1.0, x_0=x0, y_0=y0)  # at native resolution

        if unit_flux:
            s = psf_stamp.sum()
            if s > 0 and np.isfinite(s):
                psf_stamp /= s

        return psf_stamp, x_ll, y_ll

    def getPointSourcePatch(self, px: float, py: float, minval: float = 0, radius: float = None,
                            modelMask=None, stamp_size: int = 11, **kwargs) -> tractor.patch.Patch:
        """Return a patch of PSF of a point source at the given pixel
        coordinates.

        Notes
        -----
        This obtains high-resolution, oversampled PSF from SPHEREx
        instrument simulator, resamples it for sub-pixel shift, and downscales
        to get the detector-pixel-scaled PSF.

        Parameters
        ----------
        px, py : `float`
            x- and y-axis position of the point source, respectively, in
            `tractor.Image`/SPHEREx window cutout pixel coordinates.
        minval : `float`, optional
            Minimum value threshold for PSF. Default is 0.
        radius : `float`, optional
            Radius for PSF extraction. Default is None.
        modelMask : optional
            Model mask for PSF. Default is None.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        patch : `tractor.patch.Patch`
            Patch of PSF
        """
        if radius is not None:
            raise NotImplementedError("input parameter `radius` not supported yet")
        if modelMask is not None:
            raise NotImplementedError("input parameter `modelMask` not supported yet")

        # Cache patches by pixel coordinates to avoid redundant computation
        if not hasattr(self, '_patch_cache'):
            self._patch_cache = {}
        cache_key = (px, py)
        if cache_key in self._patch_cache:
            return self._patch_cache[cache_key]

        psf, X, Y = self.render_epsf_stamp_centered(self._psf, px, py)
        patch = tractor.patch.Patch(X, Y, psf)
        self._patch_cache[cache_key] = patch
        return patch

    def _padInImageHiRes(self, height: int, width: int, _px: float = None, _py: float = None,
                         img=None) -> Tuple[np.ndarray, int, int]:
        """Embed high-resolution PSF image into shape (`height`, `width`).

        Parameters
        ----------
        height, width : `int`
            Embedded image shape.
        px, py : `float`, optional
            Unused. Accepted for interface compatibility with
            `SPHERExTractorPSF._padInImageHiRes`.
        img : array-like, optional
            High-resolution PSF image to embed. Default is None.

        Returns
        -------
        pad : `numpy.ndarray`
            The embedded high-resolution PSF image.
        subimg_cx, subimg_cy : `int`
            The x- and y-coordinates, respectively, of the center of the
            subimage in the embedded image.
        """
        if img is None:
            subimg = np.asarray(self._psf.data, dtype=np.float32).copy()
            subimg /= subimg.sum()
        else:
            subimg = img.copy()

        return _subimg_center(height, width, subimg)


class PSFZoneDataTractor(SPHERExTractorPSF):
    """Tractor PSF object to work with `spx_data.psf.PSFZoneData` objects.

    Parameters
    ----------
    psf : `spx_data.psf.PSFZoneData`
        Exposure-averaged PSF grid for the detector.
    detector : {1, 2, 3, 4, 5, 6}
        SPHEREx detector/array ID.
    xshift, yshift: `int`, optional
        x- and y-axis shift in the SPHEREx detector pixel coordinates. If
        initializing the PSF for a window cutout, these are the positions of
        the `tractor.Image` origin relative to the SPHEREx detector origin.
        Default is 0.
    """

    def __init__(self, psf: PSFZoneData, detector: int, xshift: int = 0, yshift: int = 0):
        if not isinstance(psf, PSFZoneData):
            raise TypeError("Input parameter `psf` not a `spx_data.psf.PSFZoneData` object")

        # Derive the shape/radius of the PSF taking into account oversampling
        psf_shape = [
            int(x / psf.oversampling) for x in psf.get_blended_psf(0, 0).shape
        ]
        radius = max(psf_shape) * 0.5

        super().__init__(psf=psf, detector=detector, psf_shape=psf_shape, radius=radius)
        self.xshift = xshift
        self.yshift = yshift

    def __str__(self):
        """Return human-readable string representation of the PSF object."""
        return (
            f"{self.__class__.__name__}: PSF type={type(self._psf).__name__}, detector={self.detector}, "
            f"psf_shape={self.psf_shape}, radius={self.radius}, xshift={self.xshift}, yshift={self.yshift}"
        )

    def copy(self):
        return self.__class__(psf=self._psf, detector=self.detector, xshift=self.xshift, yshift=self.yshift)

    def getPointSourcePatch(
            self, px: float, py: float, minval: float = 0, radius: float = None, modelMask=None,
            **kwargs) -> tractor.patch.Patch:
        """Return a patch of PSF of a point source at the given pixel
        coordinates.


        Parameters
        ----------
        px, py : `float`
            x- and y-axis position of the point source, respectively, in
            `tractor.Image`/SPHEREx window cutout pixel coordinates.
        minval : `float`, optional
            Minimum value threshold for PSF. Default is 0.
        radius : `float`, optional
            Radius for PSF extraction. Default is None.
        modelMask : optional
            Model mask for PSF. Default is None.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        patch : `tractor.patch.Patch`
            Patch of PSF

        """
        if radius is not None:
            raise NotImplementedError("input parameter `radius` not supported yet")
        if modelMask is not None:
            raise NotImplementedError("input parameter `modelMask` not supported yet")

        # Cache patches by pixel coordinates to avoid redundant computation
        if not hasattr(self, '_patch_cache'):
            self._patch_cache = {}
        cache_key = (px, py)
        if cache_key in self._patch_cache:
            return self._patch_cache[cache_key]

        hires_psf = self._psf.get_blended_psf(px + self.xshift, py + self.yshift)

        # Calculate pixel coordinates of the lower-left corner of the high-res
        # PSF. The 0.5 offset accounts for the PSF centroid being at the
        # center of the lower-left pixel of an even-shaped image.
        x0_hires = (px + 0.5) * self._psf.oversampling - (hires_psf.shape[1] - 1) // 2 - 0.5
        y0_hires = (py + 0.5) * self._psf.oversampling - (hires_psf.shape[0] - 1) // 2 - 0.5

        # Split into integer and sub-pixel parts for Lanczos shifting
        ix0_hires = round(x0_hires)
        iy0_hires = round(y0_hires)
        dx_hires = x0_hires - ix0_hires
        dy_hires = y0_hires - iy0_hires
        hires_psf = lanczos_shift_image(hires_psf, dx_hires, dy_hires)

        # Pad the high-res PSF so it aligns with the detector-pixel boundaries
        npad_left = ix0_hires % self._psf.oversampling
        npad_bottom = iy0_hires % self._psf.oversampling
        nover = (hires_psf.shape[1] + npad_left) % self._psf.oversampling
        npad_right = self._psf.oversampling - nover if nover > 0 else 0
        nover = (hires_psf.shape[0] + npad_bottom) % self._psf.oversampling
        npad_top = self._psf.oversampling - nover if nover > 0 else 0

        h0, w0 = hires_psf.shape
        padded = np.zeros(
            (h0 + npad_bottom + npad_top, w0 + npad_left + npad_right),
            dtype=hires_psf.dtype,
        )
        padded[npad_bottom:npad_bottom + h0, npad_left:npad_left + w0] = hires_psf
        hires_psf = padded

        # Update lower-left corner coordinates after padding
        x0_hires = ix0_hires - npad_left
        y0_hires = iy0_hires - npad_bottom
        assert x0_hires % self._psf.oversampling == 0
        assert y0_hires % self._psf.oversampling == 0

        # Downsample to detector-pixel scale
        outimg = resize(hires_psf, dsize=None, fx=1/self._psf.oversampling, fy=1/self._psf.oversampling,
                        interpolation=INTER_AREA)
        outimg /= outimg.sum()
        ix0 = x0_hires // self._psf.oversampling
        iy0 = y0_hires // self._psf.oversampling

        patch = tractor.patch.Patch(ix0, iy0, outimg)
        self._patch_cache[cache_key] = patch
        return patch

    def _padInImageHiRes(self, height: int, width: int, px: float, py: float,
                         img: ArrayLike = None) -> Tuple[np.ndarray, int, int]:
        """Embed high-resolution PSF image into shape (`height`, `width`).

        Parameters
        ----------
        height, width : `int`
            Embedded image shape.
        px, py : `float`
            x- and y-axis position of the source, respectively, in
            `tractor.Image`/SPHEREx window cutout pixel coordinates.
        img : array-like, optional
            High-resolution PSF image to embed. Default is None.

        Returns
        -------
        pad : `numpy.ndarray`
            The embedded high-resolution PSF image.
        subimg_cx, subimg_cy : `int`
            The x- and y-coordinates, respectively, of the center of the
            subimage in the embedded image.
        """
        if img is None:
            psfimg = self._psf.get_blended_psf(px + self.xshift, py + self.yshift)  # use hi-res PSF image
            subimg = psfimg / psfimg.sum()
        else:
            subimg = img.copy()

        return _subimg_center(height, width, subimg)


# =============================================================================
# Fitting functions
def MJysr_to_uJy(flux_MJysr: u.Quantity, oversample: int, pix_map: np.ndarray,
                 xpos_idx: np.ndarray | list[int], ypos_idx: np.ndarray | list[int]) -> u.Quantity:
    """Convert specific intensity to spectral flux density.

    Parameters
    ----------
    flux_MJysr : `astropy.units.Quantity`
        Input source flux (MJy/sr).
    oversample : `int`
        Oversampling factor.
    pix_map : `astropy.units.Quantity`
        Solid angle pixel map (arcsec^2).
    xpos_idx : `numpy.ndarray` or `list`[`int`]
        Zero-based indexes closest to source X positions on the detector.
    ypos_idx : `numpy.ndarray` or `list`[`int`]
        Zero-based indexes closest to source Y positions on the detector.

    Returns
    -------
    `astropy.units.Quantity`
        Converted source flux (uJy).

    """
    assert (flux_MJysr.unit == "MJy/sr"), "Input flux is not in MJy/sr"
    assert (pix_map.unit == u.arcsec**2), "Input pixel map is not in arcsec^2"
    flux_uJy = flux_MJysr.to(u.uJy / u.arcsec**2) * pix_map[ypos_idx, xpos_idx] / oversample**2
    return flux_uJy


def tractor_fit_spherex(
    spx_cutout: Cutout2D,
    spx_variance: np.ndarray,
    spx_flags: np.ndarray,
    source_list: Table,
    detector: int,
    cutout_origin: tuple,
    psf: PSFZoneData | ImagePSF,
    hips_name: str,
    keep_frame_sources: bool = False,
    return_fit: bool = False,
    fit_shapes: bool = True,
    exp_id: int = 0,
) -> Table | dict[str, Table | HDUList]:
    """Compute forced photometry with Tractor of sources in a SPHEREx cutout.

    Parameters
    ----------
    spx_cutout : `astropy.nddata.Cutout2D`
        A small subregion of a SPHEREx image. WCS info should be included.
    spx_variance : `numpy.ndarray`
        Variance for the relevant subregion.
    spx_flags : `numpy.ndarray`
        Flags for the relevant subregion.
    source_list : `astropy.table.Table`
        Sources to fit. Must include columns "x_window", "y_window", "ra",
        "dec", "shape_r", "ab_ratio", "phi", and "sersic".
    detector : `int`
        Detector number (1-6).
    cutout_origin : `tuple`
        Position (x,y) in pixels of the cutout origin on full SPHEREx detector.
        Used to query the PSF.
    psf : `spx_data.psf.PSFZoneData` | `photutils.psf.ImagePSF`
        If `PSFZoneData`, contains the exposure-averaged PSF grid for the
        detector. If `ImagePSF`, contains the effective PSF (ePSF) model
        for the zone nearest the cutout center.
    hips_name : `str`
        Name of the column in `source_list` storing HEALPix indices.
    keep_frame_sources : `bool`
        If `True`, returns photometry for sources in the outer frame of the
        cutout. Default is `False`.
    return_fit : `bool`, optional
        If `True`, fit model and residual image are returned. Default is
        `False`.
    fit_shapes : bool, optional
        If `True`, treat galaxies as shapes during modeling. If False, treat
        galaxies as point sources. Default is `True`.
    exp_id : int, optional
        Per-image identifier (exposure/LVF ID) combined with each source's
        ``SPHERExRefID`` to seed the flux initialization deterministically.
        Default ``0`` seeds on source ID alone. Makes the fit reproducible
        across re-runs without coupling a source's measurements across images.

    Returns
    -------
    phot_meas_arr: `numpy.ndarray`
        Fluxes/errors of the sources provided in `source_list`. Includes
        photometry for frame sources if `keep_frame_sources` is ``True``.
    fluxfit: `dict`[`str`, `numpy.ndarray` | `list`]
        If `return_fit` is ``True``, returns a dictionary including the flux
        measurement array and a list of information for plotting residuals.

    """
    # Set up PSF for fitting
    if isinstance(psf, PSFZoneData):
        spherex_psf = PSFZoneDataTractor(psf, detector, xshift=cutout_origin[0], yshift=cutout_origin[1])
    elif isinstance(psf, ImagePSF):
        spherex_psf = EPSFTractor(psf, detector)

    # Construct source objects for Tractor
    tractor_source_objs = np.empty(shape=len(source_list), dtype=object)
    for i, (_x, _y, _ra, _dec, _rser, _ab, _phi, _nu) in enumerate(zip(
        source_list["x_window"],
        source_list["y_window"],
        source_list["ra"],
        source_list["dec"],
        source_list["shape_r"],  # LS half-light radius in arcsec
        source_list["ab_ratio"],  # derived axis ratio a/b
        source_list["phi"],  # derived position angle
        source_list["sersic"]  # LS sersic index
    )):
        # NOTE: We assign the flux assuming the same units as the input cutout,
        # MJy/sr (since model is initialized with LinearPhotCal). Deterministic
        # per-source flux initialization: seed on (SPHERExRefID, exp_id) so the
        # fit reproduces across re-runs, while each (source, image) measurement
        # still draws independently.
        _seed = (abs(int(source_list["SPHERExRefID"][i])), int(exp_id))
        _flux = Flux(np.random.default_rng(_seed).uniform(high=2))
        if (not fit_shapes) or (_rser < 1) or (_rser is None):
            # Start off with point source
            _src = PointSource(PixPos(_x, _y), _flux)
        else:
            # Start off with extended source
            ba = 1. / _ab
            cut_angle = np.arccos(spx_cutout.wcs.wcs.pc[0, 0]) * 180. / np.pi
            # Get position angle sign following "East of North" convention
            pc_det = np.linalg.det(spx_cutout.wcs.wcs.pc)
            if pc_det < 0:
                # E is CCW of N (right-handed Cartesian)
                pos_ang = _phi + cut_angle
            elif pc_det > 0:
                # E is CW of N (left-handed Cartesian)
                pos_ang = - (_phi + cut_angle)
            _src = SersicGalaxy(
                PixPos(_x, _y),
                _flux,
                GalaxyShape(_rser, ba, pos_ang),
                SersicIndex(_nu)
            )

        # Freeze position for all sources (pure forced photometry)
        _src.freezeAllRecursive()
        _src.thawParam("brightness")
        tractor_source_objs[i] = _src

    tim = tractor.Image(
        data=spx_cutout.data.value,
        invvar=1 / spx_variance,
        psf=spherex_psf,
        wcs=NullWCS(pixscale=spx_param.get_pix_scale().value),
        photocal=LinearPhotoCal(1.0),
        sky=ConstantSky(0.0),
    )
    tim.freezeAllRecursive()  # Fix all image parameters initially
    tim.thawParam("sky")  # Unfreeze the background

    # Build Tractor object and double-check source types
    if not fit_shapes:
        source_types = [type(source) for source in tractor_source_objs]
        assert SersicGalaxy not in source_types
    trac_spherex = tractor.Tractor([tim], tractor_source_objs)

    # Optimize the fit with Tractor, positions frozen
    optres = trac_spherex.optimize_forced_photometry(variance=True, sky=True)
    optres_inv_var = optres.IV

    # Get constant local background estimated by Tractor
    # NOTE: If a good spectral background estimate is available, the final
    #       background will be calculated in the Perform Forced Photometry
    #       run method as the sum of local and global backgrounds.
    src_bkgd = tim.getSky().getConstant()

    if keep_frame_sources is False:  # filter frame sources out
        tractor_source_objs = tractor_source_objs[source_list["central_source"]]
        optres_inv_var = optres_inv_var[source_list["central_source"]]
        source_list = source_list[source_list["central_source"]]

    # Precompute constant values for all sources within the window
    model_image = trac_spherex.getModelImage(0)
    residual_over_noise = np.abs(spx_cutout.data.value - model_image) / np.sqrt(spx_variance)
    bad_pixel_flags = (
        Flag.TRANSIENT | Flag.OVERFLOW | Flag.SUR_ERROR | Flag.NONFUNC | Flag.DICHROIC
        | Flag.MISSING_DATA | Flag.HOT | Flag.COLD | Flag.NONLINEAR | Flag.CROSSTALK
        | Flag.GHOST | Flag.GHOST_FPA | Flag.GHOST_EXT | Flag.STREAK | Flag.PERSIST
        | Flag.BLOOM | Flag.SNOWBALL | Flag.HALO | Flag.SATELLITE_HALO
    )
    cutout_shape = spx_cutout.data.shape

    # Extract arrays and vectorize WCS to avoid repeated lookups in loop below
    ra_arr = source_list["ra"].data
    dec_arr = source_list["dec"].data
    ref_id_arr = source_list["SPHERExRefID"].data
    x_img_arr = source_list["x_image"].data
    y_img_arr = source_list["y_image"].data
    hips_arr = source_list[hips_name].data
    pos_x_arr, pos_y_arr = spx_cutout.wcs.world_to_pixel_values(ra_arr, dec_arr)

    # Construct array of output measurements
    phot_meas_arr = np.empty(shape=len(source_list),
                             dtype=[("SPHERExRefID", "int64"),
                                    ("ra", "float64"),
                                    ("dec", "float64"),
                                    ("x_image", "float32"),
                                    ("y_image", "float32"),
                                    ("mjd", "float64"),
                                    ("lambda", "float32"),
                                    ("lambda_width", "float32"),
                                    ("flux", "float32"),
                                    ("flux_err", "float32"),
                                    ("flux_bkg", "float32"),
                                    ("ext_flg", "bool"),
                                    ("flags", "int64"),
                                    ("fit_ql", "float32"),
                                    ("deep_flg", "bool"),
                                    ("lvf_id", "int64"),
                                    ("det_id", "uint8"),
                                    (hips_name, "int32")])
    for i in range(0, len(source_list)):
        # Calculate source flux/flux error
        src_flux = tractor_source_objs[i].getParams()[0]
        src_flux_err = 1 / np.sqrt(optres_inv_var[i])

        # Get source region pixels
        source_center = PixCoord(x=pos_x_arr[i], y=pos_y_arr[i])
        reg_mask = CirclePixelRegion(center=source_center, radius=2.5).to_mask(mode="center")

        # To account for edge/corner sources: get the bounding box slices
        # overlapping the window (with respect to the cutout/bbox origin),
        # then extract masked pixels
        slices_cutout, slices_mask = reg_mask.get_overlap_slices(shape=cutout_shape)
        submask = reg_mask.data[slices_mask].astype(bool)

        # Calculate fit quality and photometry flags for the source region
        fit_ql = np.nanmean(residual_over_noise[slices_cutout][submask])
        src_reg_flags = spx_flags[slices_cutout][submask]
        phot_flag = np.bitwise_or.reduce(src_reg_flags)
        bad_phot = np.any((src_reg_flags & bad_pixel_flags) != 0)
        if bad_phot:
            phot_flag |= PhotometryFlag.CONTAINS_BAD_PIXEL

        # If fit failed, set fit quality to NaN and add fit error flag
        if (np.isnan(src_flux) or np.isnan(src_flux_err)):
            fit_ql = np.nan
            phot_flag |= PhotometryFlag.FIT_ERROR

        # Populate array element with column values
        # NOTE: To save memory, we modify flux_bkg (in global spectral case)
        #       and populate mjd, lambda, lambda_width, deep_flg, lvf_id, and
        #       det_id with values in the Perform Forced Photometry run method.
        phot_meas_arr[i] = (
            ref_id_arr[i],                                                       # SPHERExRefID
            ra_arr[i],                                                           # ra
            dec_arr[i],                                                          # dec
            x_img_arr[i],                                                        # x_image
            y_img_arr[i],                                                        # y_image
            0,                                                                   # mjd
            0,                                                                   # lambda
            0,                                                                   # lambda_width
            src_flux,                                                            # flux
            src_flux_err,                                                        # flux_err
            src_bkgd,                                                            # flux_bkg
            True if type(tractor_source_objs[i]) is SersicGalaxy else False,     # ext_flg
            phot_flag,                                                           # flags
            fit_ql,                                                              # fit_ql
            0,                                                                   # deep_flg
            0,                                                                   # lvf_id
            0,                                                                   # det_id
            hips_arr[i]                                                          # <hips_name>
        )

    if return_fit:
        # Return measurement table AND relevant info to examine residuals
        pos_str = f"{cutout_origin[0]},{cutout_origin[1]}"
        hdu_model = PrimaryHDU(trac_spherex.getModelImage(0)*spx_cutout.data.unit,
                               Header(cards={"BUNIT": spx_cutout.data.unit.to_string()}))
        hdu_cutout = ImageHDU(spx_cutout.data, name="CUTOUT",
                              header=Header(cards={"BUNIT": spx_cutout.data.unit.to_string()}))
        hdu_src = BinTableHDU(data=source_list, name="WINDOW CAT")
        hdu_var = ImageHDU(spx_variance, name="VARIANCE",
                           header=Header(cards={"BUNIT": (spx_cutout.data.unit**2).to_string()}))
        hdu_chi = ImageHDU(trac_spherex.getChiImage(0), name="CHI")
        hdul = HDUList([hdu_model, hdu_cutout, hdu_src, hdu_var, hdu_chi])
        fluxfit = {"flux_result": phot_meas_arr, "fit": [pos_str, hdul]}
        return fluxfit
    else:
        return phot_meas_arr
