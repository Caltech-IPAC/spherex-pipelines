__all__ = ["BackgroundModel", "BackgroundMesh", "NullGlobal"]

from typing import TypeVar, Any
from pathlib import Path
import abc

import numpy as np
from photutils.background import Background2D
from astropy.io import fits
from astropy import units
from astropy.utils import lazyproperty


from astropy.nddata import NDData, StdDevUncertainty


TBackgroundMesh = TypeVar("TBackgroundMesh", bound="BackgroundMesh")


class BackgroundModel(abc.ABC):
    @abc.abstractmethod
    def get_background(self) -> NDData:
        pass


class BackgroundMesh(BackgroundModel):
    """ Converts `photutils.background.Background2D` into an object with helper methods.

    Parameters
    ----------
    background_mesh : `~numpy.ndarray`
        The low-resolution background image.
    background_rms_mesh : `~numpy.ndarray`
        The low-resolution background RMS image.
    box_size : array_like (int)
        The box size along each axis.
    data_size : array_like (int)
        Full resolution image size along each axis
    unit : `~astropy.units.UnitBase` instance or str, optional
        The units of the data.
    """
    def __init__(self,
                 background_mesh,
                 background_rms_mesh,
                 box_size,
                 data_size,
                 unit=None):
        self.background_mesh = background_mesh
        self.background_rms_mesh = background_rms_mesh
        self.box_size = box_size
        self.data_size = data_size
        self.unit = unit
        self.meta = self.get_meta()  # derived, can be initialized lazily
        self.interpolator = ZoomInterpolator()

    def __eq__(self, other: Any):
        return isinstance(other, BackgroundMesh) and \
            np.array_equal(self.background_mesh, other.background_mesh) and \
            np.array_equal(self.background_rms_mesh, other.background_rms_mesh) and \
            np.array_equal(self.box_size, other.box_size) and \
            np.array_equal(self.data_size, other.data_size) and \
            self.unit == other.unit

    def _interpolate(self, mesh: np.ndarray):
        """Function that calls interpolation on mesh.

        Parameters
        ----------
        mesh : `~numpy.ndarray` or array-like
            The low resolution mesh to interpolate.

        Returns
        -------
        background : `units.quantity.Quantity`
            The high resolution background array, with units if applicable.
        """
        background = self.interpolator(mesh, self)
        if self.unit is not None:
            background <<= self.unit
        return background

    def get_background(self) -> NDData:
        """Calculates and returns an `~astropy.nddata.NDData` with high-res background.

        Returns
        -------
        nddata : `~astropy.nddata.NDData`
            The n-dimensional dataset with meta, uncertainty, and image.
        """
        background = self._interpolate(self.background_mesh)
        background_rms = self._interpolate(self.background_rms_mesh)

        # Auto includes unit since background will have it
        nddata = NDData(background, uncertainty=StdDevUncertainty(background_rms), meta=self.meta)

        return nddata

    def get_meta(self) -> fits.Header:
        """Builds and returns the metadata for background.

        Returns
        -------
        meta : `~astropy.io.fits.Header`
            The metadata for the background in a `~astropy.io.fits.Header`.
        """
        meta = fits.Header()
        meta["BUNIT"] = ("" if self.unit is None else str(self.unit),
                         "Unit of pixel value")
        meta["BKGBOX1"] = (self.box_size[0],
                           "Background estimation box size 1")
        meta["BKGBOX2"] = (self.box_size[1],
                           "Background estimation box size 2")
        meta["BKGAXIS1"] = (self.data_size[0],
                            "Shape of original image")
        meta["BKGAXIS2"] = (self.data_size[1],
                            "Shape of original image")
        return meta

    def to_hdu(self) -> fits.HDUList:
        """Generates a hdulist for the background object.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            A hdulist of all hdus.
        """
        # Create primary hdu to append to
        hdus = [fits.PrimaryHDU(data=None, header=None),
                fits.ImageHDU(data=self.background_mesh,
                              header=self.meta,
                              name="BKG MESH"),
                fits.ImageHDU(data=self.background_rms_mesh,
                              name="BKG RMS MESH", header=None)]

        hdulist = fits.HDUList(hdus)
        return hdulist

    @staticmethod
    def from_hdu(hdu: fits.HDUList) -> TBackgroundMesh:
        """Creates a `~spx_data.BackgroundMesh` object from `~astropy.io.fits.HDUList`.

        Parameters
        ----------
        hdu : `~astropy.io.fits.HDUList`
            HDUList containing all the information needed to build background.

        Returns
        -------
        background_mesh : `~spx_data.BackgroundMesh`
            Class instance created from `~astropy.io.fits.HDUList`.
        """
        header = hdu[1].header
        data_size = (header["BKGAXIS1"], header["BKGAXIS2"])
        box_size = (header["BKGBOX1"], header["BKGBOX2"])
        background_mesh = hdu[1].data
        background_rms_mesh = hdu[2].data
        unit = units.Unit(header["BUNIT"])
        background = BackgroundMesh(background_mesh, background_rms_mesh, box_size,
                                    data_size, unit=unit)
        return background

    def write(self, filename: Path, **kwargs):
        """Writes the `~spx_data.BackgroundMesh` to a fits file.

        Parameters
        ----------
        filename : `~pathlib.Path` or `str`
            The path and name for writing the fits file.
        """
        hdulist = self.to_hdu()
        hdulist.writeto(filename, **kwargs)

    @staticmethod
    def read(filename: Path, **kwargs) -> TBackgroundMesh:
        """Reads a fits file and creates a `BackgroundMesh`.

        Parameters
        ----------
        filename : `~pathlib.Path` or `str`
            The path and filename of the fits file to read.

        Returns
        -------
        bkg : `~spx_data.BackgroundMesh`
            The class instance created.
        """
        with fits.open(filename, **kwargs) as hdulist:
            background = BackgroundMesh.from_hdu(hdulist)
        return background

    @staticmethod
    def from_background2d(bkg2D: Background2D) -> TBackgroundMesh:
        return BackgroundMesh(bkg2D.background_mesh, bkg2D.background_rms_mesh,
                              bkg2D.box_size, bkg2D.data.shape, unit=bkg2D.unit)


class ZoomInterpolator:
    """
     This class generates full-sized background and background RMS images
     from lower-resolution mesh images using the `~scipy.ndimage.zoom`
     (spline) interpolator.

     This class is almost identical to photutils.BkgZoomInterpolator
     but is decoupled from the `Background2D` class.

     Parameters
     ----------
     order : int, optional
         The order of the spline interpolation used to resize the
         low-resolution background and background RMS mesh images.  The
         value must be an integer in the range 0-5.  The default is 3
         (bicubic interpolation).

     mode : {'reflect', 'constant', 'nearest', 'wrap'}, optional
         Points outside the boundaries of the input are filled according
         to the given mode.  Default is 'reflect'.

     cval : float, optional
         The value used for points outside the boundaries of the input if
         ``mode='constant'``. Default is 0.0

     grid_mode : bool, optional
         If `True` (default), the samples are considered as the centers
         of regularly-spaced grid elements. If `False`, the samples
         are treated as isolated points. For zooming 2D images,
         this keyword should be set to `True`, which makes zoom's
         behavior consistent with `scipy.ndimage.map_coordinates` and
         `skimage.transform.resize`. The `False` option is provided only
         for backwards-compatibility.

     clip : bool, optional
         Whether to clip the output to the range of values in the
         input image. This is enabled by default, since higher order
         interpolation may produce values outside the given input range.
     """

    def __init__(self, *, order=3, mode='reflect', cval=0.0, grid_mode=True,
                 clip=True):
        self.order = order
        self.mode = mode
        self.cval = cval
        self.grid_mode = grid_mode
        self.clip = clip

    def __call__(self, mesh, params: BackgroundMesh):
        """
        Resize the 2D mesh array.

        Parameters
        ----------
        mesh : `~numpy.ndarray`
            The low-resolution 2D mesh array.

        params : `~spx_data.BackgroundMesh`
            Includes box_size and data_size, which is the shape of
            the resized background or background RMS image 2D array.

        Returns
        -------
        result : `~numpy.ndarray`
            The resized background or background RMS image 2D array.
        """

        zoom_factor = params.box_size
        out_shape = params.data_size

        mesh = np.asanyarray(mesh)
        if np.ptp(mesh) == 0:
            np.zeros(out_shape, dtype=mesh.dtype)
            return np.zeros(out_shape, dtype=mesh.dtype) + np.min(mesh)

        from scipy.ndimage import zoom

        # The mesh is first resized to the larger padded-data size
        # (i.e., zoom_factor should be an integer) and then cropped
        # back to the final data size.
        result = zoom(mesh, zoom_factor, order=self.order, mode=self.mode,
                      cval=self.cval, grid_mode=self.grid_mode)
        result = result[0:out_shape[0], 0:out_shape[1]]

        if self.clip:
            minval = np.min(mesh)
            maxval = np.max(mesh)
            np.clip(result, minval, maxval, out=result)

        return result


class NullGlobal(Background2D):
    """Subclass for crowded images where `photutils.background.Background2D`
       initialization fails.
    """
    def __init__(self, data, box_size, **kwargs):
        super().__init__(data=np.full(data.shape, 0), box_size=box_size, **kwargs)

    # Override default lazy properties
    @lazyproperty
    def background(self):
        return np.full(self.data.shape, np.nan)

    @lazyproperty
    def background_mesh(self):
        mesh_shape = np.array([int(i) for i in np.divide(self.data.shape, self.box_size)])
        return np.full(mesh_shape, np.nan)
