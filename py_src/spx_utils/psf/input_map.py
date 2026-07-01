import numpy as np
from spx_utils.psf.psf_parameters import SuperresolutionPSF


class Sources(SuperresolutionPSF):
    """Select sources and regrid PSF.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage`
        Input SPHEREx image.
    cat : `Table`
        Input catalog.
    x_min : `float`
        Minimum x value for PSF zone.
    x_max : `float`
        Maximum x value for PSF zone.
    y_min : `float`
        Minimum y value for PSF zone.
    y_max : `float`
        Maximum y value for PSF zone.
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.spximg = kwargs['spximg']
        self.cat = kwargs['cat']
        self.det_id = kwargs['det_id']
        x_min = kwargs['x_min']
        x_max = kwargs['x_max']
        y_min = kwargs['y_min']
        y_max = kwargs['y_max']
        self.ra = self.cat['ra']
        self.dec = self.cat['dec']
        self.abmag = self.cat[f'mag_det{self.det_id}']
        self.selfabmag = self.cat[f'mag_det{self.det_id}']
        self.bright_neighb = self.cat['flag_brightneighbor']
        self.dist_nearest = self.cat['dist_nearest']
        self.star = self.cat['spherex_class']  # psf flag.
        self.w = self.spximg.wcs
        self.cat_pix = self.w.all_world2pix(self.ra, self.dec, 1)  # Returns a list of 2 numpy arrays

        self.cat_pix = np.vstack((self.cat_pix[0][:], self.cat_pix[1][:])).T

        sourceSize = SuperresolutionPSF.halfSourceSizeDown * 2

        # NOTE Preseve for applying Margin of sourceSize/2.
        self.whplx =\
            np.where(np.logical_and(self.cat_pix[:, 0] < (SuperresolutionPSF.gridSizeX - sourceSize / 2),
                                    self.cat_pix[:, 0] > (sourceSize / 2)))
        self.whply =\
            np.where(np.logical_and(self.cat_pix[:, 1] < (SuperresolutionPSF.gridSizeX - sourceSize / 2),
                                    self.cat_pix[:, 1] > (sourceSize / 2)))

        # Bin x/y min/max boundaries.
        self.zone_mask_x = np.logical_and(self.cat_pix[:, 0] < x_max, self.cat_pix[:, 0] > x_min)
        self.zone_mask_y = np.logical_and(self.cat_pix[:, 1] < y_max, self.cat_pix[:, 1] > y_min)
        # Gaia stars, flagged by ``class`` == 1.
        self.starmsk = (self.star & (1 << 1)) > 0  # Select only PSF calibrators
        # Select stars with no neighbor closer than 10 arcseconds
        self.distmsk = self.dist_nearest > 10.

        self.whplx = np.where(np.logical_and(self.zone_mask_x, self.starmsk, self.distmsk))
        self.whply = np.where(np.logical_and(self.zone_mask_y, self.starmsk, self.distmsk))

        self.good_coords = np.intersect1d(self.whplx, self.whply)
        try:
            self.oversamp = self.spximg.header["OVERSAMP"]
        except KeyError:
            self.oversamp = self.config.oversample

    def cat_coords(self) -> np.ndarray:
        """Down sampled catalog coordinates."""
        sourceCoordDown = self.cat_pix[self.good_coords, :]
        sourceCoordDown -= 0.5
        return sourceCoordDown

    def vmag(self):
        """AB magnitude of good source coordinates."""
        vmag = np.array(self.abmag[self.good_coords])
        return vmag

    def num_sources(self, vmag) -> int:
        """Retrieves the number of good source coordiinates."""
        # vmag = self.vmag()
        n_sources = len(vmag)
        return n_sources

    def star_mask(self):
        """Initialization of star mask."""
        starmask = np.zeros(self.spximg.flags.shape, dtype=np.bool_)
        return starmask

    def stat_mask(self):
        """Initialization of stat mask."""
        statmask = np.zeros(self.spximg.flags.shape, dtype=np.bool_)
        return statmask

    def magnitude(self, abmag):
        """Apply magnitude restriction to selected good sources."""
        # Find which source magnitudes obey desired limit and filter out the rest
        good_mags = np.where(np.logical_and(abmag[:] >= SuperresolutionPSF.mag_range_upper,
                             abmag[:] <= SuperresolutionPSF.mag_range_lower))
        return good_mags
