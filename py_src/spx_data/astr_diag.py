__all__ = ["AstrDiag", "CoarseAstrometry", "FineAstrometry", "SipModel",
           "ScampResults", "CoarseCenter", "AstrometryNetResults",
           "QuadrantStat", "QuadrantAstrometryStats"]

# Related third party imports
from pydantic import BaseModel, model_validator
from typing import Optional, List, Tuple


class WCSLinear(BaseModel):
    """A base model for WCS linear transformations
    that accepts either CD or PC formalism.
    """

    # PC Matrix formalism
    PC1_1: Optional[float] = None
    PC1_2: Optional[float] = None
    PC2_1: Optional[float] = None
    PC2_2: Optional[float] = None
    CDELT1: Optional[float] = None
    CDELT2: Optional[float] = None

    # CD Matrix formalism
    CD1_1: Optional[float] = None
    CD1_2: Optional[float] = None
    CD2_1: Optional[float] = None
    CD2_2: Optional[float] = None

    @model_validator(mode='after')
    def check_linear_transform_formalism(self) -> 'WCSLinear':
        """Ensure that exactly one of the CD or PC formalisms is present."""
        has_cd = self.CD1_1 is not None
        has_pc = self.PC1_1 is not None

        # Use an exclusive OR to check that one and only one is true
        if not (has_cd ^ has_pc):
            raise ValueError("WCS data must contain either the CD matrix "
                             "OR the PC matrix + CDELT, but not both or neither.")

        return self


class SipModel(BaseModel):
    """Pydantic model for SIP (Simple Imaging Polynomial) distortion parameters."""

    A_ORDER: int
    B_ORDER: int
    AP_ORDER: int
    BP_ORDER: int
    CRPIX: Tuple[float, float]
    A: List[List[float]]
    B: List[List[float]]
    AP: List[List[float]]
    BP: List[List[float]]


class CoarseAstrometry(WCSLinear):
    """Pydantic model for the coarse astrometry WCS solution."""

    CRVAL1: float
    CRVAL2: float
    CTYPE1: str
    CTYPE2: str
    SIP: SipModel


class FineAstrometry(WCSLinear):
    """Pydantic model for the fine astrometry WCS solution."""

    CRVAL1: float
    CRVAL2: float
    CTYPE1: str
    CTYPE2: str
    SIP: SipModel


class ScampResults(BaseModel):
    """Pydantic model for the results from SCAMP."""

    dAXIS1_arcsec: float
    dAXIS2_arcsec: float
    CONTRAST: float
    match_nstars: int
    match_nstars_sn: int


class AstrometryNetResults(BaseModel):
    """Pydantic model for the results from solve-field."""

    num_matches: int
    pixel_scale: float
    clip_trans: bool


class CoarseCenter(BaseModel):
    """Pydantic model for the location of the coarse center on the fine image."""

    x_pixel: float
    y_pixel: float
    coordinate_base: str
    description: str


class QuadrantStat(BaseModel):
    """Per-quadrant (or total) astrometric statistics in arcsec."""

    std_arcsec: Optional[float] = None
    rms_arcsec: Optional[float] = None
    n_matches: int
    # For TOTAL this can be None;
    # for quadrants it's True/False depending on min_per_quad
    sufficient_matches: Optional[bool] = None
    # New: mean residuals in RA/Dec (arcsec)
    mean_delta_ra_arcsec: Optional[float] = None
    mean_delta_dec_arcsec: Optional[float] = None


class QuadrantAstrometryStats(BaseModel):
    """Container for total and per-quadrant astrometric statistics."""

    TOTAL: QuadrantStat
    Q1: Optional[QuadrantStat] = None
    Q2: Optional[QuadrantStat] = None
    Q3: Optional[QuadrantStat] = None
    Q4: Optional[QuadrantStat] = None


class AstrDiag(BaseModel):
    """Pydantic model to store the Astrometry Diagnostics (AstrDiag) data,
    fields are optional to allow for storing partial results on failure.
    """

    # This part is always available, so it's not optional
    coarse_astrometry: CoarseAstrometry
    n_calibrators: int
    n_transients: int

    # These parts are only available on success
    fine_astrometry: Optional[FineAstrometry] = None
    scamp_results: Optional[ScampResults] = None
    astrometry_net_results: Optional[AstrometryNetResults] = None
    coarse_center_on_fine_image: Optional[CoarseCenter] = None
    roll_angle_difference_deg: Optional[float] = None
    quadrant_astrometry_stats: Optional[QuadrantAstrometryStats] = None

    # Extra astrometry / SCAMP / LDAC diagnostics
    galactic_lon_deg: Optional[float] = None
    galactic_lat_deg: Optional[float] = None
    abs_galactic_lat_deg: Optional[float] = None

    ldac_n_sources: Optional[float] = None
    ldac_mag_lower: Optional[float] = None
    ldac_mag_upper: Optional[float] = None
    ldac_min_sources_target: Optional[int] = None
    ldac_max_sources_target: Optional[int] = None
    ldac_selection_n_iter: Optional[int] = None

    scamp_position_maxerr: Optional[float] = None
    scamp_posangle_maxerr: Optional[float] = None
    scamp_pixscale_maxerr: Optional[float] = None
    scamp_cross_radius_arcsec: Optional[float] = None
    scamp_distort_degrees: Optional[float] = None
    scamp_n_iter: Optional[int] = None

    se_detect_minarea: Optional[float] = None
    se_detect_maxarea: Optional[float] = None
    se_detect_thresh_first: Optional[float] = None
    se_detect_thresh_second: Optional[float] = None

    # Status flags and error messages
    astronet_solved: Optional[bool] = None
    FINAST: int  # Will always set this, so it's required
    error_message: Optional[str] = None  # Add a field for error messages
