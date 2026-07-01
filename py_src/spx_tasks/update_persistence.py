__all__ = ["UpdatePersistenceConfig", "UpdatePersistenceTask", "PersistenceModel", "PersistenceModelState"]

# Standard library imports
import copy

# Related third party imports
from astropy.time import Time, TimeDelta
import astropy.units as u
from dataclasses import dataclass
from lsst.pex.config import Field, ListField, DictField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import (SPHERExImage, Flag, PersistenceState)
from spx_tests import sample_data
from spx_tasks import add_history
from spx_utils import log_dqa_result

# ==============================================================================


class UpdatePersistenceConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "exposure"),
                                   defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # default dataset type for input
        doc="Input level 1 image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after nonlinearity correction.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    persistence_image = cT.Output(
        name="persistence_image",
        doc="Persistence image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class UpdatePersistenceConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=UpdatePersistenceConnections):

    i_pthresh = Field(
        "Threshold above which persistence current affects slope. Detector dependent.",
        dtype=float,
        default=0.1
    )

    i_preset = Field(
        "Threshold below which persistence current is negligible. Detector dependent.",
        dtype=float,
        default=0.1
    )

    c_n = ListField(
        "Decay coefficients c_n (list of 6 floats)",
        dtype=float,
        default=[0.183, 0.029, 0.0046, 0.000729, 0.000115, 0.0000183]
    )

    tau_n = ListField(
        "Decay time constants tau_n in seconds (list of 6 floats)",
        dtype=float,
        default=[10, 63.1, 398.1, 2511.9, 15849, 100000]
    )

    flu_sat = DictField(
        "Fluence at which the persistence current saturates for each array [e-].",
        keytype=int,
        itemtype=float,
        default={1: 295000., 2: 311000., 3: 259000., 4: 183200., 5: 221100., 6: 213700.}
    )

    ip_sat = DictField(
        "The maximum persistence current observed in each array [e-/s].",
        keytype=int,
        itemtype=float,
        default={1: 81.02, 2: 91.05, 3: 77.17, 4: 27.97, 5: 42.76, 6: 38.30}
    )

    r_flu = DictField(
        "rate at which persistence current increases with fluence and is equal to ip sat/flusat [1/s].",
        keytype=int,
        itemtype=float,
        default={1: 0.0002746, 2: 0.0002928, 3: 0.0002980, 4: 0.0001527, 5: 0.0001934, 6: 0.0001792}
    )

    handle_early_trans = Field(
        "Fallback to NOT handle early transient case for persistnece flagging.",
        dtype=bool,
        default=False,
    )


def dqa_check_persistence_flux(spximg: SPHERExImage, max_fraction=0.2) -> dict:
    """Check that the number of PERSIST-flagged pixels
       is less than or equal to 20% of all pixels.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The processed SPHEREx image containing flux and flags.
    max_fraction : `float`, optional
        Maximum allowed fraction of PERSIST-flagged pixels (default: 0.2 = 20%).

    Returns
    -------
    result_dict : `dict`
        Dictionary suitable for Level2DQA with 'result' and 'description'.
    """
    total_pixels = spximg.data.size
    persist_mask = (spximg.flags & Flag.PERSIST) > 0
    n_persist = np.sum(persist_mask)
    fraction = n_persist / total_pixels

    if fraction > max_fraction:
        msg = (f"{n_persist} out of {total_pixels} pixels ({fraction:.2%}) are flagged as PERSIST, "
               f"which exceeds the allowed limit of {max_fraction:.0%}.")
        result = {"result": "FAIL", "description": msg}
    else:
        msg = (f"{n_persist} out of {total_pixels} pixels ({fraction:.2%}) are flagged as PERSIST, "
               f"within the allowed limit of {max_fraction:.0%}.")
        result = {"result": "PASS", "description": msg}

    log_dqa_result("persistence dqa", result)
    return result


def _print_flag_combinations(mask, flags, label):
    """Print unique flag combinations with binary and string names."""
    from spx_data import Flag  # adjust this import to match your code base
    unique_vals = np.unique(flags[mask])
    print(f"\n{label} – matched {np.count_nonzero(mask)} pixels:")
    for val in unique_vals:
        flag_bin = f"{val:032b}"
        try:
            flag_names = [f.name for f in Flag if (val & f) == f and f != Flag(0)]
        except ValueError:
            flag_names = ["<invalid combination>"]
        print(f"  {val:5d} | {flag_bin} | {', '.join(flag_names)}")


@dataclass
class PersistenceModelState:
    """Persistence Model to manage states.

    Attributes
    ----------
    a_n : `~numpy.ndarray`
        Coefficient for current exposure;
        array length is number of terms in model.
    b_prev2 : `~numpy.ndarray`
        Coefficient for exposure before the previous.
    spximg_prev : `~numpy.ndarray`
        Slopes from the previous and before the previous exposures.
    spximg_prev_flg : `~numpy.ndarray`
        Previous exposure flags.
    num_prev : `int`
        Number of previous exposures.
    prev1_end_time: `float`
        Previous exposure end time in MJD.
    prev2_end_time: `float`
        Exposure before previous exposure end time in MJD.
    """

    a_n: np.ndarray[float]
    b_prev2: np.ndarray[float]
    spximg_prev: np.ndarray
    spximg_prev_flg: np.ndarray
    num_prev: int
    prev1_end_time: float
    prev2_end_time: float


class PersistenceModel:
    def __init__(self,
                 c_n: np.ndarray,
                 tau_n: np.ndarray,
                 i_thresh: np.double,
                 i_reset: np.double,
                 model_state: PersistenceModelState,
                 config: UpdatePersistenceConfig):
        """Persistence model to calculate coefficients.

        Parameters
        ----------
        c_n : ~numpy.ndarray
            Unitless coefficient of decay time;
            array length is number of terms in model.
        tau_n : ~numpy.ndarray
            Time constant in seconds; array length is number of terms in model.
        i_thresh : `np.double`
            Current above which persistence affects dominate.
        i_reset : `np.double`
            Current below which persistence is considered negligible.
        model_state : `PersistenceModelState`
            Initial state of persistence model.
        """

        self.c_n = c_n
        self.tau_n = tau_n
        self.i_thresh = i_thresh
        self.i_reset = i_reset
        self._state = model_state
        self.config = config

    def compute_persistence_current(self,
                                    spximg: SPHERExImage,
                                    t_slew_2_1: float,
                                    t_int: float) -> tuple[np.ndarray, np.ndarray | None]:
        """Computes the estimated persistence current in an image,
           and updates the persistence model state.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input SPHEREx image.
        t_slew_2_1 : `float`
            Elapsed time in seconds (including t_int and t_slew), from end of
            previous exposure to end of second previous.
        t_int : `float`
            Integration time in seconds.

        Returns
        -------
        persist_current and persist_flg.: `tuple` [`~numpy.ndarray`, `~numpy.ndarray`]
            Estimated persistence current in active pixel area.

        Notes
        -----
        Assumes exposures are called to this function "in-order".
        """
        # No previous exposure.
        if self._state.num_prev == 0:
            # No previous exposure, return 0 persistence current.
            persist_current = np.zeros(spximg.data.shape)  # Active area
            persist_flg = None
            self._state.num_prev += 1
            # Save data for first previous.
            self._state.spximg_prev[0] = spximg.data
            self._state.spximg_prev_flg = spximg.flags
            # Update previous exposure end time.
            self._state.prev2_end_time = self._state.prev1_end_time
            self.update_prev1_end_time(spximg)
            return persist_current, persist_flg

        # One or more previous exposures.
        else:
            # Compute current exposure coefficient (Eq. 2).
            self._state.a_n = self._compute_coefficient(spximg, t_slew_2_1, t_int)

            # Compute persistence current, (Eq. 4)
            persist_current = self._compute_persistence_current_estimate(
                spximg.getExposureEndTime(),
                Time(self._state.prev1_end_time, format='mjd'),
                t_int,
            )

            # Set persistence flags mask.
            persist_flg = persist_current >= self.i_thresh
            # Make sure to update use: image.setFlagsFromMask(mask, Flag.PERSIST)

            self._state.spximg_prev[1] = self._state.spximg_prev[0]
            self._state.spximg_prev[0] = spximg.data
            self._state.spximg_prev_flg = spximg.flags
            self._state.num_prev += 1
            if self._state.num_prev >= 2:
                self._state.b_prev2 = self._state.a_n
            # Update previous exposure end time.
            self._state.prev2_end_time = self._state.prev1_end_time
            self.update_prev1_end_time(spximg)

            return persist_current, persist_flg

    def get_model_state(self) -> PersistenceModelState:
        """Returns the persistence model state."""
        return self._state

    def _overwrite_b_n(self, b_prev1, array_id) -> np.ndarray:
        """Logic to overwrite previous flux based on flags from previous exposure.
           (Only applied to this module).

        Returns
        -------
        overwritten_flux : ~numpy.ndarray
            Overwritten flux based on prior flags.
        """
        # overwrite flux
        b_prev1_overwrite: np.ndarray = copy.copy(b_prev1)

        # Masks from table.
        flags = self._state.spximg_prev_flg

        # Match: F_OVERFLOW=1, F_ERROR=1, F_TRANSIENT=0 or 1
        oversat_msk = ((flags & Flag.OVERFLOW) > 0) & ((flags & Flag.SUR_ERROR) > 0)

        # Match: F_OVERFLOW=1, F_ERROR=0, F_TRANSIENT=1
        sat_msk =\
            ((flags & Flag.OVERFLOW) > 0) & ((flags & Flag.SUR_ERROR) == 0) & ((flags & Flag.TRANSIENT) > 0)

        # Match: F_OVERFLOW=0, F_ERROR=1, F_TRANSIENT=1
        if self.config.handle_early_trans is True:
            early_trans = (((flags & Flag.OVERFLOW) == 0)
                           & ((flags & Flag.SUR_ERROR) > 0) & ((flags & Flag.TRANSIENT) > 0))

        # Overwrite flux based on truth table logic.
        b_prev1_overwrite[np.where(oversat_msk)] = self.config.ip_sat[array_id]
        b_prev1_overwrite[np.where(sat_msk)] = self.config.ip_sat[array_id]
        if self.config.handle_early_trans is True:
            b_prev1_overwrite[np.where(early_trans)] = self.config.ip_sat[array_id]
        # print(f"ip_sat = {self.config.ip_sat[array_id]}")

        # _print_flag_combinations(oversat_msk, flags, "oversat_msk")
        # _print_flag_combinations(sat_msk, flags, "sat_msk")
        # _print_flag_combinations(early_trans, flags, "early_trans")

        return b_prev1_overwrite

    def _lookup_table(self,
                      flux: np.ndarray,
                      band: int,
                      t_int: float,
                      use_flux: bool = True) -> np.ndarray:
        """Compute previous exposure coefficients using Bn_flux or Bn_flu from config.

        Parameters
        ----------
        flux : np.ndarray
            The flux values from the previous exposure.
        band : int
            Array ID (1–6).
        t_int : float
            Integration time in seconds.
        use_flux : bool, optional
            If True, use Bn_flux(f, t_int); otherwise use Bn_flu(flu). Default is True.

        Returns
        -------
        b_prev1 : np.ndarray
            Array of Bn values (same shape as `flux`).
        """
        values = flux if use_flux else flux * t_int
        values = np.asarray(values, dtype=np.float64)

        if use_flux:
            flu_thresh = self.config.flu_sat[band] / t_int
            bn = np.where(
                values <= 0, 0,
                np.where(
                    values <= flu_thresh,
                    self.config.r_flu[band] * values * t_int,
                    self.config.ip_sat[band]
                )
            )
        else:
            flu_thresh = self.config.flu_sat[band]
            bn = np.where(
                values <= 0, 0,
                np.where(
                    values <= flu_thresh,
                    self.config.r_flu[band] * values,
                    self.config.ip_sat[band]
                )
            )

        return bn

    def _compute_coefficient(self,
                             spximg: SPHERExImage,
                             t_slew_2_1: float,
                             t_int: float) -> np.ndarray:
        """Computes the coefficient A_n for the current exposure.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Current exposure flux.
        t_slew_2_1 : `float`
            Slew time in seconds.
        t_int : `float`
            Integration time in seconds.

        Returns
        -------
        a_n : `~numpy.ndarray`
            The persistence model coefficients for the current exposure, per pixel.
        """
        print(f"compute coefficient using t_int = {t_int}")
        # Get previous flux.
        flux_prev1: np.ndarray = copy.copy(self._state.spximg_prev[0])

        array_id = spximg.getArrayId()
        b_prev1 = self._lookup_table(flux_prev1, array_id, t_int)
        b_prev1_overwrite = self._overwrite_b_n(b_prev1, array_id)

        # Compute *current* exposure A_n coefficient from previous B_n(k-1)(Eq. 2).
        # Array of coefficients per pixel.
        # Expand b_prev1 to enable broadcasting
        b_prev1_expanded = b_prev1_overwrite[:, :, np.newaxis]

        # Calculate a_n using broadcasting
        if self._state.num_prev < 2 or self._state.b_prev2 is None:
            # No prev2 yet, skip the exponential decay term
            a_n = b_prev1_expanded
        else:
            a_n = b_prev1_expanded + self._state.b_prev2 * np.exp(-t_slew_2_1 / self.tau_n)

        # Check after 10 exposures.
        if self._state.num_prev % 10 == 0:
            # Checks if three consecutive exposures have pixels flux values
            # less than the persistence current reset threshold.
            msk_0 = spximg.data < self.i_reset
            msk_1 = self._state.spximg_prev[0] < self.i_reset
            msk_2 = self._state.spximg_prev[1] < self.i_reset
            ovrflw_1 = (spximg.flags & ~Flag.OVERFLOW) > 0
            ovrflw_2 = (self._state.spximg_prev_flg[0] & ~Flag.OVERFLOW) > 0
            ovrflw_3 = (self._state.spximg_prev_flg[1] & ~Flag.OVERFLOW) > 0
            # Resets pixels to zero based on reset threshold.
            a_n[np.where(msk_0 & msk_1 & msk_2 & ovrflw_1 & ovrflw_2 & ovrflw_3)] = 0.

        return a_n

    def _compute_persistence_current_estimate(self,
                                              mjd_end_e: Time,
                                              mjd_end_c: Time,
                                              t_int: float) -> np.ndarray:
        """Computes the estimated persistence current for an exposure.

        Parameters
        ----------
        self.a_n : `float`
            Model coefficient for the current exposure.
        mjd_end_e : `astropy.time.Time`
            End exposure time for evaluated exposure in MJD.
        mjd_end_c : `astropy.time.Time`
            End exposure time for current exposure in MJD.
        t_int : `float`
            Integration time in seconds.
        self.c_n : ~numpy.ndarray
            Unitless model coefficient of term n.
        self.tau_n : ~numpy.ndarray
            Time constant of term n in seconds.

        Returns
        -------
        persist_current : ~numpy.ndarray
            Estimted persistence current.
        """
        # Calculate the ideal decay time
        t_k = (mjd_end_e - mjd_end_c).sec - (t_int / 2)
        # Check if the calculated time is physically possible
        if t_k < 0:
            raise ValueError(
                f"Calculated negative decay time (t_k = {t_k:.2f}s). "
                f"This implies out-of-order or corrupted timestamps. "
                f"Current Exposure End: {mjd_end_e.mjd}, Previous Exposure End: {mjd_end_c.mjd}"
            )
        sum_term = self._state.a_n * self.c_n * np.exp(-t_k / self.tau_n)
        # A floating point number estimating the persistence current per pixel.
        persist_current: np.ndarray = np.sum(sum_term, axis=2)

        return persist_current

    def update_prev1_end_time(self, spximg: SPHERExImage) -> None:
        """Updates the previous state exposure end time; needed for slew time."""
        self._state.prev1_end_time = spximg.getExposureEndTime().mjd


class UpdatePersistenceTask(pipeBase.PipelineTask):
    """Implementation of L2 Update Persistence Mask and Correction.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-012

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "update-persistence"
    ConfigClass = UpdatePersistenceConfig
    full_name = "Update Persistence Mask"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-012 v2.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            persistence_state: PersistenceState | None = None
            ) -> pipeBase.Struct:
        """

        Parameters
        ----------
        spximg
            Non-linearity corrected photocurrent image (electrons/s)
        persistence_state
            Persistence model state from the previous image processing

        Returns
        -------
        output : `~lsst.pipe.base.Struct`
        """
        # If persistence state not provided, init to empty (no connections).
        # Will be handled in custom quantum graph.
        if persistence_state is None:
            persistence_state = sample_data.get_sample_persistence_state(spximg)

        c_n = self.config.c_n
        tau_n = self.config.tau_n
        i_thresh = self.config.i_pthresh
        i_reset = self.config.i_preset

        # Define the input state for the PersistenceModel.
        input_model_state = PersistenceModelState(
            a_n=persistence_state.get_model_state().a_n,
            b_prev2=persistence_state.get_model_state().b_prev2,
            spximg_prev=persistence_state.get_model_state().b_prev1_data,
            spximg_prev_flg=persistence_state.get_model_state().b_prev1_flgs,
            num_prev=persistence_state.num_prev,
            prev1_end_time=persistence_state.prev1_end_time,
            prev2_end_time=persistence_state.prev2_end_time
        )

        # Utilize `TELAPSE` instead of `TINT` to get integration time.
        t_int = TimeDelta(spximg.meta["TELAPSE"] * u.s).sec

        # Handle unset prev1_end_time
        if input_model_state.prev1_end_time == 0:
            prev1_end_time = Time(0.0, format='mjd')  # placeholder
        else:
            prev1_end_time = Time(input_model_state.prev1_end_time, format='mjd')

        if input_model_state.prev2_end_time == 0:
            t_slew_2_1 = 0.0
            prev2_end_time = Time(0.0, format='mjd')
        else:
            prev2_end_time = Time(input_model_state.prev2_end_time, format='mjd')
            t_slew_2_1 = (prev1_end_time - prev2_end_time).sec

        # Instantiate our persistence model now.
        persistence_model = PersistenceModel(c_n,
                                             tau_n,
                                             i_thresh,
                                             i_reset,
                                             input_model_state,
                                             config=self.config)

        # Get Estimated Persistence.
        persist_current, persist_flgs = persistence_model.compute_persistence_current(spximg,
                                                                                      t_slew_2_1,
                                                                                      t_int)

        # Get Persistence State.
        state = persistence_model.get_model_state()

        # Update Persistence State.
        persistence_state_out = persistence_state.update_state(state.a_n,
                                                               state.b_prev2,
                                                               state.spximg_prev,
                                                               state.spximg_prev_flg,
                                                               state.num_prev,
                                                               state.prev1_end_time,
                                                               state.prev2_end_time)

        spximg_out = copy.deepcopy(spximg)
        # output = Output(spximg_out, persist_current, persist_flgs, persistence_state)

        # Persistence Image w/ persistence current as data.
        persistence_image = SPHERExImage(data=persist_current.astype(np.float32),
                                         flags=spximg_out.flags,
                                         meta=spximg_out.meta,
                                         unit="electron / s")
        # NOTE TODO Cutout active area at end.
        # spectral wcs will most likely not be correct after cutout.
        # implement slicing for spectral wcs (separate task).
        persistence_image.flag_defs = spximg_out.flag_defs
        # Persistence Flags.
        if persist_flgs is not None:
            spximg_out.setFlagsFromMask(persist_flgs, Flag.PERSIST)
            persistence_image.setFlagsFromMask(persist_flgs, Flag.PERSIST)
        spximg_out.header["PSthresh"] = (i_thresh, "[electron/s] persistence threshold")

        # L2 DQA
        _ = dqa_check_persistence_flux(spximg_out, self.config.i_pthresh)
        # Add statistic to header
        npix_flagged_persist = np.sum((spximg_out.flags & Flag.PERSIST) > 0)
        print(f'npix_flagged_persist = {npix_flagged_persist}')
        spximg_out.header["HIERARCH L2 N_PERSIST"] = (npix_flagged_persist,
                                                      "number of pixels flagged persist")

        return pipeBase.Struct(spximg_out=spximg_out,
                               persistence_state_out=persistence_state_out,
                               persistence_image=persistence_image)
