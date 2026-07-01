__all__ = ["DetectorPixelData", "NonLinearityParameters", "PersistenceParameters",
           "PersistenceState", "PersistenceStateModelPars", "ReadNoiseParameters",
           "PersistenceLookupTable"]

# Standard library imports
import re

# Related third party imports
from deprecated.sphinx import deprecated
from astropy.io import fits
from astropy.table import Table, hstack
from dataclasses import dataclass
import numpy as np
from typing import Union

# Local application/library specific imports


class DetectorPixelData:
    """Base class for per-detector pixel-level data.
    This class should not be called by itself.
    Any class inheriting DetectorPixelData must have
    its required_extensions and optional_extensions set.

    """
    required_extensions = []
    optional_extensions = []

    def __init__(self, data, detector: Union[int, None] = None):

        """
        Post initialization (instead of __init__ as this is a dataclass)
        This method sets the hdulist property, creates a detector
        and checks if all required extensions are present.
        Parameters
        ----------
        data : `~astropy.io.fits.HDUList`
            Data representing multi extension FITS file
        """
        self.hdulist: fits.HDUList = data
        if detector is not None:
            self.detector = detector
        self._detect_detector()
        self.validate()

    def _check_optional_extensions(self):
        pass

    def _check_required_extensions(self) -> None:
        """
        Compare extensions of HDUList to required
        extensions and throwing exception in that case.
        Raises
        ------
        ValueError
            If there are missing extensions.
        """
        extnames = [hdu.header['EXTNAME'] for hdu in self.hdulist if 'EXTNAME' in hdu.header]

        missing_extension = set(self.required_extensions) - set(extnames)
        if missing_extension:
            raise ValueError(f"Required extensions are missing: {missing_extension}")

    def get_extension_data(self, extname: str) -> np.ndarray:
        """In a multi-extension FITS file, return the numpy data associated with
        a given  extension.
        Parameters
        ----------
        extname : str
            FITS reserved keyword "EXTNAME"
        Returns
        -------
        result : `~numpy.ndarray`
            Numpy data in the requested extension
        """
        return self.hdulist[extname].data

    def get_extension_header(self, extname: str = None) -> fits.header.Header:
        """In a multi-extension FITS file, return the header associated with
        a given  extension.
        Parameters
        ----------
        extname : str
            FITS reserved keyword "EXTNAME"
        Returns
        -------
        fits.header.Header
            FITS extension header.
        """
        return self.hdulist[extname].header

    @property
    def detector(self) -> int:
        """Getter for detector number/ID is contained in the PrimaryHDU header."""
        return self.hdulist[0].header.get("DETECTOR")

    @detector.setter
    def detector(self, value: int):
        """Setter for detector number/ID."""
        self.hdulist[0].header["DETECTOR"] = value

    def _detect_detector(self) -> None:
        """ Checks that the detector number/ID is contained in the PrimaryHDU header.

        Raises
        ------
        ValueError
            ValueError if detector information is missing in the PrimaryHDU header.
        """
        if self.hdulist[0].header["DETECTOR"] is None:
            raise ValueError("DETECTOR information is missing in the header.")

    def validate(self):
        self._check_required_extensions()
        self._check_optional_extensions()


class NonLinearityParameters(DetectorPixelData):
    """
    Dataclass for NonLinearity Parameters

    Parameters
    ----------
    DetectorPixelData
        DetectorPixelData base class
    """
    required_extensions = ['Q_nl', 'Qmax']
    optional_extensions = ['b1', 'b2', 'b3']

    def _check_optional_extensions(self):
        """
        Generalizable algorithm for checking if b_n-1, b_n-2 ... b_1 is present

        Raises
        ------
        ValueError
            Error for when transient parameters are discontinuous
        """
        extnames = [hdu.header['EXTNAME'] for hdu in self.hdulist if 'EXTNAME' in hdu.header]
        # Generalizable algorithm for checking if b_n-1, b_n-2 ... b_1 is present
        # when b_n is present.
        # Checking for holes in transient extensions
        transient_pattern = re.compile(r'^b\d+$')  # "b" followed by any integer

        # Match ['b1', 'b2', 'b3'] from ['Q_nl', 'Qmax', 'b1', 'b2', 'b3']
        # then [1,2 3] from ['b1', 'b2', 'b3']
        transient_index_list = [int(string[1:])
                                for string in extnames if transient_pattern.match(string)]  # ex: b2 -> 2
        # If transient extensions are detected.
        if transient_index_list:
            # max([1,2,3]) = 3, generating range(1,4) = [1,2,3]
            # comparing both sets would throw an error if [1,2] or [2,3] is provided.
            missing_tr = set([i for i in range(1, max(transient_index_list)+1)]) - set(transient_index_list)
            if missing_tr:
                raise ValueError(f"Transient extensions are missing:  b{missing_tr} \
                                 from b{transient_index_list}")


class ReadNoiseParameters(DetectorPixelData):
    """Dataclass for read noise calibration parameters.

    Parameters
    ----------
    `DetectorPixelData`
        DetectorPixelData base class
    """
    single = "READNOISE-1"
    full = "READNOISE-2"
    required_extensions = [single, full]


class PersistenceParameters(DetectorPixelData):
    """Persistence Parameters (required fixed calibration inputs).

        Notes
        -----
        - Required calibration inputs: (fixed)
            - c_n, tau_n, (2040x2040xN) => constant coefficients.
            - b_prev1 = lookup table that returns b_{K-1} in "bins", (2040x2040xN).
            - Made the following config parameters, if we want ability to easily modify.
            - f_sat = slope when overflow flag triggered => 1 value (from primary hdu).
            - f_oversat = slope when too high to be accurately determined => 1 value.
            - I_Pthreshold = threshold above which persistence current affects slope.
                Detector dependent, one value, get from primary header.
            - I_Preset = threshold below which persistence current is negligible.
                Detector dependent, one value, get from primary header.
        - Optional : None
     """

    required_extensions = ["c_n", "tau_n", "b_prev1"]

    @deprecated(
        "Deprecated in favor of module configuration parameters. Will be removed.",
        version="6.2",
        category=FutureWarning,
    )
    def __init__(self, hdulist: fits.HDUList | None, detector: Union[int, None] = None,
                 f_sat: Union[float, None] = None, f_ovrsat: Union[float, None] = None,
                 i_pthresh: Union[float, None] = None, i_preset: Union[float, None] = None):
        super().__init__(hdulist, detector)
        self._persist_model = PersistenceStateModelPars()
        if f_sat is not None:
            self.f_sat = f_sat
        if f_ovrsat is not None:
            self.f_ovrsat = f_ovrsat
        if i_pthresh is not None:
            self.i_pthresh = i_pthresh
        if i_preset is not None:
            self.i_preset = i_preset
        self._check_parameters()

    @property
    def i_pthresh(self) -> float:
        """Getter for input calibration persistence threshold, I_pthresh.
           The threshold below which persistence current is negligible.
        """
        return self.hdulist[0].header.get("I_Pthrsh", 0.)

    @i_pthresh.setter
    def i_pthresh(self, value: float):
        """Setter for persistence threshold."""
        self.hdulist[0].header["I_Pthrsh"] = value

    @property
    def i_preset(self) -> float:
        """Getter for input calibration persistence reset, I_preset.
           The threshold below which persistence current is negligible.
        """
        return self.hdulist[0].header.get("I_Preset", 0.)

    @i_preset.setter
    def i_preset(self, value: float):
        """Setter for persistence reset."""
        self.hdulist[0].header["I_Preset"] = value

    @property
    def f_sat(self) -> float:
        """Getter for slope when overflow flag triggered, saturated flux (f_sat)."""
        return self.hdulist[0].header.get('f_sat')

    @f_sat.setter
    def f_sat(self, value: float):
        """Setter for saturated flux."""
        self.hdulist[0].header["f_sat"] = value

    @property
    def f_ovrsat(self) -> float:
        """Getter for slope when too high to be accurately determined,
           oversaturated flux (f_ovrsat)."""
        return self.hdulist[0].header.get('f_ovrsat')

    @f_ovrsat.setter
    def f_ovrsat(self, value: float):
        """Setter for oversaturated flux."""
        self.hdulist[0].header["f_ovrsat"] = value

    def _check_parameters(self) -> None:
        """Check that required parameters exist in the header."""
        if self.hdulist[0].header["I_Pthrsh"] is None:
            raise ValueError("Persistence threhsold, I_Pthrsh, is missing from the header.")
        if self.hdulist[0].header["I_Preset"] is None:
            raise ValueError("Persistence reset value, I_Preset, is missing from the header.")
        if self.hdulist[0].header['f_sat'] is None:
            raise ValueError("Saturated flux, f_sat, is missing from the header.")
        if self.hdulist[0].header['f_ovrsat'] is None:
            raise ValueError("Oversaturated flux, f_ovrsat, is missing from the header.")


@dataclass
class PersistenceStateModelPars:
    """Persistence state coefficient for a L1 SPHEREx image.

    Short-term storage
    ------------------

    - ``a_n`` is a 2040 x 2040 x N array of persistence-model-state coefficients.
    - Will be written out as FITS file.

    """
    a_n: np.ndarray[float] = None
    b_prev2: np.ndarray[float] = None
    b_prev1_data: np.ndarray = None
    b_prev1_flgs: np.ndarray = None


class PersistenceState(DetectorPixelData):
    """Persistence State (updates after each image processed by module).

    Notes
    -----
    State Parameters : (mutable)

    - a_n : `~numpy.ndarray` (2040x2040xN).
    - b_prev2 : `~numpy.ndarray` (2040x2040xN) (b_{K-2}).
    - spximg.data, spximg.flags (K-1) : `~spx_data.SPHERExImage`.
    - num_prev : `int`.
    - prev1_end_time : `~astropy.time.Time`
    - There must always be a valid state, either empty or with values.

    """

    required_extensions = ["a_n", "b_prev2", "b_prev1_data", "b_prev1_flgs"]

    def __init__(self, hdulist: fits.HDUList | None, detector: Union[int, None] = None,
                 num_prev: Union[int, None] = None, prev1_end_time: Union[int, None] = None,
                 prev2_end_time: Union[int, None] = None):
        super().__init__(hdulist, detector)
        self._persist_model = PersistenceStateModelPars()
        if num_prev is not None:
            self.num_prev = num_prev
        if prev1_end_time is not None:
            self.prev1_end_time = prev1_end_time
        if prev2_end_time is not None:
            self.prev2_end_time = prev2_end_time
        self._detect_num_prev()
        self._detect_prev1_end_time()
        self._load_required_extensions()

    @property
    def num_prev(self) -> int:
        """Getter for num_prev."""
        return self.hdulist[0].header.get("NUM_PREV")

    @num_prev.setter
    def num_prev(self, value: int):
        """Setter for num_prev."""
        self.hdulist[0].header["NUM_PREV"] = value

    @property
    def prev1_end_time(self) -> np.double:
        """Getter for prev1_end_time."""
        return self.hdulist[0].header.get("PREV1END")

    @prev1_end_time.setter
    def prev1_end_time(self, value: float):
        """Setter for prev1_end_time."""
        self.hdulist[0].header["PREV1END"] = value

    @property
    def prev2_end_time(self) -> np.double:
        """Getter for prev2_end_time."""
        return self.hdulist[0].header.get("PREV2END")

    @prev2_end_time.setter
    def prev2_end_time(self, value: float):
        """Setter for prev2_end_time."""
        self.hdulist[0].header["PREV2END"] = value

    def _load_required_extensions(self):
        """
        Load data from required extensions into the _persist_model object.
        """
        for ext in self.required_extensions:
            data = self.get_extension_data(ext)
            if data is not None:
                setattr(self._persist_model, ext, data)

    def _detect_num_prev(self) -> None:
        """Checks that the number of previoius exposures is available in header.

        Raises
        ------
        ValueError
            ValueError if not in primary hdu.
        """
        if self.hdulist[0].header.get('NUM_PREV') is None:
            raise ValueError("Exposure number is missing from header.")

    def _detect_prev1_end_time(self) -> None:
        """Checks that previous exposure end time is availabe in header.
        Raises
        ------
        ValueError
            ValueError if not in primary hdu.
        """
        if self.hdulist[0].header.get('PREV1END') is None:
            raise ValueError("Previous image end time is missing from header.")

    def update_state_extension(self, updated_state_data: np.ndarray, extname: str = None):
        """Update a given state extension in PersistenceData.
        (choose what and when to writeout later)

        Parameters
        ----------
        updated_state_data : `~numpy.ndarray`
            Updated persistence state variable.
        extname : `str` | `None`, Default is `None`.
            State extension name.
        """
        self.hdulist[extname].data = updated_state_data

    def get_model_state(self):
        return self._persist_model

    def update_state(self, a_n, b_prev2, b_prev1_data, b_prev1_flgs,
                     num_prev, prev1_end_time, prev2_end_time):
        self._persist_model.a_n = a_n
        self._persist_model.b_prev2 = b_prev2
        self._persist_model.b_prev1_data = b_prev1_data
        self._persist_model.b_prev1_flgs = b_prev1_flgs
        self.num_prev = num_prev
        self.prev1_end_time = prev1_end_time
        self.prev2_end_time = prev2_end_time
        updated_persistence_state = self._update_hdulist()
        return updated_persistence_state

    def _update_hdulist(self):

        primary_header = self.hdulist[0].header.copy()

        # Update ``NUM_PREV`` and ``PREV1END`` keywords in primary header.
        primary_header['NUM_PREV'] = self.num_prev
        primary_header['PREV1END'] = self.prev1_end_time  # float(self.prev1_end_time.mjd)

        # Update state extensions.
        fits_content = [
            fits.PrimaryHDU(header=primary_header),
            fits.ImageHDU(data=self._persist_model.a_n,
                          header=fits.Header({"EXTNAME": "a_n"})),
            fits.ImageHDU(data=self._persist_model.b_prev2,
                          header=fits.Header({"EXTNAME": "b_prev2"})),
            fits.ImageHDU(data=self._persist_model.b_prev1_data,
                          header=fits.Header({"EXTNAME": "b_prev1_data"})),
            fits.ImageHDU(data=self._persist_model.b_prev1_flgs,
                          header=fits.Header({"EXTNAME": "b_prev1_flgs"})),
        ]
        # Update persistence state for pipeline.
        persistence_state = PersistenceState(
            hdulist=fits.HDUList(fits_content), detector=self.detector
        )
        return persistence_state


class PersistenceLookupTable(DetectorPixelData):
    required_extensions = ['min_fluence', 'max_fluence', 'exp_model_amps']

    @deprecated(
        "Deprecated in favor of module configuration parameters. Will be removed.",
        version="6.2",
        category=FutureWarning,
    )
    def __init__(self, data, detector: Union[int, None] = None):
        super().__init__(data, detector)
        self.c_n = [self.hdulist[0].header[f'C{i+1}'] for i in range(6)]
        self.tau_n = [self.hdulist[0].header[f'Tau{i+1}'] for i in range(6)]

    @property
    def combined_table(self) -> Table:
        """Combine the tables from specific extensions into one."""
        min_fluence_table = Table([self.get_extension_data('min_fluence')], names=['min_fluence'])
        max_fluence_table = Table([self.get_extension_data('max_fluence')], names=['max_fluence'])
        exp_model_amps_table = Table([self.get_extension_data('exp_model_amps')], names=['exp_model_amps'])

        # Combine the tables horizontally
        combined_table = hstack([min_fluence_table, max_fluence_table, exp_model_amps_table])
        return combined_table

    # Getters and setters for C* and Tau* values
    def get_c_value(self, index: int) -> float:
        return self.c_n[index - 1]

    def set_c_value(self, index: int, value: float):
        self.hdulist[0].header[f'C{index}'] = value
        self.c_n[index - 1] = value

    def get_tau_value(self, index: int) -> float:
        return self._n[index - 1]

    def set_tau_value(self, index: int, value: float):
        self.hdulist[0].header[f'Tau{index}'] = value
        self.tau_n[index - 1] = value
