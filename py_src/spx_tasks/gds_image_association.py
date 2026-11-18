__all__ = ["GdsImageAssociationConfig", "GdsImageAssociationTask"]

# Standard library imports
import copy
from datetime import datetime

# Related third party imports
from astropy.table import join, Table
from astropy.time import Time

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field
import numpy as np
from typing import Optional
import pandas as pd
from typing import Literal

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils.aggregate_gds import AggregateGDS
from spx_tasks import add_history
from spx_data import PlannedObservation

# ==============================================================================


def separate_mnemonics_by_type(sband_table: Table):
    """Separates Mnemonic values grouped by Type from an input table.

    Parameters
    ----------
    sband_table : `~astropy.table.Table`
        Input table containing 'Type' and 'Mnemonic' columns.

    Returns
    -------
    `dict`
        Dictionary where keys are unique Type values, values are Mnemonic lists.

    """
    grouped_mnemonics = {}
    for row in sband_table:
        row_type = row['Type']
        row_mnemonic = row['Mnemonic'].lower()  # Convert to lowercase
        if row_type not in grouped_mnemonics:
            grouped_mnemonics[row_type] = []
        grouped_mnemonics[row_type].append(row_mnemonic)
    return grouped_mnemonics


def populate_headers(spximg_out: SPHERExImage,
                     pos: list, vel: list,
                     val_type: Literal["START", "MID", "END"]):
    """Populate FITS header fields with spacecraft position and velocity metadata.

    This function updates the FITS header of the `spximg_out` object with
    spacecraft position (X_SC, Y_SC, Z_SC) and velocity (VX_SC, VY_SC, VZ_SC)
    values, appending a suffix based on the `val_type`. If `val_type` is "MID",
    no suffix is used. Otherwise, "_START" or "_END" is appended accordingly.

    Parameters
    ----------
    spximg_out : `spx_data.SPHERExImage`
        SPHERExImage object
    pos : `list`
        Spacecraft position [x, y, z] in kilometers.
    vel : `list`
        Spacecraft velocity [vx, vy, vz] in kilometers per second.
    val_type : {"START", "MID", "END"}
        Indicates the type of value being stored (start, midpoint, or end of exposure).

    Returns
    -------
    spximg_out : `spx_data.SPHERExImage`
        SPHERExImage object

    Notes
    -----
    The function adds six header fields:
    - Position: `X_SC`, `Y_SC`, `Z_SC` (or suffixed with `_START` or `_END`)
    - Velocity: `VX_SC`, `VY_SC`, `VZ_SC` (or suffixed with `_START` or `_END`)

    Each header key is also accompanied by a descriptive comment.

    """
    if val_type == "MID":
        suffix = ""
    else:
        suffix = "_"+val_type

    hierarch = "HIERARCH " if len(suffix) > 4 else ""
    spximg_out.header[f"{hierarch}X_SC{suffix}"] = pos[0]
    spximg_out.header.comments[f"X_SC{suffix}"] = f"[km] Exposure {val_type} x position"
    spximg_out.header[f"{hierarch}Y_SC{suffix}"] = pos[1]
    spximg_out.header.comments[f"Y_SC{suffix}"] = f"[km] Exposure {val_type} y position"
    spximg_out.header[f"{hierarch}Z_SC{suffix}"] = pos[2]
    hierarch = "HIERARCH " if len(suffix) > 3 else ""
    spximg_out.header.comments[f"Z_SC{suffix}"] = f"[km] Exposure {val_type} z position"
    spximg_out.header[f"{hierarch}VX_SC{suffix}"] = vel[0]
    spximg_out.header.comments[f"VX_SC{suffix}"] = f"[km/s] Exposure {val_type} v_x velocity"
    spximg_out.header[f"{hierarch}VY_SC{suffix}"] = vel[1]
    spximg_out.header.comments[f"VY_SC{suffix}"] = f"[km/s] Exposure {val_type} v_y velocity"
    spximg_out.header[f"{hierarch}VZ_SC{suffix}"] = vel[2]
    spximg_out.header.comments[f"VZ_SC{suffix}"] = f"[km/s] Exposure {val_type} v_z velocity"
    spximg_out.header["HIERARCH XYZ_SC_SYSTEM"] = "GEOCENTER"
    spximg_out.header.comments["HIERARCH XYZ_SC_SYSTEM"] = "Reference frame for spacecraft position (XYZ)"
    spximg_out.header["HIERARCH VXYZ_SC_SYSTEM"] = "GEOCENTER"
    spximg_out.header.comments["HIERARCH VXYZ_SC_SYSTEM"] = "Reference frame for spacecraft velocity (VXYZ)"
    spximg_out.header["PLEPHEM"] = "DE440"
    spximg_out.header.comments["PLEPHEM"] = "Planetary ephemeris used for state vector calculations"
    return spximg_out


def get_pos_velocity(julian_date: int, df: pd.DataFrame, spk: Table):
    """Get position and velocity vectors for a given Julian date.

    Parameters
    ----------
    julian_date : int
        The Julian date for which to retrieve position and velocity.
    df : pd.DataFrame
        A DataFrame containing a 'time' column of datetime objects
        for matching observations.
    spk : astropy.table.Table
        A SPICE-style table containing 'position' and 'velocity'
        columns with vector data.

    Returns
    -------
    position : list of float
        The [x, y, z] position vector corresponding to the closest time in `df`.
    velocity : list of float
        The [dx, dy, dz] velocity vector corresponding to the closest time in `df`.

    Notes
    -----
    - This function assumes the `df['time']` column is in datetime format.
    - It selects the time closest to the provided Julian date.

    """
    # Find the velocity information corresponding to a particular time
    observation_time = Time(julian_date, format='jd').to_datetime()
    closest_time_index = (df['time'] - observation_time).abs().idxmin()
    # Extract position and velocity
    position_data = spk['position']
    velocity_data = spk['velocity']
    position = position_data[closest_time_index]
    velocity = velocity_data[closest_time_index]

    x, y, z = position[0], position[1], position[2]
    dx, dy, dz = velocity[0], velocity[1], velocity[2]
    return ([x, y, z], [dx, dy, dz])
# Function to merge columns with the same logical meaning


def merge_columns(table: Table, new_col_name: str, col1: str, col2: str):
    """Merge two columns with the same logical meaning
       into a single column in an Astropy Table.

    Parameters
    ----------
    table : `astropy.table.Table`
        The input Astropy Table containing the columns to be merged.
    new_col_name : `str`
        The name of the new column that will store the merged values.
    col1 : `str`
        The name of the first column to merge.
        Values from this column are prioritized.
    col2 : `str`
        The name of the second column to merge.
        Values from this column are used if `col1` contains invalid data.

    Notes
    -----
    Operates on the combined table in-place.

    """
    merged_column = []
    for v1, v2 in zip(table[col1], table[col2]):
        if v1 != '--' and v1 != -1.0:  # Prefer numerical if valid
            merged_column.append(v1)
        elif v2 != '--':  # Otherwise, take flag value
            merged_column.append(v2)
        else:
            merged_column.append('--')  # Default for no valid data
    table[new_col_name] = merged_column
    table.remove_columns([col1, col2])


def doy_to_isot(sgt_datetime: str) -> str:
    """Convert ground track-formatted date/time strings to ISO 'T' format.

    Parameters
    ----------
    sgt_datetime : `str`
        Input ground track date/time string in year/day format
        (YYYY-DDDTHH:MM:SS)

    Returns
    -------
    `str`
        Converted date/time string in year/month/day ISO 'T' format
        (YYYY-MM-DDTHH:MM:SS).

    """
    dt = datetime.strptime(sgt_datetime, "%Y-%jT%H:%M:%S.%f")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


class GdsImageAssociationConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure"),
                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    sband = cT.Input(
        name="sband",
        doc="S-band data",
        storageClass="GdsData",
        dimensions=["instrument", "observation"],
        isCalibration=False
    )
    sband_mnemonics = cT.PrerequisiteInput(
        name="sband_mnemonics",
        doc="List of sband data to aggregate",
        storageClass="SbandMnemonics",
        dimensions=["instrument"],
        isCalibration=True
    )
    planned_observation = cT.Input(
        name="planned_observation",
        doc="planned observation data",
        storageClass="PredictedObservation",
        dimensions=["instrument", "observation"],
        isCalibration=False
    )
    sgt = cT.Input(
        name="sgt",
        doc="Reconstructed ground track data",
        storageClass="AstropyTable",
        dimensions=["instrument", "observation"],
        isCalibration=False
    )

    spk = cT.Input(
        name="spk",
        doc="Pointing ephemeris data",
        storageClass="GdsData",
        dimensions=["instrument", "observation"],
        isCalibration=False
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        # Mark each input as not required if use is not set.
        if config is not None:
            if not self.config.useSband:
                self.input.discard("sband")
                self.input.discard("sband_mnemonics")
            if not self.config.useSgt:
                self.input.discard("sgt")
            if not self.config.useSpk:
                self.input.discard("spk")
            if not self.config.usePlannedObs:
                self.input.discard("planned_observation")


class GdsImageAssociationConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=GdsImageAssociationConnections):
    """Configuration for GdsImageAssociation task."""

    useSband = Field(
        dtype=bool,
        default=False,
        doc="Whether to use S-band telemetry data (sband, sband_mnemonics)."
    )

    useSgt = Field(
        dtype=bool,
        default=False,
        doc="Whether to use SPHEREx Ground Track data."
    )

    useSpk = Field(
        dtype=bool,
        default=False,
        doc="Whether to use SPHEREx Ephemeris data."
    )
    usePlannedObs = Field(
        dtype=bool,
        default=False,
        doc="Whether to use PEF data"
    )


class GdsImageAssociationTask(pipeBase.PipelineTask):
    """Associates GDS data with a SPHEREx image.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-002

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "gds-image-association"
    ConfigClass = GdsImageAssociationConfig
    full_name = "GDS Image Association"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-002 v2.1.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            planned_observation: PlannedObservation = None,
            sgt: Optional[Table] = None,
            sband: Optional[Table] = None,
            spk: Optional[Table] = None,
            sband_mnemonics: Optional[Table] = None
            ) -> pipeBase.Struct:
        """Associates GDS data with a SPHEREx image.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input image.
        planned_observation : `PlannedObservation`
            Planned observation data for an observation.
        sgt : `astropy.table.Table`
            Table of reconstructed ground track data.
        spk : `~astropy.table.Table`
            Information about position and velocity of SPHEREx spacecraft
        sband : `astropy.table.Table`
            Sband observation table.
        sband_mnemonics : `astropy.table.Table`
            Table of sband mnemonic values to aggregate.

        Returns
        -------
        output: `~lsst.pipe.base.Struct`
            An object containing the ``spximg_out`` of type `~spx_data.SPHERExImage`
            with a binary table extension that includes GDS Sband data.

        """
        spximg_out = copy.deepcopy(spximg)

        if self.config.useSband:
            if sband is None or sband_mnemonics is None:
                raise RuntimeError("S-band data is required when config.useSband is True.")

            # Check if "Value1" and "Value2" exist in sband
            if "Value1" in sband.colnames and "Value2" in sband.colnames:
                self.log.debug("value1 and value2 column found")
                # If both exist, proceed normally
                pass
            elif "Value" in sband.colnames:
                # If only "Value" exists, rename it to "Value2"
                sband.rename_column("Value", "Value2")
                self.log.debug(f"columns = {sband.colnames}")
                # Create "Value1" as a column of zeros
                sband["Value1"] = np.zeros(len(sband))
            else:
                raise ValueError("Unexpected column structure in sband table. No 'Value' column found.")

            if "Date_Time" in sband.colnames:
                sband.rename_column("Date_Time", "Timestamp")

            # Rename "Keyword" to "Mnemonic" if it exists
            if "Keyword" in sband.colnames:
                sband.rename_column("Keyword", "Mnemonic")

            # Get description of Mnemonic
            # Special handling because description is in "Unnamed: 3"
            # in example GDS data.
            if 'Unnamed: 3' in sband_mnemonics.colnames:
                # Transfer data from 'Unnamed: 3' to 'Description'
                sband_mnemonics['Description'] = sband_mnemonics['Unnamed: 3']
            elif 'Description' in sband_mnemonics.colnames:
                # Ensure 'Description' exists
                self.log.debug("'Description' column already exists.")
            else:
                # Handle case where neither column exists
                raise ValueError("Neither 'Unnamed: 3' nor 'Description' exists in the table.")

            # Group mnemonics by type
            grouped_mnemonics = separate_mnemonics_by_type(sband_mnemonics)

            # Separate numerical and flag mnemonics
            numerical_keywords = []
            flag_keywords = []
            for key, mnemonics in grouped_mnemonics.items():
                if "D" in key:  # Check if the key contains "D" for flags
                    flag_keywords.extend(mnemonics)
                else:  # Otherwise, treat it as numerical
                    numerical_keywords.extend(mnemonics)

            # Aggregate numerical sband data
            aggregator = AggregateGDS(sband, numerical_keywords, flag_keywords)
            # Numerical aggregated dtable
            sband_nums: Table = aggregator.compute_aggregates(sband_mnemonics)

            # Create a table for flag values: first, last, mid, bitmask
            sband_flgs = aggregator.compute_flag_aggregates(sband_mnemonics)

            # Serialize 'FlagOptions' column to a string
            if 'FlagOptions' in sband_flgs.colnames:
                sband_flgs['FlagOptions'] = [
                    ";".join(flags) if isinstance(flags, (list, tuple)) else str(flags)
                    for flags in sband_flgs['FlagOptions']
                ]

            # Merge numerical_agg and flag_agg with unique prefixes
            # for overlapping columns
            combined_table = join(
                sband_nums,
                sband_flgs,
                keys='Keyword',
                join_type='outer',  # Keep all rows from both tables
                table_names=['numerical', 'flag'],  # Add prefixes to distinguish columns
                uniq_col_name='{table_name}_{col_name}'  # Use table-specific column names
            )

            # Merge the columns
            merge_columns(combined_table, 'First', 'numerical_First', 'flag_First')
            merge_columns(combined_table, 'Last', 'numerical_Last', 'flag_Last')
            merge_columns(combined_table, 'Type', 'numerical_Type', 'flag_Type')
            merge_columns(combined_table, 'Description', 'numerical_Description', 'flag_Description')
            merge_columns(combined_table, 'Variability', 'numerical_Variability', 'flag_Variability')

            # Define the desired column order
            desired_order = [
                'Keyword', 'Description', 'First', 'Last', 'Mid',
                'Min', 'Max', 'Median', 'Mean', 'Std', 'Variability',
                'Bitmap', 'FlagOptions', 'Type'
            ]

            # Ensure that all columns in `desired_order` exist in `combined_table`
            existing_columns = [col for col in desired_order if col in combined_table.colnames]

            # Assign the table with the new column order
            combined_table = combined_table[existing_columns]

            # Attach the GDS data table as a binary extension to SPHEREx image
            spximg_out.gds = combined_table

        else:
            self.log.info("Skipping S-band aggregation because useSband is False.")
        if self.config.usePlannedObs:
            # Retrieve SPS target name and flags
            target_name = planned_observation.get_target_name()
            sps_flags_str = planned_observation.get_flag_names()
            in_sband_obs = planned_observation.in_sband
            non_survey_obs = planned_observation.non_survey

            self.log.info(f"Target name: {target_name}")
            self.log.info(f"Flags: {sps_flags_str}")
            # Add this metadata to the SPHERExImage header or output structure if needed
            spximg_out.header["HIERARCH SPS_TARGET"] = target_name
            spximg_out.header.comments['SPS_TARGET'] = "Extended target ID from Survey Planning"
            # Integer target ID for ODIR/IDQR reports (removed after L2 DQA)
            spximg_out.header["HIERARCH SPS_TARGET_PEF"] = planned_observation.target_id
            spximg_out.header.comments['SPS_TARGET_PEF'] = "Integer target ID from PEF"
            spximg_out.header["HIERARCH SPS_FLAGS"] = sps_flags_str
            spximg_out.header.comments["SPS_FLAGS"] = "Flags defined by Survey Planning"
            spximg_out.header["HIERARCH OBS_IN_SBAND"] = str(in_sband_obs)
            spximg_out.header.comments["OBS_IN_SBAND"] = "Observed during S-band transmission"
            spximg_out.header["HIERARCH NON_SURVEY"] = str(non_survey_obs)
            spximg_out.header.comments["NON_SURVEY"] = "Observation is not part of a science survey"

        # Ground Track keywords
        # NOTE: Awaiting confirmation on final keywords from Gregory
        # Make sure the whole header line fits into 80 chars, e.g.
        # HIERARCH SGT_MIN_GCD2SAA = 68.044724 / [deg] Closest ground track dist to SAA

        # Ground track geocentric latitude closest to exposure midpoint time in degrees
        if self.config.useSgt:
            # Extract lat/lon closest to exposure midpoint time
            exp_midpt = Time(spximg.header["DATE-AVG"], format="isot", scale="utc")
            sgt_times = Time([doy_to_isot(date) for date in sgt["Date_Time"]], format="isot", scale="utc")
            delta_times = abs((exp_midpt - sgt_times).value)
            latlon_idx = np.argmin(delta_times).item()

            spximg_out.header['HIERARCH SGT_LAT_MIDPT'] = (
                sgt["Latitude"][latlon_idx],
                "[deg] Ground track latitude at midpoint")
        # Ground track east longitude closest to exposure midpoint time in degrees
            spximg_out.header['HIERARCH SGT_LON_MIDPT'] = (
                sgt["Longitude"][latlon_idx],
                "[deg] Ground track longitude at midpoint")
        # Ground track closest great circle distance to SAA in degrees
            spximg_out.header['HIERARCH SGT_MIN_GCD2SAA'] = (
                np.nanmin(sgt["GCD2SAA"]),
                "[deg] Closest ground track dist to SAA")
        else:
            self.log.info("Skipping SGT aggregation because useSgt is False.")

        if self.config.useSpk:
            self.log.info("Starting SPK aggregation because useSpk is True")

            df = pd.DataFrame()
            df['time'] = pd.to_datetime(spk["time"])

            # Helper to safely call get_pos_velocity
            def safe_populate(time_jd, label):
                try:
                    pos, vel = get_pos_velocity(time_jd, df, spk)
                    return populate_headers(spximg_out, pos, vel, label)
                except Exception as e:
                    self.log.warning(f"Could not populate SPK data for {label} time ({time_jd} JD): {e}")
                    return spximg_out  # unchanged

            # Use exposure start time
            exposure_start_time = spximg.getExposureStartTime()
            spximg_out = safe_populate(exposure_start_time, "START")

            exposure_mid_time = spximg.getExposureStartTime() + (spximg.getExposureDuration() / 2)
            spximg_out = safe_populate(exposure_mid_time, "MID")

            exposure_end_time = spximg.getExposureEndTime()
            spximg_out = safe_populate(exposure_end_time, "END")
        else:
            self.log.info("Skipping SPK aggregation because useSpk is False.")

        return pipeBase.Struct(spximg_out=spximg_out)
