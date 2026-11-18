import kete
from astropy.time import Time
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.utils.timer import timeMethod
from lsst.daf.butler import (
    DeferredDatasetHandle
)
from spx_data import ReadoutParameters
from typing import List
from scipy.interpolate import RectBivariateSpline
import importlib
import pandas as pd
from spx_data import ObservationId
from lsst.pex.config import Field
import numpy as np
import pyarrow as pa
from spx_data.spx_param import get_pix_scale
from astropy.table import Table
from spx_data.spx_version import SSDC_PIPELINES_VERSION

__all__ = ["SolarSystemObjectsConfig", "SolarSystemObjectsTask"]

# Conversion factor from degrees/day to arcsec/min.
DEGPERDAY2ARCSECPERMIN = 3600 / (24 * 60)

SPHEREX_PIXEL_SCALE = get_pix_scale()
SPHEREX_PIXEL_SCALE = getattr(SPHEREX_PIXEL_SCALE, "value", SPHEREX_PIXEL_SCALE)

def create_dataframe_from_headers(handles: List[DeferredDatasetHandle],
                                  column_names: list, butlerQC, max_workers=10):
    """Create a DataFrame from the FITS header values for specified column names
    across multiple FITS files using parallel threading.
    """
    # 1. Define the worker function to process a single handle.
    # We define this inside the main function to easily capture
    # 'butlerQC' and 'column_names'
    # without needing to pass them as arguments every time.
    def process_handle(handle):
        # Retrieve the full object (I/O heavy step)
        spximg = butlerQC.get(handle).get()

        # Extract data
        data = {col_name: spximg.header.get(col_name, None) for col_name in column_names}
        data["EXPSTART"] = spximg.getExposureStartTime().value
        data["EXPSTOP"] = spximg.getExposureEndTime().value
        data["OBSID"] = spximg.header.get("OBSID")
        data["DETECTOR"] = spximg.header.get("DETECTOR")
        data["WCS"] = spximg.wcs
        data["META"] = spximg.meta
        return data

    data_list = []

    for handle in handles:
        data_list.append(process_handle(handle))

    return pd.DataFrame(data_list)


def calculate_midpoint_time(header: pd.DataFrame):
    """Calculate the midpoint of exposure times and convert
    to Julian Date in TDB scale.
    """
    midpoint_mjd = (header["EXPSTART"] + header["EXPSTOP"]) / 2
    times = Time(midpoint_mjd, format="mjd", scale="utc")
    return times.tdb.jd


def create_observer_state(row, jd_tdb, obsid):
    """Create the observer's state based on position and velocity data.
    """
    position = kete.Vector(
        np.array([row["X_SC"], row["Y_SC"], row["Z_SC"]])
        / kete.constants.AU_KM,
        frame=kete.Frames.Equatorial
    )
    velocity = kete.Vector(
        np.array([row["VX_SC"], row["VY_SC"], row["VZ_SC"]])
        / kete.constants.AU_KM * 86400,
        frame=kete.Frames.Equatorial
    )

    return kete.State(
        desig=str(obsid)+"_D"+str(row["DETECTOR"]),
        jd=jd_tdb,
        pos=position,
        vel=velocity,
        center_id=399
    ).change_center(10)


def create_fovs(fits_plan, jd_tdbs, obsids, npixMargin):
    """Create a list of FOVs based on FITS plan data.
    """
    fovs = []
    for (_, row), jd_tdb, obsid in zip(fits_plan.iterrows(), jd_tdbs, obsids):
        observer_state = create_observer_state(row, jd_tdb, obsid)
        wcs = row["WCS"]

        # Most of this code is from spx_butler.skypix.skypix_utils.getSPHERExImageRegion
        rp = ReadoutParameters.from_metadata(row["META"])
        npixX, npixY = rp.Jactive, rp.Kactive
        npixMarginX, npixMarginY = (npixMargin, npixMargin)
        if abs(npixMarginX) >= npixX / 2 or abs(npixMarginY) >= npixY / 2:
            raise ValueError(f"Pixel margin {npixMargin} exceeds half-image pixel size")
        x1, x2 = (0 - npixMarginX, (npixX - 1) + npixMarginX)
        y1, y2 = (0 - npixMarginY, (npixY - 1) + npixMarginY)
        cornersX = [x1, x2, x2, x1]
        cornersY = [y1, y1, y2, y2]
        cornersCoords = wcs.pixel_to_world(cornersX, cornersY)

        vectors = []
        for coord in cornersCoords:
            vector = kete.Vector.from_ra_dec(coord.ra.deg, coord.dec.deg)
            vectors.append(vector)

        fov = kete.RectangleFOV.from_corners(vectors, observer=observer_state)
        fovs.append(fov)
    return kete.fov.FOVList(fovs)


def propagate_orbit_elements(states, fovs):
    """Propagate orbital elements to the reference time.
    """
    # Propagate orbital elements to a certain
    # reference time (arbitrarily chosen as
    # the first observation time).

    reference_jd = fovs[0].observer.jd

    return kete.propagate_n_body(states, reference_jd, include_asteroids=False)


def compute_trail(states_i, xposure):
    """Compute on-sky trail length (in pixels) and a trailed flag.

    Uses the RA/Dec rates from ``states_i.ra_dec_with_rates`` (deg/day, with
    RA' already scaled by cos(dec)) together with the exposure time to estimate
    how many pixels each object moves across during the exposure.

    Parameters
    ----------
    states_i : `kete.vector.SimultaneousStates`
        States with a FOV present, so rates can be computed.
    xposure : float
        Exposure time in seconds for this observation.

    Returns
    -------
    trail_pix : `np.ndarray`
        Trail length in pixels for each object, in the same order as the
        states.
    """
    # ra_dec_with_rates returns [RA, DEC, RA', DEC'] with rates in deg/day,
    # in the same order as the states in this object.
    radec_rates = states_i.ra_dec_with_rates
    radec_rates = np.asarray(radec_rates)
    ra_rate = radec_rates[:, 2]
    dec_rate = radec_rates[:, 3]
    # Total proper motion converted to arcsec/min.
    rate_arcsec_per_min = (
        np.sqrt(ra_rate**2 + dec_rate**2) * DEGPERDAY2ARCSECPERMIN
    )
    # Number of pixels the object moves across during the exposure.
    return rate_arcsec_per_min * (xposure / 60.0) / SPHEREX_PIXEL_SCALE


def _hgphi12(alpha__deg):
    """Compute the HG phase function components.

    Parameters
    ----------
    alpha__deg : array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    Returns
    -------
    phi1, phi2 : tuple
        The two components of the HG phase function. See `iau_hg_model`
        for more details.

    """
    alpha__rad = np.deg2rad(alpha__deg)
    sin_a = np.sin(alpha__rad)
    f_a = sin_a/(0.119+1.341*sin_a-0.754*sin_a*sin_a)
    tan_a_half = np.tan(alpha__rad*0.5)
    w = np.exp(-90.56*tan_a_half*tan_a_half)
    phi1_s = 1 - 0.986*f_a
    phi2_s = 1 - 0.238*f_a
    phi1_l = np.exp(-3.332*tan_a_half**0.631)
    phi2_l = np.exp(-1.862*tan_a_half**1.218)
    return (w*phi1_s + (1-w)*phi1_l, w*phi2_s + (1-w)*phi2_l)


def iau_hg_model(alpha__deg, gpar=0.15):
    r"""Compute the IAU HG phase function model in intensity (1 at alpha=0).

    Parameters
    ----------
    alpha__deg : array_like
        The phase angle (Sun-Target-Observer angle) in degrees.

    gpar : float, optional
        The slope parameter ($G$) in the IAU H, G modeling. See Notes.
        By default ``0.15``.

    Notes
    -----
    Semi-empirical model of the phase function of the Moon, asteroids, and
    other (airless) solar system objects. The phase function defined at
    $(0^\\circ \\le \\alpha \\le 120^\\circ)$ for the phase angle $\\alpha$. It
    is given by the following equation:

    .. math::
        \\Phi_\\mathrm{HG}(\\alpha, G) = G \\Phi_{HG1}(\\alpha) +
        (1-G) \\Phi_{HG2}(\\alpha)

    where

    .. math::
        \\Phi_{HG i}(\\alpha) =
        W \\left ( 1-\\frac{C_i \\sin \\alpha}
        {0.119+1.341 \\sin \\alpha-0.754 \\sin ^2 \\alpha} \\right )
        + (1 - W) \\times \\exp \\left \\{ -A_i \\left
        [ \\tan \\frac{\\alpha}{2} \\right ]^{B_i} \\right \\}

    and

    .. math::
        W(\\alpha) = \\exp \\left \\{ -90.56 \\tan^2 \\frac{\\alpha}{2} \\right \\}

    The parameters $A_i$, $B_i$, and $C_i$ are given by:

    .. math::
        A_1, A_2 &= 3.332, 1.862 \\sep
        B_1, B_2 = 0.631, 1.218 \\sep
        C_1, C_2 = 0.986, 0.238

    Reference: Bowell et al. 1989
    https://ui.adsabs.harvard.edu/abs/1989aste.conf..524B/abstract

    """
    hgphi1, hgphi2 = _hgphi12(np.array(np.abs(alpha__deg)))
    # Just to avoid negative alpha error
    return (1 - gpar)*hgphi1 + gpar*hgphi2


def calc_mags_bright_planets(states_i, xposure):
    """Calculate magnitudes for bright planets without brightness values."""
    obs2objs = states_i.obs_vecs
    desigs = [_r.desig for _r in states_i]
    cols = ["spkid", "H", 'G', 'dmax', 'RA', 'DEC', 'alpha', 'rh', 'ro',
         'V', 'Tmax', 'AB5_brightest', 'trail_pix', 'trailed']
    orb = pd.DataFrame(index=pd.Index(desigs, name="desig"), columns=cols)
    _orb = orb.loc[desigs]
    desigs = [_r.desig for _r in states_i]
    sun2obs = states_i.fov.observer.change_center(10).pos
    # Also do it outside the loop.

    alphas = []
    rhels = []
    robss = []
    for _state, obs2obj in zip(states_i, obs2objs):
        # Phase angle alpha
        sun2obj = _state.pos
        obj2obs = sun2obs - sun2obj
        alphas.append(obj2obs.angle_between(-sun2obj))
        # NOTE: _state.pos is heliocentric position
        # Helio/observer-centric distances
        rhels.append(sun2obj.r)
        robss.append(obs2obj.r)
    _orb["RA"] = [vec.as_equatorial.ra for vec in obs2objs]
    _orb["DEC"] = [vec.as_equatorial.dec for vec in obs2objs]
    # Trail length from on-sky motion rates (RA' already scaled by cos(dec)).
    trail_pix = compute_trail(states_i, xposure)
    _orb["trail_pix"] = trail_pix
    _orb["trailed"] = trail_pix > 1
    _orb.index = _orb.index.astype(pd.ArrowDtype(pa.string()))
    return _orb


def calc_mags(states_i, orb, tlut_5um_spl, xposure):
    """Calculate magnitudes and physical properties for solar system objects."""
    obs2objs = states_i.obs_vecs
    desigs = [_r.desig for _r in states_i]

    _orb = orb.loc[desigs][["H", "G", "spkid"]].fillna({"G": 0.15})
    # ^^ Better doing it outside the for loop
    #   - ~0.1s for `orb[orb["desig"] == desigs[i]]`, so ~1s for 10 SSOs (!?)
    #   - <0.1s order for `isin` or `.loc[desigs]`
    _orb["dmax"] = kete.conversion.compute_diameter(
        albedo=0.04, h_mag=_orb["H"])
    # ^^^^^^^^^^ Calculate (almost) maximum possible diameter by using (almost)
    # lowest possible albedo of 0.04.

    sun2obs = states_i.fov.observer.change_center(10).pos
    # Also do it outside the loop.

    alphas = []
    rhels = []
    robss = []
    for _state, obs2obj in zip(states_i, obs2objs):
        # Phase angle alpha
        sun2obj = _state.pos
        obj2obs = sun2obs - sun2obj
        alphas.append(obj2obs.angle_between(-sun2obj))
        # NOTE: _state.pos is heliocentric position
        # Helio/observer-centric distances
        rhels.append(sun2obj.r)
        robss.append(obs2obj.r)
    _orb["RA"] = [vec.as_equatorial.ra for vec in obs2objs]
    _orb["DEC"] = [vec.as_equatorial.dec for vec in obs2objs]
    _orb["alpha"] = alphas
    _orb["rh"] = rhels
    _orb["ro"] = robss
    _orb["V"] = (
        _orb["H"]
        + 5*np.log10(_orb["ro"]*_orb["rh"])
        - 2.5*np.log10(iau_hg_model(_orb["alpha"], _orb["G"]))
    )
    # The (almost) maximum possible
    # pseudo-temperature from 4-band WISE is ~ 450K
    # (Myhrvold et al. 2022 PSJ, 3, 30)
    _orb["Tmax"] = 450/np.sqrt(_orb["rh"])
    _orb["AB5_brightest"] = (
        tlut_5um_spl.ev(_orb["alpha"], _orb["Tmax"])
        - 5*np.log10(_orb["dmax"])
        + 5*np.log10(_orb["ro"])
    )
    # Trail length from on-sky motion rates (RA' already scaled by cos(dec)).
    trail_pix = compute_trail(states_i, xposure)
    _orb["trail_pix"] = trail_pix
    # Flag as trailed if it moves more than 1 pixel during the exposure.
    _orb["trailed"] = trail_pix > 1
    # Perform magnitude cuts
    vcut = 21  # magnitude limit for the V band
    tcut = 20  # thermal emission magnitude limit for the 5um band

    objects_mask = np.array((pd.isna(_orb["V"]) | (
        _orb["V"] < vcut) | (_orb["AB5_brightest"] < tcut)))
    _orb = _orb[objects_mask]
    _orb.index = _orb.index.astype(pd.ArrowDtype(pa.string()))
    return _orb


def split_dataframe_by_mjd(df, max_diff_days=3) -> List[pd.DataFrame]:
    """Split dataframe by 3 day ranges to maintain the accuracy of N body computations.

    Parameters
    ----------
    df : `pd.DataFrame`
        Dataframe containing header extracted values
    max_diff_days: `int`
        Integer representing no of days between the first and last observation
        in the resulting dataframe.

    Returns
    -------
    output: List[pd.DataFrame]
        List of Solar system object dataframes split up into 3 day periods.

    """
    # Sort the dataframe by the 'EXPSTART' column (assuming it's in MJD format)
    df = df.sort_values(by='EXPSTART')

    # Convert MJD to human-readable date and add to new column
    df['HUMAN_DATE'] = Time(df['EXPSTART'], format='mjd').iso

    # Create a list to store the resulting DataFrames
    dfs = []
    current_df = [df.iloc[0]]  # Start with the first row

    # Iterate through the rows starting from the second row
    for i in range(1, len(df)):
        # Check if adding the current row exceeds the threshold
        if df.iloc[i]['EXPSTART'] - current_df[0]['EXPSTART'] >= max_diff_days:
            # If difference exceeds 3 days, start a new group
            # Add the current group to the result
            dfs.append(pd.DataFrame(current_df))
            current_df = [df.iloc[i]]  # Reset to the current row
        else:
            # Otherwise, add the row to the current group
            current_df.append(df.iloc[i])

    # Append the last group if there are any remaining rows
    if current_df:
        dfs.append(pd.DataFrame(current_df))

    return dfs
def format_df_to_apy(df: pd.DataFrame) -> Table:
    """Convert dataframe to table with appropriate metadata."""
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric coercion first (handles int/float/None/NaN mixes).
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().any() or df[col].isna().all():
                df[col] = coerced.astype("float64")
            else:
                # Not numeric -- treat as string, NaN/None -> None
                # is fine for Arrow strings.
                df[col] = df[col].astype(str).where(df[col].notna(), None)

    sso_tab = Table.from_pandas(df, index=True)  # keep "desig" as a column
    sso_tab.meta = {"version": SSDC_PIPELINES_VERSION}
    return sso_tab


class SolarSystemObjectsConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("instrument", "spsplan"),
                                    defaultTemplates={}):
    """We run SSO pipelines for all level1 images, using
    -d  "instrument='spx' and  (detector=1 or  detector=2 or detector=3)
    and spsplan=PLAN_NAME"
    with pipetask in production.
    """

    images = cT.Input(
        name="level2a",
        doc="Input level 2 images.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
        deferLoad=True,
        multiple=True
    )
    catalog = cT.Output(
        name="ssocat",  # SSO
        doc="Output catalog with sources found in SPHEREx FOV.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "observation"),
        multiple=True
    )


class SolarSystemObjectsConfig(
    pipeBase.PipelineTaskConfig,
        pipelineConnections=SolarSystemObjectsConnections):
    """Configuration for SolarSystemObjects task."""

    update_cache = Field(
        doc="Download and update JPL SBDB catalog cache.Use existing cache if False",
        dtype=bool,
        default=False)
    pixmargin = Field(
        doc="Margin in pixels."
        "The sources will be added to the ssocat table if they fall "
        "within the image with this margin.",
        dtype=float,
        default=300.0)


class SolarSystemObjectsTask(pipeBase.PipelineTask):
    """Implementation of Compute Solar System objects.

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
        Document GitHub page: # TODO

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "compute-sso"
    ConfigClass = SolarSystemObjectsConfig
    full_name = "Compute Solar System Objects"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "TODO"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        imageList = [ref for ref in inputRefs.images]
        output = self.run(imageList=imageList, butlerQC=butlerQC)

        empty_cat = format_df_to_apy(pd.DataFrame(columns=[
            "H", "G", "spkid", "dmax", "RA", "DEC", "alpha",
            "rh", "ro", "V", "Tmax", "AB5_brightest", "trailed",
            ], index=pd.Index([], name="desig")))

        for dataset_ref in outputRefs.catalog:
            obs_id = ObservationId.unpack(dataset_ref.dataId["observation"])
            # Assuming dataset_refs is your list of DatasetRef objects
            butlerQC.put(output.get(obs_id, empty_cat), dataset_ref)

    def run(self, imageList: list[DeferredDatasetHandle],
            butlerQC: pipeBase.QuantumContext) -> dict[str, Table]:
        """Compute per-observation lists of solar system objects.

        Parameters
        ----------
        imageList: `list` [DeferredDatasetHandle]
            List of SPHERExImages in L1 format.
        butlerQC: pipeBase.QuantumContext
            Quantum context to lookup deferred datasets.

        Returns
        -------
        `dict`[`str`, `astropy.table.Table`]
            Dictionary containing solar system objects for each observation.

        """
        column_names = ["X_SC", "VX_SC", "Y_SC",
                        "VY_SC", "Z_SC", "VZ_SC", "PA_ECL", "XPOSURE"]

        # This contains NEATM-based flux (Harrris 1998 Icar, 131, 291) at λ=5um,
        # for a unit asteroid (r_hel = r_obs = 1au, diameter = 1km).
        # index: phase angle (deg),
        # columns: blackbody equilibrium temperature at r_hel = 1au (K)
        #   (== maximum == sub-solar temperature for NEATM)
        # Paper containing data: https://doi.org/10.1006/icar.1997.5865
        tlut_5um = pd.read_csv(importlib.resources.files(
            "spx_utils.data").joinpath('abmags_neatm_5um.csv'),
            index_col="alpha")
        tlut_5um_spl = RectBivariateSpline(
            tlut_5um.index.astype(float),
            tlut_5um.columns.astype(float),
            tlut_5um.values
        )
        fits_plan = create_dataframe_from_headers(imageList, column_names, butlerQC)

        orb_data = kete.horizons.fetch_known_orbit_data(update_cache=self.config.update_cache)

        orb_data["G"] = 0.15  # Default value for missing G columns
        # Removing most impacted objects
        # Logically this will not remove
        # two_body and archaic comets (old soln_date),
        # but removes most of the
        # useless objects we wanted to drop (~ 2000 objects).
        mask = (
            np.bitwise_and.reduce([pd.isna(orb_data[k])
                                  for k in
                                  ["M1", "M2", "K1", "K2", "PC", "H"]])
            | (orb_data["desig"].str.split(" ").str[0].str.endswith("D"))
        )
        orb_data = orb_data[~mask]

        ast_states = kete.mpc.table_to_states(orb_data)

        orb_data = orb_data.set_index("desig")

        # Lookup of exposure time (sec) keyed by observer designation,
        # which is built as "{OBSID}_D{DETECTOR}" in create_observer_state.
        xposure_lut = {
            f"{row['OBSID']}_D{row['DETECTOR']}": row["XPOSURE"]
            for _, row in fits_plan.iterrows()
        }

        fits_plan_3_day_chunks = split_dataframe_by_mjd(fits_plan)

        all_chunk_result = dict()
        for fits_chunk in fits_plan_3_day_chunks:
            fovs_per_obsid = []
            for obsid, per_obsid_fits in fits_chunk.groupby("OBSID"):
                # print(per_obsid_fits)
                jd_tdbs = calculate_midpoint_time(per_obsid_fits)
                # print(jd_tdbs)
                fovs = create_fovs(per_obsid_fits, jd_tdbs, per_obsid_fits["OBSID"],
                                   int(self.config.pixmargin))
                # printlen(fovs))
                fovs_per_obsid.extend(fovs)

            fovs = kete.fov.FOVList(fovs_per_obsid)
            ast_states_at_ref = propagate_orbit_elements(ast_states, fovs)
            result = kete.fov_state_check(
                obj_state=ast_states_at_ref, fovs=fovs)

            giant_planets = ["mars barycenter", "jupiter barycenter",
                             "saturn barycenter", "uranus barycenter",
                             "phoebe"]
            result_gp = kete.fov.fov_spice_check(giant_planets, fovs)

            for res in result:
                # Associating specific observations with SSO
                obs = str(res.fov.observer.desig).split("_D")[0]
                # Exposure time (sec) for this detector's observation.
                xposure = xposure_lut[str(res.fov.observer.desig)]
                # Combining all detector results in an observation
                if obs in all_chunk_result:
                    det_data = calc_mags(res, orb_data, tlut_5um_spl, xposure)

                    existing_data = all_chunk_result[obs]
                    # 2. Append to existing_data
                    final_df = pd.concat([det_data, existing_data])

                    all_chunk_result[obs] = final_df
                else:
                    all_chunk_result[obs] = calc_mags(
                        res, orb_data, tlut_5um_spl, xposure)
            # Seperate loop for giant planet results
            # As their mags do not need to be calculated.
            for res in result_gp:
                # Associating specific observations with SSO
                obs = str(res.fov.observer.desig).split("_D")[0]
                # Exposure time (sec) for this detector's observation.
                xposure = xposure_lut[str(res.fov.observer.desig)]
                # Combining all detector results in an observation
                if obs in all_chunk_result:
                    det_data = calc_mags_bright_planets(res, xposure)

                    existing_data = all_chunk_result[obs]
                    # 2. Append to existing_data
                    final_df = pd.concat([det_data, existing_data])

                    all_chunk_result[obs] = final_df
                else:
                    all_chunk_result[obs] = calc_mags_bright_planets(res, xposure)
        for key, value in all_chunk_result.items():
            # If a desig appears multiple times (due to FOV overlap margin)
            # keep only the first instance of it
            value = value.loc[~value.index.duplicated(keep='first'), :]
            # Sort by spkid
            value = value.sort_values(by=["spkid"])
            value.drop(columns=['trail_pix'], inplace=True)
            all_chunk_result[key] = value

        return {obs: format_df_to_apy(df) for obs, df in all_chunk_result.items()}
