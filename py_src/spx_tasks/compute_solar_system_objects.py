import kete
from astropy.time import Time
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.utils.timer import timeMethod
from lsst.daf.butler import (
    DeferredDatasetHandle
)
from typing import List
from scipy.interpolate import RectBivariateSpline
import importlib
import pandas as pd
from spx_data import ObservationId
from lsst.pex.config import Field
from spx_utils.coarse_astro_utils import calculate_pa_from_pc
import numpy as np
s__all__ = ["SolarSystemObjectsConfig", "SolarSystemObjectsTask"]

# Standard library imports

# Related third party imports


def create_dataframe_from_headers(handles: List[DeferredDatasetHandle],
                                  column_names: list):
    """
    Create a DataFrame from the FITS header values for specified column names
    across multiple FITS files.

    Parameters:
    - handles List[DeferredDatasetHandle]: References to SPHERExImages
    - column_names list: List of column names to extract from the FITS headers.

    Returns:
    - pd.DataFrame: Combined DataFrame with data from all FITS files.
    """
    data_list = []
    for handle in handles:
        spximg = handle.get()
        data = {col_name: spximg.header.get(
            col_name, None) for col_name in column_names}
        data["EXPSTART"] = spximg.getExposureStartTime().value
        data["EXPSTOP"] = spximg.getExposureEndTime().value
    # TODO: implement mean position and velocity
    # in estimate_coarse_astrometry
    # to ADD seperate mean values to header, and extract it.
    # Test from spherex-analysis/notebooks/L1/1.5
        data["RA_EQ"] = spximg.wcs.wcs.crval[0]
        data["DEC_EQ"] = spximg.wcs.wcs.crval[1]
        data["OBSID"] = spximg.header.get(
            "OBSID")
        data["PA_ECL"] = calculate_pa_from_pc(spximg.wcs)
        data_list.append(data)

    # Create a DataFrame from the list of dictionaries
    return pd.DataFrame(data_list)


def calculate_midpoint_time(header: pd.DataFrame):
    """
    Calculate the midpoint of exposure times and convert
    to Julian Date in TDB scale.
    """
    midpoint_mjd = (header["EXPSTART"] + header["EXPSTOP"]) / 2
    times = Time(midpoint_mjd, format="mjd", scale="utc")
    return times.tdb.jd


def create_observer_state(row, jd_tdb, obsid):
    """
    Create the observer's state based on position and velocity data.
    """
    position = kete.Vector(
        np.array([row["X_SC"], row["Y_SC"], row["Z_SC"]])
        / kete.constants.AU_KM,
        frame=kete.Frames.Ecliptic
    )
    velocity = kete.Vector(
        np.array([row["VX_SC"], row["VY_SC"], row["VZ_SC"]])
        / kete.constants.AU_KM * 86400,
        frame=kete.Frames.Ecliptic
    )

    return kete.State(
        desig=str(obsid),
        jd=jd_tdb,
        pos=position,
        vel=velocity,
        center_id=399
    )


def create_fovs(fits_plan, jd_tdbs, obsids):
    """
    Create a list of FOVs based on FITS plan data.
    """
    fovs = []
    for (_, row), jd_tdb, obsid in zip(fits_plan.iterrows(), jd_tdbs, obsids):
        observer_state = create_observer_state(row, jd_tdb, obsid)
        fov = kete.RectangleFOV(
            pointing=kete.Vector.from_ra_dec(row["RA_EQ"], row["DEC_EQ"]),
            rotation=-row["PA_ECL"],
            observer=observer_state,
            lon_width=14.0,
            lat_width=4.0
        )
        fovs.append(fov)
    return kete.fov.FOVList(fovs)


def propagate_orbit_elements(states, fovs):
    """
    Propagate orbital elements to the reference time.
    """
    # Propagate orbital elements to a certain
    # reference time (arbitrarily chosen as
    # the first observation time).
    reference_jd = fovs[0].observer.jd
    return kete.propagate_n_body(states, reference_jd, include_asteroids=False)


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
    """The IAU HG phase function model in intensity (1 at alpha=0).

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


def calc_mags(states_i, orb, tlut_5um_spl):
    obs2objs = states_i.obs_vecs
    desigs = [_r.desig for _r in states_i]

    _orb = orb.loc[desigs][["H", "G"]].fillna({"G": 0.15})
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
    # Perform magnitude cuts
    vcut = 21  # magnitude limit for the V band
    tcut = 20  # thermal emission magnitude limit for the 5um band

    objects_mask = np.array((pd.isna(_orb["V"]) | (
        _orb["V"] < vcut) | (_orb["AB5_brightest"] < tcut)))
    _orb = _orb[objects_mask]
    return _orb


def split_dataframe_by_mjd(df, max_diff_days=3) -> List[pd.DataFrame]:
    """
    Split dataframe by 3 day ranges to maintain the accuracy of N body computations.
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


class SolarSystemObjectsConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("instrument", "spsplan"),
                                    defaultTemplates={}):
    """
    We run SSO pipelines for all level2 images, using
     -d "instrument='spx' AND detector=2 "  with pipetask in production.
    """
    images = cT.Input(
        name="level2a",  # default dataset type for input
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
    update_cache = Field(
        doc="Download and update JPL SBDB catalog cache.Use existing cache if False",
        dtype=bool,
        default=False)


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

        imageList = [butlerQC.get(ref) for ref in inputRefs.images]

        output = self.run(imageList=imageList)

        for dataset_ref in outputRefs.catalog:
            # Assuming dataset_refs is your list of DatasetRef objects
            butlerQC.put(output[ObservationId.unpack(
                dataset_ref.dataId["observation"])], dataset_ref)

    def run(self,
            imageList: list[DeferredDatasetHandle]) -> dict:
        """

        Parameters
        ----------
        imageList: `list` [DeferredDatasetHandle]
            List of SPHERExImages in L1 format.

        Returns
        -------
        output: Output
            Parquet file containing asteroid data.
        """
        column_names = ["X_SC", "VX_SC", "Y_SC",
                        "VY_SC", "Z_SC", "VZ_SC", "PA_ECL"]

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
        fits_plan = create_dataframe_from_headers(imageList, column_names)
        jd_tdbs = calculate_midpoint_time(fits_plan)

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
        fits_plan_3_day_chunks = split_dataframe_by_mjd(fits_plan)
        all_chunk_result = dict()
        for fits_chunk in fits_plan_3_day_chunks:
            fovs = create_fovs(fits_plan, jd_tdbs, fits_chunk["OBSID"])
            ast_states_at_ref = propagate_orbit_elements(ast_states, fovs)
            result = kete.fov_state_check(
                obj_state=ast_states_at_ref, fovs=fovs)
            for res in result:
                # Associating specific observations with SSO
                all_chunk_result[str(res.fov.observer.desig)] = calc_mags(res, orb_data, tlut_5um_spl)
        return all_chunk_result
