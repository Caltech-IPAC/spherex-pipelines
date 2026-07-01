__all__ = ["DatasetTypes", "spherex_dataset_types"]
from dataclasses import dataclass


@dataclass
class DatasetTypes:
    abs_cal = "abs_cal"
    abs_gain_matrix = "abs_gain_matrix"  # absolute gain matrix
    astr_diag = "astr_diag"  # astrometry diagnostics
    bsh_diag_img = "bsh_diag_img"
    ck = "ck"  # C-kernel orientation data
    spk = "spk"  # SPK Ephemeris data
    nonfunc_map = "nonfunc"  # nonfunctional pixels
    nonfunc_map_v2 = "nonfunc_v2"
    dichroic_cut_map = "dichroic"  # dichroic cut pixels
    ff_calib = "flat"  # flat field calibration
    ff_est = "flat"
    dc_calib = "dark"  # dark current calibration
    dc_est = "dark"
    photo_calib = "photo_calib"  # binned photometric calibration data
    refcat = "refcat"  # reference catalog
    sip = "sip"  # distortion / SIP coefficients
    spectral_wcs = "spectral_wcs"  # spectral wcs with lookup table
    srccat = "srccat"  # refcat sources that fall within slew group images
    ssocat = "ssocat"  # precomputed Solar System Objects
    average_psf = "average_psf"  # exposure-averaged PSF
    level2 = "level2"  # Level 2 image; output of L2bPipeline
    level2a = "level2a"  # Level 2a image; output of L2aPipeline
    level1 = "level1"  # Level 1 image; output of L1Pipeline
    level0 = "raw"  # Decompressed, assembled image; output of Ka-band ingest
    level0d = "diagnostic"
    meta2 = "meta2"  # instrument Image Metadata 2 block
    meta3 = "meta3"  # instrument Image Metadata 3 block
    gain_factors = "gain_factors"  # gain factors to use for image
    sur_diag = "sur_diag"  # SUR diag
    gain_map = "gain_map"
    nonlinear_pars = "nonlinear_pars"
    readnoise_pars = "readnoise_pars"
    persistence_state = "persistence_state"
    persistence_image = "persistence_image"
    planned_observation = "planned_obs"  # planned onservation info from PEF
    phot_meas = "phot_meas"  # photometry measurement table, per HEALPix per image
    phot_meas_combined = "phot_meas_combined"  # photometry measurement table, full image
    global_bkgd = "global_bkgd"  # global background estimate
    dqa_l2 = "dqa_l2"  # level 2 DQA record
    dqa_l1 = "dqa_l1"  # level 1 DQA record
    initial_distortion_table = "initial_distortion_table"
    idqr_obs = "idqr_obs"
    idqr_agg = "idqr_agg"
    odir_obs = "odir_obs"
    odir_agg = "odir_agg"
    tractor_fits = "tractor_fits"  # dictionary of Tractor fitting results; only for local testing
    solid_angle_pixel_map = "solid_angle_pixel_map"
    sband_mnemonics = "sband_mnemonics"  # table of sband telemetry mnemonic keywords
    xtalk_coupling_matrix = "xtalk_coupling_matrix"
    sgt = "sgt"  # table of reconstructed ground track data
    allsky_cat = "allsky_cat"  # primary all-sky catalog, per HEALPix
    allsky_cat_combined = "allsky_cat_combined"  # primary all-sky catalog, combined across HEALPix
    refcat_allsky = "refcat_allsky"  # custom input reference catalog, not confined to a single HEALPix


spherex_dataset_types: DatasetTypes = DatasetTypes()
