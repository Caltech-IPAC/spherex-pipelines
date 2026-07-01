# Standard library imports

# Related third party imports

# Local application/library specific imports
from spx_data import ReadoutParameters
from spx_tasks import RemoveReferencePixelsConfig, RemoveReferencePixelsTask
from spx_tasks import CreateVarianceMapTask, CreateVarianceMapConfig
from spx_tests import sample_data


def l1img():
    config = CreateVarianceMapConfig()
    task = CreateVarianceMapTask(config=config)
    # Get sample Level 1b image.
    spximg = sample_data.get_sample_level1b_img(bad_flags=True)
    spximg.header["TCUT"] = 70
    spximg.header["TSAMP"] = 1.5349
    spximg.header["SURLIM"] = 77
    spximg.header["TRANDLY"] = 7
    spximg.header["SURDLY"] = 3
    # Get sample gain factors for array detector 1.
    gain_factors = sample_data.get_sample_gain_factors(spximg)
    # Read noise map.
    read_noise = sample_data.get_sample_read_noise_map(spximg)
    # Running variance map task to check that uncertainty array changes as expected.
    output = task.run(spximg, gain_factors, read_noise=read_noise)
    return output.spximg_out


def test_RemoveReferencePixels():
    config = RemoveReferencePixelsConfig()
    num_ref_pixels = 1
    config.num_ref_pixels = num_ref_pixels
    task = RemoveReferencePixelsTask(config=config)
    # Get sample or L1a image (phans, refs, actvs).
    # spximg_L1a = sample_data.get_sample_level1_img()
    spximg_L1a = l1img()
    rp = ReadoutParameters.from_metadata(spximg_L1a.meta)

    # Add fake L1 flag headers (if not already present) for testing
    for key in [
        "HIERARCH L1 N_TRANSIENT", "HIERARCH L1 N_OVERFLOW", "HIERARCH L1 N_SUR_ERROR",
        "HIERARCH L1 N_PHANTOM", "HIERARCH L1 N_REFERENCE", "HIERARCH L1 N_NONFUNC",
        "HIERARCH L1 N_DICHROIC", "HIERARCH L1 N_MISSING", "HIERARCH L1 N_FULLSAMPLE",
        "HIERARCH L1 N_PHANMISS"
    ]:
        spximg_L1a.header[key] = 12345  # dummy value

    # Running task.
    output = task.run(spximg_L1a)
    spximg_out = output.spximg_out
    # Output shape should be same as photosensitive pixels.
    assert spximg_out.shape == (rp.Jactive, rp.Kactive)
    assert spximg_out.data.shape == spximg_out.flags.shape
    assert spximg_out.data.shape == spximg_out.uncertainty.array.shape

    # Ensure L1-specific headers were removed
    for key in [
        "L1 N_TRANSIENT", "L1 N_OVERFLOW", "L1 N_SUR_ERROR", "L1 N_PHANTOM",
        "L1 N_REFERENCE", "L1 N_NONFUNC", "L1 N_DICHROIC", "L1 N_MISSING",
        "L1 N_FULLSAMPLE", "L1 N_PHANMISS"
    ]:
        assert key not in spximg_out.header, f"{key} was not removed from header"
