# Standard library imports
import copy
from pathlib import Path

# Related third party imports
from astropy import units as u
from astropy.time import Time
from lsst.daf.butler import Timespan

# Local application/library specific imports
from spx_tasks import EstimateCoarseAstrometryConfig, EstimateCoarseAstrometryTask
from spx_tests import sample_data
from spx_data import ReadoutParameters, PlannedObservation
from spx_butler.defaults.fake_products import initial_distortion

test_root = Path(__file__).parent.parent


def get_planned_obs() -> PlannedObservation:
    import json

    planned_obs_json = """{
      "ra":"84.22261244749616 deg",
      "dec":"-62.5437619006517 deg",
      "pa_eq":"140.95669713105121 deg",
      "timespan":[1743095339605500000,1743095456557900000],
      "obs_id":"2025W13_4B_0001_1_4",
      "target_id":277318,
      "flags":1,
      "quaternions":{
        "q1":-0.93078984083575,
        "q2":-0.2780411338954412,
        "q3":-0.0904720606196723,
        "q4":0.21940420758574125
      },
      "lat_ecl":-85.28645662176453,
      "lon_ecl":55.59780956947288,
      "pa_ecl":111.7782299680208,
      "zenith":[678.0702184219748,2549.404837344648,-6522.3639901313145],
      "moon":[346643.3145412663,-98579.49078124962,-44224.009644286096],
      "sun":[148207476.3712328,16526496.614762334,7170537.966351036],
      "vel":[4.088511468094384,-34.22437662640084,-14.448138966610523],
      "in_sband":true,
      "non_survey":false
    }"""

    data = json.loads(planned_obs_json)

    # Convert angle fields
    data["ra"] = float(data["ra"].split()[0]) * u.deg
    data["dec"] = float(data["dec"].split()[0]) * u.deg
    data["pa_eq"] = float(data["pa_eq"].split()[0]) * u.deg

    # Strip units for float-only fields
    data["lat_ecl"] = float(data["lat_ecl"])
    data["lon_ecl"] = float(data["lon_ecl"])
    data["pa_ecl"] = float(data["pa_ecl"])

    # Convert timespan from ns → seconds
    start_ns, end_ns = data["timespan"]
    start_time = Time(start_ns / 1e9, format="unix_tai", scale="tai")
    end_time = Time(end_ns / 1e9, format="unix_tai", scale="tai")
    data["timespan"] = Timespan(begin=start_time, end=end_time)

    return PlannedObservation(**data)


def test_EstimateCoarseAstrometry():
    spximg = sample_data.get_sample_level1b_img()
    spximg.header["OBSID"] = 202520204000110
    rp = ReadoutParameters.from_metadata(spximg.meta)

    # level-1 format
    expected_crpix1 = (rp.Jactive + 1)/2 + rp.jref//2 + (rp.Mphan * rp.nchannel)
    expected_crpix2 = (rp.Kactive + 1)/2 + rp.kref//2

    planned_obs = get_planned_obs()
    initial_distortion_table = initial_distortion()

    spximg_1 = copy.copy(spximg)
    spximg_1.setArrayId = 1

    spximg_2 = copy.copy(spximg)
    spximg_2.setArrayId = 2

    spximg_3 = copy.copy(spximg)
    spximg_3.setArrayId = 3

    config1 = EstimateCoarseAstrometryConfig()
    task = EstimateCoarseAstrometryTask(config=config1)

    # Detector-1
    output1 = task.run(spximg_1, planned_obs, initial_distortion_table)
    wcs_1 = output1.spximg_out.wcs
    assert spximg_1.data.shape == output1.spximg_out.data.shape, "Input and output image not same shape."
    assert wcs_1.wcs.crpix[0] == expected_crpix1, f"D1 CRPIX1 is not in level-1 format. {wcs_1.wcs.crpix[0]}"
    assert wcs_1.wcs.crpix[1] == expected_crpix2, f"D1 CRPIX2 is not in level-1 format. {wcs_1.wcs.crpix[1]}"

    # Detector-2
    output2 = task.run(spximg_2, planned_obs, initial_distortion_table)
    wcs_2 = output2.spximg_out.wcs
    assert spximg_2.data.shape == output2.spximg_out.data.shape, "Input and output image not same shape."
    assert wcs_2.wcs.crpix[0] == expected_crpix1, f"D2 CRPIX1 is not in level-1 format. {wcs_2.wcs.crpix[0]}"
    assert wcs_2.wcs.crpix[1] == expected_crpix2, f"D2 CRPIX2 is not in level-1 format. {wcs_2.wcs.crpix[1]}"

    # Detector-3
    output3 = task.run(spximg_3, planned_obs, initial_distortion_table)
    wcs_3 = output3.spximg_out.wcs
    assert spximg_3.data.shape == output3.spximg_out.data.shape, "Input and output image not same shape."
    assert wcs_3.wcs.crpix[0] == expected_crpix1, f"D3 CRPIX1 is not in level-1 format. {wcs_3.wcs.crpix[0]}"
    assert wcs_3.wcs.crpix[1] == expected_crpix2, f"D3 CRPIX2 is not in level-1 format. {wcs_3.wcs.crpix[1]}"
