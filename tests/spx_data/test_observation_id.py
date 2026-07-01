from spx_data import ObservationId


def test_observation_id():
    plan_name = "2025W37_2A"
    sps_short_id = "2025W37_2A_0486_2"
    # long id is survey planning observation id; the last number
    # is the total number of observations in a slew group
    sps_long_id = "2025W37_2A_0486_2_4"
    spx_id_int = 202537201048620
    spx_id_str = "202537201048620"
    # (pointing_id, plan_id, slewgroup_id, slewgroup_counter_id)
    all_ids = (202537201048620, 202537201, 486, 2)
    assert ObservationId.pack(sps_short_id) == spx_id_str
    assert ObservationId.pack(sps_long_id) == spx_id_str

    assert ObservationId.unpack(spx_id_int) == "2025W37_2A_0486_2"
    assert ObservationId.unpack(spx_id_str) == "2025W37_2A_0486_2"
    spx_id = ObservationId(spx_id_str)
    assert spx_id.spx_id == spx_id_str

    long_id = ObservationId(sps_long_id)
    assert (long_id == spx_id)
    assert long_id.spx_id == spx_id_str
    assert (long_id.pointing_id, long_id.plan_id,
            long_id.slewgroup_id, long_id.slewgroup_counter_id) == all_ids
    assert long_id.plan_name == plan_name

    short_id = ObservationId(sps_short_id)
    assert (short_id == long_id)
    assert spx_id.sps_id == sps_short_id

    # Test that the ID can fit in 64 bits
    test_id = "2050W52_9Z_9999_9"
    spx_id = int(ObservationId.pack(test_id))
    # Take into consideration the sign bit since it takes absolute value
    assert spx_id.bit_length() <= 63, "Cannot fit in 64 bits"
