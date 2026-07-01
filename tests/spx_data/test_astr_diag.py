import pytest
from pydantic import ValidationError


from spx_data.astr_diag import (WCSLinear, SipModel, CoarseAstrometry, FineAstrometry,
                                ScampResults, AstrometryNetResults, CoarseCenter, AstrDiag)


def get_mock_sip_model() -> SipModel:
    """Returns a valid mock SipModel object."""
    return SipModel(
        A_ORDER=2, B_ORDER=2, AP_ORDER=2, BP_ORDER=2,
        CRPIX=(1024.0, 1024.0),
        A=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        B=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        AP=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        BP=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )


def get_mock_coarse_astrometry() -> CoarseAstrometry:
    """Returns a valid mock CoarseAstrometry object using CD matrix."""
    return CoarseAstrometry(
        CRVAL1=150.1, CRVAL2=2.2,
        CTYPE1="RA---TAN", CTYPE2="DEC--TAN",
        CD1_1=-1.7e-3, CD1_2=0.0, CD2_1=0.0, CD2_2=1.7e-3,
        SIP=get_mock_sip_model()
    )


def get_mock_fine_astrometry() -> FineAstrometry:
    """Returns a valid mock FineAstrometry object using PC matrix."""
    return FineAstrometry(
        CRVAL1=150.1001, CRVAL2=2.2001,
        CTYPE1="RA---TAN-SIP", CTYPE2="DEC--TAN-SIP",
        PC1_1=1.0, PC1_2=0.0, PC2_1=0.0, PC2_2=1.0,
        CDELT1=-1.7e-3, CDELT2=1.7e-3,
        SIP=get_mock_sip_model()
    )


def get_mock_scamp_results() -> ScampResults:
    """Returns a valid mock ScampResults object."""
    return ScampResults(dAXIS1_arcsec=0.1,
                        dAXIS2_arcsec=0.2,
                        CONTRAST=10.0,
                        match_nstars=500,
                        match_nstars_sn=480,)


def get_mock_astrometry_net_results() -> AstrometryNetResults:
    """Returns a valid mock AstrometryNetResults object."""
    return AstrometryNetResults(num_matches=500, pixel_scale=6.15, clip_trans=False)


def get_mock_coarse_center() -> CoarseCenter:
    """Returns a valid mock CoarseCenter object."""
    return CoarseCenter(
        x_pixel=100.5, y_pixel=200.0,
        coordinate_base="0-based",
        description="Test location of coarse center"
    )


def test_wcs_linear_validator():
    """
    Tests the custom validator on the WCSLinear base model.
    """
    # Test success with CD matrix only
    cd_data = {"CD1_1": 1.0, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1.0}
    model_cd = WCSLinear(**cd_data)
    assert model_cd.CD1_1 == 1.0

    # Test success with PC matrix only
    pc_data = {
        "PC1_1": 1.0, "PC1_2": 0.0, "PC2_1": 0.0, "PC2_2": 1.0,
        "CDELT1": 1.0, "CDELT2": 1.0
    }
    model_pc = WCSLinear(**pc_data)
    assert model_pc.PC1_1 == 1.0

    # Test failure when NEITHER are provided
    with pytest.raises(ValidationError, match="but not both or neither"):
        WCSLinear()  # No data

    # Test failure when BOTH are provided
    both_data = {**cd_data, **pc_data}
    with pytest.raises(ValidationError, match="but not both or neither"):
        WCSLinear(**both_data)


def test_astr_diag_minimal_success():
    """
    Tests creating an AstrDiag instance with only the required fields.
    This simulates a partial or failed pipeline run.
    """
    coarse_astro = get_mock_coarse_astrometry()

    diag = AstrDiag(
        coarse_astrometry=coarse_astro,
        n_calibrators=50,
        n_transients=2,
        FINAST=4  # 4 = Insufficient calibrators
    )

    assert diag.n_calibrators == 50
    assert diag.n_transients == 2
    assert diag.FINAST == 4
    assert diag.coarse_astrometry.CRVAL1 == 150.1

    # Optional fields should be None
    assert diag.fine_astrometry is None
    assert diag.scamp_results is None
    assert diag.astrometry_net_results is None
    assert diag.astronet_solved is None
    assert diag.error_message is None


def test_astr_diag_full_success():
    """
    Tests creating an AstrDiag instance with all fields populated.
    This simulates a successful pipeline run.
    """
    diag = AstrDiag(
        coarse_astrometry=get_mock_coarse_astrometry(),
        n_calibrators=150,
        n_transients=5,
        fine_astrometry=get_mock_fine_astrometry(),
        scamp_results=get_mock_scamp_results(),
        astrometry_net_results=get_mock_astrometry_net_results(),
        coarse_center_on_fine_image=get_mock_coarse_center(),
        astronet_solved=True,
        FINAST=0,  # 0 = Good solution
        error_message=None
    )

    assert diag.n_calibrators == 150
    assert diag.FINAST == 0
    assert diag.astronet_solved is True
    assert diag.fine_astrometry.CRVAL1 == 150.1001
    assert diag.scamp_results.CONTRAST == 10.0
    assert diag.astrometry_net_results.num_matches == 500
    assert diag.coarse_center_on_fine_image.x_pixel == 100.5


def test_astr_diag_missing_required_field():
    """
    Tests that Pydantic raises a ValidationError if required fields
    (like n_calibrators, n_transients, or FINAST) are missing.
    """
    coarse_astro = get_mock_coarse_astrometry()

    # Missing 'n_calibrators'
    with pytest.raises(ValidationError) as exc_info:
        AstrDiag(
            coarse_astrometry=coarse_astro,
            n_transients=2,
            FINAST=3
        )
    # Check that the error message specifically mentions the missing field
    assert "n_calibrators" in str(exc_info.value)

    # Missing 'n_transients'
    with pytest.raises(ValidationError) as exc_info:
        AstrDiag(
            coarse_astrometry=coarse_astro,
            n_calibrators=50,
            FINAST=3
        )
    assert "n_transients" in str(exc_info.value)

    # Missing 'FINAST'
    with pytest.raises(ValidationError) as exc_info:
        AstrDiag(
            coarse_astrometry=coarse_astro,
            n_calibrators=50,
            n_transients=2
        )
    assert "FINAST" in str(exc_info.value)

    # Missing 'coarse_astrometry'
    with pytest.raises(ValidationError) as exc_info:
        AstrDiag(
            n_calibrators=50,
            n_transients=2,
            FINAST=3
        )
    assert "coarse_astrometry" in str(exc_info.value)
