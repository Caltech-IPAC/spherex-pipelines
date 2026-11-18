from spx_data import Zodi
import numpy as np


def test_zodi():
    zodi = Zodi(
        zodi_data=np.ones((3, 4), dtype=float),
        zodi_model_name="KEL98_WAVESCL_V1",
    )

    hdu = zodi.to_hdu(name="ZODI")
    hdr = hdu.header

    assert hdr["ZODIMDL"] == "KEL98_WAVESCL_V1"
    assert hdr.comments["ZODIMDL"] == "Kelsall+1998 zodi model with wavelength scaling"
    print(list(hdr["COMMENT"]))

    comments = " ".join(str(comment) for comment in hdr["COMMENT"])

    assert "Implements wavelength-dependent scaling" in comments
    assert "Kelsall et al." in comments
    assert "(1998) zodiacal light model." in comments
    assert "See SPHEREx Explanatory Supplement for full details." in comments

    assert hdr["BUNIT"] == "MJy / sr"
    assert hdu.name == "ZODI"
    assert hdu.data.dtype == np.float32
