__all__ = [
    'ReprojectionMethod',
    'ExactReprojection',
    'HistogramReprojection',
    'InterpReprojection',
    'get_method',
    'list_methods',
]

from .base import ReprojectionMethod
from .exact import ExactReprojection
from .histogram import HistogramReprojection
from .interp import InterpReprojection

_METHODS = {
    'reproject_exact': ExactReprojection,
    'reproject_hist': HistogramReprojection,
    'reproject_interp': InterpReprojection,
}


def get_method(name: str) -> ReprojectionMethod:
    """
    Get reprojection method by name.

    Parameters
    ----------
    name : str
        Method name. One of 'reproject_exact', 'reproject_hist', or 'reproject_interp'.

    Returns
    -------
    ReprojectionMethod
        Instance of the requested reprojection method.

    Raises
    ------
    ValueError
        If the method name is not recognized.

    Examples
    --------
    Reproject an image onto a target WCS::

    method = get_method("reproject_exact")
    reprojected, nhits = method.reproject(
        data, wcs_in, wcs_out, shape_out
    )
    """
    if name not in _METHODS:
        available = ', '.join(_METHODS.keys())
        raise ValueError(f"Unknown reprojection method '{name}'. Available: {available}")
    return _METHODS[name]()


def list_methods() -> list[str]:
    """List available reprojection method names."""
    return list(_METHODS.keys())
