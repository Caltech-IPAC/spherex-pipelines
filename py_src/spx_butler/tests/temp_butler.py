__all__ = ["create_temp_butler", "destroy_temp_butler"]

import tempfile
import shutil

from lsst.daf.butler import Butler
from lsst.resources import ResourcePath
from spx_butler.instrument import SimulatorInstrument

BUTLER_CONFIG = "resource://spx_butler/configs"
BUTLER_YAML = "butler.yaml"
DIMENSION_YAML = "dimensions.yaml"


def create_temp_butler(butler_config=None, butler_yaml=None,
                       dimension_yaml=None):
    """Create a temporary butler and storage.

    Parameters
    ----------
    butler_config : str, optional
        URI to configuration directory, by default None.
    butler_yaml : str, optional
        YAML file for butler configuration, by default None.
    dimension_yaml : _type_, optional
        YAML file for dimension configuration, by default None.

    Returns
    -------
    butler : lsst.daf.butler.Butler
        Butler object.
    root : str
        Path to temporary folder bulter storage is in.
    """
    # Create temp directory
    root = tempfile.mkdtemp()

    if butler_config is None:
        butler_config = BUTLER_CONFIG
    if butler_yaml is None:
        butler_yaml = BUTLER_YAML
    if dimension_yaml is None:
        dimension_yaml = DIMENSION_YAML

    config_uri = ResourcePath(butler_config, forceDirectory=True)
    butler_config = config_uri.join(butler_yaml)
    dimension_config = config_uri.join(dimension_yaml)

    Butler.makeRepo(root=root, config=butler_config,
                    dimensionConfig=dimension_config)

    butler = Butler(root, writeable=True)
    SimulatorInstrument().register(butler.registry)

    return butler, root


def destroy_temp_butler(root):
    """Removes the temporary folder and all of its contents.

    Parameters
    ----------
    root : str
        Path to the temporary folder to delete.
    """
    shutil.rmtree(root, ignore_errors=True)
