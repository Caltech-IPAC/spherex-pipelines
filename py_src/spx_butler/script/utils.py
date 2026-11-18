import sys
import lsst.pex.config as pexConfig


def apply_config_overrides(config: pexConfig.Config, config_override_file: str, dump_and_exit: bool):
    """Apply configuration overrides from a file.

    Parameters
    ----------
    config : pexConfig.Config
        Configuration object
    config_override_file : str
        Configuration override file
    dump_and_exit : bool
        If True, dump configuration and exit

    """
    if config_override_file is not None:
        config.load(config_override_file)
        config.validate()
        config.freeze()
    if dump_and_exit:
        show_config(config)
        sys.exit(0)


def show_config(config: pexConfig.Config) -> None:
    """Dump configuration to standard output.

    Parameters
    ----------
    config : `pexConfig.Config`
        The configuration object to display.

    """
    stream = sys.stdout
    config.saveToStream(stream, skipImports=True)
