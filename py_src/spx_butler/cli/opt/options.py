__all__ = (
    "in_type_option",
    "out_type_option",
    "instrument_option",
    "sps_plan_option",
    "detector_option",
    "iso_start_option",
    "iso_end_option",
    "dump_config_option"
)
import click

from lsst.daf.butler.cli.utils import MWOptionDecorator
from spx_butler.instrument import SPHERExInstrument, SimulatorInstrument

in_type_option = MWOptionDecorator(
    "--in-type",
    help="Dataset type for input."
)

out_type_option = MWOptionDecorator(
    "--out-type",
    help="Dataset type for output."
)

instrument_option = MWOptionDecorator("--instrument",
                                      type=click.Choice((SPHERExInstrument.getName(),
                                                         SimulatorInstrument.getName()),
                                                        case_sensitive=False),
                                      default=SPHERExInstrument.getName(),
                                      help=f"""\b
                                      Possible values for instrument:
                                      - {SPHERExInstrument.getName()}: SPHEREx Instrument.
                                      - {SimulatorInstrument.getName()}: Simulator Instrument.
                                      """)

detector_option = MWOptionDecorator(
    "--detector",
    type=int,
    help="Detector."
)

sps_plan_option = MWOptionDecorator(
    "--sps-plan",
    default=None,
    type=str,
    help="SPS Plan String, e.g. 2025W18_1B"
)

iso_start_option = MWOptionDecorator(
    "--iso-start",
    default=None,
    help="Start time as ISO string."
)

iso_end_option = MWOptionDecorator(
    "--iso-end",
    default=None,
    help="End time as ISO string."
)

dump_config_option = MWOptionDecorator(
    "--dump-config",
    default=False,
    help="Dump task configuration to standard output and exit.",
    is_flag=True
)
