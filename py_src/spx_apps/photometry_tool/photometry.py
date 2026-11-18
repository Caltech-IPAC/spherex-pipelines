import logging
import os
import pathlib
import subprocess
import traceback
from datetime import datetime
from importlib import resources
from .exceptions import PipetaskError


def run_photometry(temp_butler_root, query_constraint: str, log_file=None, verbose=False) -> None:
    """Run the forced photometry pipeline on the temporary butler."""
    # Pipeline definition
    l3_pipeline_yaml = resources.files("spx_apps") / "photometry_tool" / "data" / "L3ToolPipeline.yaml"

    # Run collections
    timestamp = datetime.utcnow().strftime("%Y%jT%H%M%S")
    l3_output_run = f"l3-{timestamp}"

    # run Forced Photometry pipeline
    logging.info(f"L3 Pipeline: {l3_pipeline_yaml}")
    input_coll = "*"  # use all collections in temporary butler
    n_cores = int(os.getenv("SPXTOOL_NCORES", 1))
    # Default timeout to build QGRAPH is 1 hour
    qraph_timeout_sec = int(os.getenv("SPXPHOTO_QGRAPH_TIMEOUT", 1 * 60 * 60))
    # L3 default timeout is 12 hours
    l3_timeout_sec = int(os.getenv("SPXPHOTO_L3_TIMEOUT", 12 * 60 * 60))
    logging.info("Building QGraph.")
    qgraph_path = os.path.join(temp_butler_root, f"{l3_output_run}.qg")
    subprocess_args = ["pipetask"]
    try:
        if log_file is not None:
            subprocess_args.extend(["--log-file", log_file])
        subprocess_args.extend([
            "--long-log",
            "qgraph",
            "--butler-config", temp_butler_root,
            "--pipeline", l3_pipeline_yaml,
            "--input", input_coll,
            "--output-run", l3_output_run,
            "--save-qgraph", qgraph_path,
            # Required for run-qbb below: without it the graph carries no
            # datastore records and the quantum-backed butler has to guess
            # artifact paths from the datastore template.
            "--qgraph-datastore-records",
        ])
        if query_constraint:
            logging.info(query_constraint)
            subprocess_args.extend(["--data-query", query_constraint])
        if verbose:
            dot_path = os.path.join(temp_butler_root, f"{l3_output_run}.dot")
            subprocess_args.extend(["--qgraph-dot", dot_path])
        logging.info("CREATING QGRAPH")
        logging.info(str(subprocess_args))
        result = subprocess.run(
            subprocess_args,
            check=True,
            timeout=qraph_timeout_sec,
            text=True)
    except subprocess.CalledProcessError as e:
        msg = traceback.format_exc()
        logging.error(msg)
        raise PipetaskError("Creating qgraph failed", e)
    else:
        if result is not None:
            if result.returncode != 0:
                logging.error("Preparing photometry runs failed with return code",
                              result.returncode)

    logging.info(f"Running Photometry with {n_cores} cores.")
    photometry_err = None
    subprocess_args = ["pipetask"]
    try:
        if verbose:
            log_level = logging.getLevelName(logging.getLogger().getEffectiveLevel())
            subprocess_args.extend(["--log-level",
                                    f"spx_tasks.level3a.perform_forced_photometry={log_level}"])
        if log_file is not None:
            subprocess_args.extend(["--log-file", log_file])
        subprocess_args.extend([
            "--long-log",
            "run-qbb", temp_butler_root, qgraph_path,
            "-j", str(n_cores),
        ])
        logging.info("RUNNING L3")
        logging.info(str(subprocess_args))
        l3_result = subprocess.run(
            subprocess_args,
            check=True,
            timeout=l3_timeout_sec,
            text=True)
    except Exception as e:
        msg = traceback.format_exc()
        logging.error(str(e))
        logging.error(msg)
        photometry_err = e
    else:
        if l3_result is not None:
            if l3_result.returncode != 0:
                logging.error("L3 pipeline failed with return code",
                              l3_result.returncode)

    # Stop if there are no successful photometry measurements.
    # Level 3a processing is skipped for images that failed fine astrometry.
    # Hence, it is possible to have successfully completed L3a with no results.
    phot_meas_glob = pathlib.Path(os.path.join(temp_butler_root, "phot_meas")).rglob(pattern="*.parq")
    phot_meas_path = next(phot_meas_glob, None)
    if phot_meas_path is None:
        short_desc = "No successful photometry measurements."
        if isinstance(photometry_err, subprocess.CalledProcessError):
            raise PipetaskError(short_desc, photometry_err)
        else:
            raise RuntimeError(short_desc)
    allsky_cat_glob = pathlib.Path(os.path.join(temp_butler_root, "allsky_cat")).rglob(pattern="*.parq")
    allsky_cat_path = next(allsky_cat_glob, None)
    if allsky_cat_path is None:
        short_desc = "Failed to create catalog."
        if isinstance(photometry_err, subprocess.CalledProcessError):
            raise PipetaskError(short_desc, photometry_err)
        else:
            raise RuntimeError(short_desc)
