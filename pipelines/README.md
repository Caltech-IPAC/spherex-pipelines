# Executing Pipelines

We are using Rubin/LSST Pipeline Framework to write and execute SPHEREx pipelines.
The background information on Rubin/LSST Pipeline System can be found at Section 3
of https://arxiv.org/pdf/2206.14941.pdf

The pipelines are defined as the YAML text files, and are executed with
[pipetask](https://pipelines.lsst.io/modules/lsst.ctrl.mpexec/pipetask.html) command,
defined in [lsst-ctrl-mpexec](https://github.com/lsst/ctrl_mpexec) package.

# Examples

## Update Field Distoration Pipeline
Create execution plan:
```console
$ pipetask qgraph -p ../spherex-pipelines/pipelines/UpdateFieldDistortionPipeline.yaml  --qgraph-dot distortion.dot -b <butler_repo> -i <input_collection> -o <output_chained_collection>
```

Run pipeline:
```console
$ pipetask run -b <butler_repo>/butler.yaml -p ../spherex-pipelines/pipelines/UpdateFieldDistortionPipeline.yaml -d "instrument='simu' AND detector=2" -i <input_collection> -o <output_chained_collection> --register-dataset-types
```

## Estimate Absolute Gain Matrix Pipeline
This pipeline estimates the absolute gain matrix and relative flat field.

Create execution plan:
```console
$ pipetask qgraph -b <butler_repo> -p ../spherex-pipelines/pipelines/EstimateAbsGainMatrixPipeline.yaml  --qgraph-dot ff.dot -i <input_collection> -o <output_chained_collection>
```

Run pipeline:
```console
$ pipetask run -b <butler_repo> -p ../spherex-pipelines/pipelines/EstimateAbsGainMatrixPipeline.yaml -d "instrument='simu' AND detector=2 AND exposure.dgl_qntl < 0.88 AND exposure.galactic_lat > 25.0" --register-dataset-types -i <input_collection> -o <output_chained_collection>
```

## Estimate PSF Pipeline
This pipeline estimates the PSF from multiple exposures.

Create execution plan:
```console
$ pipetask qgraph -p ../spherex-pipelines/pipelines/EstimatePSFPipeline.yaml  --qgraph-dot estimate_psf.dot -b <butler_repo> -i <input_collection> -o <output_chained_collection>
```

Run pipeline:
```console
$ pipetask run -b <butler_repo>/butler.yaml -p ../spherex-pipelines/pipelines/EstimatePSFPipeline.yaml -d "instrument='simu' AND detector=2" -i <input_collection> -o <output_chained_collection>  --register-dataset-types
```

## Create All Sky Catalog
Create execution plan:
```console
$ pipetask qgraph -b <butler_repo>  -p ../spherex-pipelines/pipelines/CreateAllSkyCatalogPipeline.yaml  --qgraph-dot allsky.dot -i <input_collection> -o <output_chained_collection>
```

Run pipeline:
```console
$ pipetask run -b <butler_repo>  -p ../spherex-pipelines/pipelines/CreateAllSkyCatalogPipeline.yaml -d "instrument='simu'" -i <input_collection> -o <output_chained_collection> --register-dataset-types
```

## L1 Pipeline
Create execution plan:
```console
$ pipetask qgraph -b <butler_repo>  -p ../spherex-pipelines/pipelines/L1Pipeline.yaml  --qgraph-dot l1.dot -i <input_collection> -o <output_chained_collection>
```

Run pipeline:
```console
$ pipetask run -b <butler_repo> -p ../spherex-pipelines/pipelines/L1Pipeline.yaml --register-dataset-types -i <input_collection> -o <output_chained_collection> -d "instrument='simu' AND detector=2"
```

## Common Commands And Tools
Convert `dot` into `pdf` ([graphviz](https://graphviz.org/) required):
```console
$ dot -Tpdf distortion.dot -o distortion.pdf
```

View `pdf` (MacOS):
```console
$ open -a Preview distortion.pdf
```
