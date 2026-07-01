# spherex-pipelines

## Introduction

The SPHEREx Science Data Center (SSDC) at Caltech/IPAC is responsible for the implementation
and operation of the Level 1 through 3 science data pipelines to produce SPHEREx public data
products, in collaboration with the SPHEREx Science Team.
The architecture of this pipeline is described in Akeson et al. (2026) and the details of the
pipeline module algorithms and data products are given in the SPHEREx Explanatory Supplement.
The data products themselves are available at the NASA/IPAC Infrared Science Archive
(IRSA; https://irsa.ipac.caltech.edu/Missions/spherex.html).
The pipeline modules are implemented primarily in Python as an application of the open-source
data access and pipeline construction middleware developed for the Vera C. Rubin Observatory
(Jenness et al. 2022).

## Purpose

This repository contains essentially all the code for the SPHEREx Level 1 through Level 3
pipelines.

Certain details of the communication with the spacecraft and the handling of its raw telemetry
are not able to be made public at this time and are maintained elsewhere.
The code and its internal documentation contain references to additional repositories required
to operate the code in production but which are not controlled by the SSDC and are not
currently public.
These repositories include the SPHEREx data simulator (Crill et al., 2025) and the SPHEREx
shared system engineering parameters.

In addition, this pipeline utilizes the community package Tractor (https://thetractor.org/,
https://thetractor.readthedocs.io/), with some modifications that are maintained in a fork
[`Caltech-IPAC/tractor-spherex`](https://github.com/Caltech-IPAC/tractor-spherex).

## Public-access repository

The [`Caltech-IPAC/spherex-pipelines` repository](https://github.com/Caltech-IPAC/spherex-pipelines)
copy of this repository contains snapshots of the code as used for each public data production,
intended to help the astronomy community understand the details of how the pipeline algorithms
are implemented.
As described in Akeson et al. (2026), this pipeline is designed to run within a secure
environment at IPAC and at the Texas Advanced Computing Center.
The SSDC does not have resources to assist users in running this pipeline in other
environments or to respond to pull requests, but will update the repository with the versions
used to create public data products.
Any questions about the pipeline functionality or data products should be submitted through
the IRSA help-desk (https://irsa.ipac.caltech.edu/docs/help_desk.html).

### Versions and Release Notes

The public-access repository includes each pipeline version used to produce public data products,
starting with the "QR2" quick-release products released beginning October 2025.
The release notes for each version include a summary of the new functionality and bug fixes
for that version.

The public-access repository receives snapshots of the code, copied over from releases
of the actual internal pipeline development repository.
We are not able to directly mirror the development repository for this mission, due to a
variety of constraints, but we make every effort to ensure that the public version is an
accurate copy of the code actually executed.
Each tag in the public-access repository is paired with a corresponding tag in the development
repository.

## References

The following links provide information on the SPHEREx mission, pipeline and data products:
* SPHERE Mission Description: "The SPHEREx Satellite Mission", Bock et al, 2026, https://ui.adsabs.harvard.edu/abs/2026ApJ...999..139B
* SPHEREx Pipeline Infrastructure and High-Level Description: "The SPHEREx Image and Spectrophotometry Processing Pipeline", Akeson et al, 2026, https://ui.adsabs.harvard.edu/abs/2025arXiv251115823A
* SPHEREx Explanatory Supplement, includes algorithm and data products details: https://irsa.ipac.caltech.edu/data/SPHEREx/docs/SPHEREx_Expsupp_QR.pdf
* SPHEREx Page at IRSA: https://irsa.ipac.caltech.edu/Missions/spherex.html
* SPHEREx Data Simulator: https://ui.adsabs.harvard.edu/abs/2025ApJS..281...10C/abstract
