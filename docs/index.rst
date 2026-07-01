#################
SPHEREx Pipelines
#################

**SPHEREx data processing.**

The SPHEREx Pipeline System is intended to perform the processing required to produce the Level 1 through 3 SPHEREx Data Products,
principally calibrated images, targeted source forced photometry, interpolated spectra, and all-sky spectral cubes,
from the NASA `SPHEREx mission <https://spherex.caltech.edu>`__, a Mid-scale Explorer (MidEx).

The scientific content of the pipeline code is organized into a series of approximately 40 *modules*,
defined by `documents <https://spherex-docs.ipac.caltech.edu/ssdc-ms>`__
agreed between the SPHEREx Science Team and the SPHEREx Science Data Center at `Caltech/IPAC <https://ipac.caltech.edu/>`__ .

Image processing is based on primitives from the `Astropy <https://astropy.org>`__ project,
elaborated for SPHEREx purposes.

Pipeline construction and data management are based on substantial elements of the "middleware" from the Vera C. Rubin Observatory's 
open-source `LSST Science Pipelines <https://pipelines.lsst.io>`__ (`arxiv:2206.14941 <https://arxiv.org/abs/2206.14941>`__).

The pipeline construction framework views the *Pipeline Task* as a basic unit of processing. 
A series of pipeline tasks with configuration parameters make a *Pipeline*.
A *Pipeline Activator* executes a *Pipeline* (or an individual *Pipeline Task*).
The Rubin Data *Butler* provides a framework for managing data consumed or produced by the pipelines.

Documentation management is provided by open-source tools derived from Rubin Observatory ones.

.. toctree::
   :hidden:

   User Guide <user-guide/index>
   API <packages/index>
   Develop <developer/index>
