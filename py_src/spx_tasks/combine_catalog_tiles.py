__all__ = ["CombineCatalogTilesConfig", "CombineCatalogTilesTask"]

# Standard library imports

# Related third party imports
from lsst.daf.butler import DeferredDatasetHandle
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import pyarrow as pa


class CombineCatalogTilesConnections(pipeBase.PipelineTaskConnections,
                                     dimensions={"instrument"},
                                     defaultTemplates={}):
    allsky_cat_list = cT.Input(
        name="allsky_cat",
        doc="Input primary all sky catalog per HEALPix.",
        storageClass="PyarrowTable",
        dimensions=["instrument", "hips6"],
        multiple=True,
        deferLoad=True)
    allsky_cat_combined = cT.Output(
        name="allsky_cat_combined",
        doc="Combined all-sky catalog (includes measurements across different "
            "HEALPix tiles).",
        storageClass="PyarrowTable",
        dimensions=["instrument"]
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class CombineCatalogTilesConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CombineCatalogTilesConnections):
    pass


class CombineCatalogTilesTask(pipeBase.PipelineTask):
    """Convert input primary/secondary all-sky catalogs to VOTable format.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-035

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    """

    _DefaultName = "combine-catalog-tiles"
    ConfigClass = CombineCatalogTilesConfig
    full_name = "Combine Catalog Tiles"
    ssdc_pipe_level = 3
    ssdc_doc_tag = "None"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, allsky_cat_list: list[DeferredDatasetHandle]) -> pipeBase.Struct:
        """Returns a condensed table of all source measurements for a set of
        HEALPix tiles.

        Parameters
        ----------
        allsky_cat_list : `list`[`lsst.daf.butler.DeferredDatasetHandle`]
            List of deferred `pyarrow.Table`, with each table containing source
            measurements for a different HEALPix tile.

        Returns
        -------
        output: `lsst.pipe.base.Struct`
            Output with component:
            - ``allsky_cat_combined`` : `pyarrow.Table`
                Contains a condensed `pyarrow.Table` with sources from distinct
                HEALPix tiles. Each row represents a single source, with array
                columns for multiple measurements.
        """
        num_cats = len(allsky_cat_list)
        self.log.debug(f"Number of input all-sky catalogs: {num_cats}")

        # Get pyarrow Table objects from deferred list and concatenate them
        pa_tabs = []
        for deferred_cat in allsky_cat_list:
            allsky_cat = deferred_cat.get()
            pa_tabs.append(allsky_cat)

        # Ignore chunking to enable data inspection by row
        pa_combined_tab = pa.concat_tables(pa_tabs).combine_chunks()

        return pipeBase.Struct(allsky_cat_combined=pa_combined_tab)
