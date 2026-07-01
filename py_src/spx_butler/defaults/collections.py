__all__ = ["Collections"]

from dataclasses import dataclass
from lsst.daf.butler import CollectionType


@dataclass(frozen=True)
class Collection:
    name: str
    collection_type: CollectionType
    description: str


class Collections():
    """Predefined collections used in SSDC pipelines."""
    refcat = Collection("refcats",
                        CollectionType.CHAINED,
                        "Reference Catalogs")
    calib = Collection("calib",
                       CollectionType.CHAINED,
                       "Calibrations")

    @classmethod
    def persistence_state(cls, collection_name) -> Collection:
        return Collection(collection_name,
                          CollectionType.TAGGED,
                          "Saved Persistent State Models")
