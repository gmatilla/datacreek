from enum import IntEnum


class DatasetStage(IntEnum):
    """Lifecycle stage for a dataset."""

    CREATED = 0
    INGESTED = 1
    GENERATED = 2
    CURATED = 3
    EXPORTED = 4
