import collections
import collections.abc


def patch_collections() -> None:
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping
