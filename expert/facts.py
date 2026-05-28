from expert._compat import patch_collections

patch_collections()

from experta import Fact


class Incident(Fact):
    pass


class WantsInfo(Fact):
    pass
