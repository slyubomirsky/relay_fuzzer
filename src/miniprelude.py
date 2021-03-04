"""
A minimal version of the Relay prelude containing only the definitions
we will actually use (i.e., no tensor array functions)

Based on the TVM prelude
"""
import tvm

class MiniPrelude:
    def __init__(self, mod=None):
        if mod is None:
            mod = tvm.IRModule()
        self.mod = mod
        self.load_prelude()

    def load_prelude(self):
        self.mod.import_from_std("prelude.rly")

        GLOBAL_DEFS = [
            "id",
            "compose",
            "flip",
            "hd",
            "tl",
            "nth",
            "update",
            "map",
            "foldl",
            "foldr",
            "foldr1",
            "concat",
            "filter",
            "zip",
            "rev",
            "map_accuml",
            "map_accumr",
            "unfoldl",
            "unfoldr",
            "sum",
            "length",
            "tmap",
            "size",
            "iterate",
        ]

        for global_def in GLOBAL_DEFS:
            setattr(self, global_def, self.mod.get_global_var(global_def))
