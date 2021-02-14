"""
Class for managing vars in scope
"""
import tvm
from tvm import relay

class ScopeLifetime:
    """
    Included to allow for the `with scope.new_scope():` construction
    """
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        self.parent.local_scopes.append([])

    def __exit__(self, a, b, c):
        self.parent.local_scopes.pop()


class VarScope:
    def __init__(self, var_stem):
        self.local_scopes = [[]]
        # global vars do not store a type so we map from var name to type
        self.global_scope = {}
        self.var_stem = var_stem
        self.var_idx = 0

    def new_local_var(self, var_type, add_to_scope=False):
        new_varname = f"{self.var_stem}_{self.var_idx}"
        self.var_idx += 1
        ret = relay.Var(new_varname, type_annotation=var_type)
        if add_to_scope:
            self.add_to_scope(ret)
        return ret

    def new_global_var(self, var_type):
        new_varname = f"global_{self.var_stem}_{self.var_idx}"
        self.var_idx += 1
        ret = relay.GlobalVar(new_varname)
        self.add_to_global_scope(ret, var_type)
        return ret

    def new_scope(self):
        return ScopeLifetime(self)

    def add_to_scope(self, local_var):
        assert isinstance(local_var, relay.Var)
        self.local_scopes[-1].append(local_var)

    def add_to_global_scope(self, global_var, var_type):
        assert isinstance(global_var, relay.GlobalVar)
        assert global_var not in self.global_scope
        self.global_scope[global_var] = var_type

    def get_global_scope(self):
        return {**self.global_scope}

    def get_local_scope(self):
        # Var names are only hints in Relay; there is no shadowing
        # at the level of the AST representation
        # (two vars are only the same if they are reference-equal),
        # so we don't have to do any checking
        ret = []
        for scope in self.local_scopes:
            ret += scope
        return ret
