"""
Parameterized generator for types, also intended to
separate decision-making policy from generating types,
though it's kept separate from the ExprConstructor
in that the ExprConstructor must build expressions to meet a type,
whereas the type generator is much less constrained
"""
from enum import Enum, auto, unique
import random

import tvm
from tvm import relay

from type_utils import instantiate

# TODO: create a config class we can use

@unique
class TypeConstructs(Enum):
    TENSOR = auto()
    TUPLE = auto()
    REF = auto()
    ADT = auto()
    FUNC = auto()

DEFAULT_CONSTRUCTS = tuple([c for c in TypeConstructs])
DEFAULT_GEN_PARAMS = {
    construct : {}
    for construct in DEFAULT_CONSTRUCTS
}

class TypeGenerator:
    def __init__(self, prelude,
                 choose_construct,
                 choose_func_arity, choose_tuple_arity, choose_adt_handle,
                 generate_dtype, generate_shape):
        self.p = prelude
        self.choose_construct = choose_construct
        self.choose_tuple_arity = choose_tuple_arity
        self.choose_func_arity = choose_func_arity
        self.choose_adt_handle = choose_adt_handle
        self.generate_dtype = generate_dtype
        self.generate_shape = generate_shape

    def adt_type_vars(self, adt_handle):
        td = self.p.mod[adt_handle]
        return td.type_vars

    def supported_adts(self):
        list_var = self.p.mod.get_global_type_var("List")
        tree_var = self.p.mod.get_global_type_var("Tree")
        option_var = self.p.mod.get_global_type_var("Option")
        return [list_var, tree_var, option_var]

    def dispatch_by_construct(self, construct, params):
        if construct == TypeConstructs.TENSOR:
            return self.generate_tensor_type(params)
        if construct == TypeConstructs.TUPLE:
            return self.generate_tuple_type(params)
        if construct == TypeConstructs.REF:
            return self.generate_ref_type(params)
        if construct == TypeConstructs.ADT:
            return self.generate_adt(params)
        if construct == TypeConstructs.FUNC:
            return self.generate_func_type(params)
        raise ValueError(f"Invalid construct {construct}")

    def generate_type(self, gen_params=None):
        """
        Entry method: Generates a type at random,
        with optional paremeters to specify.

        Parameters format:
        {TypeConstruct -> generation parameters},
        where only the specified constructs will be used
        """
        # TODO: Make the specification for the generation parameters more solid
        # and use classes instead of a dict for specifying the rules (prevent typos)
        constructs = DEFAULT_CONSTRUCTS
        params = DEFAULT_GEN_PARAMS
        if gen_params is not None:
            if len(gen_params) == 0:
                raise Exception("No generation constructs specified")
            params = gen_params
            constructs = tuple(params.keys())

        construct = self.choose_construct(constructs)
        construct_params = params[construct]
        return self.dispatch_by_construct(construct, construct_params)

    def generate_tensor_type(self, _):
        dtype = self.generate_dtype()
        shape = self.generate_shape()
        return relay.TensorType(shape, dtype)

    def generate_tuple_type(self, params):
        min_arity = 0
        if "min_arity" in params:
            min_arity = params["min_arity"]
        arity = self.choose_tuple_arity(min_arity)
        fields = [None for i in range(arity)]

        if "constrained" in params:
            for idx, ty in params["constrained"].items():
                assert isinstance(ty, relay.Type)
                if idx >= arity:
                    raise ValueError(
                        f"Constrained index {idx} is not consistent with the tuple arity {arity},"
                        " consider constraining the min arity")
                fields[idx] = ty

        for i in range(len(fields)):
            if fields[i] is not None:
                continue
            fields[i] = self.generate_type()
        return relay.TupleType(fields)

    def generate_ref_type(self, _):
        return relay.RefType(self.generate_type())

    def generate_func_type(self, params):
        if "ret_type" in params:
            ret_type = params["ret_type"]
            assert isinstance(ret_type, relay.Type)
        else:
            ret_type = self.generate_type()

        arity = self.choose_func_arity()
        arg_types = [self.generate_type() for i in range(arity)]
        return relay.FuncType(arg_types, ret_type)

    def generate_adt(self, _):
        adt_handle = self.choose_adt_handle(self.supported_adts())
        type_vars = self.adt_type_vars(adt_handle)
        base_call = relay.TypeCall(adt_handle, type_vars)
        instantiation_list = [
            (tv, self.generate_type())
            for tv in type_vars
        ]
        return instantiate(base_call, instantiation_list)
