"""
Parameterized generator for types, also intended to
separate decision-making policy from generating types.
"""
from enum import Enum, auto, unique
import random

import tvm
from tvm import relay

from type_utils import instantiate

@unique
class TypeConstructs(Enum):
    TENSOR = auto()
    TUPLE = auto()
    REF = auto()
    ADT = auto()
    FUNC = auto()

TC = TypeConstructs

DEFAULT_CONSTRUCTS = tuple([c for c in TypeConstructs])
DEFAULT_GEN_PARAMS = {
    construct : {}
    for construct in DEFAULT_CONSTRUCTS
}

def params_met(ty, params):
    """
    Given the generation params, checks that they have been met for a concrete type.
    (Provided as a single source of truth.)
    """
    if params is None:
        return True
    if isinstance(ty, relay.TensorType):
        return TC.TENSOR in params
    if isinstance(ty, relay.RefType):
        return TC.REF in params
    if isinstance(ty, relay.TypeCall):
        return TC.ADT in params
    if isinstance(ty, relay.FuncType):
        if TC.FUNC not in params:
            return False
        f_params = params[TC.FUNC]
        if "ret_type" in f_params:
            return ty.ret_type == f_params["ret_type"]
        return True
    if isinstance(ty, relay.TupleType):
        if TC.TUPLE not in params:
            return False
        t_params = params[TC.TUPLE]
        if "min_arity" in t_params:
            if len(ty.fields) < t_params["min_arity"]:
                return False
        if "constrained" in params:
            for idx, inner_ty in t_params["constrained"].items():
                if idx >= len(ty.fields):
                    return False
                if ty.fields[idx] != inner_ty:
                    return False
        return True
    raise InputError("Unsupported type")


class TypeConstructor:
    def __init__(self, prelude,
                 choose_construct,
                 choose_func_arity, choose_tuple_arity, choose_adt_handle,
                 generate_dtype, generate_shape,
                 generate_type=None):
        self.p = prelude
        self.choose_construct = choose_construct
        self.choose_tuple_arity = choose_tuple_arity
        self.choose_func_arity = choose_func_arity
        self.choose_adt_handle = choose_adt_handle
        self.generate_dtype = generate_dtype
        self.generate_shape = generate_shape
        # no generate type: naive recursion
        if generate_type is None:
            self.generate_type = self.construct_type
        else:
            self.generate_type = generate_type

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
            return self.construct_tensor_type(params)
        if construct == TypeConstructs.TUPLE:
            return self.construct_tuple_type(params)
        if construct == TypeConstructs.REF:
            return self.construct_ref_type(params)
        if construct == TypeConstructs.ADT:
            return self.construct_adt(params)
        if construct == TypeConstructs.FUNC:
            return self.construct_func_type(params)
        raise ValueError(f"Invalid construct {construct}")

    def construct_type(self, gen_params=None):
        """
        Entry method: Constructs a type,
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

    def construct_tensor_type(self, _):
        dtype = self.generate_dtype()
        shape = self.generate_shape()
        return relay.TensorType(shape, dtype)

    def construct_tuple_type(self, params):
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

    def construct_ref_type(self, _):
        return relay.RefType(self.generate_type())

    def construct_func_type(self, params):
        if "ret_type" in params:
            ret_type = params["ret_type"]
            assert isinstance(ret_type, relay.Type)
        else:
            ret_type = self.generate_type()

        arity = self.choose_func_arity()
        arg_types = [self.generate_type() for i in range(arity)]
        return relay.FuncType(arg_types, ret_type)

    def construct_adt(self, _):
        adt_handle = self.choose_adt_handle(self.supported_adts())
        type_vars = self.adt_type_vars(adt_handle)
        base_call = relay.TypeCall(adt_handle, type_vars)
        instantiation_list = [
            (tv, self.generate_type())
            for tv in type_vars
        ]
        return instantiate(base_call, instantiation_list)
