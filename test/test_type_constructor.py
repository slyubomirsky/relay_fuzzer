import itertools

import tvm
from tvm import relay

from type_constructor import TypeConstructor, params_met
from type_constructor import TypeConstructs as TC

from shared_test_generators import TestTypeGenerator

ALL_ATTEMPTS = 100
CATEGORY_ATTEMPTS = 50

def create_constructor():
    prelude = relay.prelude.Prelude()
    driver = TestTypeGenerator(prelude)
    return (driver.ctor, prelude)


def match_construct(ty):
    if isinstance(ty, relay.TensorType):
        return TC.TENSOR
    if isinstance(ty, relay.TupleType):
        return TC.TUPLE
    if isinstance(ty, relay.RefType):
        return TC.REF
    if isinstance(ty, relay.FuncType):
        return TC.FUNC
    if isinstance(ty, relay.TypeCall):
        return TC.ADT
    assert False, "Unrecognized type"

# note: we create the constructor inside the loop
# to reset the fuel for each new run

def test_fuzz_default():
    represented = set({})
    for i in range(ALL_ATTEMPTS):
        ctor, _ = create_constructor()
        ty = ctor.construct_type()
        represented.add(match_construct(ty))
    # if we have a large number of attempts, all types should be represented
    assert len(represented) == len(TC)


def test_fuzz_separately():
    constructs = [c for c in TC]
    # test single constructs, pairs, and triplets
    for i in range(1, 4):
        ctor, _ = create_constructor()
        for combo in itertools.combinations(constructs, i):
            expected = set(combo)
            conf = {c: {} for c in combo}
            for i in range(CATEGORY_ATTEMPTS):
                ty = ctor.construct_type(gen_params=conf)
                assert match_construct(ty) in expected


def test_specify_func_ret_type():
    for i in range(CATEGORY_ATTEMPTS):
        ctor, _ = create_constructor()
        ret_type = ctor.construct_type()
        gen_params = {
            TC.FUNC: {
                "ret_type": ret_type
            }
        }
        ft = ctor.construct_type(gen_params=gen_params)
        assert isinstance(ft, relay.FuncType)
        assert ft.ret_type == ret_type
        assert params_met(ft, gen_params)


def test_constrain_tuple():
    for i in range(CATEGORY_ATTEMPTS):
        ctor, _ = create_constructor()
        min_arity = 3
        first_ty = ctor.construct_type()
        second_ty = ctor.construct_type()
        constraints = {
            0: first_ty,
            2: second_ty
        }
        gen_params = {
            TC.TUPLE: {
                "min_arity": min_arity,
                "constrained": constraints
            }
        }
        tt = ctor.construct_type(gen_params=gen_params)
        assert params_met(tt, gen_params)
        assert isinstance(tt, relay.TupleType)
        assert len(tt.fields) >= min_arity
        assert tt.fields[0] == first_ty
        assert tt.fields[2] == second_ty


if __name__ == "__main__":
    test_fuzz_default()
    test_fuzz_separately()
    test_specify_func_ret_type()
    test_constrain_tuple()
