import tvm
from tvm import relay

from scope import VarScope
from shared_test_generators import TestPatternGenerator, TestTypeGenerator
from type_constructor import TypeConstructs as TC
from type_utils import instantiate_constructor

NUM_ATTEMPTS = 100

def generate_appropriate_type(prelude, gen_params):
    return TestTypeGenerator(prelude).ctor.construct_type(gen_params=gen_params)


def generate_pattern(prelude, ty):
    return TestPatternGenerator(VarScope("test_var"), prelude).generate_pattern(ty)


def validate_pattern_type(prelude, pat, ty):
    if isinstance(pat, (relay.PatternWildcard, relay.PatternVar)):
        return True
    if isinstance(pat, relay.PatternTuple):
        if not isinstance(ty, relay.TupleType):
            return False
        if len(ty.fields) != len(pat.patterns):
            return False
        return all([
            validate_pattern_type(prelude, pat.patterns[i], ty.fields[i])
            for i in range(len(pat.patterns))
        ])
    if isinstance(pat, relay.PatternConstructor):
        if not isinstance(ty, relay.TypeCall):
            return False
        adt_handle = pat.constructor.belong_to
        if ty.func != adt_handle:
            return False
        ctor_type = instantiate_constructor(prelude, pat.constructor, ty)
        assert len(ctor_type.arg_types) == len(pat.patterns)
        return all([
            validate_pattern_type(prelude, pat.patterns[i], ctor_type.arg_types[i])
            for i in range(len(pat.patterns))
        ])
    raise ValueError(f"Unrecognized pattern/type {pat} {ty}")


def test_fuzz_non_matchable():
    # all types that cannot be matched
    gen_params = {
        TC.FUNC: {},
        TC.REF: {},
        TC.TENSOR: {}
    }
    prelude = relay.prelude.Prelude()
    pat_vars = 0
    for i in range(NUM_ATTEMPTS):
        ty = generate_appropriate_type(prelude, gen_params)
        pat = generate_pattern(prelude, ty)
        assert isinstance(pat, (relay.PatternWildcard, relay.PatternVar))
        if isinstance(pat, relay.PatternVar):
            assert pat.var.type_annotation == ty
            pat_vars += 1
    # just to make sure there is some randomness
    assert pat_vars != NUM_ATTEMPTS


def test_fuzz_tuples():
    # all types that cannot be matched
    gen_params = {
        TC.TUPLE: {}
    }
    prelude = relay.prelude.Prelude()
    tup_patterns = 0
    for i in range(NUM_ATTEMPTS):
        ty = generate_appropriate_type(prelude, gen_params)
        pat = generate_pattern(prelude, ty)
        assert validate_pattern_type(prelude, pat, ty)
        if isinstance(pat, relay.PatternTuple):
            tup_patterns += 1
    assert tup_patterns != 0


def test_fuzz_list():
    prelude = relay.prelude.Prelude()
    list_var = prelude.mod.get_global_type_var("List")
    adt_patterns = 0
    for i in range(NUM_ATTEMPTS):
        list_ty = list_var(generate_appropriate_type(prelude, None))
        pat = generate_pattern(prelude, list_ty)
        assert validate_pattern_type(prelude, pat, list_ty)
        if isinstance(pat, relay.PatternConstructor):
            adt_patterns += 1
    assert adt_patterns != 0


if __name__ == "__main__":
    test_fuzz_non_matchable()
    test_fuzz_tuples()
    test_fuzz_list()
