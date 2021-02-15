import itertools
import random

import tvm
from tvm import relay

from type_constructor import TypeConstructor
from type_constructor import TypeConstructs as TC

MAX_DIM = 5
MAX_ARITY = 5

INITIAL_FUEL = 10
ALL_ATTEMPTS = 100
CATEGORY_ATTEMPTS = 50

class TestDriver:
    """
    Set a fuel parameter to limit the possible recursion depth
    """
    def __init__(self, fuel, prelude):
        self.fuel = fuel
        self.types_generated = []
        # mutual recursion!
        self.ctor = TypeConstructor(prelude,
                                    self.choose_construct,
                                    self.choose_func_arity,
                                    self.choose_tuple_arity,
                                    self.choose_adt_handle,
                                    self.generate_dtype,
                                    self.generate_shape,
                                    generate_type=self.generate_type)

    def decrease_fuel(self):
        if self.fuel > 0:
            self.fuel -= 1

    def generate_type(self):
        if self.fuel == 0 and len(self.types_generated) != 0:
            return random.choice(self.types_generated)
        self.decrease_fuel()
        ret = self.ctor.construct_type()
        self.types_generated.append(ret)
        return ret

    def choose_construct(self, available_constructs):
        self.decrease_fuel()
        return random.choice(available_constructs)

    def choose_tuple_arity(self, min_arity):
        if self.fuel == 0:
            return min_arity
        self.decrease_fuel()
        return random.randint(min_arity, max(min_arity, MAX_ARITY))

    def choose_func_arity(self):
        if self.fuel == 0:
            return 0
        self.decrease_fuel()
        return random.randint(0, MAX_ARITY)

    def choose_adt_handle(self, available_handles):
        self.decrease_fuel()
        return random.choice(available_handles)

    def generate_dtype(self):
        self.decrease_fuel()
        return random.choice(["float32", "float64", "int8", "bool"])

    def generate_shape(self):
        arity = random.randint(0, MAX_ARITY)
        self.decrease_fuel()
        return [random.randint(1, MAX_DIM) for i in range(arity)]


def create_constructor():
    prelude = relay.prelude.Prelude()
    driver = TestDriver(INITIAL_FUEL, prelude)
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
        ft = ctor.construct_type(gen_params={
            TC.FUNC: {
                "ret_type": ret_type
            }
        })
        assert isinstance(ft, relay.FuncType)
        assert ft.ret_type == ret_type


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
        tt = ctor.construct_type({
            TC.TUPLE: {
                "min_arity": min_arity,
                "constrained": constraints
            }
        })
        assert isinstance(tt, relay.TupleType)
        assert len(tt.fields) >= min_arity
        assert tt.fields[0] == first_ty
        assert tt.fields[2] == second_ty


if __name__ == "__main__":
    test_fuzz_default()
    test_fuzz_separately()
    test_specify_func_ret_type()
    test_constrain_tuple()