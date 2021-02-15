"""
Basic generators that the tests will use to ensure the components work
"""
import tvm
from tvm import relay
import random

from expr_constructor import PatternConstructor

from type_constructor import TypeConstructs as TC
from type_constructor import TypeConstructor

from type_utils import get_instantiated_constructors

MAX_DIM = 5
MAX_ARITY = 5

MAX_PATTERN_ATTEMPTS = 4

class FuelDriver:
    def __init__(self, fuel):
        self.fuel = fuel

    def decrease_fuel(self):
        if self.fuel > 0:
            self.fuel -= 1


class TestTypeGenerator(FuelDriver):
    def __init__(self, prelude, fuel=10):
        super().__init__(fuel)
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


class TestPatternGenerator(FuelDriver):
    def __init__(self, var_scope, prelude, fuel=10):
        super().__init__(fuel)
        self.p = prelude
        self.pat_ctor = PatternConstructor(var_scope, self.generate_pattern, self.choose_ctor)

    def generate_patterns(self, input_type):
        # to guarantee completeness, just put a wildcard at the end
        # (if we want to be less boring, we could do a check for completeness, but it's complicated)
        ret = []
        for i in range(MAX_PATTERN_ATTEMPTS):
            ret.append(self.generate_pattern(input_type))
            # random chance of stopping early
            if random.random() < 0.3:
                break
        ret.append(relay.PatternWildcard())
        return ret

    def generate_pattern(self, input_type):
        # we can always have a wildcard or var pattern
        if not isinstance(input_type, (relay.TupleType, relay.TypeCall)) or self.fuel == 0:
            return random.choice([
                relay.PatternWildcard(),
                self.pat_ctor.construct_var_pattern(input_type)
            ])
        self.decrease_fuel()
        # now we can choose between a wildcard, var, and either a ctor pattern or a tuple pattern depending on type
        choice = random.randint(0, 2)
        if choice == 0:
            return relay.PatternWildcard()
        elif choice == 1:
            return self.pat_ctor.construct_var_pattern(input_type)

        if isinstance(input_type, relay.TupleType):
            return self.pat_ctor.construct_tuple_pattern(input_type)
        return self.pat_ctor.construct_ctor_pattern(input_type)

    def choose_ctor(self, input_type):
        assert isinstance(input_type, relay.TypeCall)
        ctors = get_instantiated_constructors(self.p, input_type)
        return random.choice(ctors)
