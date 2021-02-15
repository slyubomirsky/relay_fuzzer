"""
Basic generators that the tests will use to ensure the componenets work
"""
import random
from type_constructor import TypeConstructs as TC
from type_constructor import TypeConstructor

MAX_DIM = 5
MAX_ARITY = 5


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
