"""
Basic generators that the tests will use to ensure the components work
"""
import numpy as np
import tvm
from tvm import relay
import random

from expr_constructor import ExprConstructor, PatternConstructor

from op_info import (ALL_BROADCASTING_OPS, ALL_IDENTITY_OPS,
                     ALL_NONSCALAR_OPS, BatchNormInfo, BatchMatmulInfo)
from relation_solver import MemoizedSolver, ILPSolver
from scope import VarScope

from type_constructor import TypeConstructs as TC
from type_constructor import TypeConstructor

from type_utils import (instantiate, get_instantiated_constructors, attempt_unify, partially_instantiate_func)

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
        self.decrease_fuel()
        if self.fuel == 0 and len(self.types_generated) != 0:
            return random.choice(self.types_generated)
        ret = self.ctor.construct_type()
        self.types_generated.append(ret)
        return ret

    def choose_construct(self, available_constructs):
        self.decrease_fuel()
        return random.choice(available_constructs)

    def choose_tuple_arity(self, min_arity):
        self.decrease_fuel()
        if self.fuel == 0:
            return min_arity
        return random.randint(min_arity, max(min_arity, MAX_ARITY))

    def choose_func_arity(self):
        self.decrease_fuel()
        if self.fuel == 0:
            return 0
        return random.randint(0, MAX_ARITY)

    def choose_adt_handle(self, available_handles):
        self.decrease_fuel()
        return random.choice(available_handles)

    def generate_dtype(self):
        self.decrease_fuel()
        return random.choice(["float32", "float64", "int8", "bool"])

    def generate_shape(self):
        self.decrease_fuel()
        if self.fuel == 0:
            return []
        arity = random.randint(0, MAX_ARITY)
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
        self.decrease_fuel()
        # we can always have a wildcard or var pattern
        if not isinstance(input_type, (relay.TupleType, relay.TypeCall)) or self.fuel == 0:
            return random.choice([
                relay.PatternWildcard(),
                self.pat_ctor.construct_var_pattern(input_type)
            ])
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
        self.decrease_fuel()
        assert isinstance(input_type, relay.TypeCall)
        ctors = get_instantiated_constructors(self.p, input_type)
        # if we don't pick an argument-free constructor eventually,
        # we can have an infinite loop
        if self.fuel == 0:
            for ctor, ft in ctors:
                if len(ft.arg_types) == 0:
                    return ctor, ft
        return random.choice(ctors)


class TestExprGenerator(FuelDriver):
    def __init__(self, prelude, fuel=10):
        super().__init__(fuel)
        self.var_scope = VarScope("gen_var")
        for gv in prelude.mod.get_global_vars():
            # not all global vars checked types are populated
            # and there is no way to check besides this...
            try:
                self.var_scope.add_to_global_scope(gv, gv.checked_type)
            except:
                continue

        self.prelude = prelude
        self.solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))

        # (eventually we should factor this out)
        # operators that can return any tensor
        self.basic_tensor_ops = [
            ctor(MAX_DIM, self.solver)
            for ctor in (ALL_IDENTITY_OPS + ALL_BROADCASTING_OPS)
        ]
        # operators that can return any nonscalar tensor
        self.all_nonscalar_ops = self.basic_tensor_ops + [ctor(MAX_DIM, self.solver)
                                                          for ctor in ALL_NONSCALAR_OPS]
        # batch matmul returns a tensor of rank exactly 3,
        # and it's probably not the only one
        self.rank_3_ops = self.all_nonscalar_ops + [BatchMatmulInfo(MAX_DIM, self.solver)]
        self.batch_norm_info = BatchNormInfo(MAX_DIM, self.solver)

        self.expr_ctor = ExprConstructor(
            self.var_scope, self.generate_expr, self.generate_type, self.choose_ctor,
            self.generate_patterns, self.generate_op)

        self.literal_chance = 0.3
        self.local_var_chance = 0.3
        self.global_var_chance = 0.05
        self.ref_write_chance = 0.05

    def generate_type(self, gen_params=None):
        if gen_params is None:
            return TestTypeGenerator(self.prelude).generate_type()
        return TestTypeGenerator(self.prelude).ctor.construct_type(gen_params=gen_params)

    def choose_ctor(self, type_call):
        # passing our fuel here because there is the potential
        # for constructing infinitely deep ADT literals
        # if we don't force termination this way
        return TestPatternGenerator(self.var_scope, self.prelude, fuel=self.fuel).choose_ctor(type_call)

    def has_available_op_calls(self, ty):
        # types supported by op calls: tensor types or tuple of a tensor and 2 vectors;
        # this is more efficient than naively looping through all the types
        if isinstance(ty, relay.TensorType):
            return True
        # batch norm
        if isinstance(ty, relay.TupleType):
            return self.batch_norm_info.supports_return_type(ty)
        return False

    def supported_ops(self, ty):
        ret = []
        if isinstance(ty, relay.TensorType):
            ret = self.basic_tensor_ops
            if len(ty.shape) != 0:
                ret = self.all_nonscalar_ops
                # batch matmul works for a rank of exactly 3
                if len(ty.shape) == 3:
                    ret += self.rank_3_ops
        # taking a shortcut for now
        if isinstance(ty, relay.TupleType):
            ret = [self.batch_norm_info]
        assert len(ret) != 0
        return ret

    def generate_patterns(self, ty):
        return TestPatternGenerator(self.var_scope, self.prelude).generate_patterns(ty)

    def generate_op(self, ty):
        return random.choice(self.supported_ops(ty))

    def generate_literal(self, ty, own_name=None):
        # if we have a variable in scope of this type, pick that instead
        if random.random() < self.local_var_chance:
            local_scope = self.var_scope.get_local_scope()
            appropriate_type = [v for v in local_scope if v.type_annotation == ty]
            if len(appropriate_type) != 0:
                return random.choice(appropriate_type)

        if isinstance(ty, relay.FuncType) and random.random() < self.global_var_chance:
            global_scope = self.var_scope.get_global_scope()
            global_choices = []
            for gv, gv_ty in global_scope.items():
                success, _ = attempt_unify(ty, gv_ty)
                if success:
                    global_choices.append(gv)
            if len(global_choices) != 0:
                return random.choice(global_choices)

        if isinstance(ty, relay.TensorType):
            # numpy doesn't handle floats the same as tensors
            dtype = ty.dtype
            if len(ty.shape) == 0:
                if dtype.startswith("float"):
                    return relay.const(random.random(), dtype=dtype)
                if dtype.startswith("bool"):
                    return relay.const(True, dtype=dtype)
                return relay.const(random.randint(0, 10), dtype=dtype)
            # numpy doesn't like TVM shapes
            conc_shape = [int(s) for s in ty.shape]
            return relay.const(np.random.rand(*conc_shape).astype(dtype), dtype=dtype)
        if isinstance(ty, relay.TupleType):
            return self.expr_ctor.construct_tuple_literal(ty.fields)
        if isinstance(ty, relay.FuncType):
            return self.expr_ctor.construct_func_literal(ty.arg_types, ty.ret_type, own_name=own_name)
        if isinstance(ty, relay.RefType):
            return self.expr_ctor.construct_ref_literal(ty.value)
        if isinstance(ty, relay.TypeCall):
            return self.expr_ctor.construct_adt_literal(ty)
        raise TypeError("Unrecognized type")

    def generate_global_call(self, ty):
        global_scope = self.var_scope.get_global_scope()
        global_choices = []
        for gv, gv_ty in global_scope.items():
            success, ft = partially_instantiate_func(gv_ty, ty)
            if not success:
                continue
            global_choices.append((gv, ft))

        if len(global_choices) == 0:
            return False, None
        gv, ft = random.choice(global_choices)
        # instantiate any remaining type vars
        instantiation_map = [
            (tv, self.generate_type())
            for tv in ft.type_params
        ]
        strip_ft = relay.FuncType(ft.arg_types, ft.ret_type)
        inst_ft = instantiate(strip_ft, instantiation_map)
        return True, relay.Call(gv, [
            self.generate_expr(arg_ty)
            for arg_ty in inst_ft.arg_types
        ])

    def generate_connective(self, ty):
        # see if we can get a global function with the right return type
        if random.random() < self.global_var_chance:
            success, call = self.generate_global_call(ty)
            if success:
                return call

        choices = [
            lambda: self.expr_ctor.construct_ref_read(ty),
            lambda: self.expr_ctor.construct_function_call(ty),
            lambda: self.expr_ctor.construct_match(ty),
            lambda: self.expr_ctor.construct_if_branch(ty),
            lambda: self.expr_ctor.construct_tuple_index(ty, random.randint(0, MAX_ARITY-1))
        ]

        # constructs available only for some types
        if ty == relay.TupleType([]) and random.random() < self.ref_write_chance:
            choices.append(self.expr_ctor.construct_ref_write)
        if self.has_available_op_calls(ty):
            choices.append(lambda: self.expr_ctor.construct_op_call(ty))

        thunk = random.choice(choices)
        return thunk()

    def generate_expr(self, ty, own_name=None):
        self.decrease_fuel()
        if self.fuel == 0:
            return self.generate_literal(ty, own_name=own_name)
        if random.random() < self.literal_chance:
            return self.generate_literal(ty, own_name=own_name)
        return self.generate_connective(ty)
