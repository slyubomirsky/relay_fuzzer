"""
Basic generators that the tests will use to ensure the components work
"""
import numpy as np
import tvm
from tvm import relay
import random

from expr_constructor import ExprConstructor, PatternConstructor

from op_info import (ALL_BROADCASTING_OPS, ALL_IDENTITY_OPS,
                     ALL_NONSCALAR_OPS, BatchNormInfo, BatchMatmulInfo, Conv2DInfo)
from relation_solver import MemoizedSolver, ILPSolver, BruteForceSolver, ProfiledSolver
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
    def __init__(self, prelude, fuel=10, max_arity=MAX_ARITY, max_dim=MAX_DIM):
        super().__init__(fuel)
        self.max_arity = max_arity
        self.max_dim = max_dim
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
        return random.randint(min_arity, max(min_arity, self.max_arity))

    def choose_func_arity(self):
        self.decrease_fuel()
        if self.fuel == 0:
            return 0
        return random.randint(0, self.max_arity)

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
        arity = random.randint(0, self.max_arity)
        return [random.randint(1, self.max_dim) for i in range(arity)]


class TestPatternGenerator(FuelDriver):
    def __init__(self, var_scope, prelude, fuel=10, pattern_attempts=MAX_PATTERN_ATTEMPTS):
        super().__init__(fuel)
        self.p = prelude
        self.pat_ctor = PatternConstructor(var_scope, self.generate_pattern, self.choose_ctor)
        self.pattern_attempts = pattern_attempts

    def generate_patterns(self, input_type):
        # to guarantee completeness, just put a wildcard at the end
        # (if we want to be less boring, we could do a check for completeness, but it's complicated)
        ret = []
        for i in range(self.pattern_attempts):
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
    def __init__(self, prelude, conf=None):
        self.config = validate_config(conf)
        super().__init__(self.config.fuel)
        self.prelude = prelude

        self.var_scope = VarScope("gen_var")
        for gv in prelude.mod.get_global_vars():
            # not all global vars checked types are populated
            # and there is no way to check besides this...

            # "main" causes problems if we are overwriting it in the fuzzer
            # (we may change its type)
            if self.config.exclude_main and gv.name_hint == "main":
                continue
            try:
                self.var_scope.add_to_global_scope(gv, gv.checked_type)
            except:
                continue

        # (eventually we should factor out dispatching operators)
        # operators that can return any tensor
        self.basic_tensor_ops = self.config.basic_tensor_ops
        # operators that can return any nonscalar tensor
        self.all_nonscalar_ops = self.config.all_nonscalar_ops
        # batch matmul returns a tensor of rank exactly 3,
        # and it's probably not the only one
        self.rank_3_ops = self.config.rank_3_ops
        # ditto conv2d for rank 4
        self.rank_4_ops = self.config.rank_4_ops
        self.batch_norm_info = self.config.batch_norm_info
        # convenient for forward solving
        self.all_ops = self.config.all_ops

        # mapping of type hashes -> arg types, additional params, op_info
        # to produce a call if the appropriate type comes up
        self.forward_queue = {}

        self.expr_ctor = ExprConstructor(
            self.var_scope, self.generate_expr, self.generate_type, self.choose_ctor,
            self.generate_patterns, self.generate_op)

        # eventually we'll want to configure these
        self.literal_chance = 0.3
        self.local_var_chance = 0.3
        self.global_var_chance = 0.05
        self.ref_write_chance = 0.05
        # operators are more fun so we will want those to happen more often
        self.operator_chance = 0.25
        # may need to fiddle with it so as not to skew too much
        self.forward_solving_chance = 0.5
        # another option: always use the forward queue but destroy it so we don't become overly reliant on it
        self.use_forward_queue_chance = 0.5

    def set_seed(self, seed):
        self.config.reset_seed(seed)

    def get_solver_profile(self):
        return self.config.get_solver_profile()

    def generate_type(self, gen_params=None):
        gen = self.config.produce_type_generator(self.prelude)
        if gen_params is None:
            # if we're free to do as we like, let's forward-solve
            if (self.config.use_forward_solving and random.random() < self.forward_solving_chance):
                return self.forward_solve()
            return gen.generate_type()
        # TODO: Fix the interface for generate type
        return gen.ctor.construct_type(gen_params=gen_params)

    def forward_solve(self):
        # pick an operator, solve for its types, and put it in the queue for the next time the type comes up
        op_info = random.choice(self.all_ops)
        arg_types, ret_type, params = op_info.sample_call()
        # terrible hack
        ty_hash = tvm.ir.structural_hash(ret_type)
        self.forward_queue[ty_hash] = arg_types, params, op_info
        return ret_type

    def hit_forward_queue(self, ty):
        if not self.config.use_forward_solving:
            return False
        if not self.has_available_op_calls(ty):
            return False
        ty_hash = tvm.ir.structural_hash(ty)
        return ty_hash in self.forward_queue

    def choose_ctor(self, type_call):
        # passing our fuel here because there is the potential
        # for constructing infinitely deep ADT literals
        # if we don't force termination this way
        gen = self.config.produce_pattern_generator(self.var_scope, self.prelude, fuel=self.fuel)
        return gen.choose_ctor(type_call)

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
                    ret = self.rank_3_ops
                # conv2d works for a rank of exactly 4
                if len(ty.shape) == 4:
                    ret = self.rank_4_ops
        # taking a shortcut for now
        if isinstance(ty, relay.TupleType):
            ret = [self.batch_norm_info]
        assert len(ret) != 0
        return ret

    def generate_patterns(self, ty):
        return TestPatternGenerator(self.var_scope, self.prelude).generate_patterns(ty)

    def generate_op(self, ty):
        if self.hit_forward_queue(ty) and random.random() < self.use_forward_queue_chance:
            ty_hash = tvm.ir.structural_hash(ty)
            _, _, op_info = self.forward_queue[ty_hash]
            return op_info
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

    def generate_forward_call(self, ty):
        # take a call out of the forward queue!
        ty_hash = tvm.ir.structural_hash(ty)
        arg_types, params, op_info = self.forward_queue[ty_hash]
        return op_info.produce_call([
            self.generate_expr(arg_ty)
            for arg_ty in arg_types
        ], additional_params=params)


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
            lambda: self.expr_ctor.construct_tuple_index(ty, random.randint(0, self.config.max_arity-1))
        ]

        # constructs available only for some types
        if ty == relay.TupleType([]) and random.random() < self.ref_write_chance:
            choices.append(self.expr_ctor.construct_ref_write)
        if self.has_available_op_calls(ty):
            # dumb way of skewing the choice
            if random.random() < self.operator_chance:
                return self.expr_ctor.construct_op_call(ty)

            choices.append(lambda: self.expr_ctor.construct_op_call(ty))

        thunk = random.choice(choices)
        return thunk()

    def generate_expr(self, ty, own_name=None):
        self.decrease_fuel()
        if self.fuel == 0:
            return self.generate_literal(ty, own_name=own_name)
        if (self.hit_forward_queue(ty)
            and random.random() < self.use_forward_queue_chance):
            return self.generate_forward_call(ty)
        if random.random() < self.literal_chance:
            return self.generate_literal(ty, own_name=own_name)
        return self.generate_connective(ty)


def validate_config(config):
    """
    Rudimentary mechanism for setting up a config object for the above test fuzzer

    If the config is None or is missing settings, this will populate them with defaults

    A configuration should be a dict containing some (or none) of the following fields:
    "max_dim": Largest dimension in any tensor shape (default: 5, real ones can be much larger)
    "max_arity": Max rank for tensor and max arity for tuples (default: 5)
                 TODO: Decouple the separate settings
    "max_pattern_attempts": Number of patterns the pattern generator will attempt to produce (default: 4)
    "solver_timeout": Seconds until the ILP solver should time out (default: 30)
                      If the solver is timing out at 30s, that's probably a bad sign
    "use_ilp": Whether to use the ILP solver; uses brute force if not (default: True)
    "memoize_solver": Whether to memoize solver results (default: True)
    "set_seed": Whether to set the random seed (default: False)
    "seed": The value for the random seed, if set to use (default: 0)
    "exclude_main": Whether to exclude the "main" variable from appearing in a generated expression.
                    This can lead to problems if reusing preludes (default: True)
    "use_forward_solving": Whether to use forward solving (sample operators to skew generator towards producing operators) (default: True)
    "fuel": Fuel parameter to control the size of generated expressions (default: 10)
    "type_fuel": Fuel parameter to control the size of generated types (default: fuel)
                 Recommendation: This should probably be lower than the general fuel,
                 as it may lead to really big programs that have to satisfy the huge types
                 but it may expose bugs to try big types
    "pattern_fuel": Fuel parameter to control the size of generated patterns (default: fuel)
                    Recommendation: Keep lower than the general fuel, but see above
    """
    class FuzzerConfig:
        def produce_type_generator(self, prelude, fuel=None):
            type_fuel = self.type_fuel
            if fuel is not None:
                type_fuel = fuel
            return TestTypeGenerator(prelude, fuel=type_fuel,
                                     max_arity=self.max_arity, max_dim=self.max_dim)

        def produce_pattern_generator(self, var_scope, prelude, fuel=None):
            pat_fuel = self.pattern_fuel
            if fuel is not None:
                pat_fuel = fuel
            return TestPatternGenerator(var_scope, prelude, fuel=pat_fuel,
                                        pattern_attempts=self.max_pattern_attempts)

        def initialize_solver(self):
            max_dim = self.max_dim
            # because the solver's random seed depends on the instance of the solver,
            # we need to configure this when we set up the ILP solver
            seed = None if not self.set_seed else self.seed
            solver = (ILPSolver(max_dim, self.solver_timeout, False, seed=seed)
                      if self.use_ilp else BruteForceSolver(max_dim))
            if self.memoize_solver:
                solver = MemoizedSolver(solver)
            self.solver = ProfiledSolver(solver)

        def initialize_ops(self):
            max_dim = self.max_dim
            solver = self.solver

            # TODO: factor out this logic and op dispatching to a dedicated class (too much coupling)
            self.basic_tensor_ops = [
                ctor(max_dim, solver)
                for ctor in (ALL_IDENTITY_OPS + ALL_BROADCASTING_OPS)
            ]
            # operators that can return any nonscalar tensor
            self.all_nonscalar_ops = self.basic_tensor_ops + [ctor(max_dim, solver)
                                                              for ctor in ALL_NONSCALAR_OPS]
            batch_matmul_info = BatchMatmulInfo(max_dim, solver)
            conv2d_info = Conv2DInfo(max_dim, solver)
            self.rank_3_ops = self.all_nonscalar_ops + [batch_matmul_info]
            self.rank_4_ops = self.all_nonscalar_ops + [conv2d_info]
            self.batch_norm_info = BatchNormInfo(max_dim, solver)
            self.all_ops = self.all_nonscalar_ops + [
                batch_matmul_info, conv2d_info, self.batch_norm_info
            ]

        def set_seeds(self):
            if not self.set_seed:
                return
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.solver.set_seed(self.seed)

        def reset_seed(self, seed):
            self.set_seed = True
            self.seed = seed
            self.set_seeds()

        def get_solver_profile(self):
            return self.solver.get_record()

    def set_field(config, inp, fieldname, default_value, validate=None):
        if inp is None or fieldname not in inp:
            setattr(config, fieldname, default_value)
            return
        if validate is not None and not validate(inp[fieldname]):
            raise ValueError(f"Invalid configuration value {listed_value}"
                             f" for field {fieldname}")
        setattr(config, fieldname, inp[fieldname])

    ret = FuzzerConfig()
    positive_int = lambda i: isinstance(i, int) and 0 < i
    is_bool = lambda b: isinstance(b, bool)

    set_field(ret, config, "max_dim", MAX_DIM, positive_int)
    set_field(ret, config, "max_arity", MAX_DIM, positive_int)
    set_field(ret, config, "max_pattern_attempts", MAX_PATTERN_ATTEMPTS, positive_int)
    set_field(ret, config, "solver_timeout", 30, positive_int)
    set_field(ret, config, "use_ilp", True, is_bool)
    set_field(ret, config, "memoize_solver", True, is_bool)
    set_field(ret, config, "set_seed", False, is_bool)
    set_field(ret, config, "random_seed", 0, lambda i: isinstance(i, int))
    set_field(ret, config, "exclude_main", True, is_bool)
    # unused for now
    set_field(ret, config, "use_forward_solving", True, is_bool)

    set_field(ret, config, "fuel", 10, positive_int)
    set_field(ret, config, "type_fuel", ret.fuel, positive_int)
    set_field(ret, config, "pattern_fuel", ret.fuel, positive_int)

    ret.initialize_solver()
    ret.initialize_ops()
    ret.set_seeds()
    return ret
