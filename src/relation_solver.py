"""
Hide away all the ILP solving so the rest of the code
doesn't have to worry about it
"""
import itertools
import math
import time

import mip
import random
from mip import Model, BINARY, INTEGER


def enumerate_all_possible_shapes(ranks, max_dim):
    all_shapes = []
    for rank in ranks:
        if rank == 0:
            # if we have a scalar, we must indicate it is an empty element or itertools omits it
            all_shapes.append(([],))
            continue
        # Yes this grows insanely quickly.
        # Combinations with replacement is lexicographically ordered
        # so we have to take permutations afterwards.
        possible_shapes = []
        for combo in itertools.combinations_with_replacement(list(range(1, max_dim+1)), rank):
            possible_shapes += itertools.permutations(combo, len(combo))
        all_shapes.append(possible_shapes)
    return itertools.product(*all_shapes)

def marshal_ilp_shapes(arg_shapes, return_shapes):
    def instantiate_ilp_vars(shape):
        return tuple([
            int(v.x) if hasattr(v, "x") else int(v)
            for v in shape
        ])

    return (
        tuple([instantiate_ilp_vars(shape) for shape in arg_shapes]),
        tuple([instantiate_ilp_vars(shape) for shape in return_shapes])
    )

def add_and_constraint(solver, a, b):
    """
    For two boolean ILP variables a and b, this adds a new var c
    that is true iff a /\ b is true.

    Based on https://cs.stackexchange.com/a/12118
    """
    c = solver.add_var(var_type=BINARY)
    # a + b - 1 is 1 only when a = b = 1
    solver += (c >= a + b - 1)
    # if a is 0, c must be 0
    solver += (c <= a)
    # if b is 0, c must be 0
    solver += (c <= b)
    return c


def add_or_constraint(solver, a, b):
    """
    For two boolean ILP vars a and b, adds a new var c
    that is true iff a \/ b is true

    Based on https://cs.stackexchange.com/a/12118
    """
    c = solver.add_var(var_type=BINARY)
    # a + b is 0 only when they are both 0
    solver += (c <= a + b)
    # if a is 1, c must be 1
    solver += (c >= a)
    # if b is 1, c must be 1
    solver += (c >= b)
    return c


def add_gt_constraint(solver, a, b, M):
    """
    For two *integer* ILP vars a and b and *constant* M where M > a and M > b,
    this creates a boolean var c that is 1 iff a > b

    Based on https://math.stackexchange.com/a/2501007
    """
    c = solver.add_var(var_type=BINARY)
    # if a > b, this is true no matter the value of c
    # if a <= b, c *must be* 0 for this to be true
    solver += (a >= b - M*(1-c))
    # if a <= b, this is true no matter the value of c.
    # if a > b, c *must be* 1 for this to be true
    solver += (a <= b + M*c)
    return c


def add_eq_constraint(solver, a, b, M):
    """
    For two *integer* ILP vars (or integer constants) a and b and *constant* M
    where M > a and M > b, this creates a boolean var c that is 1 iff a == b
    """
    # a == b <--> a <= b and a >= b
    a_lte_b = 1 - add_gt_constraint(solver, a, b, M)
    a_gte_b = 1 - add_gt_constraint(solver, b, a, M)
    return add_and_constraint(solver, a_lte_b, a_gte_b)


class Relation:
    """
    Base class for specifying a type relation, meant to be
    compatible with different solving strategies

    Each relation is responsible for specifying a format for a problem instance
    and turning the results into a solution instance
    """
    def validate(self, problem_instance):
        """
        Validity check for the given problem instance
        """
        raise NotImplementedError()

    def check(self, problem_instance, solution):
        """
        Checks the relation for a concrete shape (meant for brute force)
        """
        raise NotImplementedError()

    def all_possible_solutions(self, problem_instance):
        """
        For brute force enumeration, the relation specifies
        an enumeration of all possible solution instances to check with the above method
        """
        raise NotImplementedError()

    def convert_to_ilp_problem(self, solver, problem_instance):
        """
        Turns an input problem instance into an ILP problem instance
        """
        raise NotImplementedError()

    def produce_ilp_constraints(self, solver, ilp_problem_instance):
        """
        As the name implies, adds appropriate ILP constraints using the solver,
        based on the given problem instance.
        """
        raise NotImplementedError()

    def marshal_ilp_solution(self, ilp_problem):
        """
        Given ilp problem, it should contain a solution after running the solver;
        this will turn it into a solution instance that can be checked with check()
        """
        raise NotImplementedError()


class Solver:
    """
    Base class for solvering type relations
    """
    def solve(self, relation, problem_instance):
        raise NotImplementedError()

    def set_seed(self, seed):
        pass


class BruteForceSolver(Solver):
    """
    Can work out if the range is small
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    def solve(self, relation, problem_instance):
        if not relation.validate(problem_instance):
            raise ValueError("Relation invalid for the given problem instance")

        for solution in relation.all_possible_solutions(problem_instance):
            if relation.check(problem_instance, solution):
                return solution
        raise ValueError(f"No solution found (infeasible): {problem_instance}")


class ILPSolver(Solver):
    def __init__(self, max_dim, max_time, solver_verbose, seed=None):
        self.max_dim = max_dim
        self.max_time = max_time
        m = Model()
        m.emphasis = mip.SearchEmphasis.FEASIBILITY
        m.verbose = int(solver_verbose)
        self.m = m
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed):
        self.m.seed = seed

    def found_ilp_solution(self, solver_result):
        # OTHER: problem was trivial (as may happen with scalars),
        #        so let's not consider that an error
        return (solver_result == mip.OptimizationStatus.OPTIMAL
                or solver_result == mip.OptimizationStatus.FEASIBLE
                or solver_result == mip.OptimizationStatus.OTHER)

    def solve(self, relation, problem_instance):
        if not relation.validate(problem_instance):
            raise ValueError("Relation invalid for the given problem_instance")

        ilp_problem = relation.convert_to_ilp_problem(self.m, problem_instance)
        relation.produce_ilp_constraints(self.m, ilp_problem)

        res = self.m.optimize(max_seconds=self.max_time)
        if not self.found_ilp_solution(res):
            raise ValueError("No solution found (infeasible)")

        return relation.marshal_ilp_solution(ilp_problem)


class MemoizedSolver(Solver):
    """
    Memoizes an underlying solver.
    Note: It is very important for problem instances to be hashable
    if they are to use this interface!
    """
    def __init__(self, solver):
        self.solver = solver
        self.memo = {}

    def set_seed(self, seed):
        self.solver.set_seed(seed)

    def solve(self, relation, problem_instance):
        if not relation.validate(problem_instance):
            raise ValueError("Relation invalid for the given problem instance")

        if relation not in self.memo:
            self.memo[relation] = {}

        if problem_instance in self.memo[relation]:
            return self.memo[relation][problem_instance]
        solution = self.solver.solve(relation, problem_instance)
        self.memo[relation][problem_instance] = solution
        return solution


class ProfiledSolver(Solver):
    """
    Wrapper for solvers that records times per query
    """
    def __init__(self, solver):
        self.solver = solver
        self.record = []

    def set_seed(self, seed):
        self.solver.set_seed(seed)

    def solve(self, relation, problem_instance):
        try:
            start_time = time.time()
            success = False
            ret = self.solver.solve(relation, problem_instance)
            success = True
            return ret
        finally:
            end_time = time.time()
            self.record.append({
                "time": end_time - start_time,
                "success": success
            })

    def get_record(self):
        return self.record


# some common relations
class IdentityRelation(Relation):
    """
    Asserts all arguments have the same shape as the (single) result
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overiding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return isinstance(other, IdentityRelation) and self.max_dim == other.max_dim

    def validate(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 1:
            return False
        sol_shape = return_shapes[0]
        for rank in arg_ranks:
            if rank != len(sol_shape):
                return False
        return True

    def check(self, _, solution):
        arg_shapes, return_shapes = solution
        sol_shape = return_shapes[0]
        for arg_shape in arg_shapes:
            if tuple(arg_shape) != tuple(sol_shape):
                return False
        return True

    def all_possible_solutions(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            yield (arg_shapes, return_shapes)

    def convert_to_ilp_problem(self, solver, problem_instance):
        arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        return shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        arg_shapes, return_shapes = ilp_problem
        sol_shape = return_shapes[0]
        for shape in arg_shapes:
            for i in range(len(shape)):
                solver += (shape[i] == sol_shape[i])

    def marshal_ilp_solution(self, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        return marshal_ilp_shapes(shape_vars, return_shapes)


class BroadcastRelation(Relation):
    """
    Used for most binary elementwise operators:
    Valid for exactly 2 arguments and 1 result.
    Suppose the two arguments s1 and s2 have ranks r1 and r2.
    Let m1 be max(r1, r2), m2 be min(r1, r2).
    Let l1 be the longer of the two arguments and l2r be the shorter
    The result should be of rank m1.
    result[0:m1-m2-1] == l1[0:m1-m2-1]

    For the remaining m2 indices:
      if l1[i] == 1, result[i] == l2[i]
      if l2[i] == 1, result[i] == l1[i]
      if l1[i] != 1 and l2[i] != 1, then l1[i] must equal l2[i] and result[i] == l1[i]

    See https://github.com/apache/tvm/blob/fc48514f1d8ccffcebd12007cb6c602506975703/src/relay/op/type_relations.cc#L67
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return isinstance(other, BroadcastRelation) and self.max_dim == other.max_dim

    def validate(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 1 and len(arg_ranks) != 2:
            return False
        sol_shape = return_shapes[0]
        rank0, rank1 = arg_ranks
        return len(sol_shape) == max(rank0, rank1)

    def check(self, _, solution):
        arg_shapes, return_shapes = solution
        a0 = arg_shapes[0]
        a1 = arg_shapes[1]
        sol_shape = return_shapes[0]
        min_rank = min(len(a0), len(a1))
        max_rank = max(len(a0), len(a1))
        diff = max_rank - min_rank
        for i in range(diff):
            bcast_shape = sol_shape[i]
            if len(a0) == max_rank  and bcast_shape != a0[i]:
                return False
            if len(a1) == max_rank and bcast_shape != a1[i]:
                return False
        for i in range(diff, len(sol_shape)):
            bcast_shape = sol_shape[i]
            min_idx = i - diff
            max_idx = i
            a0_elt = a0[min_idx] if len(a0) == min_rank else a0[max_idx]
            a1_elt = a1[min_idx] if len(a1) == min_rank else a1[max_idx]

            if a0_elt == 1 and bcast_shape != a1_elt:
                return False
            if a1_elt == 1 and bcast_shape != a0_elt:
                return False
            if a0_elt != 1 and a1_elt != 1:
                if a0_elt != a1_elt:
                    return False
                if bcast_shape != a0_elt:
                    return False
        return True

    def all_possible_solutions(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            yield (arg_shapes, return_shapes)

    def convert_to_ilp_problem(self, solver, problem_instance):
        arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        return shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        a0, a1 = shape_vars[0], shape_vars[1]
        sol_shape = return_shapes[0]

        min_rank = min(len(a0), len(a1))
        max_rank = max(len(a0), len(a1))
        diff = max_rank - min_rank
        for i in range(diff):
            bcast_shape = sol_shape[i]
            if len(a0) == max_rank:
                solver += (bcast_shape == a0[i])
            if len(a1) == max_rank:
                solver += (bcast_shape == a1[i])
        for i in range(diff, len(sol_shape)):
            bcast_shape = sol_shape[i]
            min_idx = i - diff
            max_idx = i
            a0_elt = a0[min_idx] if len(a0) == min_rank else a0[max_idx]
            a1_elt = a1[min_idx] if len(a1) == min_rank else a1[max_idx]

            M = self.max_dim + 1
            a0_elt_eq_1 = add_eq_constraint(solver, a0_elt, 1, M)
            a1_elt_eq_1 = add_eq_constraint(solver, a1_elt, 1, M)
            bc_eq_a1_elt = add_eq_constraint(solver, bcast_shape, a1_elt, M)
            bc_eq_a0_elt = add_eq_constraint(solver, bcast_shape, a0_elt, M)
            a1_elt_eq_a0_elt = add_eq_constraint(solver, a0_elt, a1_elt, M)

            # a0_elt == 1 -> bc == a1_elt
            # a1_elt == 1 -> bc == a0_elt
            # neither true -> equal each other and bc == a0_elt
            # implication: a -> b is equivalent to b >= a (if a is 1, be must be 1; if b is 1, a may be 0)
            solver += (bc_eq_a1_elt >= a0_elt_eq_1)
            solver += (bc_eq_a0_elt >= a1_elt_eq_1)
            neither_is_1 = add_and_constraint(solver, 1-a0_elt_eq_1, 1-a1_elt_eq_1)
            solver += (a1_elt_eq_a0_elt >= neither_is_1)
            solver += (bc_eq_a0_elt >= neither_is_1)

    def marshal_ilp_solution(self, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        return marshal_ilp_shapes(shape_vars, return_shapes)


class DenseRelation(Relation):
    """
    Type relation for dense (matrix multiplication):

    Two cases: Either there is a unit param set (the weight is ignored) or the weight is used
    If there is a unit param set and the data shape is (d0, ..., dn), then the weight shape must be (units, dn) and the output shape is (d0, ..., dn-1, units)
    If there is not a unit param set, then if data shape is (d0, ..., dn) and the weight shape is (w0, w1), the output shape is (d0, ..., dn-1, w0) where dn must equal w1

    See https://github.com/apache/tvm/blob/26733095f5a1e0887c32d644429d430bc1f51c91/src/relay/op/nn/nn.h#L40
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return isinstance(other, DenseRelation) and self.max_dim == other.max_dim

    def all_possible_solutions(self, problem_instance):
        units_defined, arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            if not units_defined:
                yield (None, arg_shapes, return_shapes)
            if units_defined:
                for i in range(1, self.max_dim+1):
                    yield (i, arg_shapes, return_shapes)

    def validate(self, problem_instance):
        _, arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 1 and len(arg_ranks) != 2:
            return False
        sol_shape = return_shapes[0]
        d_rank = arg_ranks[0]
        w_rank = arg_ranks[1]
        # the data cannot be a scalar, the weight must be of rank 2,
        # and the output must be of the same rank as the data
        if d_rank == 0:
            return False
        if len(sol_shape) != d_rank:
            return False
        return w_rank == 2

    def check(self, problem_instance, solution):
        units_defined, _, _ = problem_instance
        units, arg_shapes, return_shapes = solution
        data = arg_shapes[0]
        weight = arg_shapes[1]
        if units_defined and units is None:
            return False
        if not units_defined and units is not None:
            return False

        # The only condition that differs when unit is defined is that w0 must match the units,
        # since the output shape will match w0 in either case
        if units is not None:
            if weight[0] != units:
                return False
        sol_shape = return_shapes[0]
        if sol_shape[-1] != weight[0]:
            return False
        if weight[1] != data[-1]:
            return False
        for i in range(len(data) - 1):
            if sol_shape[i] != data[i]:
                return False
        return True

    def convert_to_ilp_problem(self, solver, problem_instance):
        units_defined, arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        unit_var = None
        if units_defined:
            unit_var = solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
        return unit_var, shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        unit_var, shape_vars, return_shapes = ilp_problem
        data_shape = shape_vars[0]
        weight_shape = shape_vars[1]
        sol_shape = return_shapes[0]

        if unit_var is not None:
            solver += (unit_var == weight_shape[0])
        solver += (sol_shape[-1] == weight_shape[0])
        solver += (weight_shape[1] == data_shape[-1])
        for i in range(len(data_shape) - 1):
            solver += (sol_shape[i] == data_shape[i])

    def marshal_ilp_solution(self, ilp_problem):
        units, shape_vars, return_shapes = ilp_problem
        arg_shapes, ret_shapes = marshal_ilp_shapes(shape_vars, return_shapes)
        if units is not None:
            units = int(units.x)
        return units, arg_shapes, ret_shapes


class BiasAddRelation(Relation):
    """
    Type relation for bias add. Just checks that the first arg matches the return type and the second arg is a vector of the appropriate axis

    See https://github.com/apache/tvm/blob/26733095f5a1e0887c32d644429d430bc1f51c91/src/relay/op/nn/nn.cc#L52
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return (isinstance(other, BiasAddRelation) and self.max_dim == other.max_dim)

    def compute_axis_idx(self, axis, rank):
        if axis < 0:
            return rank + axis
        return axis

    def validate(self, problem_instance):
        # TODO: solve for the axis and don't take it as a given
        axis, arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 1 and len(arg_ranks) != 2:
            return False
        sol_shape = return_shapes[0]
        d_rank = arg_ranks[0]
        w_rank = arg_ranks[1]
        axis_idx = self.compute_axis_idx(axis, d_rank)
        if axis_idx < 0 or axis_idx >= d_rank:
            return False
        if len(sol_shape) != d_rank:
            return False
        return w_rank == 1

    def all_possible_solutions(self, problem_instance):
        # TODO: enumerate over axes too
        axis, arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            yield (axis, arg_shapes, return_shapes)

    def check(self, _, solution):
        axis, arg_shapes, return_shapes = solution
        data = arg_shapes[0]
        weight = arg_shapes[1]
        sol_shape = return_shapes[0]
        axis_idx = self.compute_axis_idx(axis, len(sol_shape))

        for i in range(len(sol_shape)):
            if sol_shape[i] != data[i]:
                return False
        return weight[0] == sol_shape[axis_idx]

    def convert_to_ilp_problem(self, solver, problem_instance):
        axis, arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        return axis, shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        axis, shape_vars, return_shapes = ilp_problem
        data_shape = shape_vars[0]
        weight_shape = shape_vars[1]
        sol_shape = return_shapes[0]
        axis_idx = self.compute_axis_idx(axis, len(sol_shape))

        for i in range(len(sol_shape)):
            solver += (sol_shape[i] == data_shape[i])
        solver += (weight_shape[0] == data_shape[axis_idx])

    def marshal_ilp_solution(self, ilp_problem):
        axis, shape_vars, return_shapes = ilp_problem
        arg_shapes, ret_shapes = marshal_ilp_shapes(shape_vars, return_shapes)
        return (axis, arg_shapes, ret_shapes)


class BatchMatmulRelation(Relation):
    """
    Type relation for batch matmul

    See https://github.com/apache/tvm/blob/26733095f5a1e0887c32d644429d430bc1f51c91/src/relay/op/nn/nn.cc#L901
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return (isinstance(other, BatchMatmulRelation) and self.max_dim == other.max_dim)

    def all_possible_solutions(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            yield (arg_shapes, return_shapes)

    def validate(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 1 and len(arg_ranks) != 2:
            return False
        sol_shape = return_shapes[0]
        d_rank = arg_ranks[0]
        w_rank = arg_ranks[1]
        return len(sol_shape) == 3 and d_rank == 3 and w_rank == 3

    def check(self, _, solution):
        arg_shapes, return_shapes = solution
        x, y = arg_shapes[0], arg_shapes[1]
        sol_shape = return_shapes[0]
        return (sol_shape[0] == max(x[0], y[0])
                and sol_shape[1] == x[1]
                and sol_shape[2] == y[1]
                and x[2] == y[2]
                and (y[0] == 1 or x[0] == 1 or y[0] == x[0]))

    def convert_to_ilp_problem(self, solver, problem_instance):
        arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        return shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        x_shape = shape_vars[0]
        y_shape = shape_vars[1]
        sol_shape = return_shapes[0]

        M = self.max_dim + 1
        x0_eq_o0 = add_eq_constraint(solver, x_shape[0], sol_shape[0], M)
        y0_eq_o0 = add_eq_constraint(solver, y_shape[0], sol_shape[0], M)
        x0_eq_1 = add_eq_constraint(solver, x_shape[0], 1, M)
        y0_eq_1 = add_eq_constraint(solver, y_shape[0], 1, M)
        x0_eq_y0 = add_eq_constraint(solver, y_shape[0], x_shape[0], M)

        # output[0] = x[0] or y[0]
        solver += (x0_eq_o0 + y0_eq_o0 >= 1)
        # If the output = x[0], then y[0] = x[0] or 1
        solver += (y0_eq_1 + x0_eq_y0 >= x0_eq_o0)
        # If the output = y[0], then x[0] = y[0] or 1
        solver += (x0_eq_1 + x0_eq_y0 >= y0_eq_o0)

        solver += (sol_shape[1] == x_shape[1])
        solver += (sol_shape[2] == y_shape[1])
        solver += (x_shape[2] == y_shape[2])

    def marshal_ilp_solution(self, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        return marshal_ilp_shapes(shape_vars, return_shapes)


class BatchNormRelation(Relation):
    """
    Type relation for batch norm

    See https://github.com/apache/tvm/blob/26733095f5a1e0887c32d644429d430bc1f51c91/src/relay/op/nn/nn.cc#L633
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return isinstance(other, BatchNormRelation) and self.max_dim == other.max_dim

    def all_possible_solutions(self, problem_instance):
        # TODO: enumerate over axes too
        axis, arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            yield (axis, arg_shapes, return_shapes)

    def get_axis_dim(self, axis, return_data):
        normed_rank = len(return_data)
        axis_idx = axis if axis >= 0 else normed_rank - 1
        return return_data[axis_idx]

    def is_axis_vector(self, axis_dim, target):
        return len(target) == 1 and target[0] == axis_dim

    def validate(self, problem_instance):
        # TODO: eventually we'll want to solve for the axis, not just pick one
        axis, arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 3 and len(arg_ranks) != 5:
            return False

        normed_rank = len(return_shapes[0])
        axis_dim = self.get_axis_dim(axis, return_shapes[0])

        running_mean = return_shapes[1]
        if not self.is_axis_vector(axis_dim, running_mean):
            return False
        running_var = return_shapes[2]
        if not self.is_axis_vector(axis_dim, running_var):
            return False

        input_rank = arg_ranks[0]
        if input_rank != normed_rank:
            return False
        for arg_rank in arg_ranks[1:]:
            if arg_rank != 1:
                return False
        return True

    def check(self, _, solution):
        axis, arg_shapes, return_shapes = solution
        normed_data = return_shapes[0]
        axis_dim = self.get_axis_dim(axis, normed_data)

        for i in range(len(normed_data)):
            if arg_shapes[0][i] != normed_data[i]:
                return False
        for arg_shape in arg_shapes[1:]:
            if not self.is_axis_vector(axis_dim, arg_shape):
                return False
        return True

    def convert_to_ilp_problem(self, solver, problem_instance):
        axis, arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        return axis, shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        axis, shape_vars, return_shapes = ilp_problem
        normed_data = return_shapes[0]
        axis_dim = self.get_axis_dim(axis, normed_data)

        input_data = shape_vars[0]
        input_vecs = shape_vars[1:]
        for i, d in enumerate(input_data):
            solver += (d == normed_data[i])
        for vec in input_vecs:
            solver += (vec[0] == axis_dim)

    def marshal_ilp_solution(self, ilp_problem):
        axis, shape_vars, return_shapes = ilp_problem
        arg_shapes, ret_shapes = marshal_ilp_shapes(shape_vars, return_shapes)
        return axis, arg_shapes, ret_shapes


class Conv2DRelation(Relation):
    """
    Type relation for Conv2D, which is very complicated

    See https://github.com/apache/tvm/blob/a1d43c15ac6382831370c6de141bf80888761e70/src/relay/op/nn/convolution.h#L133
    but it's really complicated so for now we'll use PyTorch's description:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d
    and the informal description:
    https://tvm.apache.org/docs/api/python/relay/nn.html#tvm.relay.nn.conv2d
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return (isinstance(other, Conv2DRelation) and self.max_dim == other.max_dim)

    def validate(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        if len(return_shapes) != 1 and len(arg_ranks) != 2:
            return False

        # all ranks must be 4
        return (len(return_shapes[0]) == 4 and arg_ranks[0] == 4 and arg_ranks[1] == 4)

    def all_possible_solutions(self, problem_instance):
        arg_ranks, return_shapes = problem_instance
        for arg_shapes in enumerate_all_possible_shapes(arg_ranks, self.max_dim):
            yield (arg_shapes, return_shapes)

    def check(self, _, solution):
        # taking a very conservative approach for now,
        # assuming that data layout is (batch_size, in_channels, H_in, W_in),
        # weight layout is (out_channels, in_channels, kernel_size[0], kernel_size[1])
        # and output layout is (batch_size, out_channels, H_out, W_in)
        # where H_out = floor(((H_in + 2*padding - dilation*(kernel_size[0]-1)-1)/stride) + 1)
        # and W_out = floor(((W_in + 2*padding - dilation*(kernel_size[1]-1)-1) / stride) + 1)
        arg_shapes, return_shapes = solution

        # fixing stride, dilation, and padding to default values for now
        # (TODO: search over these too)
        # formula from PT (doesn't seem to work)
        # math.floor((in_dim + 2*padding - dilation * (kernel_size - 1))//stride + 1)
        def compute_out_dim(in_dim, padding, dilation, kernel_size, stride):
            # hack for now based on defaults
            return in_dim - (kernel_size - 1)

        stride = (1, 1)
        dilation = (1, 1)
        padding = (0, 0)

        N_d, C_in_d, H_in, W_in = arg_shapes[0]
        C_out_w, C_in_w, k_h, k_w = arg_shapes[1]
        N_o, C_out_o, H_out, W_out = return_shapes[0]

        # agreement between values that should match exactly
        if N_o != N_d or C_out_w != C_out_o or C_in_d != C_in_w:
            return False

        expected_h_out = compute_out_dim(H_in, padding[0], dilation[0], k_h, stride[0])
        expected_w_out = compute_out_dim(W_in, padding[1], dilation[1], k_w, stride[1])
        return H_out == expected_h_out and W_out == expected_w_out

    def convert_to_ilp_problem(self, solver, problem_instance):
        arg_ranks, return_shapes = problem_instance
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        return shape_vars, return_shapes

    def produce_ilp_constraints(self, solver, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        M = self.max_dim + 1

        N_o, C_out_o, H_out, W_out = return_shapes[0]
        N_d, C_in_d, H_in, W_in = shape_vars[0]
        C_out_w, C_in_w, k_h, k_w = shape_vars[1]

        solver += (N_o == N_d)
        solver += (C_out_o == C_out_w)
        solver += (C_in_w == C_in_d)

        # TODO: will need to use linear values if we _don't_ fix stride, dilation, etc
        # to default values
        expected_h_out = H_in - (k_h - 1)
        expected_w_out = W_in - (k_w - 1)
        solver += (H_out == expected_h_out)
        solver += (W_out == expected_w_out)
        return shape_vars

    def marshal_ilp_solution(self, ilp_problem):
        shape_vars, return_shapes = ilp_problem
        return marshal_ilp_shapes(shape_vars, return_shapes)


# TODO: Add more
