"""
Hide away all the ILP solving so the rest of the code
doesn't have to worry about it
"""
import itertools

import mip
import random
from mip import Model, BINARY, INTEGER

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


def branch_constraints(solver, condition, true_branch, false_branch):
    """
    Given a boolean condition variable and boolean variables true_branch and false_branch,
    sets up constraints so that true_branch is true iff condition is true
    and false_branch is true iff condition is false
    """
    solver += (condition == true_branch)
    solver += ((1 - condition) == false_branch)


class Relation:
    """
    Base class for specifying a type relation, meant to be
    compatible with different solving strategies
    """
    def validate(self, arg_ranks, return_shapes):
        """
        Check if the ranks are possible given the return shapes
        """
        raise NotImplementedError()

    def check(self, arg_shapes, return_shapes):
        """
        Checks the relation for a concrete shape (meant for brute force)
        """
        raise NotImplementedError()

    def produce_ilp_constraints(self, solver, arg_ranks, return_shapes):
        """
        As the name implies, adds appropriate ILP constraints using the solver.
        Returns ILP variables corresponding to shape dimensions,
        grouped by shape
        """
        raise NotImplementedError()


class Solver:
    """
    Base class for filling in arg types -- takes tensor ranks
    (number of dimensions)
    and return shapes and returns the argument shapes
    """
    def solve(self, arg_ranks, return_shapes, relation):
        raise NotImplementedError()


class BruteForceSolver(Solver):
    """
    Can work out if the range is small
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

    def solve(self, arg_ranks, return_shapes, relation):
        if not relation.validate(arg_ranks, return_shapes):
            raise ValueError("Relation invalid for the given ranks and return shapes")

        all_shapes = []
        for rank in arg_ranks:
            if rank > 0:
                # Yes this grows insanely quickly.
                # Combinations with replacement is lexicographically ordered
                # so we have to take permutations afterwards.
                possible_shapes = []
                for combo in itertools.combinations_with_replacement(list(range(1, self.max_dim+1)), rank):
                    possible_shapes += itertools.permutations(combo, len(combo))
                all_shapes.append(possible_shapes)
                continue
            # if we have a scalar, we must indicate it is an empty element or itertools omits it
            all_shapes.append(([],))

        for arg_shapes in itertools.product(*all_shapes):
            if relation.check(arg_shapes, return_shapes):
                return arg_shapes
        raise ValueError(f"No solution found (infeasible): {arg_ranks} {return_shapes}")


class ILPSolver(Solver):
    def __init__(self, max_dim, max_time, solver_verbose):
        self.max_dim = max_dim
        self.max_time = max_time
        m = Model()
        m.emphasis = mip.SearchEmphasis.FEASIBILITY
        m.verbose = int(solver_verbose)
        self.m = m

    def found_ilp_solution(self, solver_result):
        return (solver_result == mip.OptimizationStatus.OPTIMAL
                or solver_result == mip.OptimizationStatus.FEASIBLE)

    def pack_vars(self, shape_vars):
        """
        Takes a list of list of ILP vars and returns their solver values
        """
        return [
            [int(v.x) for v in shape]
            for shape in shape_vars
        ]

    def solve(self, arg_ranks, return_shapes, relation):
        if not relation.validate(arg_ranks, return_shapes):
            raise ValueError("Relation invalid for the given ranks and return shapes")

        shape_vars = relation.produce_ilp_constraints(self.m, arg_ranks, return_shapes)

        # the ILP solver considers 0 variables to be an error
        # rather than a trivial solution
        no_vars = True
        for shape in shape_vars:
            if len(shape) != 0:
                no_vars = False
                break
        if no_vars:
            return [[] for shape in shape_vars]

        res = self.m.optimize(max_seconds=self.max_time)
        if not self.found_ilp_solution(res):
            raise ValueError("No solution found (infeasible)")
        ret = self.pack_vars(shape_vars)
        assert len(ret) == len(arg_ranks)
        for final_shape, rank in zip(ret, arg_ranks):
            assert len(final_shape) == rank
        return ret


class MemoizedSolver(Solver):
    """
    Memoizes an underlying solver
    """
    def __init__(self, solver):
        self.solver = solver
        self.memo = {}

    def solve(self, arg_ranks, return_shapes, relation):
        if not relation.validate(arg_ranks, return_shapes):
            raise ValueError("Relation invalid for the given ranks and return shapes")

        if relation not in self.memo:
            self.memo[relation] = {}

        # lists are not hashable
        def tuplify(l):
            if not isinstance(l, (list, tuple)):
                return l
            return tuple([
                item if not isinstance(item, (list, tuple)) else tuplify(item)
                for item in l
            ])

        query = (tuplify(arg_ranks), tuplify(return_shapes))
        if query in self.memo[relation]:
            return self.memo[relation][query]
        solution = self.solver.solve(arg_ranks, return_shapes, relation)
        self.memo[relation][query] = solution
        return solution

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

    def validate(self, arg_ranks, return_shapes):
        if len(return_shapes) != 1:
            return False
        sol_shape = return_shapes[0]
        for rank in arg_ranks:
            if rank != len(sol_shape):
                return False
        return True

    def check(self, arg_shapes, return_shapes):
        sol_shape = return_shapes[0]
        for arg_shape in arg_shapes:
            if tuple(arg_shape) != tuple(sol_shape):
                return False
        return True

    def produce_ilp_constraints(self, solver, arg_ranks, return_shapes):
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        sol_shape = return_shapes[0]
        for shape in shape_vars:
            for i in range(len(shape)):
                solver += (shape[i] == sol_shape[i])
        return shape_vars


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

    def validate(self, arg_ranks, return_shapes):
        if len(return_shapes) != 1 and len(arg_ranks) != 2:
            return False
        sol_shape = return_shapes[0]
        rank0, rank1 = arg_ranks
        return len(sol_shape) == max(rank0, rank1)

    def check(self, arg_shapes, return_shapes):
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

    def produce_ilp_constraints(self, solver, arg_ranks, return_shapes):
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        a0 = shape_vars[0]
        a1 = shape_vars[1]
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
        return shape_vars


class DenseRelation(Relation):
    """
    Type relation for dense (matrix multiplication):

    Two cases: Either there is a unit param set (the weight is ignored) or the weight is used
    If there is a unit param set and the data shape is (d0, ..., dn), then the weight shape must be (units, dn) and the output shape is (d0, ..., dn-1, units)
    If there is not a unit param set, then if data shape is (d0, ..., dn) and the weight shape is (w0, w1), the output shape is (d0, ..., dn-1, w0) where dn must equal w1

    See https://github.com/apache/tvm/blob/26733095f5a1e0887c32d644429d430bc1f51c91/src/relay/op/nn/nn.h#L40
    """
    def __init__(self, max_dim, units_defined):
        self.max_dim = max_dim
        self.units_defined = units_defined

    # overidding hash for the benefit of the memoizer
    def __hash__(self):
        return hash(self.max_dim)

    def __eq__(self, other):
        return isinstance(other, DenseRelation) and self.max_dim == other.max_dim and self.units_defined == other.units_defined

    def validate(self, arg_ranks, return_shapes):
        # we will treat the unit param as a third scalar
        expected_args = 3 if self.units_defined else 2
        if len(return_shapes) != 1 and len(arg_ranks) != expected_args:
            return False
        sol_shape = return_shapes[0]
        d_rank = arg_ranks[0]
        w_rank = arg_ranks[1]
        if self.units_defined:
            unit = arg_ranks[2]
            if unit != 1:
                return False
        # the data cannot be a scalar, the weight must be of rank 2,
        # and the output must be of the same rank as the data
        if d_rank == 0:
            return False
        if len(sol_shape) != d_rank:
            return False
        return w_rank == 2

    def check(self, arg_shapes, return_shapes):
        data = arg_shapes[0]
        weight = arg_shapes[1]
        # The only condition that differs when unit is defined is that w0 must match the units,
        # since the output shape will match w0 in either case
        if self.units_defined:
            unit = arg_shapes[2][0]
            if weight[0] != unit:
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

    def produce_ilp_constraints(self, solver, arg_ranks, return_shapes):
        shape_vars = [
            [solver.add_var(var_type=INTEGER, lb=1, ub=self.max_dim)
             for i in range(rank)]
            for rank in arg_ranks
        ]
        all_vars = shape_vars
        data_shape = shape_vars[0]
        weight_shape = shape_vars[1]
        sol_shape = return_shapes[0]

        if self.units_defined:
            unit_var = shape_vars[2][0]
            solver += (unit_var == weight_shape[0])
        solver += (sol_shape[-1] == weight_shape[0])
        solver += (weight_shape[1] == data_shape[-1])
        for i in range(len(data_shape) - 1):
            solver += (sol_shape[i] == data_shape[i])
        return shape_vars


# bias add relation
# batch matmul relation
# batch norm relation
# conv2d relation

# TODO: Add more
