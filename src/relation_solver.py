"""
Hide away all the ILP solving so the rest of the code
doesn't have to worry about it
"""
import itertools

import mip
import random
from mip import Model, BINARY, INTEGER

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
    The two arguments can have different ranks but should match each other (and the result)
    up to the length of the shorter rank.
    The longer argument must match the result.
    """
    def __init__(self, max_dim):
        self.max_dim = max_dim

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
        for i in range(len(sol_shape)):
            bcast_shape = sol_shape[i]
            if i < len(a0) and bcast_shape != a0[i]:
                return False
            if i < len(a1) and bcast_shape != a1[i]:
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
        for i in range(len(sol_shape)):
            if i < len(a0):
                solver += (a0[i] == sol_shape[i])
            if i < len(a1):
                solver += (a1[i] == sol_shape[i])
        return shape_vars

# TODO: Add more
