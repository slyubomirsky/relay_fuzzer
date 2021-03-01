import random
from relation_solver import (BruteForceSolver, ILPSolver, MemoizedSolver,
                             IdentityRelation, BroadcastRelation,
                             DenseRelation, BiasAddRelation)

MAX_RANK = 3
MAX_DIM = 4
NUM_ATTEMPTS = 5

def generate_return_shape(min_rank=0):
    rank = random.randint(min_rank, MAX_RANK)
    return [random.randint(1, MAX_DIM) for i in range(rank)]


def generate_identity_arg_ranks(max_args, ret_rank):
    num_args = random.randint(0, max_args)
    return [ret_rank for i in range(num_args)]


def generate_broadcast_arg_ranks(target_rank):
    other_rank = random.randint(0, target_rank)
    ranks = [other_rank, target_rank]
    random.shuffle(ranks)
    return ranks


def generate_dense_arg_ranks(target_rank, units_defined):
    ret = [target_rank, 2]
    if units_defined:
        ret.append(1)
    return ret


def generate_bias_add_arg_ranks(target_rank):
    return [target_rank, 1]


def solve_and_check(solver, ranks, ret_shapes, relation):
    ret = solver.solve(ranks, ret_shapes, relation)
    assert relation.check(ret, ret_shapes), (ret, ranks, ret_shapes)
    return ret


def all_solvers():
    bf = BruteForceSolver(MAX_DIM)
    ilp = ILPSolver(MAX_DIM, 30, False)
    memo_bf = MemoizedSolver(bf)
    memo_ilp = MemoizedSolver(ilp)
    return [bf, ilp, memo_bf, memo_ilp]


def check_all(solvers, ranks, ret_shapes, relation):
    for solver in solvers:
        ret = solver.solve(ranks, ret_shapes, relation)
        assert relation.check(ret, ret_shapes), (ret, ranks, ret_shapes, solver)


def solve_with_bf(ranks, ret_shapes, relation):
    return solve_and_check(BruteForceSolver(MAX_DIM),
                           ranks, ret_shapes, relation)


def solve_with_ilp(ranks, ret_shapes, relation):
    return solve_and_check(ILPSolver(MAX_DIM, 30, False),
                           ranks, ret_shapes, relation)


def test_identity_scalars():
    # all scalars is okay
    max_args = 3
    ret_shapes = [[]]

    id_rel = IdentityRelation(MAX_DIM)
    ranks = [0 for i in range(random.randint(0, max_args))]
    bf_soln = solve_with_bf(ranks, ret_shapes, id_rel)
    ilp_soln = solve_with_ilp(ranks, ret_shapes, id_rel)

    for s in bf_soln:
        assert len(s) == 0
    for s in ilp_soln:
        assert len(s) == 0


def test_bcast_scalars():
    # all scalars should work too
    max_rank = 0
    ret_shapes = [[]]

    id_rel = BroadcastRelation(MAX_DIM)
    ranks = [0, 0]
    bf_soln = solve_with_bf(ranks, ret_shapes, id_rel)
    ilp_soln = solve_with_ilp(ranks, ret_shapes, id_rel)

    for s in bf_soln:
        assert len(s) == 0
    for s in ilp_soln:
        assert len(s) == 0


def test_identity_rel_fuzz():
    max_args = 3
    id_rel = IdentityRelation(MAX_DIM)
    solvers = all_solvers()
    for i in range(NUM_ATTEMPTS):
        ret_shape = generate_return_shape()
        ranks = generate_identity_arg_ranks(max_args, len(ret_shape))
        check_all(solvers, ranks, [ret_shape], id_rel)


def test_bcast_rel_fuzz():
    bcast_rel = BroadcastRelation(MAX_DIM)
    solvers = all_solvers()
    for i in range(NUM_ATTEMPTS):
        ret_shape = generate_return_shape()
        ranks = generate_broadcast_arg_ranks(len(ret_shape))
        check_all(solvers, ranks, [ret_shape], bcast_rel)


def test_bcast_examples():
    # same example as in the TVM tests
    solvers = all_solvers()
    bcast_rel = BroadcastRelation(MAX_DIM)
    for i in range(MAX_DIM + 1):
        s1 = (MAX_DIM, i, MAX_DIM)
        s2 = (i, 1)
        expected = (MAX_DIM, i, MAX_DIM)
        assert bcast_rel.check((s1, s2), (expected,))


def test_dense_rel_fuzz():
    solvers = all_solvers()
    for units_defined in (True, False):
        dense_rel = DenseRelation(MAX_DIM, units_defined)
        for i in range(NUM_ATTEMPTS):
            ret_shape = generate_return_shape(min_rank=1)
            ranks = generate_dense_arg_ranks(len(ret_shape), units_defined)
            check_all(solvers, ranks, [ret_shape], dense_rel)


def test_bias_add_fuzz():
    solvers = all_solvers()
    for i in range(NUM_ATTEMPTS):
        ret_shape = generate_return_shape(min_rank=1)
        ret_rank = len(ret_shape)
        for j in range(NUM_ATTEMPTS):
            axis = random.randint(-(ret_rank-1), ret_rank-1)
            rel = BiasAddRelation(MAX_DIM, axis)
            ranks = generate_bias_add_arg_ranks(ret_rank)
            check_all(solvers, ranks, [ret_shape], rel)


if __name__ == "__main__":
    test_identity_scalars()
    test_bcast_scalars()
    test_identity_rel_fuzz()
    test_bcast_rel_fuzz()
    test_bcast_examples()
    test_dense_rel_fuzz()
    test_bias_add_fuzz()
