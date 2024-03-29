import random
from relation_solver import (BruteForceSolver, ILPSolver, MemoizedSolver,
                             IdentityRelation, BroadcastRelation,
                             DenseRelation, BiasAddRelation,
                             BatchMatmulRelation, BatchNormRelation,
                             Conv2DRelation)

MAX_RANK = 3
MAX_DIM = 4
NUM_ATTEMPTS = 5

def generate_return_shape(min_rank=0):
    rank = random.randint(min_rank, MAX_RANK)
    # we'll insert holes to make sure they work
    return tuple([
        (random.randint(1, MAX_DIM) if random.choice([True, False]) else None)
        for i in range(rank)
    ])


def generate_identity_arg_ranks(max_args, ret_rank):
    num_args = random.randint(0, max_args)
    return tuple([ret_rank for i in range(num_args)])


def generate_broadcast_arg_ranks(target_rank):
    other_rank = random.randint(0, target_rank)
    ranks = [other_rank, target_rank]
    random.shuffle(ranks)
    return tuple(ranks)


def generate_dense_arg_ranks(target_rank, units_defined):
    ret = [target_rank, 2]
    if units_defined:
        ret.append(1)
    return tuple(ret)


def generate_bias_add_arg_ranks(target_rank):
    return tuple([target_rank, 1])


def solve_and_check(solver, problem_instance, relation):
    assert relation.validate(problem_instance)
    ret = solver.solve(relation, problem_instance)
    assert relation.check(problem_instance, ret), (ret, problem_instance)
    return ret


def all_solvers():
    bf = BruteForceSolver(MAX_DIM)
    ilp = ILPSolver(MAX_DIM, 30, False)
    memo_bf = MemoizedSolver(bf)
    memo_ilp = MemoizedSolver(ilp)
    return [bf, ilp, memo_bf, memo_ilp]


def check_all(solvers, problem_instance, relation):
    for solver in solvers:
        solution = solve_and_check(solver, problem_instance, relation)


def check_samples(rel, params):
    for i in range(NUM_ATTEMPTS):
        problem_instance, sample = rel.sample_solution(params)
        assert rel.check(problem_instance, sample), sample


def solve_with_bf(problem_instance, relation):
    return solve_and_check(BruteForceSolver(MAX_DIM),
                           problem_instance, relation)


def solve_with_ilp(problem_instance, relation):
    return solve_and_check(ILPSolver(MAX_DIM, 30, False),
                           problem_instance, relation)


def test_identity_scalars():
    # all scalars is okay
    max_args = 3
    ret_shapes = ((),)

    id_rel = IdentityRelation(MAX_DIM)
    ranks = [0 for i in range(random.randint(0, max_args))]
    bf_soln, _ = solve_with_bf((ranks, ret_shapes), id_rel)
    ilp_soln, _ = solve_with_ilp((ranks, ret_shapes), id_rel)

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
    bf_soln, _ = solve_with_bf((ranks, ret_shapes), id_rel)
    ilp_soln, _ = solve_with_ilp((ranks, ret_shapes), id_rel)

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
        check_all(solvers, (ranks, (ret_shape,)), id_rel)
    check_samples(id_rel, (MAX_RANK, max_args))


def test_bcast_rel_fuzz():
    bcast_rel = BroadcastRelation(MAX_DIM)
    solvers = all_solvers()
    for i in range(NUM_ATTEMPTS):
        ret_shape = generate_return_shape()
        ranks = generate_broadcast_arg_ranks(len(ret_shape))
        check_all(solvers, (ranks, (ret_shape,)), bcast_rel)
    check_samples(bcast_rel, MAX_RANK)


def test_bcast_examples():
    # same example as in the TVM tests
    solvers = all_solvers()
    bcast_rel = BroadcastRelation(MAX_DIM)
    for i in range(MAX_DIM + 1):
        s1 = (MAX_DIM, i, MAX_DIM)
        s2 = (i, 1)
        expected = (MAX_DIM, i, MAX_DIM)

        problem = ((3, 2), (expected))
        soln = ((s1, s2), (expected,))
        assert bcast_rel.check(problem, soln)


def test_dense_rel_fuzz():
    solvers = all_solvers()
    dense_rel = DenseRelation(MAX_DIM)
    for units_defined in (True, False):
        for i in range(NUM_ATTEMPTS):
            ret_shape = generate_return_shape(min_rank=1)
            ranks = generate_dense_arg_ranks(len(ret_shape), units_defined)
            check_all(solvers, (units_defined, ranks, (ret_shape,)), dense_rel)
    check_samples(dense_rel, MAX_RANK)


def test_bias_add_fuzz():
    solvers = all_solvers()
    rel = BiasAddRelation(MAX_DIM)
    for i in range(NUM_ATTEMPTS):
        ret_shape = generate_return_shape(min_rank=1)
        ret_rank = len(ret_shape)
        for j in range(NUM_ATTEMPTS):
            axis = random.randint(-(ret_rank-1), ret_rank-1)
            ranks = generate_bias_add_arg_ranks(ret_rank)
            check_all(solvers, (axis, ranks, (ret_shape,)), rel)
    check_samples(rel, MAX_RANK)


def test_batch_matmul_fuzz():
    # all ranks are fixed to 3
    arg_ranks = (3, 3)
    solvers = all_solvers()
    rel = BatchMatmulRelation(MAX_DIM)
    for i in range(NUM_ATTEMPTS):
        ret_shape = tuple([random.randint(1, MAX_DIM) for i in range(3)])
        check_all(solvers, (arg_ranks, (ret_shape,)), rel)
    check_samples(rel, None)


def test_batch_norm_fuzz():
    solvers = all_solvers()
    for i in range(NUM_ATTEMPTS):
        ret_shape = generate_return_shape(min_rank=1)
        ret_rank = len(ret_shape)
        for j in range(NUM_ATTEMPTS):
            axis = random.randint(-1, ret_rank-1)
            rel = BatchNormRelation(MAX_DIM)
            axis_dim = ret_shape[axis]
            ranks = (ret_rank, 1, 1, 1, 1)
            check_all(solvers, (axis, ranks, (ret_shape, (axis_dim,), (axis_dim,))), rel)
    check_samples(rel, MAX_RANK)


def test_conv2d_fuzz():
    # all ranks are fixed to 4
    arg_ranks = (4, 4)
    solvers = all_solvers()
    rel = Conv2DRelation(MAX_DIM)
    for i in range(NUM_ATTEMPTS):
        ret_shape = tuple([random.randint(1, MAX_DIM) for i in range(4)])
        check_all(solvers, (arg_ranks, (ret_shape,)), rel)
    check_samples(rel, None)


if __name__ == "__main__":
    test_identity_scalars()
    test_bcast_scalars()
    test_identity_rel_fuzz()
    test_bcast_rel_fuzz()
    test_bcast_examples()
    test_dense_rel_fuzz()
    test_bias_add_fuzz()
    test_batch_matmul_fuzz()
    test_batch_norm_fuzz()
    test_conv2d_fuzz()
