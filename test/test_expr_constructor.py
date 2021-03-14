import time

import tvm
from tvm import relay

from shared_test_generators import TestTypeGenerator, TestExprGenerator
from expr_count import count_exprs
from miniprelude import MiniPrelude

NUM_ATTEMPTS = 100

def check_well_formed(prelude, expr, ty):
    assert relay.analysis.well_formed(expr)
    # even more of a test is that we should be able to insert into the module without problems
    mod = prelude.mod
    # if we wrap up the variable as a function, it should type check successfully
    try:
        mod["main"] = relay.Function([], expr, ret_type=ty)
        mod = relay.transform.InferType()(mod)
        full_text = mod["main"].astext(show_meta_data=True)
        checked_type = mod.get_global_var("main").checked_type.ret_type
        assert checked_type == ty, f"{full_text}\n{ty}"
    except Exception as e:
        assert False, f"{expr}\n{ty}\n{e}"


def generate_type(prelude):
    return TestTypeGenerator(prelude).generate_type()


def generate_expr(prelude, ty, seed):
    gen = TestExprGenerator(prelude)
    gen.set_seed(seed)
    ret = gen.generate_expr(ty)
    print(count_exprs(ret))
    if gen.get_solver_profile():
        print(gen.get_solver_profile())
    return ret

def generate_with_forward_solver(prelude, seed, conf=None):
    gen = TestExprGenerator(prelude, conf=conf)
    gen.set_seed(seed)
    ty = gen.forward_solve()
    expr = gen.generate_expr(ty)
    print(count_exprs(expr))
    if gen.get_solver_profile():
        print(gen.get_solver_profile())
    check_well_formed(prelude, expr, ty)

# we should try finer-grained tests than only this
def test_fuzz():
    prelude = MiniPrelude()

    for i in range(NUM_ATTEMPTS):
        start = time.time()
        ty = generate_type(prelude)
        expr = generate_expr(prelude, ty, i)
        check_well_formed(prelude, expr, ty)
        end = time.time()
        print(f"Iter time: {end - start}")

def test_fuzz_forward_solve():
    # see if forward solving first gets us more operators
    prelude = MiniPrelude()
    for i in range(NUM_ATTEMPTS):
        start = time.time()
        generate_with_forward_solver(prelude, i)
        end = time.time()
        print(f"Iter time: {end - start}")


def test_fuzz_forward_sampling():
    prelude = MiniPrelude()
    conf = {"use_forward_sampling": True}
    for i in range(NUM_ATTEMPTS):
        start = time.time()
        generate_with_forward_solver(prelude, i, conf=conf)
        end = time.time()
        print(f"Iter time: {end - start}")


def test_no_solver():
    prelude = MiniPrelude()
    conf = {
        "use_backward_solving": False,
        "use_forward_sampling": True
    }
    for i in range(NUM_ATTEMPTS):
        start = time.time()
        generate_with_forward_solver(prelude, i, conf=conf)
        end = time.time()
        print(f"Iter time: {end - start}")


if __name__ == "__main__":
    test_fuzz()
    test_fuzz_forward_solve()
    test_fuzz_forward_sampling()
    test_no_solver()
