import tvm
from tvm import relay

from shared_test_generators import TestTypeGenerator, TestExprGenerator

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
    if gen.get_solver_profile():
        print(gen.get_solver_profile())
    return ret


# we should try finer-grained tests than only this
def test_fuzz():
    import time

    for i in range(NUM_ATTEMPTS):
        prelude = relay.prelude.Prelude()
        start = time.time()
        ty = generate_type(prelude)
        expr = generate_expr(prelude, ty, i)
        check_well_formed(prelude, expr, ty)
        end = time.time()
        print(f"Iter time: {end - start}")


if __name__ == "__main__":
    test_fuzz()
