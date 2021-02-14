import tvm
from tvm import relay

from scope import VarScope

def check_var(var, expected_type, name=None):
    assert isinstance(var, relay.Var)
    if name is not None:
        assert var.name_hint == name
    assert var.type_annotation == expected_type


def check_scope(var_scope, expected_vars):
    # avoid imposing order assumptions
    scope_set = set(var_scope.get_local_scope())
    for v in expected_vars:
        if v not in scope_set:
            return False
    return True


def test_global_var_create():
    vs = VarScope("test_var")
    gv = vs.new_global_var(None)
    assert isinstance(gv, relay.GlobalVar)
    assert gv.name_hint == "global_test_var_0"
    assert vs.get_local_scope() == []
    global_scope = vs.get_global_scope()
    assert gv in global_scope and global_scope[gv] is None


def test_local_var_create_out_of_scope():
    vs = VarScope("test_var")
    v = vs.new_local_var(None, add_to_scope=False)
    check_var(v, None, name="test_var_0")
    assert vs.get_local_scope() == []

    v2 = vs.new_local_var(None, add_to_scope=False)
    check_var(v2, None, name="test_var_1")
    assert vs.get_local_scope() == []


def test_local_var_in_scope():
    vs = VarScope("test_var")
    v = vs.new_local_var(None, add_to_scope=True)
    check_var(v, None, name="test_var_0")
    assert vs.get_local_scope() == [v]

    v2 = vs.new_local_var(None, add_to_scope=True)
    check_var(v2, None, name="test_var_1")
    assert vs.get_local_scope() == [v, v2]


def test_add_fresh_to_scope():
    vs = VarScope("test_var")
    v = vs.new_local_var(None, add_to_scope=False)
    check_var(v, None, name="test_var_0")
    assert vs.get_local_scope() == []
    vs.add_to_scope(v)
    assert vs.get_local_scope() == [v]


def test_pushing_scope():
    vs = VarScope("test_var")
    with vs.new_scope():
        v = vs.new_local_var(None, add_to_scope=True)
        assert vs.get_local_scope() == [v]
    assert vs.get_local_scope() == []


def test_scope_pushes_and_pops():
    vs = VarScope("test_var")
    with vs.new_scope():
        v = vs.new_local_var(None, add_to_scope=True)
        assert vs.get_local_scope() == [v]
        with vs.new_scope():
            x = vs.new_local_var(None, add_to_scope=True)
            assert x != v
            assert check_scope(vs, [v, x])
        assert vs.get_local_scope() == [v]
        with vs.new_scope():
            y = vs.new_local_var(None, add_to_scope=True)
            assert y != v
            assert check_scope(vs, [v, y])
            with vs.new_scope():
                z = vs.new_local_var(None, add_to_scope=True)
                assert z != y
                assert check_scope(vs, [v, y, z])
            assert check_scope(vs, [v, y])
        assert vs.get_local_scope() == [v]
    assert vs.get_local_scope() == []


if __name__ == "__main__":
    test_global_var_create()
    test_add_fresh_to_scope()
    test_local_var_create_out_of_scope()
    test_local_var_in_scope()
    test_add_fresh_to_scope()
    test_pushing_scope()
    test_scope_pushes_and_pops()
