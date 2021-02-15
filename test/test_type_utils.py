import tvm
from tvm import relay

import type_utils

def check_instantiate(start, mapping, expected):
    new_type = type_utils.instantiate(start, mapping)
    return tvm.ir.structural_equal(new_type, expected)

def check_mapping(mapping, expected):
    hash_mapping = {
        tvm.ir.structural_hash(tv): ty
        for tv, ty in mapping
    }
    for tv, ty in expected:
        if not tvm.ir.structural_equal(ty, hash_mapping[tvm.ir.structural_hash(tv)]):
            return False
    return True

def test_instantiate_nothing():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    tt = relay.TupleType([a, b])
    assert check_instantiate(tt, [], tt)


def test_partial_instantiate():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    tt = relay.TupleType([a, b])
    expected_type = relay.TupleType([a, relay.TupleType([])])
    assert check_instantiate(tt, [(b, relay.TupleType([]))], expected_type)


def test_instantiate_ref():
    a = relay.TypeVar("a")
    rt = relay.RefType(a)
    replacement = [(a, relay.scalar_type("bool"))]
    expected = relay.RefType(relay.scalar_type("bool"))
    assert check_instantiate(rt, replacement, expected)


def test_instantiate_tuple_full():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    tt = relay.TupleType([a, b])
    mapping = [
        (a, relay.scalar_type("bool")),
        (b, relay.TupleType([]))
    ]
    expected_type = relay.TupleType([relay.scalar_type("bool"),
                                     relay.TupleType([])])
    assert check_instantiate(tt, mapping, expected_type)


def test_instantiate_func():
    a, b, c, d = [relay.TypeVar(l) for l in ("a", "b", "c", "d")]
    ft = relay.FuncType([a, b, c], d)
    tt = relay.TensorType((10, 20), "float32")
    mapping = [
        (a, relay.TupleType([])),
        (b, relay.scalar_type("bool")),
        (c, relay.scalar_type("int8")),
        (d, tt)
    ]
    expected_type = relay.FuncType([relay.TupleType([]),
                                    relay.scalar_type("bool"),
                                    relay.scalar_type("int8")],
                                   tt)
    assert check_instantiate(ft, mapping, expected_type)


def test_instantiate_type_call():
    gtv = relay.GlobalTypeVar("hi")
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    c = relay.TypeVar("c")
    tc = gtv(a, b, c)
    mapping = [
        (a, relay.TupleType([])),
        (b, relay.scalar_type("bool")),
        (c, relay.scalar_type("int8"))
    ]
    expected_type = gtv(relay.TupleType([]), relay.scalar_type("bool"), relay.scalar_type("int8"))
    assert check_instantiate(tc, mapping, expected_type)


def test_unify_failures():
    bool_t = relay.scalar_type("bool")
    int_t = relay.scalar_type("int32")
    cand1 = relay.TupleType([bool_t, int_t, bool_t])
    cand2 = relay.FuncType([bool_t, bool_t, bool_t, int_t], cand1)
    cand3 = relay.FuncType([bool_t], int_t)
    cand4 = relay.TupleType([int_t])
    cand5 = relay.TupleType([relay.TupleType([int_t, bool_t, int_t])])

    a, b, c, d = [relay.TypeVar(l) for l in ("a", "b", "c", "d")]
    tt = relay.TupleType([a, b])
    ft = relay.FuncType([a, b, c], d)

    for c in [bool_t, int_t, cand1, cand2, cand3, cand4, cand5]:
        success, _ = type_utils.attempt_unify(c, tt)
        assert not success
        success, _ = type_utils.attempt_unify(c, ft)
        assert not success


def test_unify_tuple():
    tt = relay.TupleType([relay.TupleType([]), relay.scalar_type("float32")])
    a, b = relay.TypeVar("a"), relay.TypeVar("b")
    template = relay.TupleType([a, b])
    success, mapping = type_utils.attempt_unify(tt, template)
    assert success
    assert check_mapping(mapping, [
        (a, relay.TupleType([])),
        (b, relay.scalar_type("float32"))
    ])


def test_unify_func():
    ft = relay.FuncType([relay.TupleType([]), relay.scalar_type("float32")],
                        relay.scalar_type("bool"))
    a, b, c = relay.TypeVar("a"), relay.TypeVar("b"), relay.TypeVar("c")
    template = relay.FuncType([a, b], c)
    success, mapping = type_utils.attempt_unify(ft, template)
    assert success
    assert check_mapping(mapping, [
        (a, relay.TupleType([])),
        (b, relay.scalar_type("float32")),
        (c, relay.scalar_type("bool"))
    ])


def test_unify_type_call():
    gtv = relay.GlobalTypeVar("hi")
    tc = relay.TypeCall(gtv, [relay.scalar_type("float32"), relay.scalar_type("bool")])
    a, b = relay.TypeVar("a"), relay.TypeVar("b")
    template = relay.TypeCall(gtv, [a, b])
    success, mapping = type_utils.attempt_unify(tc, template)
    assert success
    assert check_mapping(mapping, [
        (a, relay.scalar_type("float32")),
        (b, relay.scalar_type("bool"))
    ])


def test_instantiate_list_ctors():
    p = relay.prelude.Prelude()
    l = p.mod.get_global_type_var("List")

    inner_type = relay.TupleType([relay.scalar_type("bool"), relay.RefType(relay.scalar_type("float32"))])
    tc = l(inner_type)
    instantiated = type_utils.get_instantiated_constructors(p, tc)

    expected = {
        "Nil": relay.FuncType([], tc),
        "Cons": relay.FuncType([inner_type, tc], tc)
    }
    actual = {
        ctor.name_hint: ft
        for ctor, ft in instantiated
    }

    assert len(expected) == len(actual)
    for name, ft in actual.items():
        assert name in expected, name
        assert ft == expected[name]


def test_partially_instantiate_map():
    p = relay.prelude.Prelude()
    m = p.mod.get_global_var("map")
    l = p.mod.get_global_type_var("List")

    map_ty = m.checked_type
    concrete_list = l(relay.TupleType([]))

    success, ft = type_utils.partially_instantiate_func(map_ty, concrete_list)
    a = relay.TypeVar("a")
    expected_ft = relay.FuncType(
        [relay.FuncType([a], relay.TupleType([])), l(a)],
        concrete_list,
        type_params=[a])
    assert success
    assert ft == expected_ft


if __name__ == "__main__":
    test_instantiate_nothing()
    test_partial_instantiate()
    test_instantiate_ref()
    test_instantiate_tuple_full()
    test_instantiate_func()
    test_instantiate_type_call()
    test_unify_failures()
    test_unify_tuple()
    test_unify_func()
    test_unify_type_call()
    test_instantiate_list_ctors()
    test_partially_instantiate_map()
