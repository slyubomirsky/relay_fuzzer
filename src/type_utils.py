"""
Useful functions for dealing with types, especially quantified function types.
"""
import tvm
from tvm import relay
from tvm.relay.type_functor import TypeFunctor, TypeMutator
from tvm.relay.analysis import all_type_vars

def instantiate(ty, map_pairs):
    """
    Given a type with type variables and a map of
    type vars to assignments,
    replace the type vars with the assignments
    """
    # type vars cannot be hashed so we will use TVM's structural hash
    # as a hack

    type_map = {}
    for tv, value in map_pairs:
        type_map[tvm.ir.structural_hash(tv)] = value

    class Instantiator(TypeMutator):
        def visit_type_var(self, tv):
            tv_hash = tvm.ir.structural_hash(tv)
            if tv_hash in type_map:
                return type_map[tv_hash]
            return tv
        # this should really be a default in the mainline...
        def visit_global_type_var(self, gtv):
            return gtv

    ret = Instantiator().visit(ty)
    return ret


def attempt_unify(concrete, test):
    """
    Given a concrete candidate type and a test type
    (which may contain type vars),
    determine if they can be unified and find assignments
    to type vars if so.

    Returns (False, None) if they cannot be unified
    and (True, [(type_var, concrete type)]) if they can

    The first arg must be concrete!
    """
    assert not all_type_vars(concrete)
    assignments = {}
    class Unifier(TypeFunctor):
        def __init__(self, check_ty):
            self.check_ty = check_ty

        def compare(self, lhs, rhs):
            # very ugly but we can't change the signature of the visit functions,
            # so we have to mutate check_ty
            old_val = self.check_ty
            self.check_ty = lhs
            ret = self.visit(rhs)
            self.check_ty = old_val
            return ret

        def visit_type_var(self, tv):
            tv_hash = tvm.ir.structural_hash(tv)
            if tv_hash not in assignments:
                assignments[tv_hash] = (tv, self.check_ty)
                return True
            # if we've matched before, the match must be consistent
            return tvm.ir.structural_equal(self.check_ty, assignments[tv_hash][1])

        def visit_tuple_type(self, tt):
            check_tt = self.check_ty
            if not isinstance(check_tt, relay.TupleType):
                return False
            if len(check_tt.fields) != len(tt.fields):
                return False

            for (lhs, rhs) in zip(check_tt.fields, tt.fields):
                match = self.compare(lhs, rhs)
                if not match:
                    return False
            return True

        def visit_ref_type(self, rt):
            check_rt = self.check_ty
            if not isinstance(check_rt, relay.RefType):
                return False
            return self.compare(check_rt.value, rt.value)

        def visit_global_type_var(self, gtv):
            check_gtv = self.check_ty
            if not isinstance(check_gtv, relay.GlobalTypeVar):
                return False
            return check_gtv == gtv

        def visit_type_call(self, tc):
            check_tc = self.check_ty
            if not isinstance(check_tc, relay.TypeCall):
                return False
            if len(check_tc.args) != len(tc.args):
                return False

            match = self.compare(check_tc.func, tc.func)
            if not match:
                return False

            for (lhs, rhs) in zip(check_tc.args, tc.args):
                match = self.compare(lhs, rhs)
                if not match:
                    return False
            return True

        def visit_func_type(self, ft):
            check_ft = self.check_ty
            if not isinstance(check_ft, relay.FuncType):
                return False
            if len(check_ft.arg_types) != len(ft.arg_types):
                return False
            # ignore type constraints for our purposes, for now
            match = self.compare(check_ft.ret_type, ft.ret_type)
            if not match:
                return False

            for (lhs, rhs) in zip(check_ft.arg_types, ft.arg_types):
                match = self.compare(lhs, rhs)
                if not match:
                    return False
            return True

    unifier_res = (Unifier(concrete).visit(test))
    if not unifier_res:
        return False, None
    ret = []
    for tv, ty in assignments.values():
        ret.append((tv, ty))
    return True, ret


def instantiate_constructor(prelude, constructor, type_call):
    """
    Given a specific type call and a constructor,
    produce a function type for that constructor
    given that instantiation of the type call
    """
    func = type_call.func
    assert constructor.belong_to == func
    td = prelude.mod[func]
    ft = relay.FuncType(constructor.inputs, type_call)
    instantiation_list = [
        (td.type_vars[i], type_call.args[i])
        for i in range(len(td.type_vars))
    ]
    return instantiate(ft, instantiation_list)


def get_instantiated_constructors(prelude, type_call):
    """
    Given a type call, looks up the ADT in the prelude
    and returns a list of (constructor, constructor type)
    in which there are no type vars
    """
    func = type_call.func
    assert isinstance(func, relay.GlobalTypeVar)
    td = prelude.mod[func]
    type_vars = td.type_vars
    constructors = td.constructors
    return [
        (ctor, instantiate_constructor(prelude, ctor, type_call))
        for ctor in constructors
    ]


def partially_instantiate_func(ft, ret_type):
    """
    Given a function type (potentially with type params) and a concrete
    return type,
    return (success, function type)
    where success is true if they can be unified and false if the they cannot.
    The returned function type will be instantiated as much as possible.
    """
    # remove type params if present
    strip_ft = relay.FuncType(ft.arg_types, ft.ret_type)
    success, mapping = attempt_unify(ret_type, strip_ft.ret_type)
    if not success:
        return False, None

    new_ft = instantiate(strip_ft, mapping)
    remaining_vars = all_type_vars(new_ft)
    return True, relay.FuncType(new_ft.arg_types, new_ft.ret_type, type_params=remaining_vars)
