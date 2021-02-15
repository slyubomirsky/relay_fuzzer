"""
Classes for producing Relay expressions.
These are provided to separate producing expressions from decision-making policies.
None of these should include any kind of decision-making or randomness directly.
"""
import tvm
from tvm import relay
from tvm.relay.analysis import all_vars

from type_generator import TypeConstructs as TC

class ExprConstructor:
    def __init__(self, var_scope, generate_expr, generate_type,
                 generate_ctor, generate_patterns, generate_op):
        """
        var_scope: Responsible for producing variables
                   and tracking what's in scope
                   (lets, ifs, matches, and funcs produce new scopes)
        generate_expr: Function that takes a type and returns an expr of that type
        generate_type: As named (can specify supported types and params)
        generate_ctor: Given an ADT handle, returns a constructor for it
        generate_patterns: Given a type, generate a set of complete match patterns for it
        generate_op: Given a return type, returns the handler for an op with that return type
        """
        self.var_scope = var_scope
        self.generate_expr = generate_expr
        self.generate_type = generate_type
        self.generate_ctor = generate_ctor
        self.generate_patterns = generate_patterns
        self.generate_op = generate_op

    def constructor_tensor_literal(self, value):
        # Trivial but provided for so all expressions other than vars could be represented
        return relay.const(value)

    def construct_tuple_literal(self, member_types):
        return relay.Tuple([self.generate_expr(ty) for ty in member_types])

    def construct_ref_literal(self, inner_type):
        return relay.RefCreate(self.generate_expr(inner_type))

    def construct_func_literal(self, arg_types, ret_type, own_name=None):
        # own name: if the function is recursive, it needs to have itself in scope
        with self.var_scope.new_scope():
            arg_vars = [self.var_scope.new_local_var(ty, add_to_scope=True)
                        for ty in arg_types]
            if own_name is not None:
                self.var_scope.add_to_scope(own_name)
            body = self.generate_expr(ret_type)
            return relay.Function(arg_vars, body, ret_type=ret_type)

    def construct_adt_literal(self, type_call):
        ctor, instantiated_type = self.generate_ctor(type_call)
        return relay.Call(ctor, [self.generate_expr(input_type)
                                 for input_type in instantiated_type.arg_types])

    # connectives

    def construct_let_expr(self, ret_type):
        # handling recursive function definitions is tricky
        binder_type = self.generate_type()
        identifier = self.var_scope.new_local_var(binder_ty, add_to_scope=False)
        with self.var_scope.new_scope():
            own_name = None
            if isinstance(binder_ty, relay.FunctionType):
                own_name = identifier
            binder_expr = self.generate_expr(binder_ty, own_name=own_name)
        with self.var_scope.new_scope():
            self.var_scope.add_to_scope(identifier)
            bound_expr = self.generate_expr(ret_type)
        return relay.Let(identifier, binder_expr, bound_expr)

    def construct_tuple_index(self, ret_type, idx):
        # the tuple must be _at least_ big enough to contain the index
        # and tuple[idx] must be of the ret type
        constrained = {idx: ret_type}
        tup_type = self.generate_type(gen_params={
            TC.TUPLE: {
                "min_arity": idx+1,
                "constrained": constrained
            }
        })
        assert isinstance(tup_ty, relay.TupleType)
        return relay.TupleGetItem(self.generate_expr(tup_type), idx)

    def construct_if_branch(self, ret_type):
        # branch condition must be a boolean scalar
        cond_type = relay.scalar_type("bool")
        cond_expr = self.generate_expr(cond_type)
        # new scope for each branch
        with self.var_scope.new_scope():
            true_branch = self.generate_expr(ret_type)
        with self.var_scope.new_scope():
            false_branch = self.generate_expr(ret_type)
        return relay.If(cond_expr, true_branch, false_branch)

    def construct_function_call(self, ret_type):
        func_type = self.generate_type(gen_params={
            TC.FUNC: {
                "ret_type": ret_type
            }
        })
        assert isinstance(func_ty, relay.FuncType)
        func_expr = self.generate_expr(func_type)
        arg_exprs = [self.generate_expr(arg_types) for arg_types in func_ty.arg_types]
        return relay.Call(func_expr, arg_exprs)

    def construct_match(self, ret_type):
        # matching only defined on tuples and ADTs
        match_type = self.generate_type(gen_params={
            TC.TUPLE: {},
            TC.ADT: {}
        })
        match_val = self.generate_expr(match_type)
        match_patterns = self.generate_patterns(match_type)

        match_clauses = []
        # if there are var patterns, those vars are bound to a new scope in each clause
        for pattern in match_patterns:
            pattern_vars = all_vars(pattern)
            with self.var_scope.new_scope():
                for var in patterns:
                    self.var_scope.add_to_scope(var)
                match_expr = self.generate_expr(ret_type)
            match_clauses.append(relay.Clause(pattern, match_expr))
        return relay.Match(match_val, match_clauses)

    def construct_ref_write(self):
        # ref writes are always of type (), so there is no type param
        ref_type = self.generate_type(gen_params={TC.REF: {}})
        assert isinstance(ref_ty, relay.RefType)
        ref_expr = self.generate_expr(ref_type)
        inner_type = ref_ty.value
        assign_expr = self.generate_expr(inner_type)
        return relay.RefWrite(ref_expr, assign_expr)

    def construct_ref_read(self, ret_type):
        ref_expr = self.generate_expr(relay.RefType(ret_type))
        return relay.RefRead(ref_expr)

    def construct_op_call(self, ret_type):
        # Warning: Check that there exists an operator with the given return type first

        # Abstracting away many details of op calls because Relay ops are very varied:
        # there are ops that return just tensors, others that return tuples of tensors,
        # some that take only tensors, some that take tuples of tensors, etc.,
        # and some that take compile-time parameters (not Relay exprs),
        # so for maximum flexibility, each op should manage how it is called
        op_info = self.generate_op(ret_type)
        arg_types, additional_params = op_info.generate_arg_types(ret_type)
        arg_exprs = [self.generate_expr(arg_type) for arg_type in arg_types]
        return op_info.produce_call(arg_exprs, additional_params=additional_params)


# handle pattern generation
class PatternConstructor:
    def __init__(self, var_scope, generate_pattern, generate_ctor):
        """
        var_scope: For generating pattern vars
        generate_pattern: Given a type, generates a pattern that matches that type
        generate_ctor: Given an ADT handle, returns a constructor for it
        """
        self.var_scope = var_scope
        self.generate_pattern = generate_pattern
        self.generate_ctor = generate_ctor

    def construct_wildcard(self):
        # trivial but here for consistency
        return relay.PatternWildcard()

    def construct_var_pattern(self, var_type):
        fresh_var = self.var_scope.new_local_var(var_type, add_to_scope=False)
        return relay.PatternVar(fresh_var)

    def construct_tuple_pattern(self, tup_type):
        nested_patterns = [self.generate_pattern(field_type) for field_type in tup_type.fields]
        return relay.PatternTuple(nested_patterns)

    def construct_ctor_pattern(self, type_call):
        ctor, instantiated_type = self.generate_ctor(type_call)
        nested_patterns = [self.generate_pattern(input_type)
                           for input_type in instantiated_type.arg_types]
        return relay.PatternConstructor(ctor, nested_patterns)
