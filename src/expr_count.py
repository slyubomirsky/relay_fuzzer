"""
Simple utility for counting the number of instances of certain expressions
in the fuzzer output.
"""
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.analysis import free_vars

class ExprCounter(ExprVisitor):
    def __init__(self):
        super().__init__()
        # primarily non-literals of interest
        self.counts = {
            "recursive_func_defs": 0,
            "global_calls": 0,
            "local_calls": 0,
            "closure_calls": 0,
            "lets": 0,
            "ctor_calls": 0,
            "ref_reads": 0,
            "ref_writes": 0,
            "op_calls": 0,
            "tuple_idxs": 0
        }
        self.op_names = set({})

    def visit_call(self, call):
        called = call.op
        if isinstance(called, relay.Var):
            self.counts["local_calls"] += 1
        elif isinstance(called, relay.GlobalVar):
            self.counts["global_calls"] += 1
        elif isinstance(called, relay.Constructor):
            self.counts["ctor_calls"] += 1
        elif isinstance(called, tvm.ir.Op):
            self.counts["op_calls"] += 1
            self.op_names.add(called.name)
        else:
            self.counts["closure_calls"] += 1
        super().visit_call(call)

    def visit_let(self, let):
        self.counts["lets"] += 1
        var = let.var
        bound = let.value
        if isinstance(bound, relay.Function) and var in set(free_vars(bound)):
            self.counts["recursive_func_defs"] += 1
        super().visit_let(let)

    def visit_ref_read(self, ref_read):
        self.counts["ref_reads"] += 1
        super().visit_ref_read(ref_read)

    def visit_ref_write(self, ref_write):
        self.counts["ref_writes"] += 1
        super().visit_ref_write(ref_write)

    def visit_tuple_getitem(self, tgi):
        self.counts["tuple_idxs"] += 1
        super().visit_tuple_getitem(tgi)

def count_exprs(expr):
    """
    Given a fuzzer-generated expression, this returns a dict of counted expressions
    and a set of operator names in the expressions.

    The expressions counted are the following (because they're nontrivial):
    * Recursive function let-definitions
    * Operator calls
    * Constructor calls
    * Tuple indices
    * Ref reads
    * Ref writes
    * Let bindings
    * Calls to local variables
    * Calls to expressions evaluating to closures
    * Calls to global functions (from the prelude)
    """
    counter = ExprCounter()
    counter.visit(expr)
    return counter.counts, counter.op_names
