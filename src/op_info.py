"""
Handle generating calls and constraints for Relay operator calls
"""
import random

import tvm
from tvm import relay

from relation_solver import BroadcastRelation, IdentityRelation

class OpInfo:
    """
    Base class for operator handler, takes care of populating types
    and producing the actual call to the operator
    (to abstract over the many differences among Relay ops)
    """
    def generate_arg_types(self, ret_type):
        """
        Given the return type, solves constraints to get the arg types.
        Returns a list of arg types and optionally any additional parameters
        """
        raise NotImplementedError()

    def produce_call(self, arg_exprs, additional_params=None):
        """
        Given expression args of the appropriate type, return an op call.
        (Provided in case there are non-expression arguments that also need to be set)
        """
        raise NotImplementedError()

    def supports_return_type(self, ty):
        """
        Can this operator return the given type?
        Provided because different operators have weird rules on that
        """
        raise NotImplementedError()


class BroadcastingOp(OpInfo):
    """
    Many ops just do tensor broadcasting and keep the dtype
    """
    def __init__(self, max_dim, solver, constructor):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = BroadcastRelation(max_dim)
        self.constructor = constructor

    def generate_broadcast_ranks(self, max_rank):
        # There are only two shapes and one needs to be of the max rank.
        # In principle, we should parametrize this to permit
        # setting a generation policy, but the solver is also not very parametrized

        # rank of 0: scalar, permitted
        other_rank = random.randint(0, max_rank)
        ranks = [other_rank, max_rank]
        random.shuffle(ranks)
        return ranks

    def generate_arg_types(self, ret_type):
        assert isinstance(ret_type, relay.TensorType)
        dtype = ret_type.dtype
        shape = tuple([int(d) for d in ret_type.shape])
        arg_ranks = self.generate_broadcast_ranks(len(shape))
        arg_shapes = self.solver.solve(arg_ranks, [shape], self.relation)
        ret = [relay.TensorType(arg_shape, dtype) for arg_shape in arg_shapes]
        return ret, None

    def supports_return_type(self, ret_type):
        return isinstance(ret_type, relay.TensorType)

    def produce_call(self, arg_exprs, additional_params=None):
        return self.constructor(*arg_exprs)


class IdentityOp(OpInfo):
    """
    Some ops just ensure that the return type is the same as the args (especially unary ops)
    """
    def __init__(self, max_dim, solver, num_args, constructor):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = IdentityRelation(max_dim)
        self.num_args = num_args
        self.constructor = constructor

    def generate_arg_types(self, ret_type):
        assert isinstance(ret_type, relay.TensorType)
        dtype = ret_type.dtype
        shape = tuple([int(d) for d in ret_type.shape])
        arg_ranks = [len(shape) for i in range(self.num_args)]
        arg_shapes = self.solver.solve(arg_ranks, [shape], self.relation)
        ret = [relay.TensorType(arg_shape, dtype) for arg_shape in arg_shapes]
        return ret, None

    def supports_return_type(self, ret_type):
        # TODO: should allow for specifying non-tensor types
        # (all real ops in Relay that have the identity relation
        # only expect tensors, but the type relation can support anything)
        return isinstance(ret_type, relay.TensorType)

    def produce_call(self, arg_exprs, additional_params=None):
        return self.constructor(*arg_exprs)

"""
Wrappers to fill in default parameters for constructors
(makes it easier to define the constructors below)
"""
def define_broadcasting_op(constructor):
    def wrap_constructor(max_dim, solver):
        return BroadcastingOp(max_dim, solver, constructor)
    return wrap_constructor


def define_identity_op(num_args, constructor):
    def wrap_constructor(max_dim, solver):
        return IdentityOp(max_dim, solver, num_args, constructor)
    return wrap_constructor


def initialize_broadcasting_ops():
    constructors = [
        relay.add, relay.subtract, relay.multiply, relay.divide
    ]
    return [define_broadcasting_op(ctor) for ctor in constructors]

ALL_BROADCASTING_OPS = initialize_broadcasting_ops()

def initialize_identity_ops():
    # various unary ops (there are also identity ops that are not unary)
    constructors = [
        relay.ceil, relay.floor, relay.trunc, relay.sign, relay.logical_not
    ]
    return [define_identity_op(1, ctor) for ctor in constructors]

ALL_IDENTITY_OPS = initialize_identity_ops()

# TODO: Add more (there are tons)
