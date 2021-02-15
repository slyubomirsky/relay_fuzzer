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
    def __init__(self, max_dim, solver):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = BroadcastRelation(max_dim)

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
        shape = ret_type.shape
        arg_ranks = generate_broadcast_ranks(len(shape))
        arg_shapes = self.solver.solve(arg_ranks, shape, self.relation)
        ret = [relay.TensorType(arg_shape, dtype) for arg_shape in arg_shapes]

    def supports_return_type(self, ret_type):
        return isinstance(ret_type, relay.TensorType)


class AddInfo(BroadcastingOp):
    def produce_call(arg_exprs, additional_params=None):
        return relay.add(*arg_exprs)

class SubInfo(BroadcastingOp):
    def produce_call(arg_exprs, additional_params=None):
        return relay.subtract(*arg_exprs)

# note: elementwise mul, not matrix mul (which is nn.dense)
class MulInfo(BroadcastingOp):
    def produce_call(arg_exprs, additional_params=None):
        return relay.multiply(*arg_exprs)

class DivInfo(BroadcastingOp):
    def produce_call(arg_exprs, additional_params=None):
        return relay.divide(*arg_exprs)

# TODO: Add more (there are tons)
