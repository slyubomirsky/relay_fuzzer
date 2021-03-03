"""
Handle generating calls and constraints for Relay operator calls
"""
import random

import tvm
from tvm import relay

from relation_solver import (BroadcastRelation, IdentityRelation,
                             DenseRelation, BiasAddRelation,
                             BatchMatmulRelation, BatchNormRelation,
                             Conv2DRelation)

def random_dtype():
    # total hack until we properly parameterize the generation here
    return random.choice(["int8", "int32", "bool", "float32", "float64"])


def default_sample(op_info, min_rank=0, max_rank=None):
    # really bad default that should really be parameterized
    dtype = random_dtype()
    ret_rank = random.randint(min_rank,
                              max_rank if max_rank is not None else 5 + min_rank)
    # holes to allow free choice
    shape = tuple([None for d in range(ret_rank)])
    return op_info.solve_for_types(dtype, shape)


class OpInfo:
    """
    Base class for operator handler, takes care of populating types
    and producing the actual call to the operator
    (to abstract over the many differences among Relay ops)
    """
    def generate_arg_types(self, ret_type):
        """
        "Backward solving": Given the return type, solves constraints to get the arg types.
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

    def sample_call(self):
        """
        "Forward solving": With no target return type, produce a valid set of argument and return types.
        Returns (list of argument types, list of return types, any additional parameters)
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
        return tuple(ranks)

    def solve_for_types(self, dtype, ret_shape):
        arg_ranks = self.generate_broadcast_ranks(len(ret_shape))
        arg_shapes, return_shapes = self.solver.solve(self.relation, (arg_ranks, (ret_shape,)))
        arg_types = [relay.TensorType(arg_shape, dtype) for arg_shape in arg_shapes]
        ret_type = relay.TensorType(return_shapes[0], dtype)
        return arg_types, ret_type, None

    def generate_arg_types(self, ret_type):
        assert isinstance(ret_type, relay.TensorType)
        dtype = ret_type.dtype
        shape = tuple([int(d) for d in ret_type.shape])
        arg_types, _, _ = self.solve_for_types(dtype, shape)
        return arg_types, None

    def sample_call(self):
        return default_sample(self)

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

    def solve_for_types(self, dtype, ret_shape):
        arg_ranks = tuple([len(ret_shape) for i in range(self.num_args)])
        arg_shapes, return_shapes = self.solver.solve(self.relation, (arg_ranks, (ret_shape,)))
        arg_types = [relay.TensorType(arg_shape, dtype) for arg_shape in arg_shapes]
        ret_type = relay.TensorType(return_shapes[0], dtype)
        return arg_types, ret_type, None

    def generate_arg_types(self, ret_type):
        assert isinstance(ret_type, relay.TensorType)
        dtype = ret_type.dtype
        shape = tuple([int(d) for d in ret_type.shape])
        arg_types, _, _ = self.solve_for_types(dtype, shape)
        return arg_types, None

    def supports_return_type(self, ret_type):
        # TODO: should allow for specifying non-tensor types
        # (all real ops in Relay that have the identity relation
        # only expect tensors, but the type relation can support anything)
        return isinstance(ret_type, relay.TensorType)

    def sample_call(self):
        return default_sample(self)

    def produce_call(self, arg_exprs, additional_params=None):
        return self.constructor(*arg_exprs)

# special case: clip is an identity, unary op that also needs min and max params
class ClipInfo(IdentityOp):
    def __init__(self, max_dim, solver):
        super().__init__(max_dim, solver, 1, relay.clip)

    def generate_additional_params(self):
        # additional params: a_min and a_max (TODO: make parametrizable)
        clip_bound = 64.0 # totally arbitrary, does not affect type checking
        clip_params = [random.uniform(-clip_bound, clip_bound), random.uniform(-clip_bound, clip_bound)]
        return (min(clip_params), max(clip_params))

    def generate_arg_types(self, ret_type):
        ret, _ = super().generate_arg_types(ret_type)
        return ret, self.generate_additional_params()

    def sample_call(self):
        arg_types, ret_type, _ = super().sample_call()
        return arg_types, ret_type, self.generate_additional_params()

    def produce_call(self, arg_exprs, additional_params=None):
        return relay.clip(arg_exprs[0],
                          a_min=additional_params[0],
                          a_max=additional_params[1])

class DenseInfo(OpInfo):
    def __init__(self, max_dim, solver):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = DenseRelation(max_dim)

    def solve_for_types(self, dtype, ret_shape):
        arg_ranks = (len(ret_shape), 2)

        # TODO: parameterize this choice
        units_defined = random.choice([True, False])
        units, arg_shapes, return_shapes = self.solver.solve(
            self.relation, (units_defined, arg_ranks, (ret_shape,)))

        # data type rules: can manually set an out data type (casts), otherwise use dtype of data input
        # (for now, we will conservatively give everything the same dtype, which is by far the most common choice, but this is not inherently necessary!)
        arg_types = [
            relay.TensorType(arg_shapes[0], dtype),
            relay.TensorType(arg_shapes[1], dtype)
        ]
        ret_type = relay.TensorType(return_shapes[0], dtype)

        additional_params = None
        if units is not None:
            additional_params = {"units": units}
        return arg_types, ret_type, additional_params

    def generate_arg_types(self, ret_type):
        ret_dtype = ret_type.dtype
        ret_shape = tuple([int(d) for d in ret_type.shape])
        arg_types, _, additional_params = self.solve_for_types(ret_dtype, ret_shape)
        return arg_types, additional_params

    def produce_call(self, arg_exprs, additional_params=None):
        data = arg_exprs[0]
        weight = arg_exprs[1]
        units = None
        out_dtype = ""
        if additional_params is not None:
            if "units" in additional_params:
                units = additional_params["units"]
            if "out_dtype" in additional_params:
                out_dtype = additional_params["out_dtype"]
        return relay.nn.dense(data, weight, units=units, out_dtype=out_dtype)

    def sample_call(self):
        return default_sample(self, min_rank=1)

    def supports_return_type(self, ty):
        # supports any nonscalar
        if not isinstance(ty, relay.TensorType):
            return False
        return len(ty.shape) != 0


class BiasAddInfo(OpInfo):
    def __init__(self, max_dim, solver):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = BiasAddRelation(self.max_dim)

    def solve_for_types(self, dtype, ret_shape):
        ret_rank = len(ret_shape)
        # TODO: parameterize this choice
        axis = random.randint(-(ret_rank-1), ret_rank-1)
        arg_ranks = (ret_rank, 1)
        _, arg_shapes, return_shapes = self.solver.solve(self.relation, (axis, arg_ranks, (ret_shape,)))
        arg_types = [
            relay.TensorType(arg_shapes[0], dtype),
            relay.TensorType(arg_shapes[1], dtype)
        ]
        return_type = relay.TensorType(return_shapes[0], dtype)
        return arg_types, return_type, axis


    def generate_arg_types(self, ret_type):
        ret_dtype = ret_type.dtype
        ret_shape = tuple([int(d) for d in ret_type.shape])
        arg_types, _, axis = self.solve_for_types(ret_dtype, ret_shape)
        return arg_types, axis

    def produce_call(self, arg_exprs, additional_params=None):
        data = arg_exprs[0]
        weight = arg_exprs[1]
        axis = 0
        if additional_params is not None:
            axis = additional_params
        return relay.nn.bias_add(data, weight, axis=axis)

    def sample_call(self):
        return default_sample(self, min_rank=1)

    def supports_return_type(self, ty):
        # supports any nonscalar
        if not isinstance(ty, relay.TensorType):
            return False
        return len(ty.shape) != 0


class BatchMatmulInfo(OpInfo):
    def __init__(self, max_dim, solver):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = BatchMatmulRelation(max_dim)

    def solve_for_types(self, dtype, ret_shape):
        arg_ranks = (3, 3)
        arg_shapes, return_shapes = self.solver.solve(self.relation, (arg_ranks, (ret_shape,)))
        # TODO: technically, the type relation only checks the first arg's dtype so the second can be anything
        arg_types = [
            relay.TensorType(arg_shapes[0], dtype),
            relay.TensorType(arg_shapes[1], dtype)
        ]
        ret_type = relay.TensorType(return_shapes[0], dtype)
        return arg_types, ret_type, None


    def generate_arg_types(self, ret_type):
        ret_dtype = ret_type.dtype
        ret_shape = tuple([int(d) for d in ret_type.shape])
        arg_types, _, _ = self.solve_for_types(ret_dtype, ret_shape)
        return arg_types, None

    def produce_call(self, arg_exprs, additional_params=None):
        return relay.nn.batch_matmul(*arg_exprs)

    def sample_call(self):
        return default_sample(self, min_rank=3, max_rank=3)

    def supports_return_type(self, ty):
        # only supports tensors of rank 3
        if not isinstance(ty, relay.TensorType):
            return False
        return len(ty.shape) == 3


class BatchNormInfo(OpInfo):
    def __init__(self, max_dim, solver):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = BatchNormRelation(self.max_dim)

    def valid_axes(self, data_shape, vector_shapes):
        data_rank = len(data_shape)
        ret = []
        for i in range(data_rank):
            if all(map(
                    lambda v: (data_shape[i] is None or v[0] == data_shape[i]),
                    vector_shapes
            )):
                ret.append(i)
                if i == data_rank - 1:
                    ret.append(-1)
        return ret

    def solve_for_types(self, ret_dtype, ret_shape, vec_shapes):
        ret_rank = len(ret_shape)

        # TODO: parameterize this choice(?)
        axis = random.choice(self.valid_axes(ret_shape, vec_shapes))
        arg_ranks = (ret_rank, 1, 1, 1, 1)
        axis, arg_shapes, return_shapes = self.solver.solve(
            self.relation,
            (axis, arg_ranks, (ret_shape, *vec_shapes)))
        arg_types = [
            relay.TensorType(arg_shape, ret_dtype)
            for arg_shape in arg_shapes
        ]
        ret_type = relay.TupleType([
            relay.TensorType(solved_shape, ret_dtype)
            for solved_shape in return_shapes
        ])
        return arg_types, ret_type, axis

    def generate_arg_types(self, ret_type):
        data_type = ret_type.fields[0]
        vec_types = ret_type.fields[1:]
        ret_dtype = data_type.dtype

        ret_shape = tuple([int(d) for d in data_type.shape])
        vec_shapes = [(int(v.shape[0]),) for v in vec_types]
        arg_types, _, axis = self.solve_for_types(ret_dtype, ret_shape, vec_shapes)
        return arg_types, axis

    def produce_call(self, arg_exprs, additional_params=None):
        axis = 0
        if additional_params is not None:
            axis = additional_params
        return relay.nn.batch_norm(*arg_exprs, axis=axis).astuple()

    def sample_call(self):
        # TODO: parameterize this generation
        dtype = random_dtype()
        data_rank = random.randint(1, 5)
        vec_ranks = [1, 1]
        data_shape = tuple([None for i in range(data_rank)])
        vec_shapes = ((None,), (None,))
        return self.solve_for_types(dtype, data_shape, vec_shapes)

    def supports_return_type(self, ty):
        # tuple of length 3 where dtypes match and some axis is appropriate
        if not isinstance(ty, relay.TupleType) or len(ty.fields) != 3:
            return False
        data_type = ty.fields[0]
        vec_types = ty.fields[1:]
        if not isinstance(data_type, relay.TensorType) or len(data_type.shape) == 0:
            return False
        data_shape = tuple([int(d) for d in data_type.shape])
        data_dtype = data_type.dtype

        vec_shapes = []
        for v in vec_types:
            if not isinstance(v, relay.TensorType) or len(v.shape) != 1:
                return False
            if v.dtype != data_dtype:
                return False
            vec_shapes.append((int(v.shape[0]),))
        valid_axes = self.valid_axes(data_shape, vec_shapes)
        return len(valid_axes) != 0


class Conv2DInfo(OpInfo):
    def __init__(self, max_dim, solver):
        self.max_dim = max_dim
        self.solver = solver
        self.relation = Conv2DRelation(max_dim)

    def solve_for_types(self, dtype, ret_shape):
        # TODO: Handle the plethora of optional params eventually
        arg_ranks = (4, 4)
        arg_shapes, return_shapes = self.solver.solve(self.relation, (arg_ranks, (ret_shape,)))
        # TODO: handle dtype inference later
        arg_types = [
            relay.TensorType(arg_shape, dtype)
            for arg_shape in arg_shapes
        ]
        ret_type = relay.TensorType(return_shapes[0], dtype)
        return arg_types, ret_type, None

    def generate_arg_types(self, ret_type):
        ret_dtype = ret_type.dtype
        ret_shape = tuple([int(d) for d in ret_type.shape])
        arg_types, _, _ = self.solve_for_types(ret_dtype, ret_shape)
        return arg_types, None

    def produce_call(self, arg_exprs, additional_params=None):
        # going to leave everything at default settings for now
        return relay.nn.conv2d(*arg_exprs)

    def sample_call(self):
        return default_sample(self, min_rank=4, max_rank=4)

    def supports_return_type(self, ty):
        # tensor of rank 4
        return isinstance(ty, relay.TensorType) and len(ty.shape) == 4


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
    return list(map(define_broadcasting_op, (
        relay.add, relay.subtract, relay.multiply, relay.divide,
        relay.logical_and, relay.logical_or, relay.logical_xor
    )))


def initialize_identity_ops():
    # various unary ops (there are also identity ops that are not unary)
    def construct_unary_op_call(ctor):
        return define_identity_op(1, ctor)

    ret = list(map(construct_unary_op_call, (
        relay.ceil, relay.floor, relay.trunc, relay.sign, relay.logical_not,
        relay.log, relay.log10, relay.log2
    )))
    ret.append(ClipInfo)
    return ret


def initialize_nonscalar_ops():
    return [DenseInfo, BiasAddInfo]


ALL_BROADCASTING_OPS = initialize_broadcasting_ops()
ALL_IDENTITY_OPS = initialize_identity_ops()
ALL_NONSCALAR_OPS = initialize_nonscalar_ops()

# TODO: Add more (there are tons)
