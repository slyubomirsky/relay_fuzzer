import random

import tvm
from tvm import relay

from op_info import *
from relation_solver import (BruteForceSolver, ILPSolver, MemoizedSolver,
                             IdentityRelation, BroadcastRelation)

MAX_RANK = 10
MAX_DIM = 10
NUM_ATTEMPTS = 10

def check_op_setup(op_info, ret_type):
    # make sure the result type checks
    assert op_info.supports_return_type(ret_type)
    arg_types, additional_params = op_info.generate_arg_types(ret_type)

    arg_vars = [
        relay.Var(f"v_{i}", type_annotation=arg_type)
        for i, arg_type in enumerate(arg_types)
    ]
    call = op_info.produce_call(arg_vars, additional_params=additional_params)

    mod = tvm.IRModule()
    func = relay.Function(arg_vars, call, ret_type=ret_type)
    mod["main"] = func
    try:
        mod = relay.transform.InferType()(mod)
    except Exception as e:
        assert False, f"{func} fails to type check"


def generate_return_shape():
    rank = random.randint(0, MAX_RANK)
    return [random.randint(1, MAX_DIM) for i in range(rank)]


def generate_dtype():
    return random.choice(["int8", "float32", "float64", "int32", "int64"])


def test_broadcast_ops():
    solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))
    broadcast_ops = [
        ctor(MAX_DIM, solver)
        for ctor in (AddInfo, SubInfo, MulInfo, DivInfo)
    ]

    for op_info in broadcast_ops:
        for i in range(NUM_ATTEMPTS):
            shape = generate_return_shape()
            dtype = generate_dtype()
            ret_type = relay.TensorType(shape, dtype)
            check_op_setup(op_info, ret_type)


if __name__ == "__main__":
    test_broadcast_ops()
