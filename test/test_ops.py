import random

import tvm
from tvm import relay

from op_info import (ALL_BROADCASTING_OPS, ALL_IDENTITY_OPS, ALL_NONSCALAR_OPS,
                     BatchMatmulInfo, BatchNormInfo, Conv2DInfo)
from relation_solver import MemoizedSolver, ILPSolver

MAX_RANK = 10
MAX_DIM = 10
NUM_ATTEMPTS = 10

def validate_types(op_info, arg_types, ret_type, additional_params):
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

def assert_type_checks(op_info, arg_types, ret_type, additional_params):
    assert op_info.supports_return_type(ret_type)
    validate_types(op_info, arg_types, ret_type, additional_params)

def check_op_setup(op_info, ret_type):
    arg_types, additional_params = op_info.generate_arg_types(ret_type)
    assert_type_checks(op_info, arg_types, ret_type, additional_params)


def check_sampled_types(op_info):
    arg_types, ret_type, additional_params = op_info.sample_call(use_solver=False)
    assert_type_checks(op_info, arg_types, ret_type, additional_params)
    arg_types, ret_type, additional_params = op_info.sample_call(use_solver=True)
    assert_type_checks(op_info, arg_types, ret_type, additional_params)


def generate_return_shape(min_rank=0):
    rank = random.randint(min_rank, MAX_RANK)
    return [random.randint(1, MAX_DIM) for i in range(rank)]


def generate_dtype():
    return random.choice(["int8", "float32", "float64", "int32", "int64", "bool"])


def test_basic_ops():
    solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))
    op_info_col = [
        ctor(MAX_DIM, solver) for ctor in (ALL_BROADCASTING_OPS + ALL_IDENTITY_OPS)
    ]

    for op_info in op_info_col:
        for i in range(NUM_ATTEMPTS):
            shape = generate_return_shape()
            dtype = generate_dtype()
            ret_type = relay.TensorType(shape, dtype)
            check_op_setup(op_info, ret_type)
        for i in range(NUM_ATTEMPTS):
            check_sampled_types(op_info)


def test_nonscalar_ops():
    solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))
    op_info_col = [
        ctor(MAX_DIM, solver) for ctor in ALL_NONSCALAR_OPS
    ]

    for op_info in op_info_col:
        for i in range(NUM_ATTEMPTS):
            shape = generate_return_shape(min_rank=1)
            dtype = generate_dtype()
            ret_type = relay.TensorType(shape, dtype)
            check_op_setup(op_info, ret_type)
        for i in range(NUM_ATTEMPTS):
            check_sampled_types(op_info)


def test_batch_matmul():
    solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))
    op_info = BatchMatmulInfo(MAX_DIM, solver)

    for i in range(NUM_ATTEMPTS):
        shape = [random.randint(1, MAX_DIM) for i in range(3)]
        dtype = generate_dtype()
        ret_type = relay.TensorType(shape, dtype)
        check_op_setup(op_info, ret_type)
    for i in range(NUM_ATTEMPTS):
        check_sampled_types(op_info)


def test_batch_norm():
    solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))
    op_info = BatchNormInfo(MAX_DIM, solver)

    for i in range(NUM_ATTEMPTS):
        data_shape = generate_return_shape(min_rank=1)
        for j in range(NUM_ATTEMPTS):
            axis = random.randint(-1, len(data_shape)-1)
            vec_shape = (data_shape[axis],)
            dtype = generate_dtype()
            ret_type = relay.TupleType([
                relay.TensorType(data_shape, dtype),
                relay.TensorType(vec_shape, dtype),
                relay.TensorType(vec_shape, dtype)
            ])
            check_op_setup(op_info, ret_type)
    for i in range(NUM_ATTEMPTS):
        check_sampled_types(op_info)


def test_conv2d():
    solver = MemoizedSolver(ILPSolver(MAX_DIM, 30, False))
    op_info = Conv2DInfo(MAX_DIM, solver)
    for i in range(NUM_ATTEMPTS):
        shape = [random.randint(1, MAX_DIM) for i in range(4)]
        dtype = generate_dtype()
        ret_type = relay.TensorType(shape, dtype)
        check_op_setup(op_info, ret_type)
    for i in range(NUM_ATTEMPTS):
        check_sampled_types(op_info)


if __name__ == "__main__":
    test_basic_ops()
    test_nonscalar_ops()
    test_batch_matmul()
    test_batch_norm()
    test_conv2d()
