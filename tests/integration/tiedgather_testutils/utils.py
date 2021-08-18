# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

from typing import Iterable, Any, Callable, Tuple, Union, Mapping, Optional
import tempfile
import os
from itertools import chain
from functools import reduce

import onnx
from onnx import numpy_helper
import math
import numpy as np

import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def make_tuple(something: Any) -> Tuple:
    if isinstance(something, tuple) or isinstance(something, list):

        def concat(accl: Iterable, s: Any) -> Iterable:
            return chain(accl, make_tuple(s))

        return tuple(reduce(concat, something, ()))
    return (something, )


def run_py(proto: onnx.ModelProto,
           data: Mapping[str, np.ndarray],
           outputs: Optional[Union[str, Iterable[str]]],
           loss: Optional[str] = None,
           optimizer: Optional[popart.Optimizer] = None,
           patterns: Optional[popart.Patterns] = None,
           user_options: Optional[Mapping[str, Any]] = None,
           skip_execution: bool = False):
    batches_per_step = 1

    outputs = make_tuple(outputs)

    # Setting up the Session
    data_flow = popart.DataFlow(
        batches_per_step,
        {output: popart.AnchorReturnType("ALL")
         for output in outputs})

    if user_options is None:
        user_options = {}
    options = popart.SessionOptions()
    options.reportOptions = {"showVarStorage": "true"}
    options.enableStochasticRounding = False
    options.constantWeights = True
    options.outlineThreshold = 10.0

    for key, value in user_options.items():
        if key not in ["batchSerializationFactor", "executionPhases"]:
            setattr(options, key, value)

    replicas = user_options.get("replicatedGraphCount", 1)
    request_ipus = pow(2, math.ceil(math.log2(replicas)))
    device = tu.create_test_device(numIpus=request_ipus)

    print("Compiling graph")
    if optimizer is not None:
        session = popart.TrainingSession(fnModel=proto,
                                         deviceInfo=device,
                                         dataFlow=data_flow,
                                         userOptions=options,
                                         loss=loss,
                                         optimizer=optimizer,
                                         patterns=patterns)
    else:
        session = popart.InferenceSession(fnModel=proto,
                                          deviceInfo=device,
                                          dataFlow=data_flow,
                                          userOptions=options,
                                          patterns=patterns)

    if skip_execution:
        device.detach()
        return session

    # Compile the Poplar Graph. If it fails, return the memory stats
    try:
        session.prepareDevice()
    except popart.session.OutOfMemoryException as e:
        device.detach()
        raise e
    print("Compilation complete")

    session.weightsFromHost()
    # NOTE: If we ever use a model with random ops, we would need to call this
    # here, using the same seed given to numpy.
    # session.setRandomSeed(1984)

    anchors = session.initAnchorArrays()

    rf = user_options.get("replicatedGraphCount")
    if rf is not None and rf > 1:
        data = {k: np.repeat(v[np.newaxis], rf, 0) for k, v in data.items()}

    # Add a gradient accumulation factor dimension if needed
    af = user_options.get("accumulationFactor")
    if af is not None and af > 1:
        data = {k: np.repeat(v[np.newaxis], af, 0) for k, v in data.items()}

    stepio = popart.PyStepIO(data, anchors)
    session.run(stepio)

    with tempfile.TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "model.onnx")
        session.modelToHost(file_path)
        post_proto = onnx.load(file_path)

    # Release device
    device.detach()

    return (anchors[output] for output in outputs), post_proto, outputs


class TestFailureError(Exception):
    __test__ = False


def check_tensor(A, B, A_name, B_name, margin=1.5e-8):
    assert np.allclose(
        A, B, atol=margin), f"Check failed for (1) {A_name} (2) {B_name}"


def check_oom_failures(torch_output: np.ndarray, onnx_output: np.ndarray):
    failed_methods = []
    # Produce an error indicating which implementation ran out of memory during
    # compilation. Both could fail, so we won't print exclusively.
    if type(torch_output) == float and np.isnan(torch_output):
        failed_methods.append("Custom Operation")

    if type(onnx_output) == float and np.isnan(onnx_output):
        failed_methods.append("ONNX")

    if len(failed_methods) > 0:
        msg = "OOM in the following implementations: " + \
            ", ".join(failed_methods)

        raise TestFailureError(msg)


def check_tensors(torch_outputs: Iterable[np.ndarray],
                  onnx_outputs: Iterable[np.ndarray],
                  left_names: Iterable[str],
                  right_names: Iterable[str],
                  margin: float = 1.5e-8):
    for t_torch, t_onnx, lname, rname in zip(torch_outputs, onnx_outputs,
                                             left_names, right_names):
        check_oom_failures(t_torch, t_onnx)
        check_tensor(t_onnx.reshape(t_torch.shape),
                     t_torch,
                     lname,
                     rname,
                     margin=margin)


def onnx_to_numpy(tensor: onnx.TensorProto) -> np.ndarray:
    if tensor.data_type == onnx.TensorProto.FLOAT16:
        int_data = np.asarray(tensor.int32_data, np.int32)
        np_tensor = int_data.view(dtype=np.float16).reshape(tensor.dims)
    else:
        np_tensor = numpy_helper.to_array(tensor)
    return np_tensor


def check_onnx_model(
        model_1: onnx.ModelProto,
        model_2: onnx.ModelProto,
        onnx_to_onnx: Mapping[str, str] = {},
        transform: Mapping[str, Callable[[np.ndarray], np.ndarray]] = {},
        allow_missing: bool = True):
    model_1_weights = {}
    for weight in model_1.graph.initializer:
        model_1_weights[weight.name] = onnx_to_numpy(weight)

    if len(model_1_weights) > 0:
        for w_2 in model_2.graph.initializer:
            name = onnx_to_onnx.get(w_2.name, w_2.name)
            if name in model_1_weights.keys():
                np_w_1 = model_1_weights[name]
                if name in transform.keys():
                    np_w_1 = transform[name](np_w_1)
                elif w_2.name in transform.keys():
                    np_w_1 = transform[w_2.name](np_w_1)
                np_w_2 = onnx_to_numpy(w_2)
                check_tensor(np_w_1, np_w_2, name, w_2.name)

            else:
                if not allow_missing:
                    raise TestFailureError(
                        f"Missing weight mapping for model_2 weight {name}")
