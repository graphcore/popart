# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest
import json
import re
import onnx
import copy
from onnx import mapping

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu

np.random.seed(0)


# Attention Op from BERT
def attention_onnx(builder, qkv, mask, batch_size, sequence_length,
                   hidden_size, attention_heads, qkv_length):
    comb_shape = [batch_size, sequence_length, attention_heads, qkv_length]

    def extract_heads(tensor, index, hidden_size, transpose=False):
        tensor = builder.aiOnnxOpset9.slice([qkv],
                                            axes=[1],
                                            starts=[index * hidden_size],
                                            ends=[(index + 1) * hidden_size])
        tensor = builder.reshape_const(builder.aiOnnx, [tensor], comb_shape)
        perm = [0, 2, 1, 3] if not transpose else [0, 2, 3, 1]
        return builder.aiOnnx.transpose([tensor], perm=perm)

    q, kt, v = [extract_heads(qkv, i, hidden_size, i == 1) for i in range(3)]

    # Attention calculation
    with builder.nameScope('Z'):
        x = builder.aiOnnx.matmul([q, kt])

        c = builder.aiOnnx.constant(
            np.array(1 / np.sqrt(qkv_length)).astype(np.float32), "C")
        x = builder.aiOnnx.mul([x, c])

        x = builder.aiOnnx.add([x, mask], "ApplyMask")

        x = builder.aiOnnx.softmax([x], axis=-1)

        # x[batch_size, attention_heads, sequence_length, sequence_length] *
        # v[batch_size, attention_heads, sequence_length, qkv_length]
        z = builder.aiOnnx.matmul([x, v])

        # [batch_size, attention_heads, sequence_length, qkv_length] ->
        # [batch_size, sequence_length, attention_heads, qkv_length]
        z = builder.aiOnnx.transpose([z], perm=[0, 2, 1, 3])
        # [batch_size, sequence_length, attention_heads, qkv_length] ->
        # [batch_size*sequence_length, attention_heads*qkv_length]
        z = builder.reshape_const(builder.aiOnnx, [z],
                                  [sequence_length * batch_size, hidden_size])
    return z


@tu.requires_ipu_model
@pytest.mark.parametrize("explicit", [False, True])
def test_full_recompute_pipelining(explicit):
    batches_per_step = 5
    batch_size = 1
    hidden_size = 16
    sequence_length = 8
    attention_heads = 4
    qkv_length = hidden_size / attention_heads

    input_shape = [batch_size * sequence_length, hidden_size]
    mask_shape = [sequence_length]

    qkv_data = np.random.normal(
        0, 0.02, [hidden_size, hidden_size * 3]).astype(np.float32)

    r = np.arange(0, mask_shape[0])
    masks = []
    for i in range(batches_per_step):
        masks.append(np.less(r, i).astype(np.float32))
    mask_data = (1 - np.stack(masks)) * -1000.0

    input_data = np.random.normal(
        0, 0.02, [batches_per_step] + input_shape).astype(np.float32)

    def run_test(mode=None, verify=None, explicit=False):
        builder = popart.Builder(opsets={
            "ai.onnx": 9,
            "ai.onnx.ml": 1,
            "ai.graphcore": 1
        })

        mask = builder.addInputTensor(popart.TensorInfo("FLOAT", mask_shape),
                                      "mask")
        x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape),
                                      "x_in")

        qkv_1 = builder.addInitializedInputTensor(qkv_data, "qkv_1")
        qkv_2 = builder.addInitializedInputTensor(qkv_data, "qkv_2")
        qkv_3 = builder.addInitializedInputTensor(qkv_data, "qkv_3")

        # Recomp Mode Standard will reject "mask" as an stash op
        with builder.virtualGraph(0), builder.pipelineStage(0):
            o = builder.aiOnnx.matmul([x_in, qkv_1])
            o = attention_onnx(builder, o, mask, batch_size, sequence_length,
                               hidden_size, attention_heads, qkv_length)
        with builder.virtualGraph(1), builder.pipelineStage(1):
            o = builder.aiOnnx.matmul([o, qkv_2])
            o = attention_onnx(builder, o, mask, batch_size, sequence_length,
                               hidden_size, attention_heads, qkv_length)
        with builder.virtualGraph(2), builder.pipelineStage(2):
            o = builder.aiOnnx.matmul([o, qkv_3])
            o = attention_onnx(builder, o, mask, batch_size, sequence_length,
                               hidden_size, attention_heads, qkv_length)
            l1 = builder.aiGraphcore.l1loss([o], 0.1)

        proto = builder.getModelProto()

        dataFlow = popart.DataFlow(
            batches_per_step, {
                o:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + qkv_1:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + qkv_2:
                popart.AnchorReturnType("All"),
                popart.reservedGradientPrefix() + qkv_3:
                popart.AnchorReturnType("All"),
            })

        opts = popart.SessionOptions()
        opts.enableOutlining = False
        opts.enablePipelining = True
        opts.enableExplicitIR(explicit)

        if mode is not None:
            opts.autoRecomputation = mode
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        pat = popart.Patterns(popart.PatternsLevel.Default)

        with tu.create_test_device(numIpus=3, opts={"compileIPUCode":
                                                    False}) as device:
            session = popart.TrainingSession(fnModel=proto,
                                             dataFlow=dataFlow,
                                             userOptions=opts,
                                             loss=l1,
                                             optimizer=popart.ConstSGD(1e-9),
                                             patterns=pat,
                                             deviceInfo=device)

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()

            inputs = {x_in: input_data, mask: mask_data}
            stepio = popart.PyStepIO(inputs, anchors)

            for _ in range(10):
                session.run(stepio)

            if verify is not None:
                verify(session, explicit)

        return anchors

    def verify(session, explicit):
        ''' Verify the pipeline stashing is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        if explicit:
            # Explicit pipelining
            stashes = [
                op for graph in ir for op in ir[graph]
                if (op["type"] == "DynamicUpdate"
                    or op["type"] == "DynamicUpdateInplace")
            ]
        else:
            # Implicit pipelining
            stashes = [op for op in ir["maingraph"] if op["type"] == "Stash"]

        stashedTensors = [stash["inputs"][0]["name"] for stash in stashes]
        print(stashedTensors)

        # Need to be sorted such that if one name includes another,
        # the longer one appears first
        # Use regexp to catch variants occuring in explicit pipelining
        expectedStashes = [
            '(.*)Z/Reshape:0_c1', '(.*)x_in', '(.*)mask(.*)_c1', '(.*)mask'
        ]

        for expected in expectedStashes:
            found = ''
            for stash in stashedTensors:
                if re.match(expected, stash):
                    found = expected
                    # Avoid matching multiple times (see above)
                    stashedTensors.remove(stash)
                    break
            # Assert on name rather than true/false so it is more useful for
            # debugging
            assert expected == found

    # Verify inplacing due to problematic inplacing + implicit recompute relationship
        if explicit:
            # Not crucial since not using implicit recompute, therefore we do
            # not expect inplacing to be problematic
            pass
        else:
            # Verify inplacing due to problematic inplacing +
            # implicit recompute relationship
            inplaces = [
                op for op in ir["maingraph"] if "Inplace" in op["type"]
            ]

            bins = dict()
            for op in inplaces:
                if op["type"] in bins:
                    bins[op["type"]] += 1
                else:
                    bins[op["type"]] = 1

            print(bins)
            # T35025 : Adjusted due to inplace tensors being consumed by
            # implicit recompute
            assert "ReshapeInplace" in bins and bins["ReshapeInplace"] == 39
            assert "SliceInplace" in bins and bins["SliceInplace"] == 9
            assert "TransposeInplace" in bins and bins["TransposeInplace"] == 41
            assert "AddLhsInplace" in bins and bins["AddLhsInplace"] == 1
            assert "IdentityInplace" in bins and bins["IdentityInplace"] == 34
            assert "MulLhsInplace" in bins and bins["MulLhsInplace"] == 4
            assert "ConcatInplace" in bins and bins["ConcatInplace"] == 3
            assert "RestoreInplace" in bins and bins["RestoreInplace"] == 4
            assert len(inplaces) == 136

    # Keep one reference point always implicit to ascertain equality of the
    # two implementations
    n_anchors = run_test(explicit=False)
    s_anchors = run_test(popart.RecomputationType.Standard, explicit=explicit)
    p_anchors = run_test(popart.RecomputationType.Pipeline,
                         verify,
                         explicit=explicit)

    for key in s_anchors:
        assert np.allclose(n_anchors[key], s_anchors[key])
        assert np.allclose(n_anchors[key], p_anchors[key])
        assert np.allclose(s_anchors[key], p_anchors[key])


def fetch_weights(session, proto):
    weights = {}
    for i in range(len(proto.graph.initializer)):
        init = proto.graph.initializer[i]
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
        empty_init = np.empty(shape=init.dims, dtype=dtype)
        weights[init.name] = empty_init
    session.weightsToHost()
    weightsIo = popart.PyWeightsIO(weights)
    session.readWeights(weightsIo)
    return weights


def transfer_weights(session0, session1, proto):
    weights = {}
    for i in range(len(proto.graph.initializer)):
        init = proto.graph.initializer[i]
        dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]
        empty_init = np.empty(shape=init.dims, dtype=dtype)
        weights[init.name] = empty_init
    session0.weightsToHost()
    weightsIo = popart.PyWeightsIO(weights)
    session0.readWeights(weightsIo)
    session1.writeWeights(weightsIo)
    session1.weightsFromHost()


def compare(da, db):
    for name in da:
        print("Checking: ", name)
        print("diff", np.sum(da[name] - db[name]))
        assert np.array_equal(da[name], db[name])


@tu.requires_ipu_model
def test_implicit_pipelining_custom_fwd_only():
    """Test if running inference within the training session does not disturb
    training and results in the same output as running a separate inference
    session.
    """

    def run(inference_during_training):
        np.random.seed(1337)

        batches_per_step = 5
        accumulation_factor = 7
        batch_size = 1
        hidden_size = 16
        sequence_length = 8
        attention_heads = 4
        qkv_length = hidden_size / attention_heads

        input_shape = [batch_size * sequence_length, hidden_size]
        mask_shape = [sequence_length]

        qkv_data = np.random.normal(
            0, 0.02, [hidden_size, hidden_size * 3]).astype(np.float32)

        r = np.arange(0, mask_shape[0])
        masks = []
        for i in range(batches_per_step * accumulation_factor):
            masks.append(np.less(r, i).astype(np.float32))
        mask_data = (1 - np.stack(masks)) * -1000.0

        input_data = np.random.normal(
            0, 0.02,
            [batches_per_step, accumulation_factor] + input_shape).astype(
                np.float32)

        builder = popart.Builder(opsets={
            "ai.onnx": 9,
            "ai.onnx.ml": 1,
            "ai.graphcore": 1
        })

        mask = builder.addInputTensor(popart.TensorInfo("FLOAT", mask_shape),
                                      "mask")
        x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape),
                                      "x_in")

        qkv_1 = builder.addInitializedInputTensor(qkv_data, "qkv_1")
        qkv_2 = builder.addInitializedInputTensor(qkv_data, "qkv_2")
        qkv_3 = builder.addInitializedInputTensor(qkv_data, "qkv_3")

        # Recomp Mode Standard will reject "mask" as an stash op
        with builder.virtualGraph(0), builder.pipelineStage(0):
            o = builder.aiOnnx.matmul([x_in, qkv_1])
            o = attention_onnx(builder, o, mask, batch_size, sequence_length,
                               hidden_size, attention_heads, qkv_length)
        with builder.virtualGraph(1), builder.pipelineStage(1):
            o = builder.aiOnnx.matmul([o, qkv_2])
            o = attention_onnx(builder, o, mask, batch_size, sequence_length,
                               hidden_size, attention_heads, qkv_length)
        with builder.virtualGraph(2), builder.pipelineStage(2):
            o = builder.aiOnnx.matmul([o, qkv_3])
            o = attention_onnx(builder, o, mask, batch_size, sequence_length,
                               hidden_size, attention_heads, qkv_length)
            l1 = builder.aiGraphcore.l1loss([o], 0.1)

        proto = builder.getModelProto()
        onnxproto = onnx.load_model_from_string(proto)

        dataFlow = popart.DataFlow(batches_per_step, {
            o: popart.AnchorReturnType("All"),
            l1: popart.AnchorReturnType("All")
        })

        infDataFlow = popart.DataFlow(batches_per_step * accumulation_factor, {
            o: popart.AnchorReturnType("All"),
            l1: popart.AnchorReturnType("All")
        })

        opts = popart.SessionOptions()
        # Disable outlining to make debugging easier
        opts.enableOutlining = False
        opts.enablePipelining = True
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = accumulation_factor
        opts.autoRecomputation = popart.RecomputationType.Pipeline
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        # Option under test
        opts.createImplicitPipeliningFwdOnlyProgram = inference_during_training

        pat = popart.Patterns(popart.PatternsLevel.Default)

        with tu.create_test_device(numIpus=3, opts={"compileIPUCode":
                                                    False}) as device0:
            session = popart.TrainingSession(fnModel=proto,
                                             dataFlow=dataFlow,
                                             userOptions=opts,
                                             loss=l1,
                                             optimizer=popart.ConstSGD(1),
                                             patterns=pat,
                                             deviceInfo=device0)

            session.prepareDevice()

            session.weightsFromHost()

            anchors = session.initAnchorArrays()

            inputs = {x_in: input_data, mask: mask_data}
            stepio = popart.PyStepIO(inputs, anchors)

            session.run(stepio)

            # Catches any severe differences in the compiled training graph
            ws0 = fetch_weights(session, onnxproto)

            if inference_during_training:
                # Inference during training on the same session
                session.run("implicitPipeliningFwdOnly", stepio)
            else:
                # Separate inference session
                with tu.create_test_device(numIpus=3,
                                           opts={"compileIPUCode":
                                                 False}) as device1:
                    opts.enableGradientAccumulation = False
                    opts.accumulationFactor = 1
                    opts.createImplicitPipeliningFwdOnlyProgram = False
                    # Ensure weights can be updated after the fact
                    opts.constantWeights = False
                    inf_session = popart.InferenceSession(fnModel=proto,
                                                          dataFlow=infDataFlow,
                                                          userOptions=opts,
                                                          patterns=pat,
                                                          deviceInfo=device1)
                    inf_session.prepareDevice()
                    transfer_weights(session, inf_session, onnxproto)
                    inf_session.run(stepio)
            a = copy.deepcopy(anchors)

            # Catches if the inference during training has touched any weights
            ws1 = fetch_weights(session, onnxproto)

            session.run(stepio)

            # Catches if the inference during training has touched any
            # model state (such as accumulators)
            ws2 = fetch_weights(session, onnxproto)

            return [ws0, ws1, ws2, a]

    run0 = run(False)
    run1 = run(True)

    for i in range(len(run0)):
        compare(run0[i], run1[i])


@tu.requires_ipu_model
def test_implicit_pipelining_custom_fwd_only_no_copy():
    """Test if running inference within the training session does not add weight
    view copies between forward and backward pass
    """

    hidden_size = 5
    batches_per_step = 2
    accumulation_factor = 4
    input_shape = [hidden_size, hidden_size]

    data = np.random.normal(0, 0.02,
                            [hidden_size * hidden_size]).astype(np.float32)

    input_data = np.random.normal(
        0, 0.02, [batches_per_step, accumulation_factor] + input_shape).astype(
            np.float32)

    builder = popart.Builder(opsets={
        "ai.onnx": 9,
        "ai.onnx.ml": 1,
        "ai.graphcore": 1
    })

    x_in = builder.addInputTensor(popart.TensorInfo("FLOAT", input_shape),
                                  "x_in")

    w0 = builder.addInitializedInputTensor(data, "w0")
    w1 = builder.addInitializedInputTensor(data, "w1")

    w0r_shape = builder.aiOnnx.constant(
        np.array([hidden_size, hidden_size]).astype(np.int64))
    w1r_shape = builder.aiOnnx.constant(
        np.array([hidden_size, hidden_size]).astype(np.int64))

    with builder.virtualGraph(0), builder.pipelineStage(0):
        # Introduce infamous fake (outplace!) weight view change
        w0r = builder.aiOnnx.reshape([w0, w0r_shape])
        o = builder.aiOnnx.matmul([x_in, w0r])

    with builder.virtualGraph(1), builder.pipelineStage(1):
        # Introduce infamous fake (outplace!) weight view change
        w1r = builder.aiOnnx.reshape([w1, w1r_shape])
        o = builder.aiOnnx.matmul([o, w1r])
        l1 = builder.aiGraphcore.l1loss([o], 0.1)

    proto = builder.getModelProto()
    onnxproto = onnx.load_model_from_string(proto)

    dataFlow = popart.DataFlow(batches_per_step, {
        o: popart.AnchorReturnType("All"),
        l1: popart.AnchorReturnType("All")
    })

    infDataFlow = popart.DataFlow(batches_per_step * accumulation_factor, {
        o: popart.AnchorReturnType("All"),
        l1: popart.AnchorReturnType("All")
    })

    opts = popart.SessionOptions()
    # Disable outlining to make debugging easier
    opts.enableOutlining = False
    opts.enablePipelining = True
    opts.enableGradientAccumulation = True
    opts.accumulationFactor = accumulation_factor
    opts.autoRecomputation = popart.RecomputationType.Pipeline
    opts.virtualGraphMode = popart.VirtualGraphMode.Manual

    # Option under test
    opts.createImplicitPipeliningFwdOnlyProgram = True

    pat = popart.Patterns(popart.PatternsLevel.Default)

    with tu.create_test_device(numIpus=3, opts={"compileIPUCode":
                                                False}) as device0:
        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=opts,
                                         loss=l1,
                                         optimizer=popart.ConstSGD(1),
                                         patterns=pat,
                                         deviceInfo=device0)

        session.prepareDevice()

        # Check that there are only 4 copies
        # (regular activations and gradients being passed between the 4 pipeline
        # stages) between pipeline stages.
        # Without interipucopy.cpp::isWeightOrConstView, there would be 5 (an
        # additional copy that copies the view of the weight between the
        # last forward and first backward stage).
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))

        copies = [
            op for graph in ir for op in ir[graph] if (op["type"] == "IpuCopy")
        ]

        assert len(copies) == 2
