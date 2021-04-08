# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import pprint
import json
import platform

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
def test_full_recompute_pipelining(tmpdir):
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

    def run_test(mode=None, verify=None):
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
        if mode is not None:
            opts.autoRecomputation = mode
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual

        pat = popart.Patterns(popart.PatternsLevel.Default)

        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=opts,
                                         loss=l1,
                                         optimizer=popart.ConstSGD(1e-9),
                                         patterns=pat,
                                         deviceInfo=tu.create_test_device(
                                             numIpus=3,
                                             opts={"compileIPUCode": False}))

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        inputs = {x_in: input_data, mask: mask_data}
        stepio = popart.PyStepIO(inputs, anchors)

        for _ in range(10):
            session.run(stepio)

        if verify is not None:
            verify(session)

        return anchors

    def verify(session):
        ''' Verify the the matmul in the main graphs is correct'''
        ir = json.loads(session._serializeIr(
            popart.IrSerializationFormat.JSON))
        stashes = [op for op in ir["maingraph"] if op["type"] == "Stash"]
        stashedTensors = [stash["inputs"][0]["name"] for stash in stashes]

        assert 'x_in' in stashedTensors
        assert 'mask' in stashedTensors
        assert 'mask_c1' in stashedTensors

        # Verify inplacing
        inplaces = [op for op in ir["maingraph"] if "Inplace" in op["type"]]

        bins = dict()
        for op in inplaces:
            if op["type"] in bins:
                bins[op["type"]] += 1
            else:
                bins[op["type"]] = 1

        print(bins)
        # T35025 : Adjusted due to inplace tensors being consumed by implicit recompute
        assert "ReshapeInplace" in bins and bins["ReshapeInplace"] == 39
        assert "SliceInplace" in bins and bins["SliceInplace"] == 9
        assert "TransposeInplace" in bins and bins["TransposeInplace"] == 41
        assert "AddLhsInplace" in bins and bins["AddLhsInplace"] == 1
        assert "IdentityInplace" in bins and bins["IdentityInplace"] == 34
        assert "MulLhsInplace" in bins and bins["MulLhsInplace"] == 4
        assert "ConcatInplace" in bins and bins["ConcatInplace"] == 3
        assert "RestoreInplace" in bins and bins["RestoreInplace"] == 4
        assert len(inplaces) == 136

    n_anchors = run_test()
    s_anchors = run_test(popart.RecomputationType.Standard)
    p_anchors = run_test(popart.RecomputationType.Pipeline, verify)

    for key in s_anchors:
        assert np.allclose(n_anchors[key], s_anchors[key])
        assert np.allclose(n_anchors[key], p_anchors[key])
        assert np.allclose(s_anchors[key], p_anchors[key])
