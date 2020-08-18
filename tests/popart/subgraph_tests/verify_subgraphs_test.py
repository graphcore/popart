# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_verify_subgraph():
    builder = popart.Builder()

    # sg0
    sg0_builder = builder.createSubgraphBuilder()
    sg0_i0 = sg0_builder.addUntypedInputTensor()
    # LSTM is not outlinable
    sg0_out, _, _ = sg0_builder.aiOnnx.lstm([sg0_i0, sg0_i0, sg0_i0],
                                            3,
                                            clip=None)
    sg0_builder.addOutputTensor(sg0_out)

    # main
    i0 = builder.addInputTensor(popart.TensorInfo("FLOAT", [2, 2, 2]))
    out = builder.aiGraphcore.call([i0], 1, sg0_builder)[0]

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
            deviceInfo=tu.create_test_device())
    assert (e_info.value.args[0].endswith("are not outlineable"))


def test_subgraph_conv():
    """Test a matmul inside a subgraph vs 'regular' and asserts if they give the same output shape.
    Also ensures the conv taking the subgraph as an input does not cause any issues."""
    BATCH_SIZE = 1
    NUM_IN_CHANNELS = 80
    NUM_OUT_CHANNELS = 128
    TEMPORAL_DIM = 10

    def get_subbuilder(builder):
        """ subbuilder to do matmul of w X x"""
        subb = builder.createSubgraphBuilder()
        x = subb.addInputTensor(
            popart.TensorInfo(
                "FLOAT", [BATCH_SIZE, NUM_IN_CHANNELS, TEMPORAL_DIM]))  # input
        w = subb.addInputTensor(
            popart.TensorInfo(
                "FLOAT", [NUM_OUT_CHANNELS, NUM_IN_CHANNELS]))  # weight matrix
        o = subb.aiOnnx.matmul([w, x])  # Weight * input
        subb.addOutputTensor(o)
        return subb

    def test_main_builder(use_subbuilder_for_matmul=False):
        """ param: use_subbuilde_for_matmul - if True the subbuilder is used for matmul
        and regular matmul is used otherwise """
        builder = popart.Builder()
        if use_subbuilder_for_matmul:
            subbuilder = get_subbuilder(builder)

        x = builder.addInputTensor(
            popart.TensorInfo(
                "FLOAT", [BATCH_SIZE, NUM_IN_CHANNELS, TEMPORAL_DIM]))  # input
        w1 = builder.addInputTensor(
            popart.TensorInfo(
                "FLOAT", [NUM_OUT_CHANNELS, NUM_IN_CHANNELS]))  # weight matrix
        if use_subbuilder_for_matmul:
            o1 = builder.aiGraphcore.call([x, w1], 1, callee=subbuilder)[0]
        else:
            o1 = builder.aiOnnx.matmul([w1, x])  # Weight * input
        print("Shape of output-1:")
        print(builder.getTensorShape(o1))

        w2 = builder.addInitializedInputTensor(
            np.zeros((NUM_OUT_CHANNELS, NUM_OUT_CHANNELS)).astype(np.float32),
            'w3')
        # if this matmul is also replaced by a sub-builder call, the graph build succeeds
        # the graph build previously failed when there is was a matmul between a subbuilder call and a conv operation.
        # D25221 should have fixed this.
        o2 = builder.aiOnnx.matmul([w2, o1])

        o2 = builder.aiOnnx.unsqueeze([o2], axes=[3])
        conv_weights = builder.aiOnnx.constant(
            np.zeros([NUM_OUT_CHANNELS, NUM_OUT_CHANNELS, 3,
                      1]).astype(np.float32))
        conv_out = builder.aiOnnx.conv([o2, conv_weights],
                                       dilations=[1, 1],
                                       kernel_shape=[3, 1],
                                       strides=[1, 1],
                                       pads=[1, 0, 1, 0])

        conv_out = builder.aiOnnx.squeeze([conv_out], axes=[3])
        print("Shape of conv-out")
        print(builder.getTensorShape(conv_out))

        return builder.getTensorShape(o1)

    # # we build the graph successfully for this option
    print("Testing graph build without sub-builder")
    a1 = test_main_builder(use_subbuilder_for_matmul=False)

    # the graph build fails for this option
    print("Testing graph build with sub-builder")
    a2 = test_main_builder(use_subbuilder_for_matmul=True)

    assert a1 == a2
