# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import onnx
import itertools

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# Parameterize over supported data types using a (numpy, onnx) datatype tuple
supported_dtypes = [
    (np.float32, onnx.helper.TensorProto.FLOAT),
    (np.float16, onnx.helper.TensorProto.FLOAT16),
]


# IPU test: check that the output of randomuniform has the expected statistical properties
@tu.requires_ipu
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomuniform_stats(op_tester, dtypes):
    expected_min = np.array(-1.0, dtype=dtypes[0])
    expected_max = np.array(1.0, dtype=dtypes[0])
    two = np.array(2.0, dtype=dtypes[0])
    expected_mean = (expected_max + expected_min) / two

    def init_builder(builder):
        ru = builder.aiOnnx.randomuniform(
            shape=[10000, 1], low=expected_min, high=expected_max, dtype=dtypes[1]
        )

        actual_min = builder.aiOnnx.reducemin([ru], keepdims=False)
        builder.addOutputTensor(actual_min)

        actual_max = builder.aiOnnx.reducemax([ru], keepdims=False)
        builder.addOutputTensor(actual_max)

        actual_mean = builder.aiOnnx.reducemean([ru], keepdims=False)
        builder.addOutputTensor(actual_mean)

        return [actual_min, actual_max, actual_mean]

    def reference(_):  # ref_data is an unused argument
        return [expected_min, expected_max, expected_mean]

    op_tester.atol = 1e-02
    op_tester.run(init_builder, reference, seed=42)


# IPU test: repeatable PRNG => given the same seed, the op outputs the same data.
@tu.requires_ipu
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomuniform_repeatable(dtypes):
    seed = 8
    builder = popart.Builder()
    out = builder.aiOnnx.randomuniform(shape=[10, 1], dtype=dtypes[1])
    builder.addOutputTensor(out)

    with tu.create_test_device() as device:
        session = popart.InferenceSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
            patterns=popart.Patterns(popart.PatternsLevel.All),
            deviceInfo=device,
        )

        session.prepareDevice()
        session.setRandomSeed(seed)
        session.weightsFromHost()

        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({}, anchors)
        session.run(stepio)

        # need to copy the anchor as the next call to run will overwrite the data
        run1_out = np.copy(anchors[out])

        # Reset the seed to the same value and run the session again
        session.setRandomSeed(seed)
        session.run(stepio)
        run2_out = np.copy(anchors[out])

        assert np.array_equal(run1_out, run2_out)


# IPU test: check that having multiple randomuniform ops will sample from independent PRNG streams
@tu.requires_ipu
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomuniform_multi(op_tester, dtypes):
    np_zero = np.array(0.0, dtype=dtypes[0])

    def init_builder(builder):
        # Model evaluates:
        #   sum( |x1 - x2| ) > 0
        #  for tensors x1, x2 of the same shape generated by randomuniform op
        ru1 = builder.aiOnnx.randomuniform(
            shape=[10, 1], dtype=dtypes[1], debugContext="ru1"
        )
        ru2 = builder.aiOnnx.randomuniform(
            shape=[10, 1], dtype=dtypes[1], debugContext="ru2"
        )
        diff = builder.aiOnnx.sub([ru1, ru2])
        loss = builder.aiOnnx.reducel1([diff], keepdims=False)
        zero = builder.addInputTensor(np_zero)
        out = builder.aiOnnx.greater([loss, zero])
        builder.addOutputTensor(out)
        return [out]

    def reference(_):  # ref_data is an unused argument
        return [np.array(True)]

    op_tester.run(init_builder, reference, seed=42)


@tu.requires_ipu
@pytest.mark.parametrize("dtypes", supported_dtypes)
def test_randomuniform_repeatable_replica(dtypes):
    seed = 8
    replication_factor = 2
    builder = popart.Builder()
    out = builder.aiOnnx.randomuniform(shape=[10, 1], dtype=dtypes[1])
    loss = builder.aiGraphcore.identityloss([out])

    builder.addOutputTensor(loss)

    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = replication_factor

    with tu.create_test_device(replication_factor) as device:

        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {out: popart.AnchorReturnType("All")}),
            patterns=popart.Patterns(popart.PatternsLevel.All),
            deviceInfo=device,
            userOptions=opts,
            optimizer=popart.ConstSGD(0.1),
            loss=loss,
        )

        session.prepareDevice()
        session.setRandomSeed(seed)
        session.weightsFromHost()

        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({}, anchors)
        session.run(stepio)

        o = anchors[out]
        for ai, bi in itertools.combinations([i for i in range(replication_factor)], 2):
            print(f"Checking o[{ai}] is not equal to o[{bi}]")
            a = o[ai]
            b = o[bi]
            assert not np.allclose(a, b)


# CPU test: check error when setting the optional seed
def test_randomuniform_seederror(op_tester):
    def init_builder(builder):
        out = builder.aiOnnx.randomuniform(shape=[1000, 1], seed=1.0)
        builder.addOutputTensor(out)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None)

    assert e_info.value.args[0].endswith(
        "Optional seed attribute is not supported. Use session::setRandomSeed instead."
    )


# CPU test: check error with unsupported data type
def test_randomuniform_bad_dtype(op_tester):
    def init_builder(builder):
        out = builder.aiOnnx.randomuniform(
            shape=[1000, 1], dtype=onnx.helper.TensorProto.DOUBLE
        )
        builder.addOutputTensor(out)
        return [out]

    with pytest.raises(popart.popart_exception) as e_info:
        op_tester.run(init_builder, None)

    assert "The data type DOUBLE is not supported in Poplar" in e_info.value.args[0]
