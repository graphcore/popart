# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import numpy as np
import popart

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_training_with_var_update():

    # Use a basic training model
    builder = popart.Builder()

    input_shape = popart.TensorInfo("FLOAT", [1, 2, 4, 4])
    weight_data = np.ones([3, 2, 3, 3], np.float32)
    input = builder.addInputTensor(input_shape)
    weights = builder.addInitializedInputTensor(weight_data)
    act = builder.aiOnnx.conv([input, weights],
                              dilations=[1, 1],
                              pads=[1, 1, 1, 1],
                              strides=[1, 1])
    o = builder.aiOnnx.relu([act])

    l1 = builder.aiGraphcore.l1loss([o], 0.1)

    # Add a variable which should be incremented by 1 every time

    counter = builder.addInitializedInputTensor(np.zeros([1], np.float32))
    one = builder.aiOnnx.constant(np.asarray([1], dtype=np.float32))
    counter_added = builder.aiOnnx.add([counter, one])

    copy_alias = builder.aiGraphcore.copyvarupdate([counter, counter_added])

    anchor_names = [o, counter_added, copy_alias]

    training_dataFlow = popart.DataFlow(
        1, {
            anchor_names[0]: popart.AnchorReturnType("All"),
            anchor_names[1]: popart.AnchorReturnType("All"),
            anchor_names[2]: popart.AnchorReturnType("All"),
        })

    opts = popart.SessionOptions()
    opts.constantWeights = False  # Allow the weights to be updated

    # Create the device
    device = tu.create_test_device(numIpus=1)

    # Prepare the input data
    input_data = np.ones(input_shape.shape(), dtype=np.float32)

    # Prepare the Training session
    training_session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                              dataFlow=training_dataFlow,
                                              loss=l1,
                                              optimizer=popart.ConstSGD(0.01),
                                              userOptions=opts,
                                              deviceInfo=device)

    # training_session = popart.InferenceSession(fnModel=builder.getModelProto(),
    #                                            dataFlow=training_dataFlow,
    #                                            userOptions=opts,
    #                                            deviceInfo=device)

    training_session.prepareDevice()
    training_session.weightsFromHost()

    training_anchors = training_session.initAnchorArrays()
    training_inputs = {input: input_data}

    for i in range(4):
        training_session.run(popart.PyStepIO(training_inputs,
                                             training_anchors))
        assert training_anchors[counter_added][0] == i + 1
