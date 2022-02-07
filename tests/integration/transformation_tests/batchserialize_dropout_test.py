# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import numpy as np
import pytest
import popart

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


# TODO: Fix with T29800
@pytest.mark.skip(reason="T29800 required for batch serialized dropouts")
def test_batchserialisation_dropout():
    def model():
        np.random.seed(1984)
        input_data = np.random.rand(4, 100).astype(np.float32)
        weight_data = np.random.rand(100, 100).astype(np.float32)

        builder = popart.Builder()

        d0 = builder.addInputTensor(
            popart.TensorInfo('FLOAT', input_data.shape), 'data0')

        w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
        x = builder.aiOnnx.matmul([d0, w0])
        x1 = builder.aiOnnx.dropout([x], 1, debugContext="dropout0")[0]

        x2 = builder.aiOnnx.dropout([x], 1, debugContext="dropout1")[0]

        o = builder.aiOnnx.add([x1, x2])

        loss = builder.aiGraphcore.l1loss([o], 0.1, debugContext='loss')

        return builder.getModelProto(), {d0: input_data}, [x1, x2], loss

    def run_test(enableOutlining):
        proto, data, xs, loss = model()

        options = popart.SessionOptions()
        patterns = popart.Patterns()

        optimizer = popart.SGD({"defaultLearningRate": (0.1, True)})

        options.enableOutlining = enableOutlining
        options.autoRecomputation = popart.RecomputationType.Standard
        options.explicitRecomputation = True
        options.batchSerializationSettings.factor = 4

        device = tu.create_test_device(1, pattern=popart.SyncPattern.Full)

        dataFlow = popart.DataFlow(
            1, {x: popart.AnchorReturnType("ALL")
                for x in xs})

        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=options,
                                         loss=loss,
                                         optimizer=optimizer,
                                         patterns=patterns,
                                         deviceInfo=device)

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        stepio = popart.PyStepIO(data, anchors)

        session.run(stepio)

        device.detach()
        return [anchors[x] for x in xs]

    for enableOutlining in [False, True]:
        dropout0, dropout1 = run_test(enableOutlining)

        # Check all dropout patterns are different
        batch_outputs = [t for output in [dropout0, dropout1] for t in output]
        for i, t1 in enumerate(batch_outputs):
            for j, t2 in enumerate(batch_outputs):
                if i == j:
                    continue
                assert np.any((t1 == 0) != (t2 == 0))
