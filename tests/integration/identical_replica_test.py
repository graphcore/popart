# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# 'import test_util' requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import popart
import test_util as tu


def test_replica_bitwise_identical_update():
    def model():
        np.random.seed(1984)
        input_data = np.random.rand(2, 4).astype(np.float16)
        weight_data = np.random.rand(4, 4).astype(np.float16)

        builder = popart.Builder()

        d0 = builder.addInputTensor(
            popart.TensorInfo('FLOAT16', input_data.shape), 'data0')

        w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
        x = builder.aiOnnx.matmul([d0, w0])

        loss = builder.aiGraphcore.l1loss([x], 0.1, debugContext='loss')

        return builder.getModelProto(), {d0: input_data}, w0, loss

    def run_test():
        proto, data, w, loss = model()

        options = popart.SessionOptions()
        optimizer = popart.ConstSGD(0.1)
        options.enableStochasticRounding = True
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = 2
        options.engineOptions = {"target.deterministicWorkers": "portable"}

        data = {k: np.repeat(v[np.newaxis], 2, 0) for k, v in data.items()}

        device = tu.create_test_device(2, pattern=popart.SyncPattern.Full)

        w_updated = popart.reservedUpdatedVarPrefix() + w

        dataFlow = popart.DataFlow(
            1, {
                loss: popart.AnchorReturnType("ALL"),
                w_updated: popart.AnchorReturnType("FINAL")
            })

        session = popart.TrainingSession(fnModel=proto,
                                         dataFlow=dataFlow,
                                         userOptions=options,
                                         loss=loss,
                                         optimizer=optimizer,
                                         deviceInfo=device)

        session.prepareDevice()

        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        stepio = popart.PyStepIO(data, anchors)

        session.run(stepio)

        device.detach()

        return anchors[w_updated]

    results = run_test()
    assert np.all(results[0] == results[1])
