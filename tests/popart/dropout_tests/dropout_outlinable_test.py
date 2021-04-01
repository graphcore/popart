# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import pprint
import json
import onnx
from onnx import numpy_helper

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_dropout_outlinable(tmpdir):
    N = 20
    K = 40
    layers = 6
    dropoutRatio = 0.7

    def model():
        np.random.seed(253)
        input_data = np.random.rand(1, N, K).astype(np.float32)
        weight_data = np.random.rand(1, K, K).astype(np.float32)

        builder = popart.Builder()
        d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', (1, N, K)),
                                    'data0')
        x = d0

        for i in range(layers):
            w = builder.addInitializedInputTensor(weight_data,
                                                  debugContext='weight_%d' % i)
            x = builder.aiOnnx.matmul([x, w], debugContext='matmul_%d' % i)
            [x] = builder.aiOnnx.dropout([x],
                                         debugContext='dropout_%d' % i,
                                         num_outputs=1,
                                         ratio=dropoutRatio)

        loss = builder.aiGraphcore.l1loss([x], 0.1, debugContext='loss')
        return builder.getModelProto(), {d0: input_data}, x, loss

    def run_test():
        proto, data, x, loss = model()
        options = popart.SessionOptions()

        dataFlow = popart.DataFlow(1, {x: popart.AnchorReturnType("ALL")})
        with tu.create_test_device() as device:
            session = popart.TrainingSession(fnModel=proto,
                                             dataFlow=dataFlow,
                                             deviceInfo=device,
                                             userOptions=options,
                                             loss=loss,
                                             optimizer=popart.ConstSGD(0.01))

            session.prepareDevice()
            session.weightsFromHost()
            anchors = session.initAnchorArrays()
            stepio = popart.PyStepIO(data, anchors)
            session.run(stepio)
            ir = json.loads(
                session._serializeIr(popart.IrSerializationFormat.JSON))
            dropouts = [
                op for op in ir['maingraph'] if op['type'] == 'Dropout'
            ]
            assert (len(dropouts) <= 2)
            device.detach()
            return anchors[x]

    outputs_run1 = run_test()
