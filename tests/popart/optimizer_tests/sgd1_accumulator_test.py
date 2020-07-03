# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import itertools
import json

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

# In this test all the addGrads will undergo transformations to be converted
# into indenties then removed by PostNRepl. This causes the input grad to be the same
# for each sgd1combo which will give every accumulator the same name, throwing an error.
# To solve we name accumulators after their weightId, which is assumed to be unique.


def test_accumulators_names_dont_clash():
    np.random.seed(1984)

    builder = popart.Builder()

    input_data = np.random.rand(4, 4).astype(np.float32)
    weights = ['weight1', 'weight2', 'weight3']

    d0 = builder.addInputTensor(popart.TensorInfo('FLOAT', [4, 4]), 'data0')
    x = builder.aiOnnx.add([
        d0,
        builder.addInitializedInputTensor(
            np.random.rand(4, 4).astype(np.float32), weights[0])
    ])
    x = builder.aiOnnx.add([
        x,
        builder.addInitializedInputTensor(
            np.random.rand(4, 4).astype(np.float32), weights[1])
    ])
    x = builder.aiOnnx.add([
        x,
        builder.addInitializedInputTensor(
            np.random.rand(4, 4).astype(np.float32), weights[2])
    ])

    l1 = builder.aiGraphcore.l1loss([x], 1.0)

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {})

    opt = popart.SGD({
        "defaultLearningRate": (0.1, True),
        "defaultMomentum": (0.9, True),
        "defaultDampening": (0, True)
    })

    session = popart.TrainingSession(
        fnModel=proto,
        dataFlow=dataFlow,
        loss=l1,
        optimizer=opt,
        deviceInfo=tu.create_test_device(opts={"compileIPUCode": False}))

    ir = json.loads(session._serializeIr(popart.IrSerializationFormat.JSON))

    ops = ir["maingraph"]

    tensors = set()
    for op in ops:
        for i in op["inputs"]:
            tensors.add(i["name"])
        for o in op["outputs"]:
            tensors.add(o["name"])

    prefixes = [
        popart.reservedAcclPrefix(),
        popart.reservedAcclToUpdatePrefix(),
        popart.reservedAcclFinalOutPrefix()
    ]
    for prefix, weight in itertools.product(prefixes, weights):
        assert prefix + weight in tensors
