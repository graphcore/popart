# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import popdist
import pytest
import argparse
import pickle
import pathlib
import subprocess

import numpy as np

from test_util import create_test_device


def run_poprun_broadcast_weights():
    parser = argparse.ArgumentParser(description="Parse launch parameters.")
    parser.add_argument("--tmpdir")
    args = parser.parse_args()

    popdist.init()

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT", [1])
    i_1 = builder.addInputTensor(shape, debugContext="i_1")

    # Only initialize the weights on instance 0.
    initial_value = 42.0 if popdist.getInstanceIndex() == 0 else 0.0

    w_data = np.array([initial_value], dtype=np.float32)
    w_1 = builder.addInitializedInputTensor(w_data, debugContext="w_1")

    o = builder.aiOnnx.add([i_1, w_1])

    loss = builder.aiGraphcore.identityloss([o])
    proto = builder.getModelProto()
    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    optimizer = popart.ConstSGD(0.01)

    with create_test_device() as device:
        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=dataFlow,
            loss=loss,
            optimizer=optimizer,
            deviceInfo=device,
        )

        # Broadcast the weights from instance 0.
        session.broadcastWeights()

        session.prepareDevice()
        session.weightsFromHost()

        anchors = session.initAnchorArrays()

        inputs = {"i_1": np.array([0.0], dtype=np.float32)}
        stepio = popart.PyStepIO(inputs, anchors)

        session.run(stepio)

        tmp_path = pathlib.Path(args.tmpdir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        filename = f"test_poprun_broadcast_weights_{popdist.getInstanceIndex()}_{popdist.getNumInstances()}_{popdist.getNumTotalReplicas()}.pkl"

        # Save the anchors to a file, so we can access them outside of the MPI context.
        with open(str(tmp_path / filename), "wb") as handle:
            pickle.dump(anchors, handle, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize(
    "config",
    [
        {
            "num_instances": 2,
            "num_replicas": 2,
        },
        {
            "num_instances": 2,
            "num_replicas": 4,
        },
        {
            "num_instances": 4,
            "num_replicas": 4,
        },
    ],
)
def test_poprun_broadcast_weights(tmpdir, config):
    test_path = pathlib.Path(__file__).resolve()
    tmp_path = pathlib.Path(tmpdir)

    command = ["poprun"]
    command.append("--offline-mode=on")
    command.append("--num-instances")
    command.append(str(config["num_instances"]))
    command.append("--num-replicas")
    command.append(str(config["num_replicas"]))
    command.append("--mpi-global-args=--allow-run-as-root")
    command.append("python3")
    command.append(test_path)
    command.append("--tmpdir")
    command.append(tmpdir)

    # Launch this file through poprun (thus calling __main__).
    subprocess.run(command)

    anchors = []

    for instance_index in range(config["num_instances"]):
        filename = f"test_poprun_broadcast_weights_{instance_index}_{config['num_instances']}_{config['num_replicas']}.pkl"

        with open(str(tmp_path / filename), "rb") as handle:
            anchors.append(pickle.load(handle))

    # Anchors should be all equal over all instances when `popart.broadcast_weights` was used.
    assert all(anchor == anchors[0] for anchor in anchors)


if __name__ == "__main__":
    # When the test launches poprun in a subprocess, this subprocess will enter here.
    globals()["run_poprun_broadcast_weights"]()
