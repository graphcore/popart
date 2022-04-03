# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, List
from tempfile import TemporaryDirectory
import numpy as np
import pytest
import popart
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

BATCHES_PER_STEP = 7
BATCH_SIZE = 16
CHANNELS = 2
DATA_LEN = 3
O_DIM = 2
IPU_MAX = 16

# Learning rate 1 for easy comparison.
LEARNING_RATE = 1.0
TEMPFILE = "temporary_file"


def check_device(device, repl_config):
    if device is None:
        ipus = (int(repl_config["ipus"]) * int(repl_config["repl"]))
        fail_ceiling = 4

        if (ipus > fail_ceiling):
            pytest.skip(
                f"Test needs to run on {ipus} IPU(s), but sufficient IPU(s) were not available. "
                "As the requirement of this test was higher than {fail_ceiling} ipus it fails silently."
            )
        else:
            pytest.fail(
                f"Test needs to run on {ipus} IPU(s), but sufficient IPU(s) were not available. "
                "As the requirements of this test was {fail_ceiling} ipus or less, this is considered a fail."
            )


def getOffChipLocation(commGroup, RTS=False):
    rts = popart.ReplicatedTensorSharding.On \
          if RTS else \
          popart.ReplicatedTensorSharding.Off
    return popart.TensorLocationSettings(
        location=popart.TensorLocation(storage=popart.TensorStorage.OffChip,
                                       loadTileSet=popart.TileSet.Compute,
                                       storageTileSet=popart.TileSet.Compute,
                                       replicatedTensorSharding=rts,
                                       shardingDomain=commGroup),
        minElementsForOffChip=0,
        minElementsForReplicatedTensorSharding=2)


KEY_PARSE = {
    "commType": {
        "All": popart.CommGroupType.All,
        "Consecutive": popart.CommGroupType.Consecutive,
        "Orthogonal": popart.CommGroupType.Orthogonal,
        "None": popart.CommGroupType.Ungrouped
    },
    "retrieval": {
        "OnePerGroup": popart.VariableRetrievalMode.OnePerGroup,
        "AllReduce": popart.VariableRetrievalMode.AllReduceReplicas,
        "AllReplicas": popart.VariableRetrievalMode.AllReplicas
    }
}


def is_default_settings(var_set: popart.VariableSettings):
    return (var_set.getSharedVariableDomain().type == popart.CommGroupType.All) and \
           (var_set.getSharedVariableDomain().replicaGroupSize == 0) and \
           (var_set.getRetrievalMode() == popart.VariableRetrievalMode.OnePerGroup)


def get_variable_settings(repl_config):
    vs = popart.VariableSettings(
        popart.CommGroup(
            KEY_PARSE["commType"][
                repl_config["commType"]],  # How the groups are configured
            int(repl_config["commSize"])),  # Size of groups
        KEY_PARSE["retrieval"][
            repl_config["retrieval"]])  # How much and what to retrieve
    return vs


def get_group_idxs(config, var_set, returned):
    if var_set.getSharedVariableDomain().type == popart.CommGroupType.All:
        return [[i for i in range(int(config["repl"]))]]

    group_size = var_set.getSharedVariableDomain().replicaGroupSize
    groups = int (returned if (group_size is 0) else \
                  int(config["repl"]) / group_size)
    group_inc = 1 if var_set.getSharedVariableDomain().type \
                     != popart.CommGroupType.Orthogonal \
                  else group_size
    group_idxs = []
    for group in range(groups):
        domain = []
        gr = var_set.getGroupRepresentative(group)
        for id in range(gr, gr + (group_inc * group_size), group_inc):
            domain.append(id)
        group_idxs.append(domain)
    return group_idxs


def get_weights_array(shape: List[int], groups=[], seed=10111) -> np.array:
    np.random.seed(seed)
    reshape = []
    if len(groups) > 1:
        shape = [len(groups)] + shape

    array = np.linspace(0, 1, np.prod(shape)).astype(np.float32).reshape(shape)
    return array


# Configure SessionOptions from Repl-Config
def user_options(repl_config, location):
    opts = popart.SessionOptions()

    opts.replicatedGraphCount = int(repl_config["repl"])
    opts.enableReplicatedGraphs = int(repl_config["repl"]) != 1

    cg = popart.CommGroup(KEY_PARSE["commType"][repl_config["commType"]],
                          int(repl_config["commSize"]))

    if (location["remote"]):
        opts.weightTensorLocationSettings = getOffChipLocation(
            cg, location["RTS"])

    ipus = int(repl_config["ipus"]) * int(repl_config["repl"])

    if tu.ipu_available(ipus):
        device = tu.create_test_device(numIpus=ipus)
    else:
        print("No IPUS available for test options.")
        return None, None

    return opts, device


"""
Simple Model
"""


def get_model(repl_config: Dict,
              builder,
              groups,
              var_settings,
              location,
              session_type: str,
              initialize=True,
              opts=None,
              device=None):
    batches_per_step = BATCHES_PER_STEP if session_type == "training" else 1
    input_shape = [O_DIM, CHANNELS, DATA_LEN, DATA_LEN]

    # Create an appropriate underlying weight-array
    if initialize:
        arrays_one = get_weights_array(input_shape, seed=1337)
        arrays_two = get_weights_array(input_shape, groups)
        arrays_two_init = get_weights_array(input_shape, groups)
    else:
        re_input_shape = [len(groups)].extend(input_shape).\
                          reshape([-1].extend(input_shape))
        arrays_one = np.zeros(input_shape)
        arrays_two = np.zeros(re_input_shape)
        arrays_two_init = np.zeros(re_input_shape)

    assert np.allclose(arrays_two, arrays_two_init)

    # shapes
    data_shape = popart.TensorInfo("FLOAT", input_shape)
    lbl_shape = popart.TensorInfo("INT32", [O_DIM])
    label_shape = [O_DIM]

    # This is the tested function
    # weights_array has a distributed shape
    # var_settings hold the internal configuration
    weights_one = builder.addInitializedInputTensor(arrays_one)
    weights_two = builder.addInitializedInputTensor(arrays_two, var_settings)

    # Add input tensors
    ip = builder.addInputTensor(data_shape, "main_input_123")
    lb = builder.addInputTensor(lbl_shape, "label_input_456")

    #
    a = builder.aiOnnx.matmul([ip, weights_one])
    b = builder.aiOnnx.matmul([a, weights_two])
    o = builder.reshape_const(builder.aiOnnx, [b],
                              [O_DIM, CHANNELS * DATA_LEN * DATA_LEN])
    o = builder.aiOnnx.relu([o])
    o = builder.aiOnnx.softmax([o])

    if (session_type == "training"):
        lb = builder.addInputTensor(lbl_shape, "label_input_456")
        nll = builder.aiGraphcore.nllloss([o, lb])
    else:
        lb = None

    art = popart.AnchorReturnType("All")
    data_flow = popart.DataFlow(batches_per_step, {o: art})

    #
    if (location["remote"] and location["RTS"]
            and int(repl_config["c.idx"]) > 1):
        loc = getOffChipLocation(popart.CommGroup(popart.CommGroupType.All, 0),
                                 location["RTS"]).location
        opts.tensorLocationSettingsOverride[weights_one] = loc

    if device is None:
        ipus = (int(repl_config["ipus"]) * int(repl_config["repl"]))
        fail_ceiling = 4

        if (ipus > fail_ceiling):
            pytest.skip(
                f"Test needs to run on {ipus} IPU(s), but sufficient IPU(s) were not available. "
                "As the requirement of this test was higher than {fail_ceiling} ipus it fails silently."
            )
        else:
            pytest.fail(
                f"Test needs to run on {ipus} IPU(s), but sufficient IPU(s) were not available. "
                "As the requirements of this test was {fail_ceiling} ipus or less, this is considered a fail."
            )

    if session_type == "training":
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=data_flow,
            userOptions=opts,
            loss=nll,
            deviceInfo=device,
            optimizer=popart.ConstSGD(LEARNING_RATE))
    elif session_type == "inference":
        session = popart.InferenceSession(fnModel=builder.getModelProto(),
                                          dataFlow=data_flow,
                                          userOptions=opts,
                                          deviceInfo=device)
    else:
        pytest.fail("Unsupported session type: ", session_type)

    session.prepareDevice()

    if int(repl_config["repl"]) > 1:
        input_shape = [int(repl_config["repl"])] + input_shape
        label_shape = [int(repl_config["repl"])] + label_shape
    if batches_per_step > 1:
        input_shape = [batches_per_step] + input_shape
        label_shape = [batches_per_step] + label_shape

    tensors = [ip, lb, weights_one, weights_two]
    arrays = [arrays_one, arrays_two]

    return session, tensors, arrays, input_shape, label_shape


def run_model(session, io):
    ## load and run
    session.run(io)
    session.weightsToHost()


def verify(config, var_set, weights, groups):
    if not is_default_settings(var_set):
        # Make sure different groups have different values
        for g, group in enumerate(groups):
            for g2, group2 in enumerate(groups):
                if g < g2:
                    if var_set.getRetrievalMode(
                    ) == popart.VariableRetrievalMode.AllReplicas:
                        assert not np.allclose(weights[group[0]],
                                               weights[group2[0]])
                    else:
                        assert not np.allclose(weights[g], weights[g2])
        # Make sure members of the same group have the same value
        if (config["retrieval"] == "AllReplicas"):
            for g, group in enumerate(groups):
                for br in range(1, len(group)):
                    assert np.allclose(weights[group[0]], weights[group[br]])


# Default Replication Config
configs = [
    ## Configs that test standard cases
    {  # Default simple configuration
        "c.idx": '0',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '1',  # Replication Factor
        "commType": "All",
        "commSize": '0',
        "retrieval": "OnePerGroup",
        "locations": ["On-Chip", "OffChip"]
    },
    {  # Multiple replicas in one group, returning one
        "c.idx": '1',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "All",
        "commSize": '0',
        "retrieval": "OnePerGroup",
        "locations": ["On-Chip", "OffChip", "OffChip-Sharded"]
    },
    ## New variant returns
    {  # Grouped replicas returning all
        "c.idx": '2',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "All",
        "commSize": '0',
        "retrieval": "AllReplicas",
        "locations": ["On-Chip", "OffChip", "OffChip-Sharded"]
    },
    {  # Two ungrouped replicas, returning one each.
        "c.idx": '3',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "None",
        "commSize": '0',
        "retrieval": "OnePerGroup",
        "locations": ["On-Chip", "OffChip"]
    },
    {  # Four replicas devided in two groups
        "c.idx": '4',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '4',  # Replication Factor
        "commType": "Consecutive",
        "commSize": '2',
        "retrieval": "OnePerGroup",
        "locations": ["On-Chip", "OffChip", "OffChip-Sharded"]
    },
    {  # Four replicas devided in two groups
        "c.idx": '5',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '4',  # Replication Factor
        "commType": "Orthogonal",
        "commSize": '2',
        "retrieval": "OnePerGroup",
        "locations": ["On-Chip", "OffChip"]
        # would fail on RTS
    },
    {  # Four replicas devided in two groups
        "c.idx": '6',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '4',  # Replication Factor
        "commType": "Consecutive",
        "commSize": '2',
        "retrieval": "AllReplicas",
        "locations": ["On-Chip", "OffChip", "OffChip-Sharded"]
    },
    {  # Four replicas devided in two groups
        "c.idx": '7',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '4',  # Replication Factor
        "commType": "Orthogonal",
        "commSize": '2',
        "retrieval": "AllReplicas",
        "locations": ["On-Chip", "OffChip"]
        # would fail on RTS
    },
    {  # Cons 1
        "c.idx": '8',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "Consecutive",
        "commSize": '1',
        "retrieval": "OnePerGroup",
        "locations": ["On-Chip", "OffChip", "OffChip-Sharded"]
    },
    {  # test to verify replica-id is calculated correctly in Devicex.cpp
        "c.idx": '9',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '16',  # Replication Factor
        "commType": "Orthogonal",
        "commSize": '4',
        "retrieval": "AllReplicas",
        "locations": ["On-Chip"]
    }
]
session_types = ["training", "inference"]

remote_config = [{
    "desc": "On-Chip",
    "remote": False,
    "RTS": False
}, {
    "desc": "OffChip",
    "remote": True,
    "RTS": False
}, {
    "desc": "OffChip-Sharded",
    "remote": True,
    "RTS": True
}]

run_config = []  # debug tool, run custom set
skp_config = []  # debug tool, add to skip

if (len(run_config) == 0):
    run_config = range(len(configs))


@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("location", remote_config)
@pytest.mark.parametrize("session_type", session_types)
@tu.requires_ipu
def test_grouped_initialization(config, location, session_type):
    var_set = get_variable_settings(config)
    returned = var_set.numReplicasReturningVariable(int(config["repl"]))
    groups = [] if returned == 1 else get_group_idxs(config, var_set, returned)

    # skip if unimplemented
    if (config["c.idx"] == configs[0]["c.idx"]):
        print()
    # skip if unimplemented
    if (int(config["c.idx"]) not in run_config
            or int(config["c.idx"]) in skp_config):
        pytest.skip()
    # skip if we are using too many ipus
    if (IPU_MAX < int(config["repl"]) * int(config["ipus"])):
        pytest.skip("The test requires more IPU's than recommended.")
    #skip if the config is not supported on the location
    if (location["desc"] not in config["locations"]):
        pytest.skip(
            "This config is not supported with the given location/RTS:" +
            location["desc"])
    if (session_type == "inference" and var_set.numReplicasReturningVariable(
            int(config["repl"])) == 1):
        pytest.skip()
    if (session_type == "inference" and location["RTS"]):
        pytest.skip()

    # Get and run session

    builder = popart.Builder()
    opts, deviceContext = user_options(config, location)
    check_device(deviceContext, config)
    with deviceContext as device:
        session, tensors, arrays, input_shape, label_shape = get_model(
            config,
            builder,
            groups,
            var_set,
            location,
            session_type,
            device=device,
            opts=opts)

        #input, label, control-variable, test-tensor
        ip, lb, w1, w2 = tensors
        arrays_one, arrays_two = arrays
        buffer_one = np.zeros(arrays_one.shape).astype(np.float32)
        buffer_two = np.zeros(arrays_two.shape).astype(np.float32)

        if (config["retrieval"] == "AllReplicas"):
            shape = buffer_one.shape
            reshape = buffer_two.shape
            buffer_two = np.repeat(buffer_two.reshape((1, ) + reshape),
                                   len(groups[0]), 0)
            reshape = (len(groups) * len(groups[0]), ) + shape
            buffer_two = buffer_two.reshape(reshape)

        weightsIo = popart.PyWeightsIO({w1: buffer_one, w2: buffer_two})

        session.weightsFromHost()
        session.weightsToHost()

        # By overwriting the arrays_two before reading from device
        # we can verify that the read-write works accuratly. Not a
        # random opp.

        session.readWeights(weightsIo)

        # read worked correctly between
        if session_type == "training":
            assert np.allclose(arrays_one, buffer_one)

        if (config["retrieval"] != "AllReplicas"):
            sample = range(0, buffer_two.shape[0])
            assert np.allclose(arrays_two, buffer_two[sample])
        else:
            sample = range(0, buffer_two.shape[0], len(groups[0]))
            if (config["commType"] == "Orthogonal"):
                sample = range(0, len(groups[0]))
            arr_flat = arrays_two.flatten()
            buf_flat = buffer_two[sample].flatten()
            verify(config, var_set, buffer_two, groups)
            assert np.allclose(arr_flat, buf_flat)

        # Verify loading and unloading is correct
        # assert np.allclose(arrays_two, buffer_two)

        # In the cases where returned != 1, w1 and w2 will have different shapes
        # w2.shape = [returned, ...] <- where ... = w1.shape
        #
        # Here we assert that these shapes are correct after read
        if returned == 1:
            assert arrays_one.shape == buffer_two.shape
            assert builder.getTensorShape(w1) == builder.getTensorShape(w2)
        else:
            assert returned == buffer_two.shape[0]
            assert arrays_one.shape == buffer_two.shape[1:]

        for step in range(3):
            in_array = np.random.random_sample(input_shape).astype(np.float32)
            if (session_type == "training"):
                label_array = np.random.randint(low=0,
                                                high=20,
                                                size=label_shape).astype(
                                                    np.int32)
                io = {ip: in_array, lb: label_array}
            else:
                io = {ip: in_array}

            anchors = session.initAnchorArrays()
            stepIo = popart.PyStepIO(io, anchors)

            run_model(session, stepIo)

            session.readWeights(weightsIo)

            # Only verify this the first time, should be redundant later
            if (step == 0 and session_type == "training"):
                # checks that the weights change
                assert not np.allclose(arrays_two, buffer_two[sample])

                # checks that the gradient is different from different groups

            verify(config, var_set, buffer_two, groups)


onnx_tests = [1, 2, 3, 4, 5]
onnx_configs = []
for test in onnx_tests:
    onnx_configs += [configs[test]]


@pytest.mark.parametrize("config", onnx_configs)
@tu.requires_ipu
def test_onnx_checkpointing(config):
    if onnx_configs[0]["c.idx"] == int(config["c.idx"]):
        print()

    location = remote_config[0]
    with TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "model.onnx")

        var_set = get_variable_settings(config)
        returned = var_set.numReplicasReturningVariable(int(config["repl"]))
        groups = [] if returned == 1 else get_group_idxs(
            config, var_set, returned)

        builder = popart.Builder()
        builder.embedReplicationFactor(int(config["repl"]))
        opts, deviceContext = user_options(config, location)
        check_device(deviceContext, config)
        with deviceContext as device:
            session, tensors, arrays, input_shape, label_shape = get_model(
                config,
                builder,
                groups,
                var_set,
                location,
                "training",
                initialize=True,
                device=device,
                opts=opts)

            ip, lb, w1, w2 = tensors
            array_one, array_two = arrays

            session.weightsFromHost()

            buffer_one = np.ones(array_one.shape).astype(np.float32)
            buffer_two = np.ones(array_two.shape).astype(np.float32)
            if (config["retrieval"] == "AllReplicas"):
                old_shape = buffer_two.shape
                buffer_two = np.repeat(buffer_two.reshape((1, ) + old_shape),
                                       returned, 0)
                reshape = (returned, ) + old_shape
                buffer_two = buffer_two.reshape(reshape)
            weightsIo = popart.PyWeightsIO({w1: buffer_one, w2: buffer_two})

            session.weightsToHost()
            session.readWeights(weightsIo)

            if returned == 1:
                assert array_one.shape == buffer_two.shape
                assert builder.getTensorShape(w1) == builder.getTensorShape(w2)
            else:
                assert returned == buffer_two.shape[0]
                assert array_one.shape == buffer_two.shape[1:]

            # write to file
            session.modelToHost(tmpfile)

        # Clean Cut
        del session
        del builder

        builder = popart.Builder()
        opts, deviceContext = user_options(config, location)
        check_device(deviceContext, config)
        with deviceContext as device:
            session, tensors, array, input_shape, label_shape = get_model(
                config,
                builder,
                groups,
                var_set,
                location,
                "training",
                initialize=True,
                device=device,
                opts=opts)

            _, _, w1, w2 = tensors
            session.resetHostWeights(tmpfile, True)
            session.weightsFromHost()

            buffer_one = np.ones(array_one.shape).astype(np.float32)
            buffer_two = np.ones(array_two.shape).astype(np.float32)
            if (config["retrieval"] == "AllReplicas"):
                old_shape = buffer_two.shape
                buffer_two = np.repeat(buffer_two.reshape((1, ) + old_shape),
                                       returned, 0)
                reshape = (returned, ) + old_shape
                buffer_two = buffer_two.reshape(reshape)
            weightsIo = popart.PyWeightsIO({w1: buffer_one, w2: buffer_two})

            session.weightsToHost()
            session.readWeights(weightsIo)

        assert np.allclose(array_one, buffer_one)
        assert np.allclose(array_two, buffer_two)


DATA_SIZE = 5


def instance(repl: int,
             vs: popart.VariableSettings,
             location=None,
             length: int = 5,
             RTS: bool = False) -> np.ndarray:
    with tu.create_test_device(numIpus=repl) as device:
        shape = [O_DIM, CHANNELS, DATA_SIZE, DATA_SIZE]

        # meta
        builder = popart.Builder()

        # Make the Input
        t_info = popart.TensorInfo("FLOAT", shape)
        input_tensor = builder.addInputTensor(t_info, "input")

        # make the label
        label_shape = [O_DIM]
        label_info = popart.TensorInfo("INT32", label_shape)
        label_tensor = builder.addInputTensor(label_info, "label")

        # Make the weight
        weight = get_weights_array(shape, vs.groups(repl))
        weight_tensor = builder.addInitializedInputTensor(weight, vs)

        # graph body
        mul_tensor = builder.aiOnnx.matmul([input_tensor, weight_tensor])
        rsh_tensor = builder.reshape_const(
            builder.aiOnnx, [mul_tensor],
            [O_DIM, CHANNELS * DATA_SIZE * DATA_SIZE])
        relu_tensor = builder.aiOnnx.relu([rsh_tensor])
        out_tensor = builder.aiOnnx.softmax([relu_tensor])
        loss = builder.aiGraphcore.nllloss([out_tensor, label_tensor])

        # Dataflow elements
        art = popart.AnchorReturnType("All")
        data_flow = popart.DataFlow(BATCHES_PER_STEP, {out_tensor: art})

        # Make the session options
        options = popart.SessionOptions()
        options.replicatedGraphCount = repl
        options.enableReplicatedGraphs = True
        options.enableOutlining = False
        if (location is not None):
            options.weightTensorLocationSettings = location

        # weightIO
        buffer = np.zeros(weight.shape).astype(np.float32)
        weightsIo = popart.PyWeightsIO({weight_tensor: buffer})

        # ready session
        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFlow=data_flow,
                                         userOptions=options,
                                         loss=loss,
                                         deviceInfo=device,
                                         optimizer=popart.ConstSGD(1e-3))
        session.prepareDevice()

        session.weightsFromHost()
        session.weightsToHost()
        session.readWeights(weightsIo)

        assert np.allclose(weight, buffer)

        prefix_shape = [repl] if repl > 1 else []
        if BATCHES_PER_STEP > 1:
            prefix_shape = [BATCHES_PER_STEP] + prefix_shape

        # run
        for step in range(length):
            input = np.random.random_sample(prefix_shape + shape).astype(
                np.float32)
            label = np.random.randint(low=0, high=20,size=prefix_shape + label_shape).astype(\
                                                                            np.int32)

            input = np.arange(np.prod(shape)).astype(np.float32)
            for i in range(len(prefix_shape), 0, -1):
                input = np.tile(input, prefix_shape[i - 1])

            input = input.reshape(prefix_shape + shape)
            label = np.arange(label.size).reshape(label.shape).astype(np.int32)

            io = popart.PyStepIO(\
                    { input_tensor: input,    \
                      label_tensor: label },  \
                    session.initAnchorArrays())
            session.run(io)

        session.weightsToHost()
        session.readWeights(weightsIo)

        return buffer


@tu.requires_ipu
def test_locations():
    replication_factor = 4
    group = popart.CommGroup(popart.CommGroupType.Consecutive, 2)

    variable_setting = popart.VariableSettings(group)

    local = instance(replication_factor, variable_setting)

    remote = instance(replication_factor, variable_setting,
                      getOffChipLocation(group))

    assert np.allclose(local, remote)

    shard = instance(replication_factor,
                     variable_setting,
                     getOffChipLocation(group, True),
                     RTS=True)

    assert np.allclose(local, shard)
