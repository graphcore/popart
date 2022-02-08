# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from typing import Dict, List, Tuple
from tempfile import TemporaryDirectory
from numpy.core import shape_base
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
# Learning rate 1 for easy comparison.
LEARNING_RATE = 1.0
TEMPFILE = "temporary_file"
"""
Autogen a weight array
"""

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
        reshape = [len(groups)]
    reshape.extend(shape)

    array = np.random.random_sample(reshape).astype(np.float32)
    return array


"""
Configure SessionOptions from Repl-Config
"""


def user_options(repl_config):
    opts = popart.SessionOptions()

    opts.replicatedGraphCount = int(repl_config["repl"])
    opts.enableReplicatedGraphs = int(repl_config["repl"]) != 1

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
              initialize=True):
    """
    Get a simple test
    """
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
    nll = builder.aiGraphcore.nllloss([o, lb])

    #
    art = popart.AnchorReturnType("All")
    data_flow = popart.DataFlow(BATCHES_PER_STEP, {o: art})

    #
    opts, device = user_options(repl_config)

    if device is None:
        ipus = (int(repl_config["ipus"]) * int(repl_config["repl"]))

        if (ipus > 4):
            pytest.fail(
                f"Test needs to run on {ipus} IPU(s), but sufficient IPU(s) were not available. "
                "As the requirement of this test was higher than 4 ipus it fails silently."
            )
        else:
            pytest.skip(
                f"Test needs to run on {ipus} IPU(s), but sufficient IPU(s) were not available. "
                "As the requirements of this test was 4 ipus and less, this is considered a fail."
            )

    session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                     dataFlow=data_flow,
                                     loss=nll,
                                     optimizer=popart.ConstSGD(LEARNING_RATE),
                                     userOptions=opts,
                                     deviceInfo=device)

    session.prepareDevice()

    if int(repl_config["repl"]) > 1:
        input_shape = [int(repl_config["repl"])] + input_shape
        label_shape = [int(repl_config["repl"])] + label_shape
    if BATCHES_PER_STEP > 1:
        input_shape = [BATCHES_PER_STEP] + input_shape
        label_shape = [BATCHES_PER_STEP] + label_shape

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
        "retrieval": "OnePerGroup"
    },
    {  # Multiple replicas in one group, returning one
        "c.idx": '1',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "All",
        "commSize": '0',
        "retrieval": "OnePerGroup"
    },
    ## New variant returns
    {  # Grouped replicas returning all
        "c.idx": '2',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "All",
        "commSize": '0',
        "retrieval": "AllReplicas"
    },
    {  # Two ungrouped replicas, returning one each. 
        "c.idx": '3',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "None",
        "commSize": '0',
        "retrieval": "OnePerGroup"
    },
    {  # Four replicas devided in two groups
        "c.idx": '4',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '4',  # Replication Factor
        "commType": "Consecutive",
        "commSize": '2',
        "retrieval": "OnePerGroup"
    },
    {  # Four replicas devided in two groups
        "c.idx": '5',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '4',  # Replication Factor
        "commType": "Orthogonal",
        "commSize": '2',
        "retrieval": "OnePerGroup"
    },
    {  # Cons 1
        "c.idx": '6',  # Debug Config ID
        "ipus": '1',  # Number of IPUs to run on
        "repl": '2',  # Replication Factor
        "commType": "Consecutive",
        "commSize": '1',
        "retrieval": "OnePerGroup"
    }
]

ipu_max = 8
run_tests = []  # debug tool, run custom set
skp_tests = []  # debug tool, add to skip

if (len(run_tests) == 0):
    run_tests = range(len(configs))
"""
    Run tests
"""


@pytest.mark.parametrize("config", configs)
@tu.requires_ipu
def test_grouped_initialization(config):
    # skip if unimplemented
    if (config["c.idx"] == configs[0]["c.idx"]):
        print()
    if (int(config["c.idx"]) not in run_tests
            or int(config["c.idx"]) in skp_tests):
        pytest.skip()
    if (ipu_max < int(config["repl"]) * int(config["ipus"])):
        pytest.skip()

    var_set = get_variable_settings(config)
    returned = var_set.numReplicasReturningVariable(int(config["repl"]))
    groups = [] if returned == 1 else get_group_idxs(config, var_set, returned)

    # Get and run session

    builder = popart.Builder()
    session, tensors, arrays, input_shape, label_shape = get_model(
        config, builder, groups, var_set)

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
    assert np.allclose(arrays_one, buffer_one)

    verify(config, var_set, buffer_two, groups)

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
        label_array = np.random.randint(low=0, high=20,
                                        size=label_shape).astype(np.int32)

        anchors = session.initAnchorArrays()
        stepIo = popart.PyStepIO({ip: in_array, lb: label_array}, anchors)

        run_model(session, stepIo)

        session.readWeights(weightsIo)

        # Only verify this the first time, should be redundant later
        if (step == 0):
            # checks that the weights change
            assert not np.allclose(arrays_two, buffer_two)

            # checks that the gradient is different from different groups

        verify(config, var_set, buffer_two, groups)


onnx_tests = [1, 2, 3, 4, 5]
onnx_configs = []
for test in onnx_tests:
    onnx_configs += [configs[test]]


@pytest.mark.parametrize("config", onnx_configs)
@tu.requires_ipu
def test_onnx_checkpointing(config):
    if onnx_tests[0] == int(config["c.idx"]):
        print()
    with TemporaryDirectory() as tmpdir:
        tmpfile = os.path.join(tmpdir, "model.onnx")

        var_set = get_variable_settings(config)
        returned = var_set.numReplicasReturningVariable(int(config["repl"]))
        groups = [] if returned == 1 else get_group_idxs(
            config, var_set, returned)

        builder = popart.Builder()
        builder.embedReplicationFactor(int(config["repl"]))
        session, tensors, arrays, input_shape, label_shape = get_model(
            config, builder, groups, var_set, initialize=True)

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
        session, tensors, array, input_shape, label_shape = get_model(
            config, builder, groups, var_set, initialize=True)

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
