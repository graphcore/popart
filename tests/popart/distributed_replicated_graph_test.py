# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu
import pprint
import importlib
import os
import torch
import shutil
import json

PARTITION_NAME = 'shaurya'


def mpi4py_installed():
    try:
        from mpi4py import MPI
    except ModuleNotFoundError:
        return False
    return True


def get_mpi_params():
    try:
        from mpi4py import MPI
    except ModuleNotFoundError:
        return None
    comm = MPI.COMM_WORLD
    mpi_size = comm.Get_size()
    mpi_rank = comm.Get_rank()
    assert (mpi_size == 2)
    return (mpi_size, mpi_rank)


def grad_id(tensor_id):
    return popart.reservedGradientPrefix() + tensor_id


# TODO: We need a foolproof way of telling when we are on a IPU-Pod
def is_running_on_ipu_pod():
    ipuof_config_found = os.getenv("IPUOF_CONFIG_PATH") is not None
    vipu_cli_found = shutil.which('vipu-cli') is not None
    return ipuof_config_found and vipu_cli_found


def is_gcd_size(size):
    if not is_running_on_ipu_pod():
        return False
    gcd0_config_file = os.path.join(os.getenv('HOME'), '.ipuof.conf.d',
                                    PARTITION_NAME + '_gcd0_ipuof.conf')
    with open(gcd0_config_file, 'r') as fid:
        gcd0_config = fid.read()
    return len(json.loads(gcd0_config)) == size


# For the time being these tests can only be run on an IPU-Pod
# To allocate a VIPU partition for this test use this command:
# `vipu-cli create partition shaurya --size 2 --gcds 2`

# The tests need to be run with a command similar to this:
# mpirun --tag-output -x IPUOF_CONFIG_PATH=~/.ipuof.conf.d/shaurya_gcd0_ipuof.conf -x TEST_TARGET=Hw -np 1  python -m pytest install/popart/tests/popart/distributed_replicated_graph_test.py -s : -x IPUOF_CONFIG_PATH=~/.ipuof.conf.d/shaurya_gcd1_ipuof.conf -x TEST_TARGET=Hw -np 1  python -m pytest install/popart/tests/popart/distributed_replicated_graph_test.py -s


@pytest.mark.skipif(
    not mpi4py_installed() or not is_gcd_size(1),
    reason="mpi4py needs to be installed. Test can only be run on a IPU pod")
def test_distributed_replicated_allreduce():
    mpi_params = get_mpi_params()
    mpi_size, mpi_rank = mpi_params

    input_data = np.array(range(10), dtype=np.float32)

    builder = popart.Builder()
    t = builder.addInitializedInputTensor(input_data, "input")
    o = builder.aiGraphcore.replicatedallreduce([t])
    builder.addOutputTensor(o)
    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("All")})
    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = False
    opts.enableDistributedReplicatedGraphs = True
    opts.globalNumIpus = mpi_size
    opts.globalReplicaOffset = mpi_rank
    opts.globalReplicationFactor = 2

    numIpus = 1

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.InferenceSession(fnModel=proto,
                                      dataFeed=dataFlow,
                                      userOptions=opts,
                                      deviceInfo=device)

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {}
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    ground_truth = 2.0 * np.array(range(10), dtype=np.float32)
    assert np.allclose(anchors[o], ground_truth)


@pytest.mark.skipif(
    not mpi4py_installed() or not is_gcd_size(1),
    reason="mpi4py needs to be installed. Test can only be run on a IPU pod")
def test_distributed_replicated_weight_update():
    K = 6
    M = 7
    N = 8
    replicationFactor = 2
    lossLambda = 0.1

    np.random.seed(42)
    A_init = np.random.random((M, K)).astype(dtype=np.float32)
    B_init = np.random.random((K, N)).astype(dtype=np.float32)
    C_init = np.random.random((N, M)).astype(dtype=np.float32)
    D_init = np.random.random((M, N)).astype(dtype=np.float32)

    def ground_truth():
        A = torch.tensor(A_init, requires_grad=True)
        B = torch.tensor(B_init, requires_grad=True)
        C = torch.tensor(C_init, requires_grad=True)
        D = torch.tensor(D_init, requires_grad=True)
        E = torch.matmul(A, B)
        F = torch.matmul(E, C)
        G = torch.matmul(F, D)
        params = [A, B, C, D]
        optim = torch.optim.SGD(params, lr=1.0)
        optim.zero_grad()
        err = torch.sum(lossLambda * torch.abs(G))
        err.backward()
        A.grad = A.grad * replicationFactor
        B.grad = B.grad * replicationFactor
        C.grad = C.grad * replicationFactor
        D.grad = D.grad * replicationFactor
        optim.step()

        result = {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
        }
        return result

    mpi_params = get_mpi_params()
    mpi_size, mpi_rank = mpi_params
    builder = popart.Builder()
    A = builder.addInitializedInputTensor(A_init, "A")
    B = builder.addInitializedInputTensor(B_init, "B")
    C = builder.addInitializedInputTensor(C_init, "C")
    D = builder.addInitializedInputTensor(D_init, "D")
    E = builder.aiOnnx.matmul([A, B])
    F = builder.aiOnnx.matmul([E, C])
    loss = builder.aiGraphcore.l1loss([G], lossLambda)
    builder.addOutputTensor(loss)
    proto = builder.getModelProto()

    outputs = {
        A: popart.AnchorReturnType("All"),
        B: popart.AnchorReturnType("All"),
        C: popart.AnchorReturnType("All"),
        D: popart.AnchorReturnType("All"),
        G: popart.AnchorReturnType("All")
    }

    dataFlow = popart.DataFlow(1, outputs)
    optimizer = popart.ConstSGD(1.0)
    losses = [popart.IdentityLoss(loss, "loss")]

    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = False
    opts.enableDistributedReplicatedGraphs = True
    opts.globalNumIpus = mpi_size
    opts.globalReplicaOffset = mpi_rank
    opts.globalReplicationFactor = 2

    numIpus = 1

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=dataFlow,
                                     losses=losses,
                                     optimizer=optimizer,
                                     deviceInfo=device,
                                     userOptions=opts)

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {}
    stepio = popart.PyStepIO(inputs, anchors)

    session.weightsFromHost()

    session.run(stepio)

    torch_ground_truth = ground_truth()
    keys = ["A", "B", "C", "D"]
    for k in keys:
        assert np.allclose(anchors[k], torch_ground_truth[k].detach().numpy())


# This is a special test in which we run a hierarchical replicated graph at the instance level
# The global replication factor is 4, the local replication factor is 2
# To configure this system with VIRM use the following command:
# vipu-cli create partition shaurya --size 4 --gcds 2 --gcd-sync-replicas 4
@pytest.mark.skipif(
    not mpi4py_installed() or not is_gcd_size(2),
    reason=
    "mpi4py needs to be installed. Test can only be run on a IPU pod with gcd size 2"
)
def test_distributed_hierarchical_replicated_weight_update():
    K = 6
    M = 7
    N = 8
    replicationFactor = 4
    lossLambda = 0.1

    np.random.seed(42)
    A_init = np.random.random((M, K)).astype(dtype=np.float32)
    B_init = np.random.random((K, N)).astype(dtype=np.float32)
    C_init = np.random.random((N, M)).astype(dtype=np.float32)
    D_init = np.random.random((M, N)).astype(dtype=np.float32)

    def ground_truth():
        A = torch.tensor(A_init, requires_grad=True)
        B = torch.tensor(B_init, requires_grad=True)
        C = torch.tensor(C_init, requires_grad=True)
        D = torch.tensor(D_init, requires_grad=True)
        E = torch.matmul(A, B)
        F = torch.matmul(E, C)
        G = torch.matmul(F, D)
        params = [A, B, C, D]
        optim = torch.optim.SGD(params, lr=1.0)
        optim.zero_grad()
        err = torch.sum(lossLambda * torch.abs(G))
        err.backward()
        A.grad = A.grad * replicationFactor
        B.grad = B.grad * replicationFactor
        C.grad = C.grad * replicationFactor
        D.grad = D.grad * replicationFactor
        optim.step()

        result = {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
        }
        return result

    mpi_params = get_mpi_params()
    mpi_size, mpi_rank = mpi_params
    builder = popart.Builder()
    A = builder.addInitializedInputTensor(A_init, "A")
    B = builder.addInitializedInputTensor(B_init, "B")
    C = builder.addInitializedInputTensor(C_init, "C")
    D = builder.addInitializedInputTensor(D_init, "D")
    E = builder.aiOnnx.matmul([A, B])
    F = builder.aiOnnx.matmul([E, C])
    G = builder.aiOnnx.matmul([F, D])
    loss = builder.aiGraphcore.l1loss([G], lossLambda)
    builder.addOutputTensor(loss)
    proto = builder.getModelProto()

    outputs = {
        A: popart.AnchorReturnType("All"),
        B: popart.AnchorReturnType("All"),
        C: popart.AnchorReturnType("All"),
        D: popart.AnchorReturnType("All"),
        G: popart.AnchorReturnType("All")
    }

    dataFlow = popart.DataFlow(1, outputs)
    optimizer = popart.ConstSGD(1.0)
    losses = [popart.IdentityLoss(loss, "loss")]

    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = 2
    opts.enableDistributedReplicatedGraphs = True
    opts.globalNumIpus = 4
    opts.globalReplicaOffset = 0 if mpi_rank == 0 else 2
    opts.globalReplicationFactor = 4

    numIpus = 2

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.TrainingSession(fnModel=proto,
                                     dataFeed=dataFlow,
                                     losses=losses,
                                     optimizer=optimizer,
                                     deviceInfo=device,
                                     userOptions=opts)

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {}
    stepio = popart.PyStepIO(inputs, anchors)

    session.weightsFromHost()

    session.run(stepio)

    torch_ground_truth = ground_truth()
    keys = ["A", "B", "C", "D"]
    for k in keys:
        assert np.allclose(anchors[k], torch_ground_truth[k].detach().numpy())
