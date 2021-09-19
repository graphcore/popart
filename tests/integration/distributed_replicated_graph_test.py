# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import popart
import test_util as tu
import pprint
import importlib
import os
import sys
import torch
import shutil
import json
import pathlib
from pathlib import Path
import argparse
import subprocess
import popdist
import popdist.popart
import onnx
from onnx import numpy_helper

PARTITION_NAME = 'partition0'


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


# TODO: We need a foolproof way of telling when we are on a IPU POD
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
    j = json.loads(gcd0_config)

    return len(j['devices']) == size


# For the time being these tests can only be run on an IPU POD
# To allocate a VIPU partition for this test use this command:
# `vipu-cli create partition partition_name --size 2 --gcds 2`

# The tests need to be run with a command similar to this:
# mpirun --tag-output -x IPUOF_CONFIG_PATH=~/.ipuof.conf.d/partition_name_gcd0_ipuof.conf -x TEST_TARGET=Hw -np 1  python -m pytest install/popart/tests/popart/distributed_replicated_graph_test.py -s : -x IPUOF_CONFIG_PATH=~/.ipuof.conf.d/partition_name_gcd1_ipuof.conf -x TEST_TARGET=Hw -np 1  python -m pytest install/popart/tests/popart/distributed_replicated_graph_test.py -s


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
    opts.globalReplicaOffset = mpi_rank
    opts.globalReplicationFactor = 2

    numIpus = 1

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
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
    G = builder.aiOnnx.matmul([F, D])
    loss = builder.aiGraphcore.l1loss([G],
                                      lossLambda,
                                      reduction=popart.ReductionType.Sum)
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

    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = False
    opts.enableDistributedReplicatedGraphs = True
    opts.globalReplicaOffset = mpi_rank
    opts.globalReplicationFactor = 2

    numIpus = 1

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=dataFlow,
                                     loss=loss,
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
# vipu-cli create partition partition_name --size 4 --gcds 2 --gcd-sync-replicas 4
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
    loss = builder.aiGraphcore.l1loss([G],
                                      lossLambda,
                                      reduction=popart.ReductionType.Sum)
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

    opts = popart.SessionOptions()
    opts.enableReplicatedGraphs = True
    opts.replicatedGraphCount = 2
    opts.enableDistributedReplicatedGraphs = True
    opts.globalReplicaOffset = 0 if mpi_rank == 0 else 2
    opts.globalReplicationFactor = 4

    numIpus = 2

    device = tu.create_test_device(numIpus=numIpus)
    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=dataFlow,
                                     loss=loss,
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


def replicated_tensor_sharding_core():
    parser = argparse.ArgumentParser(description="Parse launch parameters.")
    parser.add_argument("--tensors", nargs="*")
    parser.add_argument("--optim", nargs="?")
    parser.add_argument("--tmpdir", nargs="?")
    parser.add_argument("--filename", nargs="?")
    parser.add_argument("--compute_batch", nargs="?")
    args = parser.parse_args(sys.argv[2:])

    ipus_per_replica = 1

    batches_per_step = 10
    accumulation_factor = 4
    compute_batch = int(args.compute_batch)
    hidden_size = 4
    reduction = popart.ReductionType.Sum

    deviceInfo = popdist.popart.getDevice(ipus_per_replica)
    num_local_replicas = popdist.getNumLocalReplicas()
    num_total_replicas = popdist.getNumTotalReplicas()

    builder = popart.Builder()

    np.random.seed(12321)
    weight_data = np.random.rand(hidden_size, hidden_size).astype(np.float32)

    input_data = []
    label_data = []

    for i in range(
            0, batches_per_step * num_local_replicas * accumulation_factor *
            compute_batch):
        np.random.seed(popdist.getInstanceIndex() +
                       i * popdist.getNumInstances())
        input_data += [np.random.rand(hidden_size).astype(np.float32)]
        label_data += [np.random.randint(0, hidden_size, size=1)]

    input_data = np.concatenate(input_data)
    label_data = np.concatenate(label_data)

    builder = popart.Builder()

    d0 = builder.addInputTensor(
        popart.TensorInfo("FLOAT", (compute_batch, hidden_size)), "d0")
    l0 = builder.addInputTensor(popart.TensorInfo("UINT32", (compute_batch, )),
                                "l0")

    data = {}

    data[d0] = input_data.reshape((batches_per_step, num_local_replicas,
                                   accumulation_factor, compute_batch, -1))

    w0 = builder.addInitializedInputTensor(weight_data, 'weight0')
    x = builder.aiOnnx.matmul([d0, w0])

    x = builder.aiOnnx.softmax([x])

    data[l0] = label_data.reshape((batches_per_step,
                    num_local_replicas,
                    accumulation_factor,
                    compute_batch,
                    -1))\
                .astype(np.uint32)
    loss = builder.aiGraphcore.nllloss([x, l0],
                                       reduction=reduction,
                                       debugContext='loss')

    proto = builder.getModelProto()

    dataFlow = popart.DataFlow(
        batches_per_step,
        {av: popart.AnchorReturnType("ALL")
         for av in [x, loss]})

    opts = popart.SessionOptions()
    if accumulation_factor > 1:
        opts.enableGradientAccumulation = True
        opts.accumulationFactor = accumulation_factor
    opts.explicitRecomputation = True
    opts.enableExplicitMainLoops = True
    opts.useHostCopyOps = True
    # Let popdist handle distributed settings, such as:
    # opts.enableDistributedReplicatedGraphs
    # opts.globalReplicaOffset
    # opts.globalReplicationFactor
    popdist.popart.configureSessionOptions(opts)

    for tensor in ["weight", "optimizerState", "accumulator"]:
        userOption = tensor + "TensorLocationSettings"
        print(
            f"Setting RTS: {userOption}, num_total_replicas: {num_total_replicas} num_local_replicas: {num_local_replicas}"
        )
        locationSetting = getattr(opts, userOption)
        locationSetting.minElementsForOffChip = 0
        locationSetting.minElementsForReplicatedTensorSharding = num_total_replicas
        if tensor in args.tensors:
            locationSetting.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On
        if num_total_replicas > num_local_replicas:
            locationSetting.location.shardingDomain = popart.CommGroup(
                popart.CommGroupType.Consecutive, num_local_replicas)
        setattr(opts, userOption, locationSetting)

    if args.optim == "Adam":
        optimizer = popart.Adam(
            {
                "defaultLearningRate": (0.01, False),
                "defaultBeta1": (0.9, False),
                "defaultBeta2": (0.999, False),
                "defaultEps": (1e-06, False),
                "defaultWeightDecay": (0.1, False),
                "lossScaling": (10, False),
            },
            weight_decay_mode=popart.WeightDecayMode.Decay,
            mode=popart.AdamMode.LambNoBias)
    if args.optim == "SGD":
        optimizer = popart.ConstSGD(0.01)

    session = popart.TrainingSession(fnModel=proto,
                                     dataFlow=dataFlow,
                                     deviceInfo=deviceInfo,
                                     userOptions=opts,
                                     loss=loss,
                                     optimizer=optimizer)

    session.prepareDevice()

    session.weightsFromHost()

    anchors = session.initAnchorArrays()

    stepio = popart.PyStepIO(data, anchors)

    session.run(stepio)

    tmp_path = Path(args.tmpdir)
    tmp_path.mkdir(parents=True, exist_ok=True)
    file_path = str(tmp_path / args.filename)
    session.modelToHost(file_path)
    post_proto = onnx.load(file_path)


rts_configs = [
    [
        # Baseline: 4 replicas
        {
            "filename": "rts0.onnx",
            "num_replicas": 4,
            "num_instances": 1,
            "compute_batch": 6,
        },
        # Comparison: 8 replicas
        {
            "filename": "rts1.onnx",
            "num_replicas": 8,
            "num_instances": 1,
            "compute_batch": 3,
        }
    ],
    [
        # Baseline: 8 replicas, 1 instance
        {
            "filename": "rts0.onnx",
            "num_replicas": 8,
            "num_instances": 1,
            "compute_batch": 6,
        },
        # Comparison: 16 replicas, 2 instances (2 GCD, 1 ILD)
        {
            "filename": "rts1.onnx",
            "num_replicas": 16,
            "num_instances": 2,
            "compute_batch": 3,
        }
    ],
    [
        # Baseline: 16 replicas, 1 instance
        {
            "filename":
            "rts0.onnx",
            "num_replicas":
            16,
            "num_instances":
            1,
            "compute_batch":
            6,
            "partition":
            "fabiant",
            "hosts": [
                "gbnwp-pod009-3.ipu.graphcore.ai",
                "gbnwp-pod010-3.ipu.graphcore.ai"
            ]
        },
        # Comparison: 32 replicas, 2 instances (2 GCD, 2 ILD)
        {
            "filename":
            "rts1.onnx",
            "num_replicas":
            32,
            "num_instances":
            2,
            "compute_batch":
            3,
            "partition":
            "fabiant",
            "hosts": [
                "gbnwp-pod009-3.ipu.graphcore.ai",
                "gbnwp-pod010-3.ipu.graphcore.ai"
            ]
        }
    ]
]


# This is a replicated tensor sharding test in which we run a hierarchical reduction where
# weights and optimizer states are scattered within the GCD/ILD (contiguous) and the scattered gradients all-reduced across
# GCD/ILD (orthogonal)
@pytest.mark.parametrize("configs", rts_configs)
@pytest.mark.parametrize(
    "tensors", [[], ["weight", "optimizerState"], ["optimizerState"]])
@pytest.mark.parametrize("optim", ["SGD", "Adam"])
def test_replicated_tensor_sharding(tmpdir, configs, tensors, optim):
    rtol = 1.e-3
    atol = 1.e-5

    debug = True
    reset = True
    remove = False

    for config in configs:
        test_path = pathlib.Path(__file__).resolve()

        # Set to the partition name available
        partition = None
        if "partition" in config:
            partition = config["partition"]

        # Configure if testing multi-host instances
        hosts = []
        if "hosts" in config:
            hosts = config["hosts"]
        if (len(hosts) > config["num_instances"]):
            hosts = hosts[0:config["num_instances"]]

        num_replicas = config["num_replicas"]
        ipus_per_replica = 1
        num_instances = config["num_instances"]

        command = ["poprun"]

        if debug:
            command.append("-vv")

        if len(hosts) > 1:
            command.append("--host")
            command.append(",".join([str(host) for host in hosts]))

        command.append("--num-replicas")
        command.append(str(num_replicas))
        command.append("--num-instances")
        command.append(str(num_instances))
        command.append("--ipus-per-replica")
        command.append(str(ipus_per_replica))

        if not debug:
            command.append("--only-output-from-instance")
            command.append(str(0))

        if partition is not None:
            command.append("--vipu-partition")
            command.append(partition)

        command.append("--reset-partition")
        command.append("yes" if reset else "no")
        command.append("--update-partition")
        command.append("yes")
        command.append("--remove-partition")
        command.append("yes" if remove else "no")

        command.append("--mpi-global-args=--allow-run-as-root")
        command.append(
            "--mpi-local-args=-x LD_LIBRARY_PATH -x PYTHONPATH -x PATH")
        command.append("python3")
        command.append(test_path)
        command.append("replicated_tensor_sharding_core")
        command.append("--tensor")
        for t in tensors:
            command.append(t)
        command.append("--tmpdir")
        command.append(tmpdir)
        command.append("--filename")
        command.append(config["filename"])
        command.append("--compute_batch")
        command.append(str(config["compute_batch"]))
        command.append("--optim")
        command.append(optim)

        out = subprocess.run(command)

    print(f"Testing {len(configs)} configurations")

    tmp_path = Path(tmpdir)

    gt_onnx = onnx.load(str(tmp_path / configs[0]["filename"]))

    for i in range(1, len(configs)):
        print(f"Testing run {i}: {configs[i]}")

        val_onnx = onnx.load(str(tmp_path / configs[i]["filename"]))
        for j in range(len(gt_onnx.graph.initializer)):
            print(f"Checking initializer {j}")
            gt = gt_onnx.graph.initializer[j]
            gt = numpy_helper.to_array(gt)
            val = val_onnx.graph.initializer[j]
            val = numpy_helper.to_array(val)
            print("Max difference:", np.max(np.abs(val - gt)))
            assert np.allclose(gt, val, rtol=rtol, atol=atol, equal_nan=False)


# Distributed test fixture entry point
if __name__ == '__main__':
    globals()[sys.argv[1]]()
