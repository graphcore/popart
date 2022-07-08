# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import onnx
from onnx import numpy_helper

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def run_model(
    tmpdir,
    model_file_name,
    schedule=popart.ExecutionPhaseSchedule.Interleaving,
    enable_outlining=False,
    stride=1,
    num_layers=5,
    dsize=128,
    batch_size=4,
    batch_serialize=1,
    batch_schedule=popart.BatchSerializationBatchSchedule.Isomorphic,
    num_iterations=5,
    num_replicas=2,
    optimizer=popart.Adam({"defaultLearningRate": (0.1, False)}),
):

    np.random.seed(52125)

    builder = popart.Builder()
    ip = builder.addInputTensor(popart.TensorInfo("FLOAT", [batch_size, dsize, dsize]))

    def add_layer(index, in_id):
        w = builder.addInitializedInputTensor(
            np.random.rand(dsize, dsize).astype(np.float32), f"W{index}"
        )
        matmul_id = builder.aiOnnx.matmul([in_id, w])
        return matmul_id

    out = ip
    l1 = ""
    final_loss = ""

    for i in range(num_layers):
        vgid = 0
        with builder.executionPhase(i * stride), builder.virtualGraph(vgid):
            for _ in range(3):
                out = add_layer(i, out)

        if i == num_layers - 1:
            with builder.executionPhase(i * stride), builder.virtualGraph(vgid):
                l1 = builder.aiGraphcore.l1loss([out], 0.1, popart.ReductionType.Sum)
                final_loss = builder.aiGraphcore.identityloss([l1])

    anchorIds = []

    builder.addOutputTensor(out)

    num_ipus = 1

    dfAnchors = {}
    for anchorId in anchorIds:
        dfAnchors.update({anchorId: popart.AnchorReturnType("All")})

    opts = popart.SessionOptions()

    # Cycle counting
    opts.instrumentWithHardwareCycleCounter = True

    # Outlining
    opts.enableOutlining = enable_outlining
    opts.enableOutliningCopyCostPruning = False
    opts.outlineThreshold = -np.inf
    opts.aliasZeroCopy = enable_outlining

    # Replicated graphs
    opts.replicatedGraphCount = num_replicas
    opts.enableReplicatedGraphs = True if num_replicas > 1 else False

    # IO tiles
    opts.numIOTiles = 192

    # Phased execution
    opts.executionPhaseSettings.phases = num_layers * stride
    opts.executionPhaseSettings.stages = 1
    opts.executionPhaseSettings.schedule = schedule
    opts.virtualGraphMode = popart.VirtualGraphMode.ExecutionPhases

    # Recomputation
    opts.autoRecomputation = popart.RecomputationType.Standard
    opts.explicitRecomputation = True

    # Batch serialization
    if batch_serialize > 1:
        opts.batchSerializationSettings.factor = batch_serialize
        opts.batchSerializationSettings.concatOnVirtualGraphChange = False
        opts.batchSerializationSettings.concatOnExecutionPhaseChange = False
        opts.batchSerializationSettings.concatOnPipelineStageChange = False
        opts.batchSerializationSettings.batchSchedule = batch_schedule
        # Related execution phase setting
        opts.executionPhaseSettings.activationIOSchedule = (
            popart.ExecutionPhaseIOSchedule.OnDemand
        )

    # Streaming memory
    offChipLocation = popart.TensorLocationSettings(
        location=popart.TensorLocation(
            storage=popart.TensorStorage.OffChip,
            loadTileSet=popart.TileSet.IO,
            storageTileSet=popart.TileSet.IO,
            replicatedTensorSharding=popart.ReplicatedTensorSharding.Off,
        ),
        minElementsForOffChip=0,
        minElementsForReplicatedTensorSharding=2,
    )

    offChipRtsLocation = popart.TensorLocationSettings(
        location=popart.TensorLocation(
            storage=popart.TensorStorage.OffChip,
            loadTileSet=popart.TileSet.IO,
            storageTileSet=popart.TileSet.IO,
            replicatedTensorSharding=popart.ReplicatedTensorSharding.On,
        ),
        minElementsForOffChip=0,
        minElementsForReplicatedTensorSharding=2,
    )

    opts.activationTensorLocationSettings = offChipLocation
    opts.weightTensorLocationSettings = offChipRtsLocation
    opts.optimizerStateTensorLocationSettings = offChipRtsLocation

    proto = builder.getModelProto()

    with tu.create_test_device(
        num_replicas * num_ipus, pattern=popart.SyncPattern.Full
    ) as device:

        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=popart.DataFlow(1, dfAnchors),
            optimizer=optimizer,
            loss=final_loss,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            userOptions=opts,
            deviceInfo=device,
        )

        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()

        for i in range(num_iterations):
            ip_data = np.random.rand(num_replicas, batch_size, dsize, dsize).astype(
                np.float32
            )
            stepio = popart.PyStepIO({ip: ip_data}, anchors)
            session.run(stepio)

        cycles = session.getCycleCount()

        print("anchors:")
        print(anchors)
        session.modelToHost(str(tmpdir / model_file_name))

        return cycles


def check_model(lhs_model, rhs_model):
    for i in range(len(lhs_model.graph.initializer)):
        lhs = lhs_model.graph.initializer[i]
        for j in range(len(rhs_model.graph.initializer)):
            rhs = rhs_model.graph.initializer[j]
            if rhs.name == lhs.name:
                print(f"Checking initializer {i} ({lhs.name} - {rhs.name})")
                lhsa = numpy_helper.to_array(lhs)
                rhsa = numpy_helper.to_array(rhs)
                assert np.allclose(lhsa, rhsa, rtol=1.0e-4, atol=1.0e-6)


@tu.requires_ipu
def test_phase_overlap(tmpdir):
    cycles_interleaving = run_model(
        tmpdir, "interleaving.onnx", schedule=popart.ExecutionPhaseSchedule.Interleaving
    )

    cycles_clustered_io = run_model(
        tmpdir,
        "batch_clustered_io.onnx",
        schedule=popart.ExecutionPhaseSchedule.BatchClusteredIO,
    )

    cycles_clustered_io_outline = run_model(
        tmpdir,
        "batch_clustered_io_outline.onnx",
        schedule=popart.ExecutionPhaseSchedule.BatchClusteredIO,
        enable_outlining=True,
    )

    interleaving = onnx.load(str(tmpdir / "interleaving.onnx"))
    clustered_io = onnx.load(str(tmpdir / "batch_clustered_io.onnx"))
    clustered_io_outline = onnx.load(str(tmpdir / "batch_clustered_io_outline.onnx"))

    print(
        f"Cycles: {cycles_interleaving} {cycles_clustered_io} {cycles_clustered_io_outline}"
    )

    check_model(interleaving, clustered_io)
    # Test requires T26754 to pass
    check_model(interleaving, clustered_io_outline)

    assert cycles_clustered_io < 0.9 * cycles_interleaving
    # Test requires T26968 to pass
    assert cycles_clustered_io_outline < 0.9 * cycles_interleaving


@tu.requires_ipu
def test_batch_overlap(tmpdir):
    cycles_isomorphic = run_model(
        tmpdir,
        "isomorphic.onnx",
        stride=4,
        batch_serialize=4,
        batch_schedule=popart.BatchSerializationBatchSchedule.Isomorphic,
    )

    cycles_overlap_on_compute = run_model(
        tmpdir,
        "overlap_on_compute.onnx",
        stride=4,
        batch_serialize=4,
        batch_schedule=popart.BatchSerializationBatchSchedule.OverlapOnCompute,
    )

    cycles_overlap_on_compute_outline = run_model(
        tmpdir,
        "overlap_on_compute_outline.onnx",
        stride=4,
        batch_serialize=4,
        batch_schedule=popart.BatchSerializationBatchSchedule.OverlapOnCompute,
        enable_outlining=True,
    )

    isomorphic = onnx.load(str(tmpdir / "isomorphic.onnx"))
    overlap_on_compute = onnx.load(str(tmpdir / "overlap_on_compute.onnx"))
    overlap_on_compute_outline = onnx.load(
        str(tmpdir / "overlap_on_compute_outline.onnx")
    )

    print(
        f"Cycles: {cycles_isomorphic} {cycles_overlap_on_compute} {cycles_overlap_on_compute_outline}"
    )

    check_model(isomorphic, overlap_on_compute)
    # Test requires T26754 to pass
    check_model(isomorphic, overlap_on_compute_outline)

    assert cycles_overlap_on_compute < 0.9 * cycles_isomorphic
    # Test requires T26968 to pass
    assert cycles_overlap_on_compute_outline < 0.9 * cycles_isomorphic
