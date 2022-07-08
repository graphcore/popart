# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import pytest
import popart._internal.ir as _ir
from utils import make_main_graph

# Remove as soon as implemented.
transforms = [
    # "AutoVirtualGraph",
    # # "Autodiff", tested below
    # "BatchSerialize",
    # "ClipWeightGradientsByNorm",
    # "DecomposeGradSum",
    # "DecomposeLoops",
    # "ExplicitRecompute",
    # "GroupMatMuls",
    # "HostIOSetup",
    # "HostReduce",
    # "InferPipelineStages",
    # "InplaceAccumulateGradPartialsIntoOptimizerAccumTensor",
    # "InterIpuCopy",
    # "IoComputeTileCopy",
    # "MergeCopies",
    # "MergeDuplicateOps",
    # "MergeLoops",
    # "MergeVarUpdates",
    # "MergeAllVarUpdates",
    # "MergeAuto",
    # "MergeTightThreshold",
    # "MergeLooseThreshold",
    "Prune",
    # "RandomSetup",
    # "RemoteSetup",
    # "SerializeMatMuls",
    # "StreamingMemory",
    # "MainLoops",
    # "Pipeline",
    # "AutomaticLossScale",
    # "SubgraphOutline",
    # "DynamicOpTransform",
    # "PreAutomaticLossScale",
    # "AccumulateOuterFragmentParallelizer",
    # "MergeExchange",
]


@pytest.mark.parametrize("transform_name", transforms)
def test_apply_transforms(transform_name: str) -> None:
    """Test some of the above simpler transforms"""
    ir, _, _ = make_main_graph()
    g = ir.getMainGraph()

    transform = getattr(_ir.transforms, transform_name)
    print(f"Applying transform {transform().getName()}")
    assert transform().apply(g)


@pytest.mark.parametrize(
    "conn_type", [_ir.ExpectedConnectionType.Fwd, _ir.ExpectedConnectionType.FwdGrad]
)
@pytest.mark.parametrize(
    "stitch_strategy",
    [
        _ir.transforms.AutodiffStitchStrategy.RecomputeMinimal,
        _ir.transforms.AutodiffStitchStrategy.RecomputeAllNonInputs,
        _ir.transforms.AutodiffStitchStrategy.AddFwdOutputs,
        _ir.transforms.AutodiffStitchStrategy.SafeAddFwdOutputs,
    ],
)
def test_autodiff(
    conn_type: _ir.ExpectedConnectionType,
    stitch_strategy: _ir.transforms.AutodiffStitchStrategy,
) -> None:
    """Special test for more complex autodiff transform"""
    ir, outs_, weights = make_main_graph()
    out_ = outs_[0]
    weight = weights[0]
    g = ir.getGraph("fwd")

    t = _ir.transforms.Autodiff()

    a = _ir.ExpectedConnection(out_.id, conn_type)
    bwd_info = _ir.ExpectedConnection(weight.id, conn_type)

    fwd_bwd = dict()
    fwd_bwd["bwd"] = _ir.BwdGraphInfo(g.id, [a], [bwd_info])

    # Apply the autodiff
    result = t.apply(
        ir, g.id, [out_.id], _ir.OptionalTensors(), fwd_bwd, stitch_strategy
    )
    for gid, bwd_info in result.items():
        assert gid.str() in ["", "fwd", "bwd"]  # main graph  # fwd graph  # bwd graph
        if not gid.str():
            # main graph, no bwd info.
            assert len(bwd_info.expectedInputs) == 0
            assert len(bwd_info.expectedOutputs) == 0
        elif gid.str() == "fwd":
            assert len(bwd_info.expectedInputs) == (
                3
                if stitch_strategy
                == _ir.transforms.AutodiffStitchStrategy.RecomputeAllNonInputs
                else 2
            )
            assert len(bwd_info.expectedOutputs) == 2
        elif gid.str() == "bwd":
            for expected in bwd_info.expectedInputs:
                assert expected.fwdId == out_.id
                assert expected.type == conn_type
            for expected in bwd_info.expectedOutputs:
                assert expected.fwdId == weight.id
                assert expected.type == conn_type
    bwd = ir.getGraph(result[g.id].bwdGraphId)
    assert len(bwd.getOpIds()) == (
        8
        if stitch_strategy
        == _ir.transforms.AutodiffStitchStrategy.RecomputeAllNonInputs
        else 6
    )
