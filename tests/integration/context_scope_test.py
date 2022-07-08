# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import pytest


def test_get_pipeline_stage():
    builder = popart.Builder()

    # If the scope hasn't been set, hasPipelineStage will return False
    assert builder.hasPipelineStage() is False

    # If the scope hasn't been set, getPipelineStage will throw a runtime error
    with pytest.raises(popart.popart_exception) as e_info:
        _ = builder.getPipelineStage()  # stage
    assert "Pipeline stage not set in current scope" in e_info.value.args[0]

    # When we enter a pipelineStage context, the builder should return that stage
    with builder.pipelineStage(0):
        assert builder.hasPipelineStage() is True
        assert builder.getPipelineStage() == 0

    with builder.pipelineStage(1):
        assert builder.hasPipelineStage() is True
        assert builder.getPipelineStage() == 1

    # Nested pipeline scopes are not currently supported. See below:
    with builder.pipelineStage(0):
        assert builder.hasPipelineStage() is True
        assert builder.getPipelineStage() == 0

        with builder.pipelineStage(1):
            assert builder.hasPipelineStage() is True
            assert builder.getPipelineStage() == 1

        assert builder.hasPipelineStage() is True
        assert builder.getPipelineStage() == 0


def test_get_virtual_graph():
    builder = popart.Builder()

    # If the scope hasn't been set, hasVirtualGraph will return False
    assert builder.hasVirtualGraph() is False

    # If the scope hasn't been set, getVirtualGraph will throw a runtime error
    with pytest.raises(popart.popart_exception) as e_info:
        _ = builder.getVirtualGraph()  # stage
    assert "Virtual graph not set in current scope" in e_info.value.args[0]

    # When we enter a VirtualGraph context, the builder should return that stage
    with builder.virtualGraph(0):
        assert builder.hasVirtualGraph() is True
        assert builder.getVirtualGraph() == 0

    with builder.virtualGraph(1):
        assert builder.hasVirtualGraph() is True
        assert builder.getVirtualGraph() == 1

    # Nested virtual graph scopes are not currently supported. See below:
    with builder.virtualGraph(0):
        assert builder.hasVirtualGraph() is True
        assert builder.getVirtualGraph() == 0

        # Virtual graph is stacked now
        with builder.virtualGraph(1):
            assert builder.hasVirtualGraph() is True
            assert builder.getVirtualGraph() == 1

        assert builder.hasVirtualGraph() is True
        assert builder.getVirtualGraph() == 0


def test_get_name_scope():
    builder = popart.Builder()

    # If the scope hasn't been set, getNameScope will return an empty string
    assert builder.getNameScope() == ""

    # If the scope hasn't been set, getNameScope will return the provided string with no delimiter
    string = "foo"
    assert builder.getNameScope(string) == string

    with builder.nameScope("foo"):
        assert builder.getNameScope() == "foo/"

    with builder.nameScope("foo"):
        assert builder.getNameScope("bar") == "foo/bar"

    # Nested namescopes are supported. See below:
    with builder.nameScope("foo"):
        with builder.nameScope("bar"):
            assert builder.getNameScope() == "foo/bar/"
