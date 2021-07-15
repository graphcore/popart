# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import json
import pytest
import popart
import pytest
import test_util as tu
from test_session import PopartTestSession


def test_subgraph_ir_name():

    i1 = np.array([3]).astype(np.float32)
    i2 = np.array([6]).astype(np.float32)
    cond = np.array([True]).astype(np.float32)
    main_name = "main_graph"
    then_name = "then_branch"
    else_name = "else_branch"

    def init_builder(builder):
        builder.setGraphName(main_name)
        a = builder.addInputTensor(i1)
        b = builder.addInputTensor(i2)
        cond_var = builder.addInputTensor(cond)

        bbuilder = builder.createSubgraphBuilder()
        bbuilder.setGraphName(then_name)
        bbuilder.addInputTensorFromParentGraph(a)
        bbuilder.addInputTensorFromParentGraph(b)
        bbuilder.addOutputTensor(bbuilder.aiOnnxOpset10.add([a, b]))

        ebuilder = builder.createSubgraphBuilder()
        ebuilder.setGraphName(else_name)
        ebuilder.addInputTensorFromParentGraph(a)
        ebuilder.addInputTensorFromParentGraph(b)
        ebuilder.addOutputTensor(ebuilder.aiOnnxOpset10.sub([b, a]))

        o = builder.aiOnnx.logical_if([cond_var], 1, ebuilder, bbuilder)[0]
        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.prepare(init_builder)

    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))
    assert (len(ir[then_name]) == 1)
    assert (len(ir[else_name]) == 1)


def test_main_graph_ir_name():

    np.random.seed(1)
    input_data = np.random.rand(2).astype(np.float32)
    graph_name = "body_graph"

    def init_builder(builder):
        builder.setGraphName(graph_name)
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.add([i1, i1])
        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.prepare(init_builder)

    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))
    assert (len(ir[graph_name]) == 1)


def test_empty_graph_ir_name():

    np.random.seed(1)
    input_data = np.random.rand(2).astype(np.float32)

    def init_builder(builder):
        i1 = builder.addInputTensor(input_data)
        o = builder.aiOnnx.add([i1, i1])
        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.prepare(init_builder)

    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))
    assert (len(ir["maingraph"]) == 1)
