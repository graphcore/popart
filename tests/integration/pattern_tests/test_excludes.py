# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import pytest
import numpy as np
import popart
import json

# importing test_session requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from test_session import PopartTestSession
import test_util as tu


# Check that an error is thrown when adding an invalid name to excludePatterns
def test_name_checking():
    np.random.seed(1)

    input_data = np.random.rand(2, 2).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(input_data, 'data0')

        x = builder.aiOnnx.identity([d0])
        builder.excludePatterns(x, ["ThisPatternDoesNotExist"])

        builder.addOutputTensor(x)
        return [x]

    session = PopartTestSession()

    with tu.create_test_device() as device:
        with pytest.raises(popart.popart_exception) as e_info:
            session.prepare(init_builder, device=device)

    assert e_info.value.args[0].startswith(
        "Invalid pattern name 'ThisPatternDoesNotExist'")


# Exclude an identity op from PostNRepl and check it is in the final ir.
def test_basic_exclusion():
    np.random.seed(1)

    input_data = np.random.rand(2, 2).astype(np.float32)
    const_data = np.random.rand(2, 2).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(input_data, 'data0')
        c0 = builder.aiOnnx.constant(const_data)

        x = builder.aiOnnx.add([d0, c0])

        # this would normally be removed by PreUniReplPattern
        x = builder.aiOnnx.identity([x])
        builder.excludePatterns(x, ["PostNRepl"])

        x = builder.aiOnnx.add([x, c0])

        builder.addOutputTensor(x)
        return [x]

    session = PopartTestSession()
    with tu.create_test_device() as device:
        session.prepare(init_builder, device=device)

    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))

    main_graph = ir['maingraph']
    assert len(main_graph) == 3

    ops = [i['type'] for i in main_graph]
    assert 'IdentityInplace' in ops


# Create two identity ops and exclude one of them from PostNRepl. Only one should be in the final ir.
def test_remove_one_identity():
    np.random.seed(1)

    input_data = np.random.rand(2, 2).astype(np.float32)
    const_data = np.random.rand(2, 2).astype(np.float32)

    def init_builder(builder):
        d0 = builder.addInputTensor(input_data, 'data0')
        c0 = builder.aiOnnx.constant(const_data)

        x = builder.aiOnnx.add([d0, c0])

        # this would normally be removed by PreUniReplPattern
        x = builder.aiOnnx.identity([x])
        builder.excludePatterns(x, ["PostNRepl"])

        x = builder.aiOnnx.identity([x])

        x = builder.aiOnnx.add([x, c0])

        builder.addOutputTensor(x)
        return [x]

    session = PopartTestSession()
    with tu.create_test_device() as device:
        session.prepare(init_builder, device=device)

    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))

    main_graph = ir['maingraph']
    assert len(main_graph) == 3

    ops = [i['type'] for i in main_graph]

    identities = [i for i in ops if i.startswith('Identity')]
    assert len(identities) == 1


# Create a chain of identity ops.
# All ops exclude themselves from PostNRepl.
# Only the middle op excludes itself from InPlace.
# Check all ops except the middle and output identities are inplace.
def test_inplace_exclude():
    np.random.seed(1)

    input_data = np.random.rand(2, 2).astype(np.float32)

    exclude_inplace_id = ''

    def init_builder(builder):
        nonlocal exclude_inplace_id

        d0 = builder.addInputTensor(input_data, 'data0')

        x = d0
        for i in range(10):
            x = builder.aiOnnx.identity([x])

            if i == 5:
                builder.excludePatterns(x, ["InPlace", "PostNRepl"])
                exclude_inplace_id = x
            else:
                builder.excludePatterns(x, ["PostNRepl"])

        builder.addOutputTensor(x)
        return [x]

    session = PopartTestSession()
    with tu.create_test_device() as device:
        session.prepare(init_builder, device=device)

    ir = json.loads(
        session._session._serializeIr(popart.IrSerializationFormat.JSON))

    main_graph = ir['maingraph']
    assert len(main_graph) == 10

    for i in main_graph:
        o = i['outputs'][0]['name']
        if o == exclude_inplace_id:
            assert i['type'] == 'Identity'
        else:
            assert i['type'] == 'IdentityInplace'
