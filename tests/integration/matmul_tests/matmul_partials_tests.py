# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import re
import popart

# Add parent dir (tests/popart) to path.
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import test_util as tu
from test_session import PopartTestSession


@tu.requires_ipu_model
def test_per_op_partials():
    # Use parameters such that the number of accumulations per output (in each pass) is greater than 16. If the number of
    # accumulations is less than or equal to the number of input channels supported by AMP the planner can ignore
    # the partial type option since no partials are formed outside of the AMP unit.
    data_size = 32
    data1 = np.random.rand(data_size, data_size).astype(np.float16)
    data2 = np.random.rand(data_size, data_size).astype(np.float16)
    data3 = np.random.rand(data_size, data_size).astype(np.float16)
    partials_type = ['', '']

    def init_builder(builder):
        t1 = builder.addInputTensor(data1, 'data_in_1')
        t2 = builder.addInputTensor(data2, 'data_in_2')
        t3 = builder.addInputTensor(data3, 'data_in_3')

        m1 = builder.aiOnnx.matmul([t1, t2], 'mul_1')
        m2 = builder.aiOnnx.matmul([m1, t3], 'mul_2')

        builder.setPartialsType(m1, partials_type[0])
        builder.setPartialsType(m2, partials_type[1])

        o = builder.aiOnnx.add([m1, m2])

        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.device = 'ipu_model'

    # check both convs are using half partials
    partials_type[0] = 'HALF'
    partials_type[1] = 'HALF'
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['half'], ['float'])

    # check both convs are using float partials
    partials_type[0] = 'FLOAT'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['float'], ['half'])

    # check both float and half partials are used
    partials_type[0] = 'HALF'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['half', 'float'], [])


@tu.requires_ipu_model
def test_per_op_partials_train():
    # Use parameters such that the number of accumulations per output (in each pass) is greater than 16. If the number of
    # accumulations is less than or equal to the number of input channels supported by AMP the planner can ignore
    # the partial type option since no partials are formed outside of the AMP unit.
    data_size = 32
    data1 = np.random.rand(data_size, data_size).astype(np.float16)
    data2 = np.random.rand(data_size, data_size).astype(np.float16)
    data3 = np.random.rand(data_size, data_size).astype(np.float16)
    partials_type = ['', '']

    def init_builder(builder):
        t1 = builder.addInputTensor(data1, 'data_in_1')
        t2 = builder.addInputTensor(data2, 'data_in_2')
        t3 = builder.addInputTensor(data3, 'data_in_3')

        m1 = builder.aiOnnx.matmul([t1, t2], 'mul_1')
        m2 = builder.aiOnnx.matmul([m1, t3], 'mul_2')

        builder.setPartialsType(m1, partials_type[0])
        builder.setPartialsType(m2, partials_type[1])

        o = builder.aiOnnx.add([m1, m2])
        builder.addOutputTensor(o)

        loss = builder.aiGraphcore.identityloss([o])

        return [
            loss,
            popart.reservedGradientPrefix() + t1,
            popart.reservedGradientPrefix() + t2,
            popart.reservedGradientPrefix() + t3
        ]

    session = PopartTestSession()
    session.mode = 'train'
    session.device = 'ipu_model'

    # check both convs are using half partials
    partials_type[0] = 'HALF'
    partials_type[1] = 'HALF'
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['half'], ['float'])

    # check both convs are using float partials
    partials_type[0] = 'FLOAT'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['float'], ['half'])

    # check both float and half partials are used
    partials_type[0] = 'HALF'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['half', 'float'], [])


@tu.requires_ipu_model
def test_global_partials():
    # Use parameters such that the number of accumulations per output (in each pass) is greater than 16. If the number of
    # accumulations is less than or equal to the number of input channels supported by AMP the planner can ignore
    # the partial type option since no partials are formed outside of the AMP unit.
    data_size = 32
    data1 = np.random.rand(data_size, data_size).astype(np.float16)
    data2 = np.random.rand(data_size, data_size).astype(np.float16)
    partials_type = ['', '']

    def init_builder(builder):
        t1 = builder.addInputTensor(data1, 'data_in_1')
        t2 = builder.addInputTensor(data2, 'data_in_2')

        o = builder.aiOnnx.matmul([t1, t2], 'mul_1')

        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.device = 'ipu_model'

    # check convs are using half partials
    session.options.partialsTypeMatMuls = "half"
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['half'], ['float'])

    # check convs are using float partials
    session.options.partialsTypeMatMuls = "float"
    session.prepare_and_run(init_builder)
    _check_for_matmul_partials(session, ['float'], ['half'])


# check the summary report to see which conv partials are being used
# TODO
@tu.requires_ipu_model
def _check_for_matmul_partials(sess, includes, excludes):
    sr = sess._session.getSummaryReport()
    sr = sr.splitlines()
    # get to the memory usage section
    for i in range(len(sr)):
        line = sr[i]
        if line.startswith('Memory Usage:'):
            sr = sr[i:]
            break

    # get to the vertex data
    for i in range(len(sr)):
        line = sr[i].strip()
        if line.startswith('Vertex Data ('):
            sr = sr[i:]
            break

    # get to the by type
    for i in range(len(sr)):
        line = sr[i].strip()
        if line.startswith('By Type:'):
            sr = sr[i + 1:]
            break

    # get this whole of this section
    for i in range(len(sr)):
        line = sr[i].strip()
        if line == '':
            sr = sr[:i]
            break

    types = [i.split()[0] for i in sr]
    for item in includes:
        # It is difficult to know which option the convolution planner will choose,
        # so we check for any.
        # MatMuls are decomposed by Poplar into 1x1 convolutions.
        r = re.compile(f"poplin::ConvPartial1x1Out.*<half,{item}.*")
        assert (len(list(filter(r.match, types))) != 0)

    for item in excludes:
        r = re.compile(f"poplin::ConvPartial1x1Out.*<half,{item}.*")
        assert (len(list(filter(r.match, types))) == 0)
