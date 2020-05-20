# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import popart
import test_util as tu
from test_session import PopartTestSession


@tu.requires_ipu_model
def test_per_op_partials():
    data_size = 4
    kernel_size = 3
    datas = np.random.rand(1, 2, data_size, data_size).astype(np.float16)
    kernel = np.random.rand(3, 2, kernel_size, kernel_size).astype(np.float16)
    partials_type = ['', '']

    def init_builder(builder):
        d1 = builder.addInputTensor(datas, 'data_in')
        k = builder.addInputTensor(kernel)

        c1 = builder.aiOnnx.conv([d1, k],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1])
        c2 = builder.aiOnnx.conv([d1, k],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1])

        builder.setPartialsType(c1, partials_type[0])
        builder.setPartialsType(c2, partials_type[1])

        o = builder.aiOnnx.add([c1, c2])

        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.device = 'ipu_model'

    # check both convs are using half partials
    partials_type[0] = 'HALF'
    partials_type[1] = 'HALF'
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['half'], ['float'])

    # check both convs are using float partials
    partials_type[0] = 'FLOAT'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['float'], ['half'])

    # check both float and half partials are used
    partials_type[0] = 'HALF'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['half', 'float'], [])


@tu.requires_ipu_model
def test_per_op_partials_train():
    data_size = 4
    kernel_size = 3
    datas = np.random.rand(1, 2, data_size, data_size).astype(np.float16)
    kernel = np.random.rand(3, 2, kernel_size, kernel_size).astype(np.float16)
    partials_type = ['', '']

    def init_builder(builder):
        d1 = builder.addInputTensor(datas, 'data_in')
        k = builder.addInputTensor(kernel)

        c1 = builder.aiOnnx.conv([d1, k],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1])
        c2 = builder.aiOnnx.conv([d1, k],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1])

        builder.setPartialsType(c1, partials_type[0])
        builder.setPartialsType(c2, partials_type[1])

        o = builder.aiOnnx.add([c1, c2])
        loss = builder.aiGraphcore.identityloss([o])

        builder.addOutputTensor(o)
        return [
            loss,
            popart.reservedGradientPrefix() + d1,
            popart.reservedGradientPrefix() + k
        ]

    session = PopartTestSession()
    session.mode = 'train'
    session.device = 'ipu_model'

    # check both convs are using half partials
    partials_type[0] = 'HALF'
    partials_type[1] = 'HALF'
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['half'], ['float'])

    # check both convs are using float partials
    partials_type[0] = 'FLOAT'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['float'], ['half'])

    # check both float and half partials are used
    partials_type[0] = 'HALF'
    partials_type[1] = 'FLOAT'
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['half', 'float'], [])


@tu.requires_ipu_model
def test_global_partials():
    data_size = 4
    kernel_size = 3
    datas = np.random.rand(1, 2, data_size, data_size).astype(np.float16)
    kernel = np.random.rand(3, 2, kernel_size, kernel_size).astype(np.float16)

    def init_builder(builder):
        d1 = builder.addInputTensor(datas, 'data_in')
        k = builder.addInputTensor(kernel)

        o = builder.aiOnnx.conv([d1, k],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1])

        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()
    session.device = 'ipu_model'

    # check convs are using half partials
    session.options.convolutionOptions = {'partialsType': 'half'}
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['half'], ['float'])

    # check convs are using float partials
    session.options.convolutionOptions = {'partialsType': 'float'}
    session.prepare_and_run(init_builder)
    _check_for_conv_partials(session, ['float'], ['half'])


# check the summary report to see which conv partials are being used
@tu.requires_ipu_model
def _check_for_conv_partials(sess, includes, excludes):
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
        # so we check for both.
        line_1 = f'poplin::ConvPartialHorizontalMac<half,{item},true>'
        # Ignore last parameter for line_2 before D22061 gets landed
        line_2 = f'poplin::ConvPartial1x1Out<half,{item},true,false'
        print(f'Checking {line_1} or {line_2} is present')
        assert (line_1 in types
                or (len([s for s in types if line_2 in s]) != 0))

    for item in excludes:
        line_1 = f'poplin::ConvPartialHorizontalMac<half,{item},true>'
        # Ignore last parameter for line_2 before D22061 gets landed
        line_2 = f'poplin::ConvPartial1x1Out<half,{item},true,false'
        print(f'Checking {line_1} and {line_2} is not present')
        assert ((line_1 not in types)
                and (len([s for s in types if line_2 in s]) == 0))
