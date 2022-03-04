# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import numpy as np
import re
import popart
import test_util as tu
from test_session import PopartTestSession


@tu.requires_ipu_model
def test_per_op_partials():
    # Use parameters such that the number of accumulations per output (in each pass) is greater than 16. If the number of
    # accumulations is less than or equal to the number of input channels supported by AMP the planner can ignore
    # the partial type option since no partials are formed outside of the AMP unit.
    batch_size = 5
    input_channels = 4
    output_channels = 3
    data_size = 4
    kernel_size = 3
    datas = np.random.rand(batch_size, input_channels, data_size,
                           data_size).astype(np.float16)
    kernel = np.random.rand(output_channels, input_channels, kernel_size,
                            kernel_size).astype(np.float16)
    partials_type = ['', '']

    def init_builder0(builder):
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

    def init_builder1(builder):
        d1 = builder.addInputTensor(datas, 'data_in')
        k = builder.addInputTensor(kernel)

        [c1, c2] = builder.aiGraphcore.multiconv([[d1, k], [d1, k]],
                                                 dilations=[[1, 1], [1, 1]],
                                                 pads=[[1, 1, 1, 1],
                                                       [1, 1, 1, 1]],
                                                 strides=[[1, 1], [1, 1]],
                                                 partialsTypes=partials_type)
        o = builder.aiOnnx.add([c1, c2])

        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()

    # check both convs are using half partials
    partials_type[0] = 'HALF'
    partials_type[1] = 'HALF'
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder0, device=device)
        _check_for_conv_partials(session, ['half'], ['float'])
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder1, device=device)
        _check_for_conv_partials(session, ['half'], ['float'])

    # check both convs are using float partials
    partials_type[0] = 'FLOAT'
    partials_type[1] = 'FLOAT'
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder0, device=device)
        _check_for_conv_partials(session, ['float'], ['half'])
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder1, device=device)
        _check_for_conv_partials(session, ['float'], ['half'])

    # check both float and half partials are used
    partials_type[0] = 'HALF'
    partials_type[1] = 'FLOAT'
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder0, device=device)
        _check_for_conv_partials(session, ['half', 'float'], [])
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder1, device=device)
        _check_for_conv_partials(session, ['half', 'float'], [])


@tu.requires_ipu_model
def test_per_op_partials_train():
    # Use parameters such that the number of accumulations per output (in each pass) is greater than 16. If the number of
    # accumulations is less than or equal to the number of input channels supported by AMP the planner can ignore
    # the partial type option since no partials are formed outside of the AMP unit.
    batch_size = 5
    input_channels = 4
    output_channels = 3
    data_size = 4
    kernel_size = 3
    datas = np.random.rand(batch_size, input_channels, data_size,
                           data_size).astype(np.float16)
    kernel = np.random.rand(output_channels, input_channels, kernel_size,
                            kernel_size).astype(np.float16)
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

    # check both convs are using half partials
    partials_type[0] = 'HALF'
    partials_type[1] = 'HALF'
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder, device=device)
        _check_for_conv_partials(session, ['half'], ['float'])

    # check both convs are using float partials
    partials_type[0] = 'FLOAT'
    partials_type[1] = 'FLOAT'
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder, device=device)
        _check_for_conv_partials(session, ['float'], ['half'])

    # check both float and half partials are used
    partials_type[0] = 'HALF'
    partials_type[1] = 'FLOAT'
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder, device=device)
        _check_for_conv_partials(session, ['half', 'float'], [])


@tu.requires_ipu_model
def test_global_partials():
    # Use parameters such that the number of accumulations per output (in each pass) is greater than 16. If the number of
    # accumulations is less than or equal to the number of input channels supported by AMP the planner can ignore
    # the partial type option since no partials are formed outside of the AMP unit.
    batch_size = 5
    input_channels = 4
    output_channels = 3
    data_size = 4
    kernel_size = 3
    datas = np.random.rand(batch_size, input_channels, data_size,
                           data_size).astype(np.float16)
    kernel = np.random.rand(output_channels, input_channels, kernel_size,
                            kernel_size).astype(np.float16)

    def init_builder0(builder):
        d1 = builder.addInputTensor(datas, 'data_in')
        k = builder.addInputTensor(kernel)

        o = builder.aiOnnx.conv([d1, k],
                                dilations=[1, 1],
                                pads=[1, 1, 1, 1],
                                strides=[1, 1])

        builder.addOutputTensor(o)
        return [o]

    def init_builder1(builder):
        d1 = builder.addInputTensor(datas, 'data_in')
        k = builder.addInputTensor(kernel)

        [o] = builder.aiGraphcore.multiconv([[d1, k]], pads=[[1, 1, 1, 1]])

        builder.addOutputTensor(o)
        return [o]

    session = PopartTestSession()

    # check convs are using half partials
    session.options.convolutionOptions = {'partialsType': 'half'}
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder0, device=device)
        _check_for_conv_partials(session, ['half'], ['float'])
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder1, device=device)
        _check_for_conv_partials(session, ['half'], ['float'])

    # check convs are using float partials
    session.options.convolutionOptions = {'partialsType': 'float'}
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder0, device=device)
        _check_for_conv_partials(session, ['float'], ['half'])
    with tu.create_test_device() as device:
        session.prepare_and_run(init_builder1, device=device)
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
        # so we check for any.
        r = re.compile(f"poplin::ConvPartial.*<half,{item}.*")
        assert (len(list(filter(r.match, types))) != 0)

    for item in excludes:
        r = re.compile(f"poplin::ConvPartial.*<half,{item}.*")
        assert (len(list(filter(r.match, types))) == 0)
