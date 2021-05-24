# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# THIS IS AN AUTOGENERATED FILE, DO NOT EDIT DIRECTLY
#
# To regenerate this file run:
#     python popart/tests/popart/operators_test/generate_keras_lstm_models.py
#
# File generated using TensorFlow version 2.3.0
# and keras2onnx version 1.7.0

import numpy as np
from numpy import array, float32
import popart
import onnx
import pytest

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def _run_comparison_test(data, result, proto, expected_activations,
                         lstm_op_pattern):
    model = onnx.load_from_string(proto)

    if expected_activations:
        lstms = [i for i in model.graph.node if i.op_type == 'LSTM']
        assert len(lstms) == 1
        activations = [
            i for i in lstms[0].attribute if i.name == 'activations'
        ]
        assert len(activations) == 1
        activations = activations[0].strings
        assert len(activations) == len(expected_activations)
        for expected, actual in zip(expected_activations, activations):
            assert expected == actual.decode('utf-8').lower()

    outId = model.graph.output[0].name
    inId = model.graph.input[0].name

    dataFlow = popart.DataFlow(1, {outId: popart.AnchorReturnType("All")})
    patterns = popart.Patterns(popart.PatternsLevel.Default)
    patterns.enablePattern('LSTMOp', lstm_op_pattern)
    session = popart.InferenceSession(fnModel=proto,
                                      dataFlow=dataFlow,
                                      deviceInfo=tu.create_test_device(),
                                      patterns=patterns)

    session.prepareDevice()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({inId: data}, anchors)
    session.run(stepio)

    assert np.allclose(anchors[outId], result)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_basic(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[-0.02019063, -0.193365, 0.00335996, -0.17738362]],
                   dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xa5\x08\n?\n\nlstm_input\x12\x06lstm_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xa6\x01\n\x06lstm_X\n\x06lstm_W\n\x06lstm_R\n\x06lstm_B\n\x00\n\x00\n\x00\n\x00\x12\x06lstm_Y\x12\x08lstm_Y_h\x12\x08lstm_Y_c\x1a\x04lstm"\x04LSTM*%\n\x0bactivationsJ\x07SigmoidJ\x04TanhJ\x04Tanh\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n3\n\x08lstm_Y_h\x12\x04lstm\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\nsequential*\x93\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02TYi\xbe,\x18\xe8>\x00\x06j\xbex<\x8d=\xd4\x93\xa4\xbePS5>N\x0f\xe0\xbe>\x1b\xab>\xd0\xcb\x1e=\xde\x08\xbb>\xba\x83\xbb\xbe\x80I\xbc=\xf0m\x89=7X\xe9\xbe\x1c?\x97>\xf6\x97\xde\xbe\x08\x1b\xa2=\xa8.\xfc\xbe\x004\x07?H\xc0\xed>\xfc\xb5\x17\xbe\xe01\x8d>L\xd8\xb5\xbe\xfaq\xd2>\xa4P\xb2\xbd\x9c\x90\xb7\xbe4\xd7\xfe\xbe\x90\xc2\x0f\xbd\xbc\xe3\x11>\x84\xf5R\xbe\xe0\xd4\xda=\x90\xfb\x06\xbd<\xf0\xba\xbd@\rw\xbeX8\xc0=\\\x88\xf9>\x9c\xa6\xac>\x1f\xdd\x07\xbf\x00,\x07?\x12\xdes\xbe\x800\xf3\xbbl\xee}>\x98\xaa\xe8>0$m=>\xb1\x0b?\xa0\xee\x94>z\xfa\xc8\xbe@Yw\xbeT\xaf\\>L\xc3I>l\xb6\xe1\xbe\x9d\xd7\xeb\xbe\xae\x1d\xd2\xbe0\xd9\x12=\xc0n\xee=\xbc\xea\xf7\xbd\xe0[k>\xda\xde\x8f>\xb2\xa4\xb4\xbe`G\xf1\xbe\xfc/6>&\xd6\xfd\xbe \xff\xa1<\x08u\x00\xbfB\x06lstm_W*\x93\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\x08\x84i>\xec\xb4\xc2=*\xa1\x0b\xbe\x82\xb4\x14>mo\x1b\xbd\r^\x84\xbe\xcc\xfa\x02\xbf\xb6\xc9"?\x85\xbc/>\x10\x9ed\xbe\x10\xdc\xbd\xbd\xb80L=\x8d\x1f\x17\xbf,\x9d\xff>d\xb1\x00\xbeVE2>\x11\x17\x04?)\xa7&>n\x1a\xa1<\x16\xc0\x8e>N\x1a\xd3\xbd\xb4*c\xbb\xde\xf4#=\r\x8a\xcc>\x12\xae\xbf;\xb4f<\xbe\t\xb7\x1b=\t\xeaJ\xbe\xb82\x16>\x10\x0f\x8f\xbd\xaeD\x9f>\xea\xd6\x07>Y\x85\xa4\xbe.8\x9f\xbe~\xae\xd6>(\x8c;>D\x89\xe9\xbc\xabU\x9f<\xdf!\xf9\xbe\xb4\xc8\xae\xbeI\x90\xdf\xbc\\\xc0\xaa>*`\xfa\xbd\xee\xceE\xbd\xfb\xbe\x1e>\xbe\xb9\xfb\xbcj\xe8\x8f\xbeO5\x1b\xbd\x91r\x97\xbe\xa1z\x99\xbe\xa8\x02\x8d\xbb\xaeH\xad=\xd9\xdb(=\x14\x9d\xf7>\xbce&>\xefXH>\x18\xd8,\xbd\x84\xbf\x0f=\x1e!\xc6<L\xdc;\xbe\x99\xe0e\xbe\xef\xc2#\xbe\x94\xf6}\xbe\xca\xab\xf7\xbdB\x06lstm_R*\x91\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x06lstm_BZ \n\nlstm_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x16\n\x04lstm\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, [], lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_relu_relu(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[0., 0., 0., 0.03986165]], dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xc6\x08\nC\n\x0clstm_1_input\x12\x08lstm_1_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xb3\x01\n\x08lstm_1_X\n\x08lstm_1_W\n\x08lstm_1_R\n\x08lstm_1_B\n\x00\n\x00\n\x00\n\x00\x12\x08lstm_1_Y\x12\nlstm_1_Y_h\x12\nlstm_1_Y_c\x1a\x06lstm_1"\x04LSTM*"\n\x0bactivationsJ\x04ReluJ\x04ReluJ\x04Relu\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n7\n\nlstm_1_Y_h\x12\x06lstm_1\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\x0csequential_1*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xde\xf1\xaa\xbe\x84dQ\xbe$\x07\xd1>\xa4\x917>\x98\xb4\x86>\x1c>\'>\xd4\xaf">0,$>@\xeeZ\xbc\x02G\x0b?\xb4\xe7\x1e>\xb8y.>P\x81\xd2>c\xd0\xfa\xben\xae\xc1>N7\x86>D\xe6\\>\x1a\xab\x01\xbf\x92\xdb\x0b?T\xc5\xe1\xbeb\x0f\x0b\xbf@\x9aQ\xbc\x88{6\xbe,\x13\xfc\xbdVv\x18\xbe\x80\xce\xd7;H\x99\x08>DN\x03\xbf0\xc1\x92=\xa4\xe1\xfd>\xc0\xf9\x08<B\xb2\x03\xbf\x80c\xd0\xbd\xa89\xdb>\xccG\xab>&N\xd9>\x1e#\x08\xbfGW\x82\xbe\x00\x9a\xbc:\xef\xd0\xc8\xbe\xf0\x0ck><1\x85\xbd\x04\xd3r>\xd0\xb9C=\xac\xdc\x16\xbe.\xf4\x84>|,\xa4>8\x88\x1f\xbe\xc8e\xb5=^5\xb9\xbe\x02\x02\x86>\x00\xa2\xce\xbcZ\xd1\xe0>\x0e\xb0\x91\xbe\xaf\xa8\xf1\xbe@0\xef\xbe\xb8\x7f\xce\xbe\x8c\xbe\x04?\xd8\n\xb3\xbe\x9b\x8e\xee\xbeP\xc4I=$q\x07?\x90\xb5>\xbe\x9c\xa0h\xbeB\x08lstm_1_W*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02d<\xaa>S\xf0r\xbe\xfb\xb8y\xbb\xa4\x08\x82=\xc5L\xa8>\xdaT\x88<_\x84t\xbe\x1f\xf1\x0e>\x8b]Z<\xe9!\x80\xbe\x10\xdef>`\x94\x97\xbe\'a\x1e\xbe\xd3\xdc\xf9>GP\x94\xbe\x8e%V\xbe\xdb\xfc\x8a\xbe\xa7\xe6(>\x84\xbf\xbb=5<@\xbe\xafR\xc0\xbe|*\x0f>\xde\xe4\xd1\xbdD\xf7\xa0;MW">\xc9\xe0\x80>])\xd1\xbc\x8c\xc8\xf8\xbe\x12\x1bK=P\xaf\xf3>\xcc\x92n>\xb1\xd5\xb2>D\xc1?>)\xc5\xe7>P\x88\x03?_\xc3<>T\xc4<\xbe\x92\xe5R<P\xbf\xf9\xbd\xee\xda\x80\xbe\xa5\x0c\x0c\xbe\xbe\x8ag\xbd\xb767\xbe\x84\xbe\xe7\xbb\xd7T\x8b\xbd\x16W\xe9\xbd\xd0r\xa9>\x0f\xad\xbd\xbd\xd2\xb3\x0e\xbf3h\xac\xbd\x0c\xf8F\xbb0"\xc1>\xd1\xf5\x97>X\xee[>\x95\xbe\xf3\xbe+\xf9\xc1=\xbeC\xba\xbd\x9b\xa4+>\x90T\x87=B\x95\xc6\xbeG\x84\xfe\xbdjTp=}z\x93\xbe\xd0\x02W>B\x08lstm_1_R*\x93\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x08lstm_1_BZ"\n\x0clstm_1_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x18\n\x06lstm_1\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['relu', 'relu', 'relu'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_relu_sigmoid(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[0.14236511, 0., 0.17311893, 0.]], dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xc9\x08\nC\n\x0clstm_2_input\x12\x08lstm_2_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xb6\x01\n\x08lstm_2_X\n\x08lstm_2_W\n\x08lstm_2_R\n\x08lstm_2_B\n\x00\n\x00\n\x00\n\x00\x12\x08lstm_2_Y\x12\nlstm_2_Y_h\x12\nlstm_2_Y_c\x1a\x06lstm_2"\x04LSTM*%\n\x0bactivationsJ\x07SigmoidJ\x04ReluJ\x04Relu\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n7\n\nlstm_2_Y_h\x12\x06lstm_2\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\x0csequential_2*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xf5k\x08\xbf\x98\xcd\x80=\xcco >\xfe7\xb6>\x84m\xe4\xbeD\xc3\xc5\xbe\xc0\xc5\x88<\xd4|\xd2>\xc0\x1ds\xbc\x8ah\xc0>\x19\x08\xb2\xbe\xfcV\x97>\x9c\x0bf>\x00\xfc\xb6\xbc\xa0\xba\x11=\x0c;\x1c>\xf2\x90\xa3\xbe\\\xb5o>\xca\xe5\x08\xbe\xb0\xc9n\xbd*%\xa5>\xdet\x8d\xbe\xd2\xb2\x94>@C\x19\xbe\x14\xf5\xef>\xa2\xd3\x00\xbfb4\x01\xbft\xee\xe2\xbe\xe8\xfb\xa6=\x80.\'\xbd\xdaUc\xbe{J\xab\xbev\x9e \xbe\xb2\xb2\x01\xbf\xfe\x9d\xbe>\xca\xb7x\xbe\xa0\x06x=(\x96\x00?\xde\xe1\xa4>\xa44>>\xdc\xc5V>,P\x93\xbe\xb0\x18\x7f=\xc4\x08^\xbeD\x14\xb1\xbd\xf0\xeed>\xc4!\xdd\xbedA\xb8\xben\xcb\xa3\xbe\xfe6\x99>\x0e\xcd\x86>\x84Q\xc2>\xfc\xb4\xd4\xbe\xf2D\x0f\xbe\xe4\xb7S\xbe\xf49\x9e>\xd00N\xbe\x80\x8b\x0b?\xfbm\xfd\xbeb\xfc\xb8>\x02\xff\xff\xbe\x12Ln\xbe\xd0\xf00\xbd\xb2\x8e\xaa\xbeB\x08lstm_2_W*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02`&\x85\xbee\xa0\x88\xbe\x04\xda\xb0<v]\x1a?\x80\xdf\xe7\xbd\x85\xd0@\xbe\xa1\x16\n\xbe\x1e~\x86=\xc2)$>\x0b.\xf6>\xcd\xf3\xb9\xbeB\x08E=?D\xcf>\x89\xd9\xed\xbd\x0b\x81\t?\xb2\xfc\xab\xbe\x01\xeb\xae=8Q\xeb\xbdp\x97O\xbeVF\x1d\xbe\xea\xe7\xcb\xbe\x94\x19\xbc\xbe\xa2OX\xbe\x14\x0e\xb3\xbe\xc0I\x95\xbd\x8f\xd8\x0f>\x1fx;>\x9a\xbc\x9f=\xab\xb3)?\x1c\xc4\xc4\xbe9\x8b\xb7\xbe\xb8%G>%\x11\x92\xbd_\x19\xb1\xbe\xd7\xe1\x1a>\xe6\xb8\xba=C\xa7\xf8=B6\x86\xbd\xe4F\x0f\xbe\xba\x06\xd4=\x18\x81M\xbep\x1c\xc9\xbb|o%\xbd8`\n\xbc\xcc\x95\xcb\xbd\xbb\xf9\x97>V\x95z<?\rY>\x89\xa2\x84\xbd\xbf\x9au\xbeU\xf0\x8a\xbd\x0b\x04\xb0\xbe~\xb8\xee\xba\xbfPe>\xfd\xb4\xa1\xbe\x89O\xb0\xbe|tX>\xb8\x9d\x02<\xc8O\xe0=*\xbc$>\xde\xa2\xaf\xbc}\xa4\xeb\xbd\xf8\xf4\xc7\xbeO\xbf\xb0<B\x08lstm_2_R*\x93\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x08lstm_2_BZ"\n\x0clstm_2_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x18\n\x06lstm_2\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['sigmoid', 'relu', 'relu'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_relu_tanh(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[-0., -0., -0., 0.01629485]], dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xc6\x08\nC\n\x0clstm_4_input\x12\x08lstm_4_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xb3\x01\n\x08lstm_4_X\n\x08lstm_4_W\n\x08lstm_4_R\n\x08lstm_4_B\n\x00\n\x00\n\x00\n\x00\x12\x08lstm_4_Y\x12\nlstm_4_Y_h\x12\nlstm_4_Y_c\x1a\x06lstm_4"\x04LSTM*"\n\x0bactivationsJ\x04TanhJ\x04ReluJ\x04Relu\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n7\n\nlstm_4_Y_h\x12\x06lstm_4\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\x0csequential_4*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xf0e\x00=\xa0\x87\xe1\xbe\xd4\xa3R>\xa0\n{=Pd\xce\xbe\xc0o_\xbdF\x85\x90\xbe\x05\xb3\xc1\xbe0\nu>\xfch\xce\xbd\x90<\x84>68\x02\xbf\xa0g\x89=d\xce+>\xe8\xacE\xbdZ\x15\xd3>X_\xd1\xben\xf6\x92\xbe\x066\x9d>\x90\x87\x04=\x04j\xa8>w<\xf2\xbe\x0e\xce\xdd>P\x8an\xbe\x80k\x9a\xbe\x10\xd4\x93=F\x8e\x84\xbe<D\x98>\xf0\xc3\xe2>@\r.\xbd\x82\x95\xe7\xbe\x14\xf4\x88>\x08:\x9c=h\xff\xd4\xbe\xa0\xec\x1f=6G\xcf\xbe\x82#\x00\xbf\x94\nU>\x00\x99\x16\xbb\xbe4\xb1>(\xad\xda\xbd\x93\x9d\x08\xbfL\xa8\x86\xbd\'\xb5\x08\xbf\xb0\xc66\xbd:\x8d\xfc\xbe\x8b\x1b\xf7\xbe\xb0\x10l=\x02\x86\xc7>\x18\xec\xa4\xbd\x90\x98;\xbdLu2\xbe\xee]\xb5\xbe@h\x0b?";\xb5>\xa14\x89\xbe`\x0e\\=\xce\xab\xaf\xbe\x00R\xab\xbd\xbc9\xc9\xbe\xd0\x07\xf2>\xf8\xe1\x17\xbe4\x93g>\x93\x05\xb3\xbeB\x08lstm_4_W*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xd8\x8c\x82>\xa0\xbb\x1a>)\x1e\x80>\xc5}\x11?\xd7m\x10\xbe\x11o]\xbeC\xb4\\\xbcs\x1a\xb6\xbd\xb7\x9f\xd1=J\x13\x04>_\x8e*=\x86[d\xbe\x94-\xc5\xbex8\xf8\xbdD\xea8>aS\x7f>\xecw\xac>\x94\x17\x9e\xbe]\x0c\xed>\x98!\xab\xbd\x1a\xed\xa9\xbd\xed9\xa9>\xce\xb9s\xbe\xe2*\x9d\xbeL\xc0\xfd\xbdU?\x1f\xbeE\x04#\xbe\x16\x17/\xbdi\xa7\\\xbe\xe1 \xdf\xbc#y\xdc>\xf2\xaa\xcd\xber)\xc4\xbe\xaem)>\xec\x01I\xbd(3\x83\xbd\x04V\xef=i\x91\xef>0\x8c~\xbb\x93v\x93\xbely\xd0\xbdzWS\xbe\x9e\xf7\x92>\x08mG\xbeQ\x9cu\xbe85\xe3>/\xfe\xed>,U\xd8<YT\xd4\xbd\x94NJ\xbe\xb7\x1dh>\xdd\xe0I\xbe\xbe\x06\xd4\xbe1!\x02\xbe\xe8\r$\xbe\xdb0\x15>5!\xc6\xbeb\xed\xcd\xbd ;\x17\xbbo\xa3\x19>\xe3\xf0\xf0\xbd\xa24\xb0>\x12\xb3T>\x1e\xd8\x94>B\x08lstm_4_R*\x93\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x08lstm_4_BZ"\n\x0clstm_4_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x18\n\x06lstm_4\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['tanh', 'relu', 'relu'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_sigmoid_relu(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[0., 0., 0.17980874, 0.]], dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xcc\x08\nC\n\x0clstm_5_input\x12\x08lstm_5_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xb9\x01\n\x08lstm_5_X\n\x08lstm_5_W\n\x08lstm_5_R\n\x08lstm_5_B\n\x00\n\x00\n\x00\n\x00\x12\x08lstm_5_Y\x12\nlstm_5_Y_h\x12\nlstm_5_Y_c\x1a\x06lstm_5"\x04LSTM*(\n\x0bactivationsJ\x04ReluJ\x07SigmoidJ\x07Sigmoid\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n7\n\nlstm_5_Y_h\x12\x06lstm_5\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\x0csequential_5*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02.2\xe3\xbe\xd47\xa2>\xc8\xf5\xa9=\xd4\xa6\xa2\xbd\xf4\xdf\x9d>\xfec\xe5\xbeno\x05\xbf6\x91\t\xbf@\xf1\xb5>\xbb\xaa\xf4\xbe\xb0\x92\xff=\\\xb9\xbc\xbe\x90\xfa(=\x1e\xfd\x06\xbe\x80\xcd\xeb>\xcb\x15\x08\xbf\x97\xe6\x97\xbe\\\xd1)\xbe\xd4#\x19\xbe\xbf\xf0\x92\xbeX\xc1\xf4=Mx\xa9\xbe\xe0d\x19>\xcf\xd5\xc0\xbeL\xfa?>$\x8c:>\xa2\xa9\x9e\xbe@Q\xf2>X\x14v\xbe\xbd\xf4\x03\xbf0\x8eq>P\xf9\n=4\xae\xaf>\xbc\xbf\xef\xbd\xc87\xa4>8\xa7%>\x94$P\xbe\xc8\xf2\x00?D:\xac\xbe\x04\xbb5>\xe8_\x92=\xc8\x84_>\x9e\xc5\xc4\xbe\x84?\xb7>\xc4\'\x0c>XL\xe1=\x03\\\xf4\xbe\xd8\xa1\xc8\xbelw\xbd>\x00\xc7k<\x046\n?j\xf0\xf8\xbe\x80V\xe2\xbc\xdc\xd0\x01\xbe\xc8g!\xbe>\x0c\x9d\xbehz\x9c>\xdcF\x93>\xa0G\xe6\xbc\xc0\x94\x0b=\x0e\x08\xe1>\xa9t\xa0\xbe\xc0\x07\xe9>\xd2\xc2\x13\xbeB\x08lstm_5_W*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xd0\x89\xa2\xbeWTv\xbd;\x0b\xda\xbe\x83[\xb6>\xf9\x90\x9e\xbej\xed)=\nH}\xbe\x90J\r>4c\xbb\xbd\xc8\xa3\xdb=V\x98\xeb\xbe1\x97\x9a>\xe8\xbf\x9b=\x9b\xfei>\xe5_\x13>\xa2\x1e\xa8\xbd\x1f\xcbz>\x05\x83\xc7=\xc0\xfe\x8a\xbe~+\x91\xbe5\xa5$\xbe\xad\x7f\x92\xbe\xe7\xb1\xbe\xbd\x031\x1b\xbeQ\x08\x81\xbe\xa6\xa2\xd2>\x87k\xbc>Q\xc4\xcd=+\xb0p>l\x96\xfb\xbd\xa2-V=\x02?\x8d>\xb5\xaaG\xbdj\xf7\x87>\xe07\x8d>\x853\xb2>\xbbL$\xbeu\xc9\xfa<=\x03\x05\xbe\x02p\xc4\xbd\xb2\xae\x9a>\xa4\xb7\xdf>hf\x90\xbe\x80qP>*\xc4\xe3=Xs\xbe\xbd\x10\xe0\x8e>\xcd)Q>\x90;\xde\xbcC\xdeM>\xbbN\x1e\xbeos\t\xbf-"\x0c?\xe4p\x05>\xe4\xbd\xc0\xbd)6.>+B\xbc\xbe\x9a\xff\x04>\xdd\xb0\x04>\xb9\x07T=\xc2]\x06\xbe\xa8\x85\x0e?\xe6l\x8a\xbdz\xb3:\xbeB\x08lstm_5_R*\x93\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x08lstm_5_BZ"\n\x0clstm_5_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x18\n\x06lstm_5\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['relu', 'sigmoid', 'sigmoid'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_sigmoid_sigmoid(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[0.34316674, 0.34707376, 0.42337856, 0.39620247]],
                   dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xcf\x08\nC\n\x0clstm_6_input\x12\x08lstm_6_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xbc\x01\n\x08lstm_6_X\n\x08lstm_6_W\n\x08lstm_6_R\n\x08lstm_6_B\n\x00\n\x00\n\x00\n\x00\x12\x08lstm_6_Y\x12\nlstm_6_Y_h\x12\nlstm_6_Y_c\x1a\x06lstm_6"\x04LSTM*+\n\x0bactivationsJ\x07SigmoidJ\x07SigmoidJ\x07Sigmoid\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n7\n\nlstm_6_Y_h\x12\x06lstm_6\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\x0csequential_6*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02V\xfe\x08\xbf\x9c-\xd1\xbd\xfa\x94\x87>\xc8\xa1p>\xc0\xbd\x11<\xe6\xcaR\xbe\xacD|\xbe(tf>Dz\x88>p^b=\xb4\xa4\x7f>N5\xcd\xbe\xe6\xc8\x05?\x1c\xf7\x05?\xc2\xaa\xa1>\xec\xeb\xfb\xbe$\xa9\xf7>hM^\xbe\x80e#>\xa0TA>XhR>\xa0\xef\xeb=$\xec\x7f>\xd5\xf8\x87\xbe\xd2\x0e\x0b?\xf4\xa7\x16\xbe\xfcx\n\xbe\xd2>\xda>\x1a#\xdd>\xe0=\x15>|\xa8\xd9>\xab*\xe5\xbe\x98\x15\xf7>&\n\xac\xbe\xbe\xb4\xf2\xbe\xeaN\xba\xbe\x08\xdbn>sr\x94\xbe\xf0\xf8\x18=\xc4\x13\'>\xac\x1b\x04?\xf0\xf5:=\x04`\x9d\xbd\x18/\xe4\xbd\x00$c;/\xab\x8c\xbe`\x05#>:x\xda\xbeXg\xa0\xbe\x00Le<&$ \xbe\xe0\xf1\xba>$\xa2\xe7\xbe\xd0\xf5\xa4=\xc8\xe57\xbe}\xcf\xe0\xbe\x93f\xa2\xbe\xe8\xc3\x0b>\xb9C\t\xbfP\xffa>h\x12\x81>\xaa\xba\xec\xbe\x90r\xb7\xbd\x8a\x90\x01\xbfB\x08lstm_6_W*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xe0\xb1;>\xa7\xda\xbd\xbd\xecs\x16\xbe\xcd\xfbS>\xc1O1=\xa3]\xae=S\xc0\xb7>$\xd6L\xbe\xd6\xc0\xd1=\x9a\xb7k=>\xb8u>a\t\x9c>\x82\x83\x8b>5\xce9>\x9e\xf5\xf0\xbenZ\x80> \x84\x01\xbf\xb8\xb1\x87=\x9e\x00\xda=\xe8\xaf\xed>\xc8\xfd\xd2\xbcB\xbfP=\xd4uX\xbe\xf7\xcd\xe0=\xbf\xe4\x08\xbe\xa7\x92\xb7>\x8a,\r>\xd0\x14\xc3=u#\x03>\xf1\xd1\x88>\xd2=\xda=\x05\xb9\xe8\xbe[^m>\xe4t\x8f>\\\xa0\xe3\xbdx\xd3\x1b\xbc\x15V"?f\r\xae\xbd\x08#4\xbd\x07KC>\x9cd\xcd<\xea\t\xfa\xbd\xa0[v>|*O>\x165\x8e>n\x1c\x8d\xbd\xe4\xa5\xfa>n.\xfe=\xad\x16\x8d\xbc\xe2j\xf3\xbev\xf3\xf6=ZvK>\x92\x85\xfe\xbd\x80\xab\xf3>V5\xfc\xbdk#\x83>\x89\xfcI>c\x8d\xbf>\xcb\x80\xb5>\xbbw\x84>\x1b\xa4\xcc<\xfe\x92T\xbe\x8da\x05\xbe-\x86l>B\x08lstm_6_R*\x93\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x08lstm_6_BZ"\n\x0clstm_6_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x18\n\x06lstm_6\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model,
                         ['sigmoid', 'sigmoid', 'sigmoid'], lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_sigmoid_tanh(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[0.22958073, -0.02352869, -0.24502626, -0.27619946]],
                   dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xcc\x08\nC\n\x0clstm_8_input\x12\x08lstm_8_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xb9\x01\n\x08lstm_8_X\n\x08lstm_8_W\n\x08lstm_8_R\n\x08lstm_8_B\n\x00\n\x00\n\x00\n\x00\x12\x08lstm_8_Y\x12\nlstm_8_Y_h\x12\nlstm_8_Y_c\x1a\x06lstm_8"\x04LSTM*(\n\x0bactivationsJ\x04TanhJ\x07SigmoidJ\x07Sigmoid\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n7\n\nlstm_8_Y_h\x12\x06lstm_8\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\x0csequential_8*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02h\x85\xb5\xbd\xd4#i>:\xe4\x8d>\xf4\xbe\xe8> z8=\xba\xab\x89>H\xf3\x0c>(\x0e\x84>\x9a\x11\xd3>\xf8\xb9\x97\xbe\xfcj\x07?\xe4\x0b\x1e\xbe\x1c\xf4N>\xb8\x7f\xce>\x00\x1c\x12=\xe8\xd51>\xe0,\'\xbd\xb2\xe4\x08?\x909\x04\xbf\xa0\x11\xc8\xbe<\x80$>\xab\x07\x84\xbe1\xa4\xb5\xbe\xc0[\x84<6\xb2\xb9\xbexh\xb1\xbe\x8cE[>PX\xbc>\xfen\xc8>v\x7f\xf4\xbe\xf0\xc1\x12\xbeB\xdb9\xbe\xa8:\x00\xbe`do>\xb6?0\xbe^\xce\x89>\xa1f\x93\xbe\x18ic>f+\xc8\xbe\x81\xdb\xb6\xbe\x1e\xaf\xc3\xbe\xa0\xf3\x06=\xf0\xa2\xcd\xbe@h\xcb=\x80<\xda<\xa0vE>\x88\xce8\xbeES\x95\xbeh~X\xbe8\xdcO\xbd,\xdf\xbc\xbd\x80\xf8\x06\xbe\xfb\xb4\x85\xbe\xd4(\xf8\xbd\xe4\xc6g>\x8c\x17\xdb>\x11\xb1\xe5\xbe\xb1J\xf6\xbe\xc8\xb4\xe9>4\xe6\xf0>*\x88\x86>$\xd1\xf6>l=\x06?\x08J\xe8\xbdB\x08lstm_8_W*\x95\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\x10(\xa9\xbe@\x91e>g\xa9L\xbe\xde\x02\xb5;|;\x9e>kh\xa6=\x98H\xa7\xbd\xef\x15\xa0>!\xc6G\xbb|\xc6\xb5\xbe\x80\xe1L>\x83[.>\xb6\x1d\xf2=\x0ct\x96\xbe\x1bw\x1e>\xae\xd2\xe3\xbe\xa7H\x15\xbe4i\xbc=\xcdV\xe1>\xe2\x05#>2<\xec\xbe:\x86\xf4<bnn\xbeg\n5>j\x89\x11\xbe\xac%\x0f>\x15\xe0\xda=\x8d\xf75>^\xf9\xf1>\xc7\x89\x8e>\xcc\xb8\x0e\xbe\xdf\x1fh=\xf3*\xc7=\x13`\xb1\xbe\x07\xc8\x16>\xf3\x193>@\xeeQ\xbe!\xef\xb1>p9\x99\xbe\xfb\xb7\xac\xbc|\xbai>E\x87&\xbe?\xd4\x90\xbe\x14h\xbd=\xe1\xeb\xdd\xbd[\xba\xa7\xbe\xe4R\xd3\xbe\xf7\xc6\x8f>3\r\xb9\xbd}\xd4\r>\xc0g\x1b9\xfb\x98\x1f\xbf\x9b\xd8\xcd\xbe\xd4\x0c0\xbe.\xad\x87>\xb4\xf0\x93=\xf3?\x9c\xbc\xf2\x9ee>\xe0\x05\xa7>a\xec>>\'\xff\t\xbe-\xcc\xc6\xbea\xc0\x89\xbe5FD\xbeB\x08lstm_8_R*\x93\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\x08lstm_8_BZ"\n\x0clstm_8_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x18\n\x06lstm_8\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['tanh', 'sigmoid', 'sigmoid'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_tanh_relu(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[0., 0., 0., 0.]], dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xd8\x08\nE\n\rlstm_13_input\x12\tlstm_13_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xbb\x01\n\tlstm_13_X\n\tlstm_13_W\n\tlstm_13_R\n\tlstm_13_B\n\x00\n\x00\n\x00\n\x00\x12\tlstm_13_Y\x12\x0blstm_13_Y_h\x12\x0blstm_13_Y_c\x1a\x07lstm_13"\x04LSTM*"\n\x0bactivationsJ\x04ReluJ\x04TanhJ\x04Tanh\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n9\n\x0blstm_13_Y_h\x12\x07lstm_13\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\rsequential_13*\x96\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\\A\x05\xbf\xe0\x14\xb1>"\xc8+\xbed\x87\xc3\xbez\xf1\x04?\xb5\xc4\xe8\xbe\xe0\xa3\xbb\xbc\xa3c\xb4\xbe\xc0\x11\x94\xbc@s\xeb<L\x84\xe6\xbedz\xcd\xbd\xc0?\xe3="\xdf\xfb\xbe\x82\xa3i\xbe\x00\x9fA;`,\xdb\xbe\xcahA\xbe\x10\x7f\xab>\xf0ZD>,\xfe\x8f>\xb5\xb5\x02\xbfl<\xed\xbeE\xbd\xe5\xbe\xbc\x8bk>\x10p7>|8\xef>`\xf5S>`\xce\xc7>\xb0u\xa2\xbe\x94\xfe.>\x90\x12\x03\xbd\xbd\xea\x90\xbe\xe0f\xe0\xbe@\xc7\xb7\xbd\xff\x86\x80\xbe6\xf7T\xbe\x90\x99\xeb=\xc8A\x05?\x06\xb7\xb7\xbe\xa2o\x88>\xcc\xe5\xd7\xbd\xc6ZO\xbe\'\xba\x06\xbfp\x91\x04>\xa4\xe5a>,C\xcf\xbe>\xd5\xec\xbe@\xeb\xd6<\xb4\x0b\xf4\xbe\xe4\x00\x06\xbf&t\xaa\xbe\xa0\xdd\xcb>\xfc\x18\xa8\xbdL\xe7z>\xaeL\x0f\xbe\x92J;\xbe\x98\xb4\x02\xbf\xe2\xc3\xcf>\x0e\x81N\xbe\x88x5>0\xbeZ\xbd\xa4\xca\x82>ho\xe2=B\tlstm_13_W*\x96\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x020O\x8a\xbe\xd8\xf3\xd6\xbc\x8e\xd6\xb6;\xd0E\x8d\xbd|\xbfZ\xbe\xb4\xfb\xb4\xbeZ\x0bK>x \x0b\xbf\x0fD\xcc>N\xbb\x8a\xbd\x9a\xcc\xff\xbeQ%\xaf=\xf6w\x07>(8\xa5=j\xdb\x89>\xa3\x00\xd0\xbe8\xde\xee<\xf8\xda\x87\xbe\xb5\x14\xca\xbd@\x11\xe9\xbdZ\xb1\xd5>E\xb0\x8d=m\nL=DF\xbb\xbde\x9ex\xbd\xf3\x9b\xfc<\x10\x9a\xae=!\xe7\x88>\x86D\x91\xbe\t\xc0\xb8\xbd#\x8f\xa4>\xa0~\xa1>\xf0\xec\x9c\xbe\xd7\xb4\x1d>23\x1b\xbe\xc11\x9a\xbe\xef*\xb5\xbd\xeb\xec[\xbdl\xf7\x97>z\xbc->Jx\xb8=`\x89c>\x0f\xd0\xcf=]]1\xbe\xacF\x87\xbd\x03\x1d\xf1>\xfc\xd6\xac=\xaa\xfb\x0b\xbd\x03\xd8\xeb\xbd\xa1\xff\xd1\xbe\xf9\xb3\x94=\x12\xfd\x89\xbd"\x89\xd1\xbeK\x05\\\xbe\xf5\xf6\xd7\xbe\xa5\xbfo>\xe5<\x1f\xbe\xe9^\xa8>L\xb0w>\x1dhs>\xbb\xf9\xb5\xbe\xc8\xf7\xcb>vY\xc3\xbeZ8\x86\xbeB\tlstm_13_R*\x94\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\tlstm_13_BZ#\n\rlstm_13_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x19\n\x07lstm_13\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['relu', 'tanh', 'tanh'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_tanh_sigmoid(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[-0.09767307, -0.18018779, -0.01351621, -0.18635663]],
                   dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xdb\x08\nE\n\rlstm_14_input\x12\tlstm_14_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xbe\x01\n\tlstm_14_X\n\tlstm_14_W\n\tlstm_14_R\n\tlstm_14_B\n\x00\n\x00\n\x00\n\x00\x12\tlstm_14_Y\x12\x0blstm_14_Y_h\x12\x0blstm_14_Y_c\x1a\x07lstm_14"\x04LSTM*%\n\x0bactivationsJ\x07SigmoidJ\x04TanhJ\x04Tanh\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n9\n\x0blstm_14_Y_h\x12\x07lstm_14\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\rsequential_14*\x96\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\x18U\xf2\xbe\xf2\xbe\xb7>\xe7\x85\x97\xbe\xd0\xa6\x95>\x17\x0e\x0c\xbf0\xe4\xb2\xbd\xcc\xc3\x06?@>\x1c\xbd\xcck\x0b\xbe8]\x06\xbe\x02Y\n?\x00\x93\xcf<\x8e\x12\x82\xbe \xf6I=D1\xc7>^R\xb4\xbeN\xfaJ\xbe\xc2\x8e\xd5\xbeV\xfc\xea\xbe\xdc\x171>\xd0r5\xbe\xacv~>\x0e@\xb6>j\x1a\xe0>\x00\xe9\xcc\xbcPM\xe5\xbe`\xe7\xe5><\xa9!\xbe\xe8V\x10>\xe4\xb8,\xbe:M\xc1\xbe\xcc\xc7H>\x10o\xcc>\x92SQ\xbe\xdaa\x00\xbe:\xac\xf7\xbe\xc8\xe8\xaa\xbe\xa6<\x9f\xbe\xb0\xa8\xca=\xb6\xd8m\xbe\x02\xe0\xab>\xb8\x04\xcc\xbe\x9e\xce\x0c\xbe\x94\xa7\xa9\xbe\xd6\xac\xa8>\x14N|>\x926\x12\xbe\xda\xfd\xd8>>k\xb1>\xb2\x9e\x10\xbe\x92\'o\xbes\xce\x83\xbe\xedQ\xbe\xbe\x08|\xe6=\xec\xa3\xe1\xbdr\xd4\xac\xbe\xee\x80\xdb>`\xa2\xd5=\xd0\x12\xf3\xbe\xa4\xf6\xd7\xbeX\xd5\x84=\x02\xda\xe1\xbe\xb0\xe0\n\xbd\xf0y\x12\xbdB\tlstm_14_W*\x96\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xd0K\xff>gne\xbe\x9c\x17\xf4=,\xdf\xb7\xbd\x13\x08V\xbe\xd0i5\xbc\xccq\xa6\xbemd\x05\xbe\xa7C\xa9\xbd\xa9T\xe4\xbe\xd0\xb5\x88=a#8>\xfe\xb6\x12>\xa7\x89\xe3=\xd8\x00\xd3\xbe\x9c\xc1\xa0>\x9a\xad\xb4\xbe\xd7D\xb6>\x19\xfc\x9a<\\\x94\x11>\xd8|\xfd\xbd\x83\xbf\x14\xbeC3\xec=\x18\xda\x11\xbe\xf0\xcf\xca=\xe4\xb9\x14>\xfd>\xd6\xbe\x8a\x9dY\xbeg,j>\x14\xde\xc8<@\x1b~:\xc7Mw\xbe\x9d\xe4r\xbc\x12\x1c\xd6<\x18\xbd\xbc>\x81&\x18\xbfl:j>\x90\xdd\xd0>\x8b\xa1\xeb\xbc\xe6\xe4\xbb;\x96b\xfe>\x02i\x87\xbebu\xb8\xbd\x82\x15\xa4=\x88\xbcZ\xbe\xdb?\xad\xbe\x11\xd9u\xbel\x86\x10\xbe$5C>N3\x96>{N\xe5>\x0e\x9b\xc4>\xa7\xfe\xee\xbc\xa4t\x8a>\xcc\xc0\xf2\xbd\xf8kU\xbe\x9b\xcaW\xbe&\xe9o\xbe\x1a\xbe\x83=3U\xb6>\xa1\xf0h>\xecjk=\xb5|\x9e\xbeX\x8fV=B\tlstm_14_R*\x94\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\tlstm_14_BZ#\n\rlstm_14_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x19\n\x07lstm_14\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['sigmoid', 'tanh', 'tanh'],
                         lstm_op_pattern)


@pytest.mark.parametrize("lstm_op_pattern", [False, True])
def test_tanh_tanh(lstm_op_pattern):
    data = array([[[0.5488135, 0.71518934, 0.60276335, 0.5448832],
                   [0.4236548, 0.6458941, 0.4375872, 0.891773],
                   [0.96366274, 0.3834415, 0.79172504, 0.5288949],
                   [0.56804454, 0.92559665, 0.07103606, 0.0871293]]],
                 dtype=float32)
    result = array([[-0.06280307, 0.02255315, 0.02322592, 0.04542083]],
                   dtype=float32)
    model = b'\x08\x06\x12\nkeras2onnx\x1a\x051.8.1"\x0bonnxmltools(\x002\x00:\xd8\x08\nE\n\rlstm_16_input\x12\tlstm_16_X\x1a\tTranspose"\tTranspose*\x0f\n\x04perm@\x01@\x00@\x02\xa0\x01\x072\x00:\x00\n\xbb\x01\n\tlstm_16_X\n\tlstm_16_W\n\tlstm_16_R\n\tlstm_16_B\n\x00\n\x00\n\x00\n\x00\x12\tlstm_16_Y\x12\x0blstm_16_Y_h\x12\x0blstm_16_Y_c\x1a\x07lstm_16"\x04LSTM*"\n\x0bactivationsJ\x04TanhJ\x04TanhJ\x04Tanh\xa0\x01\x08*\x17\n\tdirection"\x07forward\xa0\x01\x03*\x12\n\x0bhidden_size\x18\x04\xa0\x01\x02:\x00\n9\n\x0blstm_16_Y_h\x12\x07lstm_16\x1a\x07Squeeze"\x07Squeeze*\x0b\n\x04axes@\x00\xa0\x01\x072\x00:\x00\x12\rsequential_16*\x96\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02\xf4D]\xbe\xe2;\xd9>\x80\xb9N\xbc\xc8\xff\xca=\\\x14\xec>\x96.\x9c\xbeZF\xc1>\xfc\x9d+>@H\xcd>R1\x83>\x08K\xf2>4\xfe\xbc>\x08q1\xbe.\xa3\xbe\xbeL\xec\xdf\xbd&\x16=\xbe\xe6\x0b\x0c?\xd6\x80\xbd>\xc0\xfe\xb8>\xa8\xef\xce=\xaa\xf9\x00?\x9c\x01\xc6\xbd`\x96D=\x88\xa2W\xbe\x80\xb1M\xbd\xc0\xb5\xc2\xbe\x00\x89\x14\xbbR$\x0f\xbeV\x02\x92\xbep\xb11\xbd\x10\x86\x16\xbd4\x7f\xf2>S\xe1\x8a\xbeD\xe5\x11>\xc8\xe9\x98\xbe\xe0\xa9\x97\xbch7\xc7\xbe\x98y\xa4\xbex\xa7\xc2=\x06y\x01\xbe\x0ee\xe5>\x96\xc9\x06?\xcc\x1d@>PcA\xbe\x1c\r\xe5>\xf4?\x8c>p8\xcb>B\xa8\xa9>{\xd1\x9e\xbe\xb0\xb9/\xbe\xb8\xa3\x9f\xbeP\xb6\xc6>d\xaam>\xe0\xe8\xa5\xbd\x1c\x95\xbc>\xf8\x05\xf3=\x98V\x80=\x01m\xe6\xbe\x12O\xbc>\xb0\x1e\xa6=\x86>a\xbe\x80V\xf1\xbc\x8c\xe5\xc3>\x06`\x02?B\tlstm_16_W*\x96\x02\x08\x01\x08\x10\x08\x04\x10\x01"\x80\x02P\x0e\x1e\xbeV\xd4&\xbc\x1dh\xb3\xbd\xa9\x95H>f8\xf4\xbe}\xdc\xba\xbe\x7f(>>Cp\x9a>\x98\xc6 \xbd\xd9\x90Q\xbe\xb8\x1d_<0\xfb\x02>\x81\xee\xcb\xbc\xbe\xe2\xc8\xbc\x00\xac\x0b\xbe\x08\xb4\x19>E\xf2y\xbe\xdc3\x9f\xbey\xaeC>\xd8\x89Y\xbe\xd5\\/\xbb\xc1\x8f\xc7\xbe\xe9\x87<\xbfv-\xb9\xbcZ\xf9:\xbe\xc2V\x8e>M\xde\xb1\xbde\xf1\x8a\xbe\x0fu$>)H\xd1>\xd0\x95\x03\xbe\xa2\x07&>\r[\x8b>\x99\xba\xbc=\x0f\x08#\xbe\x91\x01I>e\x8f\xb3>tj*\xbe\xfa\xde\xe3=;\xca\x17\xben\xdfc\xbe\xe0\xf3\xdb>\x7f<\x11>C7g>\xb0r\xb7>\x8bJu\xbe\xdd\xad\xe2>\x02#}>\xda\xfe7>C\xfb\x16\xbepsE:\xcd;\xea\xbes)\xab>\x18{R=i\xd7F=\x0c\t7\xbe\xe0QO>\x94}F\xbd\xcc\xdf\x80\xbe\xf8\x07\xeb>TI\x85\xbez[&>\xfc\xbc\x03\xbe\x86-~\xbeB\tlstm_16_R*\x94\x01\x08\x01\x08 \x10\x01"\x80\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00B\tlstm_16_BZ#\n\rlstm_16_input\x12\x12\n\x10\x08\x01\x12\x0c\n\x02\x08\x01\n\x02\x08\x04\n\x02\x08\x04b\x19\n\x07lstm_16\x12\x0e\n\x0c\x08\x01\x12\x08\n\x02\x08\x01\n\x02\x08\x04B\x04\n\x00\x10\x0b'
    _run_comparison_test(data, result, model, ['tanh', 'tanh', 'tanh'],
                         lstm_op_pattern)