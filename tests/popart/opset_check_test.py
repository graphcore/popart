import torch
import torch.nn as nn
import torch.nn.functional as F
import popart
import onnx
import test_util as tu
import pytest

resize_op_error = (
    "For an opset 11 Model, the ONNX spec stipulates that a "
    "Resize op must be version 11. The highest version we "
    "have implemented less than or equal to 11 is 10, so bailing.")


def test_opset():
    class X(nn.Module):
        def __init__(self):
            super(X, self).__init__()

        def forward(self, x):
            return F.upsample(x, scale_factor=[2, 2])

    model = X()
    data = torch.randn(1, 1, 3, 3)
    torch.onnx.export(model, data, 'foo.onnx', verbose=True, opset_version=11)

    model = onnx.load_model('foo.onnx')
    # This test only works if there is a resize in the model
    assert 'Resize' in [node.op_type for node in model.graph.node]

    dataFlow = popart.DataFlow(
        1, {model.graph.output[0].name: popart.AnchorReturnType("All")})
    with pytest.raises(popart.popart_exception) as e_info:
        sess = popart.InferenceSession(fnModel='foo.onnx',
                                       dataFlow=dataFlow,
                                       deviceInfo=tu.create_test_device())

    assert resize_op_error in str(e_info.value)


def test_strictOpVersions():
    def run_test(strictOpVersions):
        class X(nn.Module):
            def __init__(self):
                super(X, self).__init__()

            def forward(self, x):
                return F.upsample(x, scale_factor=[2, 2])

        model = X()
        data = torch.randn(1, 1, 3, 3)
        torch.onnx.export(model,
                          data,
                          'foo.onnx',
                          verbose=True,
                          opset_version=11)

        model = onnx.load_model('foo.onnx')
        # This test only works if there is a resize in the model
        assert 'Resize' in [node.op_type for node in model.graph.node]

        dataFlow = popart.DataFlow(
            1, {model.graph.output[0].name: popart.AnchorReturnType("All")})
        opts = popart.SessionOptions()
        opts.strictOpVersions = strictOpVersions

        try:
            sess = popart.InferenceSession(fnModel='foo.onnx',
                                           dataFlow=dataFlow,
                                           deviceInfo=tu.create_test_device(),
                                           userOptions=opts)
        except popart.popart_exception as e:
            return str(e)

        return ""

    # Check that the version error is only returned when strictOpVersions is True
    assert resize_op_error in run_test(True)
    assert resize_op_error not in run_test(False)
