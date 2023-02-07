# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart

import torch.onnx
import torchvision

input_ = torch.FloatTensor(torch.randn(4, 3, 224, 224))
model = torchvision.models.alexnet(pretrained=False)

output_name = "output"


def onnx_opset_version():
    popart_max_onnx_opset_version = 11
    default_onnx_opset_version = 0

    try:
        from torch.onnx.symbolic_helper import _export_onnx_opset_version
    except ImportError:
        try:
            from torch.onnx._globals import GLOBALS
        except ImportError:
            default_onnx_opset_version = popart_max_onnx_opset_version
        else:
            default_onnx_opset_version = GLOBALS.export_onnx_opset_version
    else:
        default_onnx_opset_version = _export_onnx_opset_version

    return min(popart_max_onnx_opset_version, default_onnx_opset_version)


torch.onnx.export(
    model,
    input_,
    "alexnet.onnx",
    output_names=[output_name],
    opset_version=onnx_opset_version(),
)

# Create a runtime environment
anchors = {output_name: popart.AnchorReturnType("All")}
dataFlow = popart.DataFlow(100, anchors)

# Append an Nll loss operation to the model
builder = popart.Builder("alexnet.onnx")
labels = builder.addInputTensor("INT32", [4], "label_input")
nlll = builder.aiGraphcore.nllloss([output_name, labels])

optimizer = popart.ConstSGD(0.001)

# Run session on CPU
device = popart.DeviceManager().createCpuDevice()
session = popart.TrainingSession(
    builder.getModelProto(),
    deviceInfo=device,
    dataFlow=dataFlow,
    loss=nlll,
    optimizer=optimizer,
)
