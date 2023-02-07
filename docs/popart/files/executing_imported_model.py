# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart

import torch.onnx
import torchvision

import numpy as np
import onnx

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

# Obtain inputs/outputs name of loaded model
loaded_model = onnx.load("alexnet.onnx")
inputs_name = [node.name for node in loaded_model.graph.input]
outputs_name = [node.name for node in loaded_model.graph.output]

print("Iputs name:", inputs_name)
print("Outputs name:", outputs_name)

# Create a runtime environment
anchors = {output_name: popart.AnchorReturnType("All")}
dataFlow = popart.DataFlow(100, anchors)
device = popart.DeviceManager().createCpuDevice()

session = popart.InferenceSession("alexnet.onnx", dataFlow, device)

session.prepareDevice()

input_1 = np.random.randn(4, 3, 224, 224).astype(np.float32)

stepio = popart.PyStepIO({"input.1": input_1}, session.initAnchorArrays())
