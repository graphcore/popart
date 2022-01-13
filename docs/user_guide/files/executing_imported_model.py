# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart

import torch.onnx
import torchvision

import numpy as np
import onnx

input_ = torch.FloatTensor(torch.randn(4, 3, 224, 224))
model = torchvision.models.alexnet(pretrained=True)

output_name = "output"

torch.onnx.export(model, input_, "alexnet.onnx", output_names=[output_name])

# Obtain inputs/outputs name of loaded model
loaded_model = onnx.load('alexnet.onnx')
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

stepio = popart.PyStepIO({'input.1': input_1}, session.initAnchorArrays())
