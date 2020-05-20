# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart

import torch.onnx
import torchvision

input_ = torch.FloatTensor(torch.randn(4, 3, 224, 224))
model = torchvision.models.alexnet(pretrained=False)

output_name = "output"

torch.onnx.export(model, input_, "alexnet.onnx", output_names=[output_name])

# Create a runtime environment
anchors = {output_name: popart.AnchorReturnType("All")}
dataFlow = popart.DataFlow(100, anchors)

# Append an Nll loss operation to the model
builder = popart.Builder("alexnet.onnx")
labels = builder.addInputTensor("INT32", [4])
nlll = builder.aiGraphcore.nllloss([output_name, labels])

optimizer = popart.ConstSGD(0.001)

# Run session on CPU
device = popart.DeviceManager().createCpuDevice()
session = popart.TrainingSession(builder.getModelProto(),
                                 deviceInfo=device,
                                 dataFlow=dataFlow,
                                 loss=nlll,
                                 optimizer=optimizer)
