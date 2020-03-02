import popart

import torch.onnx
import torchvision

input_ = torch.FloatTensor(torch.randn(4, 3, 224, 224))
model = torchvision.models.alexnet(pretrained=False)

labels_name = "labels"
output_name = "output"

torch.onnx.export(model, input_, "alexnet.onnx", output_names=[output_name])

# Describe the labels input shape
inputShapeInfo = popart.InputShapeInfo()
inputShapeInfo.add(labels_name, popart.TensorInfo("INT32", [4]))

# Create a runtime environment
anchors = {output_name: popart.AnchorReturnType("ALL")}
dataFeed = popart.DataFlow(100, anchors)

losses = [popart.NllLoss(output_name, labels_name, "loss")]
optimizer = popart.ConstSGD(0.001)

# Run session on CPU
device = popart.DeviceManager().createCpuDevice()
session = popart.TrainingSession("alexnet.onnx",
                                 deviceInfo=device,
                                 dataFeed=dataFeed,
                                 losses=losses,
                                 optimizer=optimizer,
                                 inputShapeInfo=inputShapeInfo)
