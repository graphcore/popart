import os
import numpy as np
import torch
import torch.utils
import torch.utils.data
from writer import NetWriter
from optimizers import SGD
from datafeeds import FromTxtFiles
import pywillow


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class PytorchNetWriter(NetWriter):
    def __init__(self, inNames, outNames, losses, optimizer, anchors,
                 willowTest, dataFeed, module):
        """
        module:
          -- pytorch module (whose forward does not have the loss layers)
        all others:
          -- parameters passed to base class.
        """
        NetWriter.__init__(self, inNames, outNames, losses, optimizer, anchors,
                          willowTest, dataFeed)

        self.module = module

    def getTorchDataset(self):
        """
        convert willow's DataFeed to a torch Dataset
        """
        if isinstance(self.dataFeed, FromTxtFiles):
            tensors = [torch.from_numpy(x) for x in self.dataFeed.streamVals()]
            return torch.utils.data.dataset.TensorDataset(*tensors)

        else:
            raise RuntimeError("unrecognised DataFeed type")

    def getTorchOptimizer(self):
        """
        convert willow's Optimizer to a torch Optimizer
        """
        if isinstance(self.optimizer, SGD):
            return torch.optim.SGD(
                self.module.parameters(),
                lr=self.optimizer.learnRate,
                momentum=0.0)
        else:
            raise RuntimeError("unrecognised optimizer")

    def getTorchLossTarget(self, streamMap, outMap):
        """
        Build the torch extension for computing the
        loss described by pywillow's losses
        """
        lossValues = []
        for loss in self.losses:
            if isinstance(loss, pywillow.NllLoss):
                criterion = torch.nn.NLLLoss()
                lossValues.append(
                    criterion(outMap[loss.probsTensorId()],
                              streamMap[loss.labelTensorId()]))
            elif isinstance(loss, pywillow.L1Loss):
                lossValues.append(
                    loss.getLambda() * torch.norm(outMap[loss.getInputId()], 1))

        return sum(lossValues)

    def writeOnnxModel(self, dirname, t_step, streamMap):
        # write ONNX model
        fnModel = os.path.join(dirname, "model%d.onnx" % (t_step, ))
        print("writing ONNX model (t=%d) to protobuf file" % (t_step, ))
        # now jump into eval mode, just to write the onnx model
        # note that this might do strange things with batch-normalisation
        self.module.eval()
        print("  --writing %s" % (fnModel, ))
        torch.onnx.export(
            self.module, [streamMap[inName] for inName in self.inNames],
            fnModel,
            verbose=False,
            input_names=self.inNames,
            output_names=self.outNames)

    def getStreamMap(self, data):
        # unpack the stream input from "data":
        streamMap = {}
        # this should return [image0, image1, label] for model 0
        for j, streamName in enumerate(self.dataFeed.streamNames()):
            streamMap[streamName] = data[j]
        return streamMap

    def writeOnnx(self, dirname):
        """
        TODO : sort out case of tuples of tuples for outputs

        w.r.t. parameter names. It is possible to flatten
        named_parameters of module, but it is not clear that
        the order will always correspond to the order of the
        onnx "trace", so I won't.
        """

        dataset = self.getTorchDataset()

        trainloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.dataFeed.batchsize,
            shuffle=False,
            num_workers=0)

        if not self.willowTest:
            data = iter(trainloader).next()
            self.writeOnnxModel(dirname,0, self.getStreamMap(data))

        # note for other frameworks. If willowTest is always False,
        # this elimimates most of the work: don't need to worry about
        # Optimizer, losses, maybe not even data stream.
        else:
            torchOptimizer = self.getTorchOptimizer()

            for i, data in enumerate(trainloader, 0):
                streamMap = self.getStreamMap(data)
                self.writeOnnxModel(dirname,i, streamMap)

                #forwards - backwards - update
                self.module.train()
                torchOptimizer.zero_grad()
                outputs = self.module(
                    [streamMap[name] for name in self.inNames])
                outMap = {}
                for j, outName in enumerate(self.outNames):
                    outMap[outName] = outputs[j]
                lossTarget = self.getTorchLossTarget(streamMap, outMap)
                lossTarget.backward()
                torchOptimizer.step()
                for p in self.module.parameters():
                    print(p.max())
