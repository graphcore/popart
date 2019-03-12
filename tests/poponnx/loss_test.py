import math
import numpy as np
import poponnx
import pytest
import torch


def test_3d_nll_loss_input():
    ## input data
    Batchsize = 2
    ExtraDim = 4  # e.g. sequence length in language model
    Classes = 3

    dshape = [Batchsize, ExtraDim, Classes]
    lshape = [Batchsize, ExtraDim]

    ip_data = np.random.rand(Batchsize, ExtraDim, Classes).astype(np.float32)
    lb_data = np.random.randint(Classes, size=lshape)

    ###
    # Poponnx
    ###
    builder = poponnx.Builder()
    ip = builder.addInitializedInputTensor(ip_data)
    lb = builder.addInputTensor(poponnx.TensorInfo("INT32", lshape))
    out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))
    builder.addOutputTensor(out)

    session = poponnx.Session(
        fnModel=builder.getModelProto(),
        dataFeed=poponnx.DataFlow(
            1, {
                "loss": poponnx.AnchorReturnType("ALL"),
                out: poponnx.AnchorReturnType("ALL")
            }),
        optimizer=poponnx.ConstSGD(0.001, 0.01),
        losses=[poponnx.NllLoss(out, lb, "loss")],
        passes=poponnx.Patterns(poponnx.PatternsLevel.ALL))

    session.setDevice(poponnx.DeviceManager().createCpuDevice())
    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    stepio = poponnx.PyStepIO({lb: lb_data.astype(np.int32)}, anchors)
    session.evaluate(stepio)

    ###
    # Pytorch
    ###
    softmax = torch.nn.Softmax(dim=1)
    loss = torch.nn.NLLLoss(reduction='none')

    # Swap Classes, ExtraDim axes
    # This is because pytorch NllLoss expects inputs of the format:
    #   Probs  - [BatchSize, Classes, ExtraDim0, ... , ExtraDimN]
    #   Labels - [BatchSize, ExtraDim0, ... , ExtraDimN]
    # whereas Poponnx expects (same as Tensorflow):
    #   Probs  - [BatchSize, ExtraDim0, ... , ExtraDimN, Classes]
    #   Labels - [BatchSize, ExtraDim0, ... , ExtraDimN]
    ip_data = ip_data.transpose([0, 2, 1])
    input = torch.tensor(ip_data, requires_grad=True)
    target = torch.tensor(lb_data)
    sm_out = softmax(input)
    logsm = torch.log(sm_out)
    output = loss(logsm, target)

    ###
    # Compare
    ###
    print("Torch loss\n:", output.data.numpy())
    print("Poponnx loss\n:", anchors["loss"])

    def getTensorError(tA, pA):
        # pA, tA are corresponding tensors from two models
        ss_err = np.sum((np.array(pA) - np.array(tA))**2)
        ss_pA = np.sum(np.array(pA)**2)
        ss_tA = np.sum(np.array(tA)**2)
        return ss_err / (math.sqrt(ss_pA * ss_tA) + 1.0e-8)

    def checkResult(result, margin):
        if np.isnan(result):
            raise Exception(str(result) + " is NaN")
        elif (result > margin):
            raise Exception(str(result) + " is greater than " + str(margin))

    result = getTensorError(output.data.numpy(), anchors["loss"])
    checkResult(result, 1e-8)
