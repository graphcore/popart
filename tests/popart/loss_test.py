import math
import numpy as np
import popart
import pytest
import torch


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
    # Popart
    ###
    builder = popart.Builder()
    ip = builder.addInitializedInputTensor(ip_data)
    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))
    builder.addOutputTensor(out)

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFeed=popart.DataFlow(
            1, {
                "loss": popart.AnchorReturnType("ALL"),
                out: popart.AnchorReturnType("ALL")
            }),
        optimizer=popart.ConstSGD(0.001, 0.01),
        losses=[popart.NllLoss(out, lb, "loss")],
        passes=popart.Patterns(popart.PatternsLevel.ALL),
        deviceInfo=popart.DeviceManager().createCpuDevice())

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({lb: lb_data.astype(np.int32)}, anchors)
    session.run(stepio)

    ###
    # Pytorch
    ###
    softmax = torch.nn.Softmax(dim=1)
    loss = torch.nn.NLLLoss(reduction='none')

    # Swap Classes, ExtraDim axes
    # This is because pytorch NllLoss expects inputs of the format:
    #   Probs  - [BatchSize, Classes, ExtraDim0, ... , ExtraDimN]
    #   Labels - [BatchSize, ExtraDim0, ... , ExtraDimN]
    # whereas Popart expects (same as Tensorflow):
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
    print("Popart loss\n:", anchors["loss"])

    result = getTensorError(output.data.numpy(), anchors["loss"])
    checkResult(result, 1e-8)


def test_nll_loss_with_ignored_index():
    ## input data
    Batchsize = 2
    Classes = 8

    dshape = [Batchsize, Classes]
    lshape = [Batchsize]

    ip_data = np.random.rand(Batchsize, Classes).astype(np.float32)
    lb_data = np.array([1, 7])

    # Samples whose target class index is equal to ignoreInd should
    # not contribute to the loss (or loss gradient)
    ignoreInd = 1

    ###
    # Popart
    ###
    builder = popart.Builder()
    ip = builder.addInitializedInputTensor(ip_data)
    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))
    builder.addOutputTensor(out)

    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFeed=popart.DataFlow(1, {"loss": popart.AnchorReturnType("ALL")}),
        optimizer=popart.ConstSGD(0.001, 0.01),
        losses=[popart.NllLoss(out, lb, "loss", ignore_index=ignoreInd)],
        passes=popart.Patterns(popart.PatternsLevel.ALL),
        deviceInfo=popart.DeviceManager().createCpuDevice())

    session.prepareDevice()
    session.weightsFromHost()

    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({lb: lb_data.astype(np.int32)}, anchors)
    session.run(stepio)

    ###
    # Pytorch
    ###
    softmax = torch.nn.Softmax(dim=1)
    loss = torch.nn.NLLLoss(reduction='none', ignore_index=ignoreInd)

    input = torch.tensor(ip_data, requires_grad=True)
    target = torch.tensor(lb_data)
    sm_out = softmax(input)
    logsm = torch.log(sm_out)
    output = loss(logsm, target)

    ###
    # Compare
    ###
    torch_loss = output.data.numpy()
    popart_loss = anchors["loss"]
    print("Torch loss\n:", torch_loss)
    print("Popart loss\n:", popart_loss)

    for sampleInd, labelInd in enumerate(lb_data):
        if labelInd == ignoreInd:
            assertStr = "losses for ignoreInd samples should be zero"
            assert (torch_loss[sampleInd] == 0), assertStr
            assert (popart_loss[sampleInd] == 0), assertStr

    result = getTensorError(torch_loss, popart_loss)
    checkResult(result, 1e-8)


def test_nll_loss_grad_with_ignored_index():
    ## input data
    Batchsize = 3
    Classes = 8

    dshape = [Batchsize, Classes]
    lshape = [Batchsize]

    ip_data = np.random.rand(Batchsize, Classes).astype(np.float32)
    lb_data = np.array([1, 7, 4])

    # Samples whose target class index is equal to ignoreInd should
    # not contribute to the loss (or loss gradient)
    ignoreInd = 1

    ###
    # Popart
    ###
    builder = popart.Builder()
    ip = builder.addInitializedInputTensor(ip_data)
    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))
    builder.addOutputTensor(out)

    ## 2 sessions: one with "SoftmaxGradDirect" pattern, one without
    def getPreparesSession(patterns):
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(
                1, {
                    popart.reservedGradientPrefix() + ip:
                    popart.AnchorReturnType("ALL")
                }),
            optimizer=popart.ConstSGD(0.001, 0.01),
            losses=[
                popart.NllLoss(out,
                               lb,
                               "loss",
                               ignore_index=ignoreInd,
                               reduction=popart.ReductionType.Mean)
            ],
            passes=popart.Patterns(patterns),
            deviceInfo=popart.DeviceManager().createCpuDevice())

        session.prepareDevice()
        session.weightsFromHost()
        session.optimizerFromHost()
        return session

    inputs = {lb: lb_data.astype(np.int32)}

    # 1)
    session_SMD = getPreparesSession(["PreUniRepl", "SoftmaxGradDirect"])
    anchors_SMD = session_SMD.initAnchorArrays()
    stepio_SMD = popart.PyStepIO(inputs, anchors_SMD)
    session_SMD.run(stepio_SMD)

    # 2)
    session_NoSMD = getPreparesSession(["PreUniRepl"])
    anchors_NoSMD = session_NoSMD.initAnchorArrays()
    stepio_NoSMD = popart.PyStepIO(inputs, anchors_NoSMD)
    session_NoSMD.run(stepio_NoSMD)

    ###
    # Pytorch
    ###
    # function to extract grad
    def set_grad(var):
        def hook(grad):
            var.grad = grad

        return hook

    softmax = torch.nn.Softmax(dim=1)
    loss = torch.nn.NLLLoss(reduction="mean", ignore_index=ignoreInd)

    input = torch.tensor(ip_data, requires_grad=True)
    target = torch.tensor(lb_data)
    sm_out = softmax(input)
    sm_out.register_hook(set_grad(sm_out))
    logsm = torch.log(sm_out)
    output = loss(logsm, target)
    output.backward(retain_graph=True)

    ###
    # Compare
    ###
    torch_ip_grad = input.grad.numpy()
    px_smd_ip_grad = anchors_SMD[popart.reservedGradientPrefix() + ip]
    px_no_smd_ip_grad = anchors_NoSMD[popart.reservedGradientPrefix() + ip]

    for sampleInd, labelInd in enumerate(lb_data):
        if labelInd == ignoreInd:
            assertStr = "loss grads for ignoreInd samples should be zero"
            zero = np.zeros(Classes)
            assert (np.equal(torch_ip_grad[sampleInd], zero).all()), assertStr
            assert (np.equal(px_smd_ip_grad[sampleInd], zero).all()), assertStr
            assert (np.equal(px_no_smd_ip_grad[sampleInd],
                             zero).all()), assertStr

    checkResult(getTensorError(torch_ip_grad, px_smd_ip_grad), 1e-8)
    checkResult(getTensorError(torch_ip_grad, px_no_smd_ip_grad), 1e-8)
