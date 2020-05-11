# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import math
import numpy as np
import popart
import test_util as tu
import pytest
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2


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
    def run(withBuilderOp):
        # fix the random seed for this test
        np.random.seed(0)
        ## input data
        Batchsize = 2
        ExtraDim = 4  # e.g. sequence length in language model
        Classes = 3

        dshape = [Batchsize, ExtraDim, Classes]
        lshape = [Batchsize, ExtraDim]
        flat_lshape = [Batchsize * ExtraDim]

        ip_data = np.random.rand(Batchsize, ExtraDim,
                                 Classes).astype(np.float32)
        lb_data = np.random.randint(Classes, size=lshape)

        ###
        # Popart
        ###
        builder = popart.Builder()
        ip = builder.addInitializedInputTensor(ip_data)
        lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
        out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))

        if withBuilderOp == True:
            nll0 = builder.aiGraphcore.nllloss([out, lb])
            nll1 = builder.reshape_const(builder.aiOnnx, [nll0], flat_lshape)
            loss = popart.IdentityLoss(nll1, "loss")
        else:
            loss = popart.NllLoss(out, lb, "loss")

        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(1, ["loss", out]),
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            losses=[loss],
            patterns=popart.Patterns(popart.PatternsLevel.All),
            deviceInfo=tu.create_test_device())

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
        if withBuilderOp == True:
            output = output.reshape(flat_lshape)

        ###
        # Compare
        ###
        print("Torch loss\n:", output.data.numpy())
        print("Popart loss\n:", anchors["loss"])

        result = getTensorError(output.data.numpy(), anchors["loss"])
        checkResult(result, 1e-8)

    run(True)
    run(False)


def test_nll_loss_with_ignored_index():
    # fix the random seed for this test
    np.random.seed(0)
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
        dataFeed=popart.DataFlow(1, {"loss": popart.AnchorReturnType("All")}),
        optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
        losses=[popart.NllLoss(out, lb, "loss", ignore_index=ignoreInd)],
        patterns=popart.Patterns(popart.PatternsLevel.All),
        deviceInfo=tu.create_test_device())

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
    # fix the random seed for this test
    np.random.seed(0)
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
                    popart.AnchorReturnType("All")
                }),
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            losses=[
                popart.NllLoss(out,
                               lb,
                               "loss",
                               ignore_index=ignoreInd,
                               reduction=popart.ReductionType.Mean)
            ],
            patterns=popart.Patterns(patterns),
            deviceInfo=tu.create_test_device())

        session.prepareDevice()
        session.weightsFromHost()

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


def test_id_loss_error():
    # fix the random seed for this test
    np.random.seed(0)

    ## input data
    Batchsize = 2
    ExtraDim = 7
    Classes = 8

    ip_data = np.random.rand(Batchsize, ExtraDim, Classes).astype(np.float32)

    ###
    # Popart
    ###
    builder = popart.Builder()

    # Prepare input data
    ip = builder.addInitializedInputTensor(ip_data, "input")
    out = builder.aiOnnx.exp([ip])

    builder.addOutputTensor(out)

    art = popart.AnchorReturnType("All")
    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(1, {"loss": art}),
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            losses=[popart.IdentityLoss(out, "loss")],
            patterns=popart.Patterns(popart.PatternsLevel.All),
            deviceInfo=tu.create_test_device())

    assert (e_info.value.args[0].startswith(
        f"The identity loss Op(ai.onnx.Identity:1, inputs=[Exp:0]," +
        f" outputs=[loss]) (shape [{Batchsize} {ExtraDim} {Classes}]) is expecting a tensor"
    ))


def test_id_nllloss_train():
    # fix the random seed for this test
    np.random.seed(0)
    # input data
    Batchsize = 8
    Classes = 32

    def get_model(ip_data, lb_data, w_data, id_loss):

        ###
        # Popart
        ###
        builder = popart.Builder()
        # Prepare input data
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", ip_data.shape),
                                    "input")
        lb = builder.addInputTensor(popart.TensorInfo("INT32", lb_data.shape),
                                    "label")
        w0 = builder.addInitializedInputTensor(w_data, "weight")

        c0 = builder.aiOnnx.conv([ip, w0],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1],
                                 debugPrefix="conv")

        r0 = builder.reshape_const(builder.aiOnnx, [c0], [Batchsize, Classes])
        if id_loss:
            depth = builder.aiOnnx.constant(
                np.array(Classes).astype(np.int32), "depth")
            eps = builder.aiOnnx.constant(
                np.array(1.0e-7).astype(np.float32), "eps")
            values = builder.addInputTensor(popart.TensorInfo("INT32", [2]),
                                            "values")

            values_data = np.array([0, 1]).astype(np.int32)

            # 'Manually' calculate NLLLoss
            sm = builder.aiOnnx.softmax([r0],
                                        axis=np.size(lb_data.shape),
                                        debugPrefix="output")
            lb = builder.aiOnnx.onehot([lb, depth, values],
                                       axis=np.size(lb_data.shape))
            lb = builder.aiOnnx.cast([lb], "FLOAT")

            mul = builder.aiOnnx.mul([sm, lb])
            red = builder.aiOnnx.reducesum([mul],
                                           axes=[np.size(lb_data.shape)],
                                           keepdims=False)
            add = builder.aiOnnx.add([red, eps])
            log = builder.aiOnnx.log([add])
            out = builder.aiOnnx.neg([log])

            losses = [popart.IdentityLoss(out, "loss")]
        else:
            sm = builder.aiOnnx.softmax([r0], axis=np.size(lb_data.shape))
            losses = [popart.NllLoss(sm, lb, "loss")]
        # Output
        builder.addOutputTensor(sm)

        opts = popart.SessionOptions()

        art = popart.AnchorReturnType("All")
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(1, {
                "loss": art,
                w0: art,
                "label": art
            }),
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            losses=losses,
            patterns=popart.Patterns(popart.PatternsLevel.Default),
            deviceInfo=tu.create_test_device(),
            userOptions=opts)

        session.prepareDevice()
        session.weightsFromHost()

        anchors = session.initAnchorArrays()
        if id_loss:
            stepio = popart.PyStepIO(
                {
                    ip: ip_data,
                    "label": lb_data.astype(np.int32),
                    values: values_data
                }, anchors)
        else:
            stepio = popart.PyStepIO(
                {
                    ip: ip_data,
                    "label": lb_data.astype(np.int32)
                }, anchors)

        return session, stepio, anchors

    dshape = [Batchsize, 2, 4, 4]
    lshape = [Batchsize]
    wshape = [2, 2, 3, 3]

    ip_data = np.random.random_sample(size=dshape).astype(np.float32)
    lb_data = np.random.randint(Classes, size=lshape)
    w_data = np.random.random_sample(size=wshape).astype(np.float32)

    ###
    # Pytorch
    ###
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(2, 2, 3, padding=[1, 1], bias=False)
            self.conv.weight.data = torch.tensor(w_data)
            self.sm = nn.Softmax(dim=np.size(lb_data.shape))

        def forward(self, x, y):
            x = self.conv(x)
            x = torch.reshape(x, [Batchsize, Classes])
            x = self.sm(x)
            # Manual calculation of Nll loss. Pytorch's reduction is different to
            # popart, so we calculate manually.
            x = torch.mul(x, y)
            x = torch.sum(x, dim=[np.size(lb_data.shape)])
            x = torch.log(x)
            x = -1 * x
            return x

    net = Net()
    criterion = nn.Identity(reduction="sum")
    optimizer = optim.SGD(net.parameters(),
                          lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY)
    input_ = torch.tensor(ip_data, requires_grad=True)
    # No 'onehot' op in pytorch so send in onehot tensor as input.
    onehot = np.eye(Classes)[lb_data]
    label = torch.tensor(onehot, requires_grad=False)

    ###
    # Compare
    ###

    id_sess, id_steio, id_anchors = get_model(ip_data,
                                              lb_data,
                                              w_data,
                                              id_loss=True)
    n_sess, n_stepio, n_anchors = get_model(ip_data,
                                            lb_data,
                                            w_data,
                                            id_loss=False)

    for i in range(5):
        # Pytorch
        optimizer.zero_grad()
        outputs = net(input_, label)
        loss = criterion(torch.sum(outputs))
        loss.backward()
        optimizer.step()
        # Popart
        id_sess.run(id_steio)
        n_sess.run(n_stepio)
        print(f"Step {i}")
        print("ID Loss:", id_anchors["loss"].sum())
        print("Normal Loss:", n_anchors["loss"].sum())
        print("Pytorch Loss:", loss.item())
        print("ID weight:", id_anchors["weight"].sum())
        print("Normal weight:", n_anchors["weight"].sum())
        # Checks

        assert (id_anchors["loss"].sum() - n_anchors["loss"].sum()) < 1e-4
        assert (id_anchors["loss"].sum() - loss.item()) < 1e-4

        result = getTensorError(id_anchors["loss"], n_anchors["loss"])
        result_w = getTensorError(id_anchors["weight"], n_anchors["weight"])
        checkResult(result, 1e-8)
        checkResult(result_w, 1e-8)


def test_id_l1loss_train():
    # fix the random seed for this test
    np.random.seed(0)
    ## input data
    Batchsize = 4
    ExtraDim = 7
    Classes = 32

    def get_model(ip_data, w_data, id_loss):

        ###
        # Popart
        ###
        builder = popart.Builder()
        # Prepare input data
        ip = builder.addInputTensor(popart.TensorInfo("FLOAT", ip_data.shape),
                                    "input")
        w0 = builder.addInitializedInputTensor(w_data, "weight")

        c0 = builder.aiOnnx.conv([ip, w0],
                                 dilations=[1, 1],
                                 pads=[1, 1, 1, 1],
                                 strides=[1, 1],
                                 debugPrefix="conv")

        r0 = builder.reshape_const(builder.aiOnnx, [c0], [Batchsize, Classes])
        out = builder.aiOnnx.relu([r0], "relu")

        if id_loss:
            redl1 = builder.aiOnnx.reducel1([out], axes=[1], keepdims=False)
            losses = [popart.IdentityLoss(redl1, "loss")]
        else:
            losses = [popart.L1Loss(out, "loss", 1.0)]
        # Output
        builder.addOutputTensor(out)

        opts = popart.SessionOptions()

        art = popart.AnchorReturnType("All")
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFeed=popart.DataFlow(
                1, {
                    "loss": art,
                    w0: art,
                    out: art,
                    popart.reservedGradientPrefix() + out: art
                }),
            optimizer=popart.ConstSGD(LEARNING_RATE),
            losses=losses,
            patterns=popart.Patterns(popart.PatternsLevel.All),
            deviceInfo=tu.create_test_device(),
            userOptions=opts)

        session.prepareDevice()
        session.weightsFromHost()

        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({
            ip: ip_data,
        }, anchors)

        return session, stepio, anchors, out

    dshape = [Batchsize, 2, 4, 4]

    ip_data = np.random.random_sample(size=dshape).astype(np.float32)
    w_data = np.ones([2, 2, 3, 3]).astype(np.float32)

    ###
    # Pytorch
    ###
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv = nn.Conv2d(2, 2, 3, padding=[1, 1], bias=False)
            self.conv.weight.data = torch.tensor(w_data)
            self.relu = nn.ReLU()
            self.l1 = nn.L1Loss(reduction="sum")

        def forward(self, x, y):
            x = self.conv(x)
            x = torch.reshape(x, [Batchsize, Classes])
            x = self.relu(x)
            x = self.l1(x, y)
            return x

    net = Net()
    criterion = nn.Identity(reduction='sum')
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
    input_ = torch.tensor(ip_data, requires_grad=True)
    target = torch.tensor(np.zeros(shape=[Batchsize, Classes]).astype(
        np.float32),
                          requires_grad=False)

    ###
    # Compare
    ###

    id_sess, id_steio, id_anchors, out = get_model(ip_data,
                                                   w_data,
                                                   id_loss=True)
    n_sess, n_stepio, n_anchors, out = get_model(ip_data,
                                                 w_data,
                                                 id_loss=False)

    for i in range(5):
        # Pytorch
        optimizer.zero_grad()
        outputs = net(input_, target)
        loss = criterion(outputs)
        loss.backward()
        optimizer.step()
        # Popart
        id_sess.run(id_steio)
        n_sess.run(n_stepio)
        print(f"Step {i}")
        print("ID Loss:", id_anchors["loss"].mean())
        print("Normal Loss:", n_anchors["loss"].mean())
        print("Pytorch Loss:", loss.item() / Batchsize)
        # Checks
        assert (id_anchors["loss"].mean() - n_anchors["loss"].mean()) < 1e-4
        assert (id_anchors["loss"].mean() - (loss.item() / Batchsize)) < 1e-4

        result = getTensorError(id_anchors["loss"], n_anchors["loss"])
        result_w = getTensorError(id_anchors["weight"], n_anchors["weight"])

        checkResult(result, 1e-8)
        checkResult(result_w, 1e-8)
