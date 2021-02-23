# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import math
import numpy as np
import popart
import pytest
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from op_tester import op_tester

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

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


def get_torch_reduction_type(popart_reduction_type):
    if popart_reduction_type == popart.ReductionType.Mean:
        return "mean"

    if popart_reduction_type == popart.ReductionType.Sum:
        return "sum"

    if popart_reduction_type == popart.ReductionType.NoReduction:
        return "none"


def get_pytorch_equivalent_loss(torch_loss_fn,
                                popart_reduction_type,
                                loss_inputs,
                                extra_args={}):

    reduction = get_torch_reduction_type(popart_reduction_type)
    return torch_loss_fn(reduction=reduction, **extra_args)(*loss_inputs)


def get_pytorch_equivalent_identity_loss(popart_reduction_type, input):
    if popart_reduction_type == popart.ReductionType.Sum:
        return input.sum()
    elif popart_reduction_type == popart.ReductionType.Mean:
        return input.mean()
    else:
        assert popart_reduction_type == popart.ReductionType.Sum


def popart_reduction_type(str):
    if str == "mean":
        return popart.ReductionType.Mean
    if str == "sum":
        return popart.ReductionType.Sum
    assert str == "none"
    return popart.ReductionType.NoReduction


def run_3d_nll_loss_input(popart_reduction_type, with_patterns):
    # fix the random seed for this test
    np.random.seed(0)
    ## input data
    Batchsize = 2
    ExtraDim = 4  # e.g. sequence length in language model
    Classes = 3

    dshape = [Batchsize, ExtraDim, Classes]
    lshape = [Batchsize, ExtraDim]
    flat_lshape = [Batchsize * ExtraDim]

    ip_data = np.random.rand(*dshape).astype(np.float32)
    lb_data = np.random.randint(Classes, size=lshape)

    ###
    # Popart
    ###
    builder = popart.Builder()
    ip = builder.addInitializedInputTensor(ip_data)
    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))

    nll0 = builder.aiGraphcore.nllloss([out, lb], popart_reduction_type)

    patterns = (popart.Patterns(
        popart.PatternsLevel.All) if with_patterns else popart.Patterns(
            popart.PatternsLevel.NoPatterns).enableRuntimeAsserts(False))

    if popart_reduction_type == popart.ReductionType.NoReduction:
        with pytest.raises(popart.popart_exception) as e_info:
            popart.TrainingSession(fnModel=builder.getModelProto(),
                                   dataFlow=popart.DataFlow(1, [nll0, out]),
                                   optimizer=popart.ConstSGD(
                                       LEARNING_RATE, WEIGHT_DECAY),
                                   loss=nll0,
                                   patterns=patterns,
                                   deviceInfo=tu.create_test_device())
        assert (e_info.value.args[0].endswith("must be a scalar tensor"))
        return
    else:
        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFlow=popart.DataFlow(
                                             1, [nll0, out]),
                                         optimizer=popart.ConstSGD(
                                             LEARNING_RATE, WEIGHT_DECAY),
                                         loss=nll0,
                                         patterns=patterns,
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
    output = get_pytorch_equivalent_loss(
        torch.nn.NLLLoss, popart_reduction_type, [logsm, target])

    ###
    # Compare
    ###
    print("Torch loss\n:", output.data.numpy())
    print("Popart loss\n:", anchors[nll0])

    result = getTensorError(output.data.numpy(), anchors[nll0])
    checkResult(result, 1e-8)


def run_nll_loss_with_ignored_index(popart_reduction_type, with_patterns):
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
    nll = builder.aiGraphcore.nllloss([out, lb],
                                      ignoreIndex=ignoreInd,
                                      reduction=popart_reduction_type)

    patterns = (popart.Patterns(
        popart.PatternsLevel.All) if with_patterns else popart.Patterns(
            popart.PatternsLevel.NoPatterns).enableRuntimeAsserts(False))

    if popart_reduction_type == popart.ReductionType.NoReduction:
        with pytest.raises(popart.popart_exception) as e_info:
            session = popart.TrainingSession(
                fnModel=builder.getModelProto(),
                dataFlow=popart.DataFlow(1, [nll]),
                optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
                loss=nll,
                patterns=patterns,
                deviceInfo=tu.create_test_device())
        assert (e_info.value.args[0].endswith("must be a scalar tensor"))
        return
    else:
        session = popart.TrainingSession(fnModel=builder.getModelProto(),
                                         dataFlow=popart.DataFlow(1, [nll]),
                                         optimizer=popart.ConstSGD(
                                             LEARNING_RATE, WEIGHT_DECAY),
                                         loss=nll,
                                         patterns=patterns,
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

    input = torch.tensor(ip_data, requires_grad=True)
    target = torch.tensor(lb_data)
    sm_out = softmax(input)
    logsm = torch.log(sm_out)
    output = get_pytorch_equivalent_loss(
        torch.nn.NLLLoss,
        popart_reduction_type, [logsm, target],
        extra_args={'ignore_index': ignoreInd})

    ###
    # Compare
    ###
    torch_loss = output.data.numpy()
    popart_loss = anchors[nll]

    result = getTensorError(torch_loss, popart_loss)
    checkResult(result, 1e-8)


def run_nll_loss_grad_with_ignored_index(popart_reduction_type):
    # fix the random seed for this test
    np.random.seed(0)
    ## input data
    Batchsize = 3
    Classes = 8

    dshape = [Batchsize, Classes]
    lshape = [Batchsize]

    ip_data = np.random.rand(*dshape).astype(np.float32)
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
    nll = builder.aiGraphcore.nllloss([out, lb],
                                      ignoreIndex=ignoreInd,
                                      reduction=popart_reduction_type)

    ## 2 sessions: one with "SoftmaxGradDirect" pattern, one without
    def getPreparesSession(patterns):
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1,
                                     [popart.reservedGradientPrefix() + ip]),
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            loss=nll,
            patterns=patterns,
            deviceInfo=tu.create_test_device())

        session.prepareDevice()
        session.weightsFromHost()

        return session

    inputs = {lb: lb_data.astype(np.int32)}

    # 1)
    session_SMD = getPreparesSession(
        popart.Patterns(["PreUniRepl",
                         "SoftmaxGradDirect"]).enableRuntimeAsserts(False))
    anchors_SMD = session_SMD.initAnchorArrays()
    stepio_SMD = popart.PyStepIO(inputs, anchors_SMD)
    session_SMD.run(stepio_SMD)

    # 2)
    session_NoSMD = getPreparesSession(
        popart.Patterns(["PreUniRepl"]).enableRuntimeAsserts(False))
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

    input = torch.tensor(ip_data, requires_grad=True)
    target = torch.tensor(lb_data)
    sm_out = softmax(input)
    sm_out.register_hook(set_grad(sm_out))
    logsm = torch.log(sm_out)
    output = get_pytorch_equivalent_loss(
        torch.nn.NLLLoss,
        popart_reduction_type, [logsm, target],
        extra_args={'ignore_index': ignoreInd})

    output.sum().backward(retain_graph=True)

    ###
    # Compare
    ###
    torch_ip_grad = input.grad.numpy()
    px_smd_ip_grad = anchors_SMD[popart.reservedGradientPrefix() + ip]
    px_no_smd_ip_grad = anchors_NoSMD[popart.reservedGradientPrefix() + ip]

    checkResult(getTensorError(torch_ip_grad, px_smd_ip_grad), 1e-8)
    checkResult(getTensorError(torch_ip_grad, px_no_smd_ip_grad), 1e-8)


def test_nll_loss_input_with_invalid_input():
    # fix the random seed for this test
    np.random.seed(0)

    ## input data
    Batchsize = 2
    ExtraDim = 4  # e.g. sequence length in language model
    Classes = 3

    dshape = [Batchsize, ExtraDim, Classes]
    lshape = [Batchsize, ExtraDim + 1]  # Doesn't match!

    ip_data = np.random.rand(*dshape).astype(np.float32)
    lb_data = np.random.randint(Classes, size=lshape)

    ###
    # Popart
    ###
    builder = popart.Builder()
    ip = builder.addInitializedInputTensor(ip_data)
    lb = builder.addInputTensor(popart.TensorInfo("INT32", lshape))
    out = builder.aiOnnx.softmax([ip], axis=np.size(lshape))

    nll0 = builder.aiGraphcore.nllloss([out, lb],
                                       popart.ReductionType.NoReduction)

    patterns = popart.PatternsLevel.NoPatterns

    with pytest.raises(popart.popart_exception) as e_info:
        session = popart.TrainingSession(
            fnModel=builder.getModelProto(),
            dataFlow=popart.DataFlow(1, [nll0]),
            optimizer=popart.ConstSGD(LEARNING_RATE, WEIGHT_DECAY),
            loss=nll0,
            patterns=popart.Patterns(patterns).enableRuntimeAsserts(False),
            deviceInfo=tu.create_test_device())

    assert (e_info.value.args[0].startswith(
        "The label tensor (INT32   [2 5]) must have shape [2 4] to match all but the final dimension of the probabilities tensor (FLOAT   [2 4 3])"
    ))


def run_all_combinations(test_fn):
    for reduction in (popart.ReductionType.Mean,
                      popart.ReductionType.NoReduction,
                      popart.ReductionType.Sum):
        for patterns in (False, True):
            print(reduction)
            print(patterns, flush=True)

            test_fn(reduction, patterns)


def test_3d_nll_loss_input():
    run_all_combinations(run_3d_nll_loss_input)


def test_nll_loss_with_ignored_index():
    run_all_combinations(run_nll_loss_with_ignored_index)


def test_nll_loss_grad_with_ignored_index():
    run_nll_loss_grad_with_ignored_index(popart.ReductionType.Mean)
    run_nll_loss_grad_with_ignored_index(popart.ReductionType.Sum)


@pytest.mark.parametrize("ignore_index", (None, 1))
def test_loss_scaling(ignore_index, op_tester):
    nll_scale = 1.2
    l1_scale = 1.5

    for popart_reduction_type in (popart.ReductionType.Mean,
                                  popart.ReductionType.Sum):

        ## input data
        Batchsize = 2
        Classes = 3

        dshape = [Batchsize, Classes]
        lshape = [Batchsize]
        flat_lshape = [Batchsize]

        ip_data = np.random.rand(*dshape).astype(np.float32)
        lb_data = np.random.randint(Classes, size=lshape)

        def init_builder(builder):
            ip = builder.addInitializedInputTensor(ip_data)
            lb = builder.addInputTensor(lb_data.astype(np.int32))

            sm = builder.aiOnnx.softmax([ip], axis=np.size(lshape))
            nll = builder.aiGraphcore.nllloss([sm, lb],
                                              reduction=popart_reduction_type,
                                              ignoreIndex=ignore_index)
            nll_scaled = builder.aiGraphcore.scale([nll], nll_scale)

            l1 = builder.aiGraphcore.l1loss([ip],
                                            1.0,
                                            reduction=popart_reduction_type)
            l1_scaled = builder.aiGraphcore.scale([l1], l1_scale)

            out = builder.aiOnnx.add([nll_scaled, l1_scaled])
            builder.addOutputTensor(out)

            result = [
                out,
                popart.reservedGradientPrefix() + ip,
                popart.reservedGradientPrefix() + out
            ]
            return result

        def reference(ref_data):
            input = torch.tensor(ip_data, requires_grad=True)
            target = torch.tensor(lb_data, requires_grad=False)

            logsm = torch.nn.LogSoftmax()(input)
            extra_args = {'ignore_index': ignore_index} if ignore_index else {}
            nll = get_pytorch_equivalent_loss(torch.nn.NLLLoss,
                                              popart_reduction_type,
                                              [logsm, target],
                                              extra_args=extra_args)
            nll_scaled = nll * nll_scale

            l1 = get_pytorch_equivalent_loss(
                torch.nn.L1Loss, popart_reduction_type,
                [input, torch.zeros_like(input)])
            l1_scaled = l1 * l1_scale

            out = nll_scaled + l1_scaled

            d__o = ref_data.getOutputTensorGrad(0)
            out.backward(torch.tensor(d__o))

            result = [out, input.grad, None]
            return result

        op_tester.setPatterns([], enableRuntimeAsserts=False)
        op_tester.run(init_builder, reference, 'train')


def test_nllloss_reduction_equiv(op_tester):
    dshapes = ([2, 3], [2, 4, 4], [5, 1, 3], [1, 1])
    for dshape in dshapes:
        lshape = dshape[:-1]
        classes = dshape[-1]
        ip_data = np.random.rand(*dshape).astype(np.float32)
        lb_data = np.random.randint(classes, size=lshape)

        def test(patternsList):
            def getAnchors(extraReduction):
                builder = popart.Builder()
                ip = builder.addInitializedInputTensor(ip_data)
                lb = builder.addInputTensor("INT32", lshape)

                sm = builder.aiOnnx.softmax([ip], axis=np.size(lshape))
                if extraReduction == True:
                    nll = builder.aiGraphcore.nllloss(
                        [sm, lb], reduction=popart.ReductionType.NoReduction)
                    loss = builder.aiOnnx.reducesum([nll])
                else:
                    loss = builder.aiGraphcore.nllloss(
                        [sm, lb], reduction=popart.ReductionType.Sum)

                anchors = [popart.reservedGradientPrefix() + ip]
                # Always test 'loss' too, except for when we want to test with
                # the SoftmaxGradDirect pattern, which requires 'loss' to be
                # anchored
                if 'SoftmaxGradDirect' not in patternsList or 'NlllWithSoftmaxGradDirect' in patternsList:
                    anchors.append(loss)

                session = popart.TrainingSession(
                    fnModel=builder.getModelProto(),
                    loss=loss,
                    dataFlow=popart.DataFlow(1, anchors),
                    optimizer=popart.ConstSGD(0.1),
                    deviceInfo=tu.create_test_device(),
                    patterns=popart.Patterns(
                        patternsList).enableRuntimeAsserts(False))
                session.prepareDevice()
                session.weightsFromHost()
                anchors = session.initAnchorArrays()
                stepio = popart.PyStepIO({lb: lb_data.astype(np.int32)},
                                         anchors)
                session.run(stepio)
                return anchors

            # perform sum reduction of individual losses inside nllloss op
            lr_anchors = getAnchors(False)

            # perform sum reduction of individual losses outside nllloss op
            er_anchors = getAnchors(True)

            # check they are equivalent
            for (id0, a0), (id1, a1) in zip(lr_anchors.items(),
                                            er_anchors.items()):
                checkResult(getTensorError(a0, a1), 1e-8)

        test(['PreUniRepl'])  # Nll, NllGrad and SoftmaxGrad Ops
        test(['PreUniRepl', 'SoftmaxGradDirect'])  # SoftmaxGradDirect Op
        test(['PreUniRepl', 'SoftmaxGradDirect',
              'NlllWithSoftmaxGradDirect'])  # NllWithSoftmaxGradDirect Op


@tu.requires_ipu_model
def test_nll_no_underflow():
    dtype = np.float16

    # Input probabilities
    probs_np = np.array(
        [[1., 0., 0., 0., 0.], [0.5, 0.5, 0., 0., 0.],
         [1 / 3.0, 1 / 3.0, 1 / 3.0, 0., 0.], [0.25, 0.25, 0.25, 0.25, 0.],
         [0.2, 0.2, 0.2, 0.2, 0.2]],
        dtype=dtype)

    labels_np = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    builder = popart.Builder()
    probs = builder.addInitializedInputTensor(probs_np, "probs")
    builder.addOutputTensor(builder.aiOnnx.identity([probs]))
    dprobs = popart.reservedGradientPrefix() + probs
    labels = builder.addInputTensor(popart.TensorInfo("INT32", [5]))
    loss = builder.aiGraphcore.nllloss([probs, labels],
                                       popart.ReductionType.Sum,
                                       debugPrefix="nllLossVal")

    anchor_desc = {
        dprobs: popart.AnchorReturnType("ALL"),
        loss: popart.AnchorReturnType("ALL")
    }
    dataFlow = popart.DataFlow(1, anchor_desc)
    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        loss=loss,
        deviceInfo=popart.DeviceManager().createIpuModelDevice({}),
        optimizer=popart.ConstSGD(0.00001),
        dataFlow=dataFlow)
    session.prepareDevice()
    session.weightsFromHost()
    anchors = session.initAnchorArrays()
    stepio = popart.PyStepIO({labels: labels_np}, anchors)
    session.run(stepio)

    assert not np.isnan(anchors[loss]).any()
    assert not np.isnan(anchors[dprobs]).any()


def test_nll_input_is_log_probability(op_tester):
    np.random.seed(0)
    data = np.random.rand(3, 5).astype(np.float32)
    target = np.random.randint(5, size=3)

    def init_builder(builder):
        P = builder.addInitializedInputTensor(data)
        T = builder.addInitializedInputTensor(target.astype(np.int32))
        logP = builder.aiOnnx.logsoftmax([P], axis=1)
        nll = builder.aiGraphcore.nllloss([logP, T], inputIsLogProbability=1)
        builder.addOutputTensor(nll)
        return [nll]

    def reference(ref_data):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.nn.NLLLoss()
        p = torch.tensor(data)
        t = torch.tensor(target)
        nll = loss(logsoftmax(p), t)
        return [nll]

    op_tester.run(init_builder, reference, seed=8)


def test_nll_input_is_log_probability_training(op_tester):
    np.random.seed(0)
    data = np.random.rand(3, 5).astype(np.float32)
    target = np.random.randint(5, size=3)

    def init_builder(builder):
        P = builder.addInitializedInputTensor(data)
        T = builder.addInputTensor(target.astype(np.int32))
        logP = builder.aiOnnx.logsoftmax([P], axis=1)
        nll = builder.aiGraphcore.nllloss([logP, T], inputIsLogProbability=1)
        builder.addOutputTensor(nll)
        return [
            nll,
            popart.reservedGradientPrefix() + P,
            popart.reservedGradientPrefix() + nll,
        ]

    def reference(ref_data):
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.nn.NLLLoss()
        p = torch.tensor(data, requires_grad=True)
        t = torch.tensor(target, requires_grad=False)
        nll = loss(logsoftmax(p), t)
        d__nll = ref_data.getOutputTensorGrad(0)
        nll.backward(torch.tensor(d__nll))
        return [nll, p.grad, None]

    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train', seed=8)


@pytest.mark.parametrize("blank", [0, 1])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_ctc_loss(op_tester, blank, reduction):
    np.random.seed(0)
    reductionTypeMap = {
        "none": popart.ReductionType.NoReduction,
        "mean": popart.ReductionType.Mean,
        "sum": popart.ReductionType.Sum
    }

    # max input length
    T = 6
    # batch size
    N = 4
    # number of classes (including blank)
    C = 3
    # max target length
    S = 2

    # fixed params
    zero_infinity = False

    # [T, N, C] data (not yet logsoftmax'ed)
    logits_data = np.random.rand(T, N, C).astype(np.float32)
    # [N, S] targets (not blank=0)
    targets_data = np.random.randint(1, C, size=(N, S)).astype(np.uint32)
    # Lengths of inputs.
    input_lengths_data = np.full(shape=(N, ), fill_value=T).astype(np.uint32)
    # Lengths of targets.
    target_lengths_data = np.random.randint(0, S + 1,
                                            size=(N, )).astype(np.uint32)

    if blank != 0:
        targets_data = np.where(targets_data == blank, 0,
                                targets_data).astype(np.uint32)

    def init_builder(builder):
        logits = builder.addInputTensor(logits_data)
        targets = builder.addInputTensor(targets_data)
        input_lengths = builder.addInputTensor(input_lengths_data)
        target_lengths = builder.addInputTensor(target_lengths_data)

        log_probs = builder.aiOnnx.logsoftmax([logits], axis=2)

        ctc = builder.aiGraphcore.ctcloss(
            [log_probs, targets, input_lengths, target_lengths],
            reductionTypeMap[reduction], blank)
        builder.addOutputTensor(ctc)

        # NOTE: There are versions of pytorch that include an unnecessary
        # logsoftmax inside the CTC loss. If this is the case the gradient of
        # log_probs would nott match (but the gradient of logits should still
        # match when this is the case as logsoftmax is idempotent).
        return [
            ctc, logits, log_probs,
            popart.reservedGradientPrefix() + logits,
            popart.reservedGradientPrefix() + ctc
        ]

    def reference(ref_data):

        logsoftmax = torch.nn.LogSoftmax(dim=2)
        loss = torch.nn.CTCLoss(blank=blank,
                                reduction=reduction,
                                zero_infinity=zero_infinity)

        logits = torch.tensor(logits_data, requires_grad=True)
        targets = torch.tensor(targets_data.astype(np.int32))
        input_lengths = torch.tensor(input_lengths_data.astype(np.int32))
        target_lengths = torch.tensor(target_lengths_data.astype(np.int32))

        log_probs = logsoftmax(logits)  # [T, N, C]
        log_probs.retain_grad()
        ctc = loss(log_probs, targets, input_lengths, target_lengths)
        ctc.retain_grad()

        d__ctc = ref_data.getOutputTensorGrad(0)
        ctc.backward(torch.tensor(d__ctc))
        return [ctc, logits, log_probs, logits.grad, ctc.grad]

    op_tester.atol = 1e-07
    op_tester.setPatterns([], enableRuntimeAsserts=False)
    op_tester.run(init_builder, reference, step_type='train', seed=8)
