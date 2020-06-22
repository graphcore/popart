# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import popart
import test_util as tu

import numpy as np
import torch


def loss_scaling_test(constLossScaling):

    # In previous implementations of popart loss scaling only worked for
    # nll/l1 losses. This test is to ensure it works for any loss.
    #
    # For a computation matmul(c1,p1) where c1 is a constant matrix and
    # p1 is a parameter, calculate loss as reducesum(matmul(c1,p1)) and
    # update p1 using a optimizer that 'uses' loss scaling. Do this
    # for both pytorch and popart and compare the result.

    np.random.seed(1)
    sgd_learning_rate = 0.1
    sgd_moment = 0.9
    c1_shape = (1, 1)
    p1_shape = (1, 1)
    c1_init = np.random.rand(*c1_shape).astype(np.float32)
    p1_init = np.random.rand(*p1_shape).astype(np.float32)
    out_init = np.matmul(c1_init, p1_init)

    def get_updated_p1_popart():
        builder = popart.Builder()

        # Computation is out = matmul(i1, i2)
        c1 = builder.addInputTensor(popart.TensorInfo("FLOAT", c1_shape))
        p1 = builder.addInitializedInputTensor(p1_init)
        out = builder.aiOnnx.matmul([c1, p1])

        # Set up a training session.
        device = tu.create_test_device()
        dataFlow = popart.DataFlow(
            1, {
                c1: popart.AnchorReturnType("Final"),
                p1: popart.AnchorReturnType("Final"),
                out: popart.AnchorReturnType("Final")
            })

        # We're testing losses other than nll/l1 work.
        loss = builder.aiOnnx.reducesum([out])
        optimizer = popart.SGD({
            "defaultLearningRate": (sgd_learning_rate, True),
            "defaultMomentum": (sgd_moment, False),
            "lossScaling": (200, constLossScaling)
        })
        session = popart.TrainingSession(builder.getModelProto(),
                                         deviceInfo=device,
                                         dataFlow=dataFlow,
                                         loss=loss,
                                         optimizer=optimizer)

        session.prepareDevice()
        session.weightsFromHost()

        # Run the popart session to get an answer.
        anchors = session.initAnchorArrays()
        stepio = popart.PyStepIO({c1: c1_init}, anchors)
        session.run(stepio)
        return anchors[c1], anchors[p1], anchors[out]

    def get_updated_p1_pytorch():

        # Computation is out = matmul(i1, i2)
        c1 = torch.tensor(c1_init, requires_grad=False)
        p1 = torch.tensor(p1_init, requires_grad=True)
        out = torch.matmul(c1, p1)

        # Set up optimizer, compute loss.
        optimizer = torch.optim.SGD([p1],
                                    lr=sgd_learning_rate,
                                    momentum=sgd_moment)
        optimizer.zero_grad()
        loss = torch.sum(out)

        # Compute gradient and optimize.
        loss.backward()
        optimizer.step()

        # Return the tensors.
        return c1.detach().numpy(), p1.detach().numpy(), out.detach().numpy()

    c1_popart, p1_popart, out_popart = get_updated_p1_popart()
    c1_pytorch, p1_pytorch, out_pytorch = get_updated_p1_pytorch()

    # We're not expecting changes in c1 or out, check anyway.
    assert (np.isclose(c1_popart, c1_init))
    assert (np.isclose(c1_pytorch, c1_init))
    assert (np.isclose(out_popart, out_init))
    assert (np.isclose(out_pytorch, out_init))

    # We expect p1 to match.
    assert (np.isclose(
        p1_popart, p1_pytorch
    )), f'Expected p1_popart={p1_popart} to match p1_pytorch={p1_pytorch}'


def test_loss_scaling_with_const():
    loss_scaling_test(True)


def test_loss_scaling_with_nonconst():
    loss_scaling_test(False)
