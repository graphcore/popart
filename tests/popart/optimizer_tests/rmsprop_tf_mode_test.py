# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import numpy as np
import pytest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu
import rmsprop_update_numpy as rpnp

adaptive_modes = [
    popart.AdaptiveMode.RMSProp,
    popart.AdaptiveMode.CenteredRMSProp,
]


@pytest.mark.parametrize("use_tf_variant", [True, False])
@pytest.mark.parametrize("const_lr", [True, False])
@pytest.mark.parametrize("adaptive_mode", adaptive_modes)
@pytest.mark.parametrize("momentum", [0.0, 0.9])
@pytest.mark.parametrize("weight_decay", [0.0, 1e-5])
@pytest.mark.parametrize("const_weight_decay", [True, False])
def test_rmsprop_tf_mode(use_tf_variant, const_lr, adaptive_mode, momentum,
                         weight_decay, const_weight_decay):
    np.random.seed(0)
    input_dim = 3
    num_steps = 15
    batches_per_step = 10
    samples_per_batch = 8
    # Optimizer parameters.
    learning_rates = np.linspace(0.02, 0.00001, num_steps)
    weight_decays = np.linspace(weight_decay, weight_decay + 0.01, num_steps)
    alpha = 0.95
    eps = 0.001

    # Initial weights and inputs.
    w0_data = np.random.randn(input_dim, input_dim).astype(np.float32)
    w1_data = np.random.randn(input_dim, input_dim).astype(np.float32)
    input_data = [
        np.random.randn(
            batches_per_step,
            samples_per_batch,
            input_dim,
            input_dim,
        ).astype(np.float32) for _ in range(num_steps)
    ]

    # Build the model.
    #
    # input  w0     w1
    #     \  |      |
    #      mul  -  add - L1 loss
    builder = popart.Builder()
    input = builder.addInputTensor(
        popart.TensorInfo("FLOAT", [samples_per_batch, input_dim, input_dim]))
    w0 = builder.addInitializedInputTensor(w0_data)
    w1 = builder.addInitializedInputTensor(w1_data)
    mm0 = builder.aiOnnx.mul([input, w0])
    mm1 = builder.aiOnnx.add([mm0, w1])
    l1 = builder.aiGraphcore.l1loss([mm1], 1.0)

    dataflow = popart.DataFlow(batches_per_step, {})
    session = popart.TrainingSession(
        fnModel=builder.getModelProto(),
        dataFlow=dataflow,
        loss=l1,
        optimizer=get_rmsprop(
            learning_rates[0],
            const_lr,
            alpha,
            momentum,
            weight_decay,
            const_weight_decay,
            eps,
            adaptive_mode,
            use_tf_variant,
        ),
        deviceInfo=tu.create_test_device(),
    )
    anchor_arrays = session.initAnchorArrays()
    session.prepareDevice()
    session.weightsFromHost()

    # Run popart training and retrieve the weights.
    for step in range(num_steps):
        stepio = popart.PyStepIO({input: input_data[step]}, anchor_arrays)
        session.run(stepio)
        if step < num_steps - 1:
            # Update optimizer from host in case lr or wd are non-const.
            need_to_update_optimizer = False
            lr = learning_rates[0]
            wd = weight_decays[0]
            if not const_lr:
                lr = learning_rates[step + 1]
                need_to_update_optimizer = True
            if not const_weight_decay:
                wd = weight_decays[step + 1]
                need_to_update_optimizer = True
            if need_to_update_optimizer:
                session.updateOptimizerFromHost(
                    get_rmsprop(
                        lr,
                        const_lr,
                        alpha,
                        momentum,
                        wd,
                        const_weight_decay,
                        eps,
                        adaptive_mode,
                        use_tf_variant,
                    ))

    session.weightsToHost()
    w0_popart = np.zeros((input_dim, input_dim), dtype=np.float32)
    w1_popart = np.zeros((input_dim, input_dim), dtype=np.float32)
    weights_read = popart.PyWeightsIO({w0: w0_popart, w1: w1_popart})
    session.readWeights(weights_read)

    # Run numpy training.
    centered = adaptive_mode == popart.AdaptiveMode.CenteredRMSProp
    w0_np = w0_data.copy()
    w1_np = w1_data.copy()
    mg0 = np.zeros(w0_np.shape, dtype=w0_np.dtype)
    mg1 = np.zeros(w1_np.shape, dtype=w1_np.dtype)
    rms0 = np.ones(w0_np.shape, dtype=w0_np.dtype)
    rms1 = np.ones(w1_np.shape, dtype=w1_np.dtype)
    mom0 = np.zeros(w0_np.shape, dtype=w0_np.dtype)
    mom1 = np.zeros(w1_np.shape, dtype=w1_np.dtype)

    for step in range(num_steps):
        lr = learning_rates[0] if const_lr else learning_rates[step]
        wd = weight_decays[0] if const_weight_decay else weight_decays[step]
        for batch in range(batches_per_step):
            w0_grad = np.zeros(w0_np.shape, dtype=w0_np.dtype)
            w1_grad = np.zeros(w1_np.shape, dtype=w1_np.dtype)

            for sample in range(samples_per_batch):
                x = input_data[step][batch][sample]
                w0_grad_sample, w1_grad_sample = model_grad(w0_np, w1_np, x)
                w0_grad += (1.0 / samples_per_batch) * w0_grad_sample
                w1_grad += (1.0 / samples_per_batch) * w1_grad_sample

            w0_np, mg0, rms0, mom0 = rpnp.rmsprop_update_numpy(
                w0_np,
                w0_grad,
                mg0,
                rms0,
                mom0,
                lr,
                alpha,
                momentum,
                wd,
                eps,
                centered=centered,
            )
            w1_np, mg1, rms1, mom1 = rpnp.rmsprop_update_numpy(
                w1_np,
                w1_grad,
                mg1,
                rms1,
                mom1,
                lr,
                alpha,
                momentum,
                wd,
                eps,
                centered=centered,
            )

    # Compare the resulting paramaters.
    if use_tf_variant:
        np.testing.assert_allclose(w0_popart, w0_np, rtol=1e-02, atol=1e-05)
        np.testing.assert_allclose(w1_popart, w1_np, rtol=1e-02, atol=1e-05)
    else:
        assert not np.allclose(w0_popart, w0_np, rtol=1e-02, atol=1e-05)
        assert not np.allclose(w1_popart, w1_np, rtol=1e-02, atol=1e-05)


def get_rmsprop(lr, const_lr, alpha, momentum, weight_decay,
                const_weight_decay, eps, mode, tf_variant):
    return popart.Adaptive(
        {
            'defaultLearningRate': (lr, const_lr),
            'defaultAlpha': (alpha, True),
            'defaultMomentum': (momentum, True),
            'defaultEps': (eps, True),
            'defaultWeightDecay': (weight_decay, const_weight_decay),
        },
        mode=mode,
        weight_decay_mode=popart.WeightDecayMode.L2Regularization,
        rmsprop_tf_variant=tf_variant,
    )


def model_grad(w0, w1, x):
    num_elems = w0.shape[0] * w0.shape[1]
    fwd_act = x * w0 + w1
    w0_grad = x.copy() / num_elems
    w0_grad[fwd_act < 0] *= -1
    w1_grad = np.ones(w1.shape) / num_elems
    w1_grad[fwd_act < 0] *= -1
    return w0_grad, w1_grad
