# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import popart


from abc import ABC
import test_util as tu
import pytest


class BuilderUtils:
    """
    Provides methods to simplify common builder/onnx ops.
    """

    def __init__(self, builder):
        self.builder = builder
        self.aiOnnx = builder.aiOnnx
        self.aiGraphcore = builder.aiGraphcore

    def const(self, t, dtype):
        return self.aiOnnx.constant(np.array(t).astype(dtype))

    def const_int32(self, t):
        return self.const(t, dtype=np.int32)

    def one_hot(self, t, depth):
        return self.aiOnnx.onehot(
            [t, self.const_int32(depth), self.const_int32([0, 1])]
        )

    def matmul(self, lhs, rhs, debugContext):
        return self.aiOnnx.matmul([lhs, rhs], debugContext=debugContext)

    def add(self, lhs, rhs, debugContext):
        return self.aiOnnx.add([lhs, rhs], debugContext=debugContext)

    def softmax(self, t, debugContext):
        return self.aiOnnx.softmax([t], axis=1, debugContext=debugContext)

    def cast(self, t, dtype):
        return self.aiOnnx.cast([t], to=self.get_type_string(dtype))

    def add_initialized_input_tensor(self, tensor_info, debug_context):
        return self.builder.addInitializedInputTensor(tensor_info, debug_context)

    def get_type_string(self, dtype):
        return {
            np.float32: "FLOAT",
            np.float16: "FLOAT16",
            np.int32: "INT32",
            np.uint32: "UINT32",
        }[dtype]


class SwitchFunction(ABC, BuilderUtils):
    """
    Assignment using a linear + softmax layer and top-1 choice
    """

    def __init__(self, builder, FORCE_FP32=True):
        BuilderUtils.__init__(self, builder)
        self.builder = builder
        # Calculated after __call__() is executed
        self.expert_weights = None
        self.balanced_loss = None
        self.assignment_trainable = True

        self.force_fp32 = FORCE_FP32

        if self.force_fp32:
            self.dtype = np.float32
        else:
            self.dtype = np.float16

    def _initialise_params(self):
        W_value = np.random.normal(size=(8, 1)).astype(self.dtype)
        b_value = np.random.normal(size=(1,)).astype(self.dtype)
        if self.assignment_trainable:
            W = self.add_initialized_input_tensor(W_value, "W")
            b = self.add_initialized_input_tensor(b_value, "b")
        else:
            W = self.const(W_value, dtype=self.dtype)
            b = self.const(b_value, dtype=self.dtype)
        return W, b

    def __call__(self, hiddens: str) -> str:
        # hiddens shape [64, 8]

        # Initialise router params
        W, b = self._initialise_params()
        # W shape [8, 1], b shape [1]
        debug_context = ""

        # Cast to fp32
        if self.force_fp32:
            hiddens = self.cast(hiddens, dtype=self.dtype)

        # Calculate token assignments
        router_logits = self.matmul(hiddens, W, debug_context)  # shape [64, 1]
        router_logits = self.add(router_logits, b, debug_context)
        router_probs = self.softmax(router_logits, debug_context)
        token_assignments = self.aiOnnx.argmax(
            [router_logits], axis=-1, keepdims=0
        )  # shape [64]

        # Calculate balancing loss term
        expert_mask = self.one_hot(token_assignments, 1)  # shape [64, 1]
        expert_mask = self.cast(
            expert_mask, dtype=self.dtype
        )  # cast to float for reducemean

        density = self.aiOnnx.reducemean(
            [expert_mask], axes=[0], keepdims=0
        )  # mean of actual assignments
        density_proxy = self.aiOnnx.reducemean(
            [router_probs], axes=[0], keepdims=0
        )  # mean of probabilities

        num_experts_squared = self.const(1.0, dtype=self.dtype)

        # Auxiliary loss
        density_product = self.aiOnnx.mul(
            [density, density_proxy]
        )  # if balanced, each elem 1 / N^2
        density_dot_product = self.aiOnnx.reducemean(
            [density_product], axes=[0], keepdims=0
        )  # if balanced = 1 / N^2
        balanced_loss = density_dot_product
        balanced_loss = self.aiOnnx.mul(
            [balanced_loss, num_experts_squared]
        )  # if balanced = 1

        # Cast back to fp16
        if self.force_fp32:
            balanced_loss = self.cast(balanced_loss, dtype=np.float16)

        self.balanced_loss = balanced_loss

        return token_assignments


@pytest.mark.parametrize("FORCE_FP32", [True, False])
@pytest.mark.parametrize("WEIGHT_FINAL_LOSS", [True, False])
def test_bert_test(FORCE_FP32: bool, WEIGHT_FINAL_LOSS: bool):
    def bert_session_options():
        options = popart.SessionOptions()
        options.virtualGraphMode = popart.VirtualGraphMode.Manual
        options.enableStochasticRounding = True
        partials_type = "half"
        options.partialsTypeMatMuls = partials_type
        options.convolutionOptions = {"partialsType": partials_type}
        options.enablePipelining = True
        options.autoRecomputation = popart.RecomputationType.Pipeline
        options.enableGradientAccumulation = True
        options.accumulationFactor = 8
        options.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        return options

    builder = popart.Builder(opsets={"ai.onnx": 11, "ai.onnx.ml": 1, "ai.graphcore": 1})
    expert_assignment_function = SwitchFunction(builder, FORCE_FP32)
    aux_losses = {}

    # build graph
    # add input tensor
    x = builder.addInputTensor(popart.TensorInfo("FLOAT16", [64, 8]), "input_x")

    # 2 'encoder' layers, 1 on each IPU
    for i in range(2):
        with builder.pipelineStage(i), builder.virtualGraph(i), builder.nameScope("FF"):
            expert_assignment_function(x)
            aux_loss = expert_assignment_function.balanced_loss
            aux_losses[i] = aux_loss
            x = builder.aiOnnx.add([x, x], "")

    # final loss
    with builder.pipelineStage(2), builder.virtualGraph(0), builder.nameScope(
        "FinalAuxLoss"
    ):
        aux_loss = builder.aiOnnx.sum(list(aux_losses.values()), "AuxLoss")
        final_loss = aux_loss
        if WEIGHT_FINAL_LOSS:
            aux_loss_weight = builder.aiOnnx.constant(np.array(1.0).astype(np.float16))
            final_loss = builder.aiOnnx.mul([aux_loss_weight, final_loss])

    # add output anchors
    outputs = {}
    for aux_loss in aux_losses.values():
        outputs[aux_loss] = popart.AnchorReturnType("SUM")

    # popart stuff
    with tu.create_test_device(
        numIpus=2,
        pattern=popart.SyncPattern.SinglePipeline,
        connectionType=popart.DeviceConnectionType.Always,
    ) as device:
        session_kwargs = dict(
            fnModel=builder.getModelProto(),
            loss=final_loss,
            deviceInfo=device,
            optimizer=popart.ConstSGD(0.01),
            dataFlow=popart.DataFlow(1, outputs),
            patterns=popart.Patterns(),
            userOptions=bert_session_options(),
        )
        session = popart.TrainingSession(**session_kwargs)
        session.prepareDevice()
        session.weightsFromHost()
        anchors = session.initAnchorArrays()
        print("FINISHED COMPILATION")

        # training one step
        data = {"input_x": np.random.rand(8, 64, 8).astype(np.float16)}
        stepio = popart.PyStepIO(data, anchors)
        session.run(stepio)
        print("FINISHED RUNNING")
        device.detach()
