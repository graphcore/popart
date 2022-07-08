# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import popart
import pytest
from tempfile import TemporaryDirectory

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def test_reset_host_weights_with_extra_tensor_in_onnx_model():
    """
    1. Create a training session, and a corresponding validation session
    2. The training session must contain some feauture that means when writing
       the ONNX model back to the host, it contains extra initializers compared
       with the original (builder-generated) model. In this case we achieve this
       by using an SGD optimizer with momentum.
    3. Try resetting the weights of the validation session using the ONNX model
       with the additional momentum tensor (call resetHostWeights)
    4. Observe that a PopART exception is thrown
    5. Try again, but with ignoreWeightsInModelWithoutCorrespondingHostWeight.
    6. Observe that it succeeds
    """

    def getModelWithRandomWeights():
        builder = popart.Builder()
        dShape = [2, 2]
        i0 = builder.addInputTensor(popart.TensorInfo("FLOAT", dShape))
        wData = np.random.rand(*dShape).astype(np.float32)
        w0 = builder.addInitializedInputTensor(wData)
        o = builder.aiOnnx.matmul([i0, w0])
        loss = builder.aiGraphcore.l1loss([o], 0.1)
        builder.addOutputTensor(loss)
        return builder

    with tu.create_test_device() as device:
        tr_builder = getModelWithRandomWeights()
        o = tr_builder.getOutputTensorIds()[0]

        # 1. & 2.
        # Training
        tr_opt = popart.SGD({"defaultMomentum": (0.01, True)})
        tr_sess = popart.TrainingSession(
            fnModel=tr_builder.getModelProto(),
            dataFlow=popart.DataFlow(1, []),
            loss=o,
            optimizer=tr_opt,
            deviceInfo=device,
        )
        tr_sess.prepareDevice()
        with TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tr_model.onnx")
            tr_sess.modelToHost(tmpfile)

            # Validation (with different model proto weights)
            va_builder = getModelWithRandomWeights()
            va_opts = popart.SessionOptions()
            va_opts.constantWeights = False
            va_sess = popart.InferenceSession(
                fnModel=va_builder.getModelProto(),
                dataFlow=popart.DataFlow(1, [o]),
                deviceInfo=device,
                userOptions=va_opts,
            )
            va_sess.prepareDevice()

            # 3. Try reset validation weights with training weights
            wId = [
                w for w in va_builder.getInputTensorIds() if va_builder.isInitializer(w)
            ][0]
            missing_tensor_name = popart.reservedAcclPrefix() + wId
            with pytest.raises(popart.popart_exception) as e_info:
                va_sess.resetHostWeights(tmpfile)
            # 4.
            assert (
                e_info.value.args[0]
                == "resetWeights, no tensor '" + missing_tensor_name + "' in tensors"
            )

            # 5. & 6. Try again, but this time ignore the missing tensor
            va_sess.resetHostWeights(
                tmpfile, ignoreWeightsInModelWithoutCorrespondingHostWeight=True
            )
