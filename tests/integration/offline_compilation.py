# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
import numpy as np
import os
import popart
import tempfile
import pytest
import glob


class TestOfflineCompilation:
    def _init_data(self, data_type=np.float32):
        self.data_a = np.random.rand(3, 2).astype(data_type)
        self.data_b = np.random.rand(3, 2).astype(data_type)

    def _init_builder(self, data_type="FLOAT"):
        self.builder = popart.Builder()

        self.data_shape = popart.TensorInfo(data_type, [2])

        self.input_a = self.builder.addInputTensor(self.data_shape)
        self.input_b = self.builder.addInputTensor(self.data_shape)

        self.output = self.builder.aiOnnx.add([self.input_a, self.input_b])
        self.builder.addOutputTensor(self.output)

    def _init_session(self):
        self._init_builder()

        device = popart.DeviceManager().createOfflineIPUDevice({})

        self.session = popart.InferenceSession(
            fnModel=self.builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {self.output: popart.AnchorReturnType("All")}),
            deviceInfo=device,
        )

    def test_compileAndExport_model(self):
        self._init_builder()
        device = popart.DeviceManager().createIpuModelDevice({})
        session = popart.InferenceSession(
            fnModel=self.builder.getModelProto(),
            dataFlow=popart.DataFlow(1, {self.output: popart.AnchorReturnType("All")}),
            deviceInfo=device,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            with pytest.raises(popart.popart_exception) as e:
                session.compileAndExport(tmpdirname)
            assert "Executables for device type ipu-model cannot be saved" in str(
                e.value
            )

    def test_compileAndExport_offline_ipu_dir(self):
        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            assert os.path.isdir(tmpdirname)
            self.session.compileAndExport(tmpdirname)
            files = glob.glob(f"{tmpdirname}/*")
            assert len(files) == 1, "Expected exactly 1 file"

    def test_compileAndExport_offline_ipu_file(self):
        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            assert os.path.isdir(tmpdirname)
            model_file = os.path.join(tmpdirname, "subfolder", "my_model.popart")
            self.session.compileAndExport(model_file)
            assert os.path.exists(model_file)
