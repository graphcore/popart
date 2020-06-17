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
            dataFlow=popart.DataFlow(
                1, {self.output: popart.AnchorReturnType("All")}),
            deviceInfo=device)

    def test_exportDataset(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_builder()
        self._init_data()

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.builder.exportDataset(
                {
                    self.input_a: self.data_a,
                    self.input_b: self.data_b
                }, 3, os.path.join(tmpdirname, "data.bin"))

            files = glob.glob("%s/*" % tmpdirname)
            assert len(
                files
            ) == 1, "Expected a single 'data.bin' file containing input data"
            assert os.path.basename(files[0]) == "data.bin"

    def test_exportDataset_not_enough_data(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_builder()
        self._init_data()

        with tempfile.TemporaryDirectory() as tmpdirname:
            with pytest.raises(popart.popart_exception) as e:
                self.builder.exportDataset(
                    {
                        self.input_a: self.data_a,
                        self.input_b: self.data_b
                    }, 5, os.path.join(tmpdirname, "data.bin"))
            assert "Unexpectedly reached the end" in str(e.value)

    def test_exportDataset_shapes_mismatch(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_builder()

        self.data_a = np.random.rand(3, 1).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            with pytest.raises(popart.popart_exception) as e:
                self.builder.exportDataset({self.input_a: self.data_a}, 3,
                                           os.path.join(
                                               tmpdirname, "data.bin"))
            assert "The shape provided [1] didn't match the one expected [2]" in str(
                e.value)

    def test_exportDataset_types_mismatch(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_builder("FLOAT")
        self._init_data(np.float16)

        with tempfile.TemporaryDirectory() as tmpdirname:
            with pytest.raises(popart.popart_exception) as e:
                self.builder.exportDataset({self.input_a: self.data_a}, 3,
                                           os.path.join(
                                               tmpdirname, "data.bin"))
            assert "Type discrepency for tensor input" in str(e.value)

    def test_exportDataset_JSON_check(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.session.compileAndExport(tmpdirname)
            files = glob.glob("%s/*" % tmpdirname)
            assert len(
                files) == 2, "Expected 2 files: one 'json' and one 'bin'"
            assert os.path.splitext(files[0])[1] == ".bin"
            assert os.path.splitext(files[1])[1] == ".json"
            assert os.path.splitext(files[1])[0] == os.path.splitext(
                files[0])[0]
            json_file = files[1]

            with tempfile.TemporaryDirectory() as tmpdataset:
                self._init_data()
                self.builder.exportDataset(
                    {
                        self.input_a: self.data_a,
                        self.input_b: self.data_b
                    }, 3, os.path.join(tmpdataset, "data.bin"), json_file)

                files = glob.glob("%s/*" % tmpdataset)
                assert len(
                    files
                ) == 1, "Expected a single 'data.bin' file containing input data"
                assert os.path.basename(files[0]) == "data.bin"

            # Create a new builder that uses INT instead of FLOAT
            self._init_builder("INT32")
            self._init_data(np.int32)

            # Check it works without validation:
            with tempfile.TemporaryDirectory() as tmpdataset:
                self.builder.exportDataset(
                    {
                        self.input_a: self.data_a,
                        self.input_b: self.data_b
                    }, 3, os.path.join(tmpdataset, "data.bin"))
                files = glob.glob("%s/*" % tmpdataset)
                assert len(
                    files
                ) == 1, "Expected a single 'data.bin' file containing input data"
                assert os.path.basename(files[0]) == "data.bin"

            # Check it fails with JSON validation:
            with tempfile.TemporaryDirectory() as tmpdataset:
                with pytest.raises(popart.popart_exception) as e:
                    self.builder.exportDataset(
                        {
                            self.input_a: self.data_a,
                            self.input_b: self.data_b
                        }, 3, os.path.join(tmpdataset, "data.bin"), json_file)
                assert "doesn't match the info from the metadata" in str(
                    e.value)

    def test_exportInputs(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdataset:
            self._init_data()
            self.session.exportInputs(
                {
                    self.input_a: self.data_a,
                    self.input_b: self.data_b
                }, 3, os.path.join(tmpdataset, "data.bin"))

            files = glob.glob("%s/*" % tmpdataset)
            assert len(
                files
            ) == 1, "Expected a single 'data.bin' file containing input data"
            assert os.path.basename(files[0]) == "data.bin"

    def test_compileAndExport_model(self):
        self._init_builder()
        device = popart.DeviceManager().createIpuModelDevice({})
        session = popart.InferenceSession(
            fnModel=self.builder.getModelProto(),
            dataFlow=popart.DataFlow(
                1, {self.output: popart.AnchorReturnType("All")}),
            deviceInfo=device)

        with tempfile.TemporaryDirectory() as tmpdirname:
            with pytest.raises(popart.popart_exception) as e:
                session.compileAndExport(tmpdirname, tmpdirname)
            assert "Offline compilation is not supported" in str(e.value)

    def test_compileAndExport_offline_ipu(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.session.compileAndExport(tmpdirname, tmpdirname)
            assert os.path.exists(tmpdirname)
            assert (not os.path.exists(os.path.join(tmpdirname, "test_file"))
                    ), "The file for testing writing should be deleted"

        # Try calling prepare device after
        with pytest.raises(popart.popart_exception) as e:
            self.session.prepareDevice()
        assert "Cannot run on an offline-ipu" in str(e.value)

    def test_compileAndExport_both_None(self):
        self._init_session()

        with pytest.raises(popart.popart_exception) as e:
            self.session.compileAndExport(None, None)
        assert "At least one" in str(e.value)

    def test_compileAndExport_weights_only(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.session.compileAndExport("", tmpdirname)
            files = glob.glob("%s/*" % tmpdirname)
            assert len(
                files) == 1, "Expected a single 'bin' file containing weights"
            assert os.path.splitext(files[0])[1] == ".bin"

    def test_compileAndExport_executable_only(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.session.compileAndExport(tmpdirname)
            files = glob.glob("%s/*" % tmpdirname)
            assert len(
                files) == 2, "Expected 2 files: one 'json' and one 'bin'"
            assert os.path.splitext(files[0])[1] == ".bin"
            assert os.path.splitext(files[1])[1] == ".json"
            assert os.path.splitext(files[1])[0] == os.path.splitext(
                files[0])[0]

    def test_compileAndExport_both(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory() as tmpdirname:
            self.session.compileAndExport(tmpdirname, tmpdirname)
            files = glob.glob("%s/*" % tmpdirname)
            assert len(files) == 3, "Expected 2 bin + 1 json"

    def test_compileAndExport_separate_folders(self):
        if not popart.exporterIsAvailable():
            pytest.skip("Exporter support needs to be compiled in")

        self._init_session()

        with tempfile.TemporaryDirectory(
        ) as tmpExe, tempfile.TemporaryDirectory() as tmpWeights:
            self.session.compileAndExport(tmpExe, tmpWeights)
            files = glob.glob("%s/*" % tmpExe)
            assert len(
                files) == 2, "Expected 2 files: one 'json' and one 'bin'"
            assert os.path.splitext(files[0])[1] == ".bin"
            assert os.path.splitext(files[1])[1] == ".json"
            assert os.path.splitext(files[1])[0] == os.path.splitext(
                files[0])[0]
            files = glob.glob("%s/*" % tmpWeights)
            assert len(
                files) == 1, "Expected a single 'bin' file containing weights"
            assert os.path.splitext(files[0])[1] == ".bin"
