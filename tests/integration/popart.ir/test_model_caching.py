# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import popart.ir as pir
import popart.ir.ops as ops
from popart.ir import dtypes
import numpy as np
import shutil

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu


def build_and_run(cache_path):
    ir = pir.Ir()
    main = ir.main_graph
    with main:
        # Test host to device
        x_h2d = pir.h2d_stream((1, ), pir.float32, name='x_stream')
        x = ops.host_load(x_h2d, 'x')

        # Test variable
        y = pir.variable(4.0, name='y')
        z = ops.add(x, y)

        # Test random op
        seed_h2d = pir.h2d_stream(shape=(2, ),
                                  dtype=dtypes.uint32,
                                  name='seed_stream')
        seed = ops.host_load(seed_h2d, 'seed')
        r = ops.random_normal(seed, (1, ))

        z = ops.add(z, r)

        # Test device to host
        z_d2h = pir.d2h_stream(z.shape, z.dtype, name="z_stream")
        ops.host_store(z_d2h, z)

    # Create seed
    parent_seed = 1984
    seed_tensors = pir.create_seeds(parent_seed, batches_per_step=1)

    ## Run the program
    ir = ir._pb_ir  # Internal ir

    dataFlow = popart.DataFlow(
        batchesPerStep=1,
        anchorTensors={z_d2h.tensor_id: popart.AnchorReturnType("All")})
    ir.setDataFlow(dataFlow)
    ir.updateVertices()

    opts = ir.getSessionOptions()
    opts.useHostCopyOps = True
    opts.enableExplicitMainLoops = True
    opts.aliasZeroCopy = True
    opts.explicitRecomputation = True

    # Enable engine caching
    opts.enableEngineCaching = True
    opts.cachePath = cache_path

    device = tu.create_test_device()
    session = popart.InferenceSession.fromIr(ir=ir, deviceInfo=device)

    session.prepareDevice()

    # Create buffers for anchors
    anchors = session.initAnchorArrays()

    inputs = {
        x_h2d.tensor_id: np.array(3.0, dtype='float32'),
        seed_h2d.tensor_id: seed_tensors,
    }

    # Run the model
    stepio = popart.PyStepIO(inputs=inputs, outputs=anchors)
    session.weightsFromHost()
    session.run(stepio)
    output = anchors['z_stream']

    device.detach()

    return output


@tu.requires_ipu
def test_model_caching(tmp_path, capfd):

    # Check the log output to see if an engine was compiled,
    # or if a cached engine was used.
    def loaded_saved_executable():
        _, stderr = capfd.readouterr()
        startedEngineCompilation = False
        loadedPoplarExecutable = False
        for line in stderr.splitlines():
            if 'Starting compilation' in line:
                startedEngineCompilation = True
            elif 'Loading serialized PopART executable' in line:
                loadedPoplarExecutable = True

        assert startedEngineCompilation != loadedPoplarExecutable
        return startedEngineCompilation

    popart.getLogger().setLevel('DEBUG')
    cache_path = str(tmp_path / 'model_caching')

    result0 = build_and_run(cache_path)
    assert loaded_saved_executable() is True

    result1 = build_and_run(cache_path)
    assert loaded_saved_executable() is False
    assert result0 == result1

    shutil.rmtree(cache_path)

    build_and_run(cache_path)
    assert loaded_saved_executable() is True
