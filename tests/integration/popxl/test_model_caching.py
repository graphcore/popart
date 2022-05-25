# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import popart
import popxl
import popxl.ops as ops
from popxl import dtypes
import numpy as np
import shutil


def build_and_run(cache_path, engine_options=None):
    ir = popxl.Ir()
    main = ir.main_graph
    with main:
        # Test host to device
        x_h2d = popxl.h2d_stream((1, ), popxl.float32, name='x_stream')
        x = ops.host_load(x_h2d, 'x')

        # Test variable
        y = popxl.variable(4.0, name='y')
        z = ops.add(x, y)

        # Test random op
        seed_h2d = popxl.h2d_stream(shape=(2, ),
                                    dtype=dtypes.uint32,
                                    name='seed_stream')
        seed = ops.host_load(seed_h2d, 'seed')
        r = ops.random_normal(seed, (1, ))

        z = ops.add(z, r)

        # Test device to host
        z_d2h = popxl.d2h_stream(z.shape, z.dtype, name="z_stream")
        ops.host_store(z_d2h, z)

    # Create seed
    parent_seed = 1984
    seed_tensors = popxl.create_seeds(parent_seed, batches_per_step=1)

    opts = ir._pb_ir.getSessionOptions()

    if engine_options is not None:
        for k, v in engine_options.items():
            opts.engineOptions[k] = v

    # Enable engine caching
    opts.enableEngineCaching = True
    opts.cachePath = cache_path

    session = popxl.Session(ir, "ipu_hw")

    with session:
        outputs = session.run({
            x_h2d: np.array(3.0, dtype='float32'),
            seed_h2d: seed_tensors
        })

    return outputs[z_d2h]


def loaded_saved_executable(capfd):
    """ Check the log output to see if an engine was compiled, or if a cached
        engine was used. """
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


def test_model_caching(tmp_path, capfd):
    """ Test if the first time we run a model we get a cache miss, and the
    second time we get a cache hit. """

    popart.getLogger().setLevel('DEBUG')
    cache_path = str(tmp_path / 'model_caching')

    result0 = build_and_run(cache_path)
    assert loaded_saved_executable(capfd) is True

    result1 = build_and_run(cache_path)
    assert loaded_saved_executable(capfd) is False
    assert result0 == result1

    shutil.rmtree(cache_path)

    build_and_run(cache_path)
    assert loaded_saved_executable(capfd) is True


def test_model_caching_miss_on_engine_option_change(tmp_path, capfd):
    """ Test that if we change engine options that affect the executable between
    runs then we don't get a cache hit. """

    popart.getLogger().setLevel('DEBUG')
    cache_path = str(tmp_path / 'model_caching')

    result0 = build_and_run(cache_path,
                            engine_options={"opt.enableInlining": "false"})
    assert loaded_saved_executable(capfd) is True

    result1 = build_and_run(cache_path,
                            engine_options={"opt.enableInlining": "true"})
    assert loaded_saved_executable(capfd) is True
    assert result0 == result1
