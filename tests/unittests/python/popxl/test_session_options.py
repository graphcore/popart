# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

import popxl
import os


def test_cache_env():
    os.environ['POPXL_CACHE_DIR'] = 'PATH_TO_CACHE'

    ir = popxl.Ir()
    opts = ir._pb_ir.getSessionOptions()

    assert opts.enableEngineCaching
    assert opts.cachePath == 'PATH_TO_CACHE'

    del os.environ['POPXL_CACHE_DIR']
