# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import popart
import popxl
from popxl import ops


class TestWriteVariablesData:
    def test_write_variables_data_no_cache(self):
        ir = popxl.Ir()
        with ir.main_graph:
            v = popxl.variable(1, popxl.float32)
            d2h = popxl.d2h_stream(v.shape, v.dtype)
            ops.host_store(d2h, v)

        with popxl.Session(ir, 'ipu_hw') as sess:
            assert sess.run()[d2h] == 1
            sess.write_variable_data(v, 2)
            assert sess.run()[d2h] == 2
            sess.write_variable_data(v, 3)
            assert sess.run()[d2h] == 3

    def test_write_variables_data_cache(self, tmp_path):
        opts = popart.SessionOptions()
        popart.getLogger().setLevel('DEBUG')
        os.environ['POPART_CACHE_DIR'] = str(tmp_path / 'cache')
        ir = popxl.Ir()
        with ir.main_graph:
            v = popxl.variable(1, popxl.float32)
            d2h = popxl.d2h_stream(v.shape, v.dtype)
            ops.host_store(d2h, v)

        with popxl.Session(ir, 'ipu_hw') as sess:
            assert sess.run()[d2h] == 1
            sess.write_variable_data(v, 2)
            assert sess.run()[d2h] == 2

        del os.environ['POPART_CACHE_DIR']
